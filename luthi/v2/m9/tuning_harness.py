"""Staleness / re-eval tuning harness (2026-07-06).

The combined tuning pass — F2 consistency thresholds (staleness.py / kills.py),
the living-band join, the eye source, the resolution decays, and the gain's
rise/cap — is BLOCKED on two things: a trained checkpoint AND a
staleness-producing regime. Fable's §6 re-review (2026-07-05) pinned the second
blocker precisely: a 60-cycle smoke probe produced ZERO natural re-evals,
because at ``mcts_max_depth=1`` the advanced root is childless, the tree
rebuilds each cycle, and node stamps never age past the window — so the
consistency-deviation distribution the F2 thresholds must be tuned against never
forms.

This module removes that blocker with two entries, both checkpoint-AGNOSTIC
(they take an already-built ``M9Trainer`` so the tuning pass can hand over a
trained-checkpoint-loaded one):

  * :func:`run_natural_regime` — the REAL tuning entry. Drives the persistent
    tree under real θ movement at ``mcts_max_depth >= 2`` and collects natural
    re-evals + deviation samples. Honest caveat learned building this: natural
    aging needs a node to survive >= consistency_window cycles WITHOUT
    re-stamping, but ``advance_root`` prunes off-trajectory branches each cycle,
    so on an UNTRAINED policy (which churns the acted child near-randomly) the
    retained subtree is always freshly-stamped and re-evals can be zero. A
    stable TRAINED policy re-traverses the same subtree, so nodes persist and
    age. So this needs the checkpoint not just for realistic deviation VALUES
    but to produce the signal at all — a deeper reason the pass is checkpoint-
    blocked than previously recorded.

  * :func:`run_aged_node_regime` — the aged-node alternative Fable sanctioned
    ("max_depth >= 2 OR an aged-node harness"). Advances ``theta_version``
    between plans without re-planning, so retained nodes age deterministically
    on ANY model. Exercises the full re-eval / priority / budget / deviation /
    failover machinery NOW, ahead of the checkpoint — the plumbing signal, not
    realistic values.

Neither sets any threshold — tuning against real checkpoints is the pass's job,
and choosing the numbers is the designers' call. This only makes the signal
observable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class StalenessRegimeStats:
    """Per-cycle staleness-pass observations collected across a regime run.

    ``deviation_samples`` is the distribution the F2 thresholds
    (failover_consistency_threshold / recovery_consistency_threshold /
    kills.consistency_max) must be set against — a scale-normalized fraction of
    natural re-eval-vs-cached disagreement, meaningful only on a TRAINED model.
    ``reevaluated_per_cycle`` sums to ``total_reevaluated``; its being > 0 is
    the proof the regime produced natural aging (the smoke check).
    """

    reevaluated_per_cycle: list[int] = field(default_factory=list)
    deviation_samples: list[float] = field(default_factory=list)
    failover_cycles: list[int] = field(default_factory=list)
    theta_versions: list[int] = field(default_factory=list)

    def record(self, staleness_pass: dict, cycle: int) -> None:
        reeval = int(staleness_pass.get("reevaluated", 0))
        self.reevaluated_per_cycle.append(reeval)
        dev = staleness_pass.get("median_deviation", None)
        # The producer emits median_deviation=0.0 (not None) on cycles where
        # the re-eval slice ran empty (_staleness_pass's default stats dict),
        # so the reeval > 0 guard is what actually keeps sentinel zeros out of
        # the tuning distribution; the None check covers a pass that never ran
        # (fresh trainer's empty dict). (Fable review 2026-07-15.)
        if dev is not None and reeval > 0:
            self.deviation_samples.append(float(dev))
        if staleness_pass.get("in_failover", False):
            self.failover_cycles.append(cycle)
        self.theta_versions.append(int(staleness_pass.get("theta_version", 0)))

    @property
    def total_reevaluated(self) -> int:
        return sum(self.reevaluated_per_cycle)

    @property
    def any_failover(self) -> bool:
        return bool(self.failover_cycles)

    def summary(self) -> dict:
        n = len(self.deviation_samples)
        mean_dev = sum(self.deviation_samples) / n if n else None
        return {
            "cycles": len(self.reevaluated_per_cycle),
            "total_reevaluated": self.total_reevaluated,
            "deviation_n": n,
            "deviation_mean": mean_dev,
            "deviation_min": min(self.deviation_samples) if n else None,
            "deviation_max": max(self.deviation_samples) if n else None,
            "failover_cycles": list(self.failover_cycles),
        }


def run_natural_regime(
    trainer,
    next_batch: Callable[[int], tuple[str, dict]],
    next_state: Callable[[int], tuple],
    *,
    cycles: int,
) -> StalenessRegimeStats:
    """Drive the persistent-tree staleness pass under REAL θ movement.

    Requires ``mcts_persistent_tree=True`` and ``mcts_max_depth >= 2`` (a
    ValueError otherwise; at depth 1 the advanced root is childless and nothing
    can age). Each cycle: ``train_step`` (advances θ → ``theta_version`` ticks →
    the persistent tree falls behind it), then ``select_action`` (plans; the
    plan-§4 staleness pass re-evaluates naturally-aged nodes).

    IMPORTANT — this is the tuning pass's real entry, and it needs a TRAINED
    checkpoint to be meaningful, for a reason deeper than realistic deviation
    values: natural aging requires a node to survive >= consistency_window
    cycles WITHOUT re-stamping, but ``advance_root`` prunes off-trajectory
    branches every cycle. On an untrained (smoke) policy the acted child churns
    near-randomly, so the retained subtree is always freshly-stamped and
    ``total_reevaluated`` can legitimately be 0. A STABLE trained policy keeps
    re-traversing the same deep subtree, so those nodes persist and age. Hence
    this function does not (cannot) guarantee re-evals on an arbitrary model —
    for a deterministic exercise of the re-eval + deviation path today, use
    :func:`run_aged_node_regime`.

    ``next_batch(cycle) -> (modality, batch)`` and ``next_state(cycle) ->
    (s_t, context_latents)`` are caller-supplied so the harness is
    checkpoint-agnostic. Returns the collected :class:`StalenessRegimeStats`.
    """
    cfg = trainer.m9_config
    if not cfg.mcts_persistent_tree:
        raise ValueError(
            "run_natural_regime requires mcts_persistent_tree=True; the "
            "reset-per-cycle path cannot produce cross-cycle aging."
        )
    if cfg.mcts_max_depth < 2:
        raise ValueError(
            f"run_natural_regime requires mcts_max_depth >= 2 (got "
            f"{cfg.mcts_max_depth}); at depth 1 the advanced root is childless, "
            "no node ages, and zero natural re-evals form — the exact blocker "
            "this harness exists to remove."
        )

    stats = StalenessRegimeStats()
    for cycle in range(cycles):
        modality, batch = next_batch(cycle)
        trainer.train_step(modality, batch)          # advances θ
        s_t, ctx = next_state(cycle)
        trainer.select_action(s_t, context_latents=ctx)   # plans + re-evals
        stats.record(trainer.last_staleness_pass, cycle)
    return stats


def run_aged_node_regime(
    trainer,
    next_state: Callable[[int], tuple],
    *,
    cycles: int,
    age_ticks: int | None = None,
) -> StalenessRegimeStats:
    """Deterministically exercise the re-eval + consistency-deviation path.

    The aged-node alternative Fable sanctioned ("max_depth >= 2 OR an aged-node
    harness"). Instead of waiting for a stable policy to let nodes age
    naturally, this advances ``theta_version`` between plans WITHOUT re-planning
    (``observe_drift`` ticks the θ clock but does not touch the tree), so the
    retained nodes stamped at the earlier clock are past the staleness window
    when the next ``select_action`` runs its pass. This is the same mechanism
    the existing budget test uses; it produces re-evals on any model, so the
    machinery — priority ordering, budget carve-out, the scale-normalized
    deviation, failover accounting — is exercisable and testable NOW, ahead of
    the trained checkpoint. It does NOT produce realistic deviation VALUES
    (those need a trained model); it produces the plumbing signal.

    ``age_ticks`` defaults to ``consistency_window + 1`` (just past the staleness
    cutoff). Requires ``mcts_persistent_tree=True``. Returns the stats.

    **Treat the trainer as expended after this regime** (Fable review
    2026-07-15): every ``observe_drift(0.0)`` tick also pushes a synthetic
    zero into the θ drift band, so after ``cycles * age_ticks`` ticks the
    band's median/MAD are flooded toward zero. Within the regime that is
    self-consistently benign (no real update ever enters the band, so
    nothing can read as a spike), but if the SAME trainer is afterwards
    driven with real training steps, the first genuine ``||Δθ||`` reading
    lands on a zeroed band and registers as a spurious drift spike —
    cache wipe + failover on a healthy model. Build a fresh trainer for
    any post-regime work; do not run the natural regime after the aged
    regime on one trainer.
    """
    cfg = trainer.m9_config
    if not cfg.mcts_persistent_tree:
        raise ValueError(
            "run_aged_node_regime requires mcts_persistent_tree=True; the "
            "reset-per-cycle path discards the tree, so there is nothing to age."
        )
    if age_ticks is None:
        age_ticks = trainer.staleness.config.consistency_window + 1

    stats = StalenessRegimeStats()
    # Plant an initial tree to age against.
    s0, ctx0 = next_state(0)
    trainer.select_action(s0, context_latents=ctx0)
    for cycle in range(cycles):
        # Age the retained nodes past the window WITHOUT re-planning (θ clock
        # only; the tree is untouched, so stamps stay put and go stale).
        for _ in range(age_ticks):
            trainer.staleness.observe_drift(0.0)
        s_t, ctx = next_state(cycle + 1)
        trainer.select_action(s_t, context_latents=ctx)   # pass re-evals aged
        stats.record(trainer.last_staleness_pass, cycle)
    return stats
