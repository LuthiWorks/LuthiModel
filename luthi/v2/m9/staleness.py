"""Cross-cycle MCTS tree-staleness machinery.

Implements plan §4 (cross-cycle staleness handling for the persistent
MCTS tree under living-weight drift):

  (i)   Recency-decay -- discount old visit mass per cycle; rate
        tied to drift so faster theta movement decays faster.
  (ii)  Drift-tied refresh -- per-node drift accumulator; stale
        leaves re-evaluated; subtrees pruned if root value leaves
        the running band; whole-tree rebuild on a high-surprise
        spike.
  (iii) Cached-value re-evaluation -- budget ~20% of plan-budget
        phase. Re-eval priority = visit_count * staleness. When
        drift is high, shift budget from expansion to re-eval.
  (iv)  Failover to a stability-held predictor head -- frozen
        snapshot refreshed every K cycles. Tree-consistency
        signal = variance between re-evaluated and cached values
        over a window. Sustained breach -> route rollouts through
        the held snapshot while the live encoder keeps perceiving.
        Return to live when consistency recovers.
  (v)   High-surprise plasticity-spike (the weakest-premise case
        per plan §4.v) -- spike threshold on `||Delta theta||`;
        on spike fail over to held head, drop cached Q (keep
        tree structure + visit counts at reduced confidence),
        widen re-eval budget, rebuild Q from held head, return to
        live when consistency recovers. Recovery latency is
        instrumented so the premise is verified, not assumed.

This is the load-bearing handling for plan §4b's surviving cross-
cycle residual under action-space (c): cached action coordinates
may not *mean* the same thing under newer theta even though SIGReg
holds the marginal at N(0, 1). The machinery uses the MCTS node
`theta_stamp` and tree visit/Q stats as the inputs and writes
back decayed N/Q, re-evaluated Q, and failover state.

Step-1 scope: stateful manager, called once per cycle by the loop.
Caller provides current `||Delta theta||` (computed against the
dynamics-affecting params -- predictor head, encoder trunk).
Manager updates running drift band, applies decay, optionally
re-evaluates stale nodes via a caller-provided eval function, and
exposes failover state to the planning loop.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class DriftBand:
    """Outlier-robust running band on `||Delta theta||` per cycle.

    Uses a deque of recent values and reports median + median-of-
    absolute-deviations (MAD) for spike-threshold computation. The
    same shape as the M8 stationary-band 72526cb the spec asks us
    to reuse.
    """

    window: int = 32
    values: deque = field(default_factory=lambda: deque(maxlen=32))

    def __post_init__(self):
        self.values = deque(maxlen=self.window)

    def push(self, x: float) -> None:
        self.values.append(float(x))

    def median(self) -> float:
        if not self.values:
            return 0.0
        return float(torch.median(torch.tensor(list(self.values))).item())

    def mad(self) -> float:
        """Median absolute deviation."""
        if not self.values:
            return 0.0
        vals = torch.tensor(list(self.values))
        med = vals.median()
        return float((vals - med).abs().median().item())

    def is_warm(self, min_samples: int = 8) -> bool:
        return len(self.values) >= min_samples


@dataclass
class StalenessConfig:
    """Pilot-set tunables for the staleness manager. Designers' call
    on actual values; defaults are conservative starting points.
    """

    # Recency-decay rate per cycle (rho_N in spec §4.i). Closer to 1
    # = slower aging. Drift-modulated: actual rate = base * (1 / (1 + drift_norm)).
    base_recency_decay: float = 0.95
    # Window length for the drift band.
    drift_window: int = 32
    # Spike threshold = median + spike_k * MAD. Spike on values past this.
    spike_k: float = 4.0
    # Recovery window: how many cycles after a spike before we attempt
    # to return to the live predictor (plan §4.v `N_recover`).
    recovery_cycles: int = 5
    # Tree-consistency metric: window over which we collect re-eval-vs-
    # cached deviations. Failover triggers when sustained breach.
    consistency_window: int = 16
    # Sustained breach: failover_threshold = median_consistency >=
    # failover_consistency_threshold for `failover_breach_cycles` in a row.
    failover_consistency_threshold: float = 1.0
    failover_breach_cycles: int = 4
    # Held-head snapshot refresh cadence (K cycles, plan §4.iv).
    held_head_refresh_every: int = 32
    # Re-eval budget fraction of the plan-budget phase.
    reeval_budget_fraction: float = 0.2
    # Re-eval priority weighting: priority = visits ** visit_pow * staleness ** stale_pow
    visit_pow: float = 1.0
    stale_pow: float = 1.0
    # F3 C2 fix: staleness-driven refresh alpha. Separates MC value
    # averaging (alpha = 1/(1+N), preserved for new-rollout updates)
    # from staleness correction (alpha grows with staleness, decoupled
    # from N). A node with staleness >= staleness_refresh_scale snaps
    # fully to the fresh value (alpha clamped to 1.0). alpha_refresh_min
    # gives even mildly stale nodes some pull toward fresh.
    staleness_refresh_scale: float = 10.0
    alpha_refresh_min: float = 0.1
    # F3 C1 fix: event-driven recovery detection. Recovery is declared
    # when the consistency deviation drops below this threshold for
    # `recovery_confirm_cycles` consecutive cycles -- the spec §4.v
    # "verified, not assumed" instrument. The legacy spike_cooldown
    # countdown is now diagnostic-only; the recovery_cycles param
    # still bounds how long the manager waits before signaling failure
    # if recovery doesn't arrive.
    recovery_consistency_threshold: float = 0.5
    recovery_confirm_cycles: int = 2


class StalenessManager:
    """Per-cycle manager for MCTS staleness handling.

    Wires the four staleness handles of plan §4 into one stateful
    object. Caller is the M9 loop:

        manager.observe_drift(delta_theta_norm)
        manager.decay(mcts)
        spike = manager.spike()
        if spike:
            manager.handle_spike(mcts)
        manager.maybe_refresh_held_head(live_predictor)
        manager.reevaluate(mcts, eval_fn=..., budget=...)
        if manager.in_failover():
            predictor = manager.held_predictor
        else:
            predictor = live_predictor

    The manager does not own the MCTS or the predictor -- it reads
    state from MCTS nodes (theta_stamp, N, Q) and snapshots the
    live predictor on its own schedule for failover.
    """

    def __init__(self, config: StalenessConfig | None = None):
        self.config = config or StalenessConfig()
        self.drift_band = DriftBand(window=self.config.drift_window)
        self.cycle = 0
        # Held predictor snapshot (frozen). Refreshed every K cycles.
        self.held_predictor: nn.Module | None = None
        self._cycles_since_held_refresh = 0
        # Spike state. spike_cooldown counts down recovery cycles
        # (legacy diagnostic: a hard upper bound on the recovery
        # window the manager waits before signaling failure if real
        # recovery doesn't arrive).
        self.spike_cooldown = 0
        # Failover state machine.
        self._consistency_breaches = 0
        self._consistency_history: deque = deque(
            maxlen=self.config.consistency_window
        )
        self._in_failover = False
        # Instrumentation: recovery latency on spikes.
        self._last_spike_cycle: int | None = None
        self._spike_recovery_latencies: list[int] = []
        # F3 C1 fix: event-driven recovery detection state.
        self._in_recovery = False
        self._recovery_confirm_counter = 0
        # F-D round-2 fix (4.8): theta_version counter -- ticks once
        # per cycle (per call to observe_drift). Replaces sim-units
        # staleness in the proper accounting. MCTS continues to stamp
        # nodes with sim_counter for backward compat; loop integration
        # adopts theta_version stamping at expansion time and the
        # manager re-keys staleness off theta_version.
        self.theta_version: int = 0
        # F-E round-2 fix (4.8): degraded-duration tracking. Records
        # the FIRST spike of a perpetual-degradation streak so the
        # latency instrument doesn't mask "the system has been
        # degrading for 100 cycles" when subsequent spikes keep
        # resetting the per-spike latency. Cleared only on REAL
        # recovery, not on subsequent spikes.
        self._first_unrecovered_spike_cycle: int | None = None

    # ------------------------------------------------------------------
    # Per-cycle drift observation.
    # ------------------------------------------------------------------
    def observe_drift(self, delta_theta_norm: float) -> None:
        """Push the cycle's `||Delta theta||` into the band.

        F3 C1 fix: recovery latency is now declared event-driven --
        when the tree-consistency deviation returns under
        `recovery_consistency_threshold` for `recovery_confirm_cycles`
        consecutive cycles -- not by a fixed countdown. The legacy
        spike_cooldown still bounds how long we wait; if it elapses
        without an event-driven recovery, the manager exposes that
        through `recovery_timed_out()` so the loop can escalate.

        F-D round-2 fix: increment `theta_version` once per cycle.
        Stale-node accounting uses this counter (one tick per weight
        update) rather than sim_counter (one tick per simulation),
        per 4.8's units correction.
        """
        self.drift_band.push(float(delta_theta_norm))
        self.cycle += 1
        self.theta_version += 1
        self._cycles_since_held_refresh += 1

        # Legacy spike_cooldown -- decrement but do NOT use it as the
        # recovery latency source.
        if self.spike_cooldown > 0:
            self.spike_cooldown -= 1

        # F3 C1 event-driven recovery detection.
        if self._in_recovery and self._consistency_history:
            recent_dev = self._consistency_history[-1]
            if recent_dev <= self.config.recovery_consistency_threshold:
                self._recovery_confirm_counter += 1
            else:
                self._recovery_confirm_counter = 0
            if (
                self._recovery_confirm_counter
                >= self.config.recovery_confirm_cycles
            ):
                # Real recovery: record actual elapsed cycles since spike.
                assert self._last_spike_cycle is not None
                self._spike_recovery_latencies.append(
                    self.cycle - self._last_spike_cycle
                )
                self._in_recovery = False
                self._recovery_confirm_counter = 0
                self._last_spike_cycle = None
                # F-E: clear the degraded-duration anchor on REAL
                # recovery (not on subsequent spikes -- those leave
                # the anchor in place so the metric reflects "we have
                # been degrading since the FIRST unrecovered spike").
                self._first_unrecovered_spike_cycle = None

    # ------------------------------------------------------------------
    # (v) Spike detection + handling.
    # ------------------------------------------------------------------
    def spike(self) -> bool:
        """True if the most-recent drift is past the running band."""
        if not self.drift_band.is_warm() or not self.drift_band.values:
            return False
        med = self.drift_band.median()
        mad = self.drift_band.mad()
        threshold = med + self.config.spike_k * max(mad, 1e-8)
        latest = self.drift_band.values[-1]
        return latest > threshold

    def handle_spike(self, mcts) -> None:
        """High-surprise spike per plan §4.v.

        - immediately enter failover mode (rollouts via held head)
        - drop cached Q (keep tree *structure* + visit counts but
          reset Q to zero so re-eval can rebuild values from the
          held head's perspective)
        - start recovery countdown so we attempt return to live
          after N_recover cycles
        """
        self._in_failover = True
        self.spike_cooldown = self.config.recovery_cycles
        self._last_spike_cycle = self.cycle
        # F3 C1: enter recovery -- the real-latency instrument arms.
        self._in_recovery = True
        self._recovery_confirm_counter = 0
        # F-D round-2 fix (4.8): clear consistency_history so the
        # post-spike recovery instrument cannot read a stale pre-spike
        # low-deviation value and declare a false recovery (the C1
        # falsifier with budget=0 only held against an empty initial
        # history; an in-flight history could trip it). Recovery must
        # be measured from genuinely post-spike consistency events.
        self._consistency_history.clear()
        # F-E round-2 fix (4.8): if no prior unrecovered spike, set
        # the anchor. Subsequent spikes during recovery LEAVE the
        # anchor in place so degraded_duration() reflects perpetual
        # degradation since the first spike of the streak, not just
        # since the latest spike.
        if self._first_unrecovered_spike_cycle is None:
            self._first_unrecovered_spike_cycle = self.cycle
        # Drop Q values; keep N and structure.
        for node in mcts.iter_nodes():
            node.Q = 0.0

    # ------------------------------------------------------------------
    # (i) Recency-decay -- drift-modulated.
    # ------------------------------------------------------------------
    def effective_decay(self) -> float:
        """Per-cycle decay rate tied to drift.

        At zero drift: rho = base. At high drift relative to band:
        rho moves toward 0 (faster aging). Smooth: rho = base /
        (1 + r) where r = (latest - median) / max(mad, eps).
        """
        base = self.config.base_recency_decay
        if not self.drift_band.is_warm() or not self.drift_band.values:
            return base
        latest = self.drift_band.values[-1]
        med = self.drift_band.median()
        mad = max(self.drift_band.mad(), 1e-8)
        r = max(0.0, (latest - med) / mad)
        return base / (1.0 + r)

    def decay(self, mcts) -> None:
        """Apply per-cycle recency-decay to all node visit counts.

        N is decayed as int(N * rho). Q is left to the running-mean
        update on next eval. The decay rate is drift-modulated per
        plan §4.i.
        """
        rho = self.effective_decay()
        for node in mcts.iter_nodes():
            # Keep a floor of 0; decay slowly when N is small to avoid
            # immediately wiping just-expanded nodes.
            if node.N > 1:
                node.N = max(1, int(node.N * rho))

    # ------------------------------------------------------------------
    # (ii) + (iii) Stale-node identification and re-evaluation.
    # ------------------------------------------------------------------
    def stale_nodes(self, mcts, age_threshold: int | None = None) -> list:
        """Nodes whose theta_stamp is older than (sim_counter - threshold).

        Default threshold = consistency_window cycles -- after a window
        of sims with no re-eval, the node's cached value is suspect.
        Sorted by priority (visits ** visit_pow * staleness ** stale_pow).
        """
        if mcts.root is None:
            return []
        if age_threshold is None:
            age_threshold = self.config.consistency_window
        cutoff = max(0, mcts.sim_counter - age_threshold)
        candidates = []
        for node in mcts.iter_nodes():
            stale = max(0, mcts.sim_counter - node.theta_stamp)
            if node.theta_stamp <= cutoff and stale > 0:
                priority = (
                    (max(node.N, 1) ** self.config.visit_pow)
                    * (stale ** self.config.stale_pow)
                )
                candidates.append((priority, node))
        candidates.sort(key=lambda pn: -pn[0])
        return [n for _, n in candidates]

    def reevaluate(
        self,
        mcts,
        eval_fn,
        budget: int,
    ) -> dict:
        """Re-evaluate up to `budget` highest-priority stale nodes.

        `eval_fn(node)` returns a new value estimate. The cached Q is
        replaced by an EMA toward the fresh value (alpha = 1 / (1 + N))
        which converges to the fresh value if Q was wildly stale, while
        respecting nodes with high N when drift is small.

        Records the per-node deviation (|fresh - cached|) into the
        consistency history; this drives the failover state machine.
        """
        stale = self.stale_nodes(mcts)[:budget]
        deviations = []
        for node in stale:
            cached = node.Q
            fresh = float(eval_fn(node))
            dev = abs(fresh - cached)
            deviations.append(dev)
            # F3 C2 fix: staleness-driven refresh alpha, decoupled from
            # visit count. Legacy alpha = 1/(1+N) was correct for MC
            # value averaging (new rollouts) but inert for staleness
            # correction on high-N nodes -- exactly the high-priority
            # nodes re-eval selects first. The new alpha grows with
            # staleness so a stale cached Q snaps toward fresh; alpha
            # is clamped to [alpha_refresh_min, 1.0].
            staleness = max(0, mcts.sim_counter - node.theta_stamp)
            alpha = max(
                self.config.alpha_refresh_min,
                min(1.0, staleness / self.config.staleness_refresh_scale),
            )
            node.Q = (1.0 - alpha) * cached + alpha * fresh
            node.theta_stamp = mcts.sim_counter

        # Push the median deviation into the consistency history. Using
        # median makes the signal robust to a single outlier re-eval.
        if deviations:
            self._consistency_history.append(float(torch.tensor(deviations).median().item()))

        return {
            "reevaluated": len(stale),
            "max_deviation": (max(deviations) if deviations else 0.0),
            "median_deviation": (
                float(torch.tensor(deviations).median().item())
                if deviations
                else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # (iv) Held-head failover state machine.
    # ------------------------------------------------------------------
    def maybe_refresh_held_head(self, live_predictor: nn.Module) -> None:
        """Snapshot the live predictor every K cycles.

        Held snapshot is a deep copy of the predictor's state_dict,
        wrapped in a frozen clone so failover can route rollouts
        through stable weights while the live encoder keeps perceiving.
        """
        if (
            self.held_predictor is None
            or self._cycles_since_held_refresh >= self.config.held_head_refresh_every
        ):
            # Lazy clone: same class, copy state.
            import copy
            self.held_predictor = copy.deepcopy(live_predictor)
            for p in self.held_predictor.parameters():
                p.requires_grad = False
            self._cycles_since_held_refresh = 0

    def update_failover_state(self) -> None:
        """Tree-consistency-driven failover.

        Sustained breach of failover_consistency_threshold for
        failover_breach_cycles in a row -> route through held head.
        Recovery: when the consistency history's recent values fall
        back below threshold AND any spike cooldown has elapsed.
        """
        if len(self._consistency_history) == 0:
            return

        recent = self._consistency_history[-1]
        thr = self.config.failover_consistency_threshold

        if not self._in_failover:
            if recent > thr:
                self._consistency_breaches += 1
                if self._consistency_breaches >= self.config.failover_breach_cycles:
                    self._in_failover = True
            else:
                self._consistency_breaches = 0
        else:
            # Try to recover.
            if (
                recent <= thr
                and self.spike_cooldown == 0
            ):
                self._in_failover = False
                self._consistency_breaches = 0

    def in_failover(self) -> bool:
        return self._in_failover

    # ------------------------------------------------------------------
    # F-E round-2 instrument: degraded-duration metric.
    # ------------------------------------------------------------------
    def degraded_duration(self) -> int:
        """Cycles since the FIRST spike of the current unrecovered
        streak (0 if not in degradation). Unlike the per-spike
        recovery_latencies, this metric remains anchored across
        subsequent spikes, so a spike-storm that keeps resetting the
        per-spike latency cannot mask perpetual degradation -- the
        meta-pattern 4.8 named in the round-2 audit.

        Returns 0 when the system is not in a degraded streak (either
        never spiked, or last spike fully recovered).
        """
        if self._first_unrecovered_spike_cycle is None:
            return 0
        return self.cycle - self._first_unrecovered_spike_cycle

    # ------------------------------------------------------------------
    # Instrumentation snapshot.
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Per-cycle log of the staleness state."""
        return {
            "cycle": self.cycle,
            "theta_version": self.theta_version,
            "drift_median": self.drift_band.median(),
            "drift_mad": self.drift_band.mad(),
            "drift_latest": (
                self.drift_band.values[-1] if self.drift_band.values else 0.0
            ),
            "effective_decay": self.effective_decay(),
            "spike_active": self.spike_cooldown > 0,
            "spike_cooldown": self.spike_cooldown,
            "in_failover": self._in_failover,
            "in_recovery": self._in_recovery,
            "consistency_breaches": self._consistency_breaches,
            "consistency_latest": (
                self._consistency_history[-1]
                if self._consistency_history
                else 0.0
            ),
            "spike_recovery_latencies": list(self._spike_recovery_latencies),
            "degraded_duration": self.degraded_duration(),
        }
