"""Mode-compatibility matrix for the living substrate (2026-07-15).

One auditable place for every "these modes cannot be combined" rule in the
v2 substrate. Before this module the failure surface was scattered inline
raises (living_layer_pc forward, jepa_loss lived path) that had to be
grepped for; a reviewer auditing "what combinations are forbidden and why"
had no single source. Now the rules are DECLARED here and ENFORCED at the
site that can actually see the condition:

- Rules whose inputs are constructor arguments are checked at
  construction (nothing to add today -- the current forbidden combos all
  involve a caller-side runtime choice like gradient checkpointing, which
  a layer cannot see at __init__).
- Rules involving runtime state (recompute replay, freeze_plasticity)
  keep their raise at forward/backward time, but the raise routes through
  :func:`raise_incompatible` so the message, the reason, and the
  enforcement point live in the table below rather than in string
  literals at N call sites.

House rule this serves (docs/KEY_FINDINGS.md #4): prefer crashes over
silent corruption -- incompatible combinations raise loud RuntimeError
rather than producing wrong results quietly.

Adding a rule: add a ModeIncompatibility to INCOMPATIBILITIES, route the
enforcement site through raise_incompatible(name, ...), and add the cell
to tests/test_mode_matrix.py's forbidden-cells sweep so the rule is pinned
by a test that proves it fires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn


@dataclass(frozen=True)
class ModeIncompatibility:
    """One forbidden mode combination.

    name: stable key the enforcement site raises by.
    modes: the flags/contexts that collide (documentation, not enforcement).
    enforced_at: where the raise lives and why it cannot be earlier.
    reason: the mechanism that makes the combination unsound -- this is
        the text a confused 3 AM debugger reads, so it names the corruption
        that WOULD have happened.
    """

    name: str
    modes: tuple[str, ...]
    enforced_at: str
    reason: str


INCOMPATIBILITIES: tuple[ModeIncompatibility, ...] = (
    ModeIncompatibility(
        name="ipc_x_grad_checkpoint",
        modes=("inference_steps_per_forward > 1", "gradient checkpointing"),
        enforced_at=(
            "PredictiveCodingLayer.forward, recompute branch (backward "
            "time) -- the layer cannot know at construction whether a "
            "caller will wrap it in torch.utils.checkpoint"
        ),
        reason=(
            "iPC evolves the living weight T times WITHIN one forward; the "
            "single cached snapshot cannot reproduce that trajectory on the "
            "checkpoint replay, so recomputed activations (and therefore "
            "gradients) would silently diverge from the original forward. "
            "Disable one or the other."
        ),
    ),
    ModeIncompatibility(
        name="recompute_without_original",
        modes=(
            "gradient-checkpoint recompute",
            "no cached snapshot from the original forward",
        ),
        enforced_at=(
            "PredictiveCodingLayer.forward, recompute branch (backward "
            "time) -- the condition is runtime cache state"
        ),
        reason=(
            "A checkpoint replay reached this layer but no "
            "_fwd_weight_snapshot exists from the original forward. Two "
            "known causes, both misuse: (1) the original forward ran under "
            "freeze_plasticity() (frozen path caches nothing; the freeze "
            "exits before backward, so the replay lands here on the normal "
            "path) -- the lived re-encode is incompatible with gradient "
            "checkpointing, see jepa_loss.compute_lived_loss's guard; "
            "(2) clear_forward_cache() was called before backward() "
            "completed. Without this guard the replay would either crash "
            "cryptically or, worse, silently reuse a STALE snapshot from an "
            "earlier step and produce wrong gradients."
        ),
    ),
    ModeIncompatibility(
        name="lived_reencode_x_grad_checkpoint",
        modes=("compute_lived_loss re-encode", "encoder gradient checkpointing"),
        enforced_at=(
            "JEPALoss.compute_lived_loss, before the frozen re-encode "
            "(forward time) -- the encoder-level flag is visible there"
        ),
        reason=(
            "Checkpoint replay runs in backward(), AFTER freeze_plasticity "
            "has exited, so the recomputed forward would either fire "
            "pc_self_modify (double-plasticity, if the wrap omits "
            "luthi_context_fn) or trip the recompute_without_original "
            "guard. The frozen-plasticity guarantee does not hold on the "
            "recompute pass. Disable encoder gradient checkpointing for "
            "the lived path (or add snapshot-based recompute support "
            "first)."
        ),
    ),
)

_BY_NAME = {rule.name: rule for rule in INCOMPATIBILITIES}


def get_rule(name: str) -> ModeIncompatibility:
    """Look up a declared incompatibility by name (KeyError if undeclared --
    an enforcement site may only raise rules that exist in the table)."""
    return _BY_NAME[name]


def raise_incompatible(name: str, extra: str = "") -> NoReturn:
    """Raise the declared incompatibility as a RuntimeError.

    ``extra`` appends site-specific context (tensor shapes, config values)
    after the declared reason.
    """
    rule = get_rule(name)
    msg = f"Incompatible modes {' x '.join(rule.modes)}: {rule.reason}"
    if extra:
        msg = f"{msg} [{extra}]"
    raise RuntimeError(msg)


__all__ = [
    "ModeIncompatibility",
    "INCOMPATIBILITIES",
    "get_rule",
    "raise_incompatible",
]
