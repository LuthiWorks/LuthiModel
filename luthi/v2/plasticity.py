"""Frozen-plasticity context for the Item #6 lived JEPA re-encode.

The lived world-model gradient (Item #6) needs a forward pass through the
encoder trunk that produces grad-capable output WITHOUT self-modifying the
living substrate. Perception already self-modified once -- online, during the
cycle's generation forward -- so the learner's offline re-encode must not do
it a second time ("no double-plasticity").

Scope -- what this DOES and does NOT close (Window A audit, 2026-06-28):
freeze_plasticity stops the learner's re-encode from WRITING living state
(no pc_self_modify, no episode store). It does NOT make the re-encode's READS
of living state consistent: the frozen forward reads ``self.weight`` directly
and recalls the episode buffers, and under Plan §4's async actor/learner split
the actor's perception forward WRITES those same buffers in place
concurrently -- a torn read the freeze cannot prevent. Closing that read-race
is §4's job (a detached snapshot of the living buffers taken under the actor's
write-lock, re-encoded outside the lock), NOT this context manager's. Do not
read the freeze as a concurrency guarantee.

``freeze_plasticity(root)`` sweeps the whole module tree under ``root`` and
flips the ``_plasticity_frozen`` flag on every living-state writer it finds --
both the :class:`~luthi.v2.living_layer_pc.PredictiveCodingLayer` (its frozen
forward skips ``pc_self_modify`` and its layer-level episode write while still
letting the gradient flow to ``self.weight``) and the block-level
:class:`~luthi.episode_store.EpisodeStore` (skips ``store()``, keeps
recall+blend so the re-encoded latents retain perception's memory structure).

Using a module-tree sweep -- rather than threading a flag through ``forward``
arguments -- is what guarantees coverage of EVERY living layer in the trunk,
not just the top block. Prior per-module flag state is restored on exit, so
nesting and already-partially-frozen trees are safe.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import torch.nn as nn

from luthi.episode_store import EpisodeStore
from luthi.v2.living_layer_pc import PredictiveCodingLayer

# The two living-state writers an encode pass touches. Kept as a module-level
# tuple so callers (and tests) can introspect exactly what gets frozen.
FROZEN_TYPES: tuple[type[nn.Module], ...] = (PredictiveCodingLayer, EpisodeStore)


@contextlib.contextmanager
def freeze_plasticity(root: nn.Module) -> Iterator[None]:
    """Suspend living-state self-modification under ``root`` for the duration
    of the ``with`` block.

    Args:
        root: Any module whose subtree contains the living layers /
            episode stores to freeze (typically the encoder /
            ``MultimodalPredictiveCodingLM``).

    On exit, every toggled module's prior ``_plasticity_frozen`` value is
    restored, so an already-frozen module stays frozen and the context
    composes under nesting.
    """
    toggled: list[tuple[nn.Module, bool]] = []
    for module in root.modules():
        if isinstance(module, FROZEN_TYPES):
            toggled.append((module, module._plasticity_frozen))
            module._plasticity_frozen = True
    try:
        yield
    finally:
        for module, prev in toggled:
            module._plasticity_frozen = prev


# Default wake-phase attenuation. NOT a round number picked for feel --
# it comes from the step-count asymmetry between day and night, and the
# arithmetic is the reason "greatly lessened" has to mean far more than it
# sounds.
#
# A 16-hour day at Sanctuary's 10 Hz loop is ~576,000 forwards. A rest
# phase replays the day's episodes -- order 2,000-3,000 of them at
# ~0.1x pc_rate (`consolidation_rate_factor`), so order 200-300
# rate-units of integration. Day-integrated plasticity is
# 576,000 * attenuation rate-units. For the NIGHT to be the primary
# integrator rather than a rounding error on the day:
#
#     attenuation = 1e-2  -> day  5,760 units vs night ~230   day wins 25x
#     attenuation = 1e-3  -> day    576 units vs night ~230   comparable
#     attenuation = 1e-4  -> day     58 units vs night ~230   night wins 4x
#
# So a "10x reduction" would not remotely deliver the intent; the day
# would still dominate by two orders of magnitude purely on step count.
# 1e-3 is the point where the two channels are the same order and the day
# is genuinely material-to-be-pondered rather than the main event, with
# the night given the decisive share once the entity's keep/toss curation
# concentrates replay on what mattered.
#
# TUNE-ME, and it belongs to the combined tuning pass (§7-D) against a
# trained checkpoint and the real cycle rate -- like the gain's rise/cap
# and the F2 thresholds. It is a starting point derived from arithmetic,
# not a ruled constant.
WAKE_ATTENUATION_DEFAULT: float = 1e-3


def _sweep_attenuation(root: nn.Module, factor: float) -> list:
    if not 0.0 <= factor <= 1.0:
        raise ValueError(
            f"wake attenuation must lie in [0, 1]; got {factor}. "
            "Values > 1 would AMPLIFY waking plasticity, which is the "
            "opposite of this regime's purpose."
        )
    toggled = []
    for module in root.modules():
        if isinstance(module, PredictiveCodingLayer):
            toggled.append((module, module._wake_attenuation))
            module._wake_attenuation = float(factor)
    return toggled


@contextlib.contextmanager
def wake_attenuated(
    root: nn.Module, factor: float = WAKE_ATTENUATION_DEFAULT,
) -> Iterator[None]:
    """The waking day: the substrate still moves, but far less.

    Brian's ruling, 2026-08-14: *"I don't think the waking state should be
    completely frozen, but ... the susceptibility to change should be
    greatly lessened so the lesson of the day can be intentionally
    pondered rather than purely reacted to."*

    Under this regime the living weights, prediction and set-point keep
    moving at ``factor`` times their ordinary rates, and **everything that
    decides what the day meant runs at full strength**: precision,
    ``error_acc``, the surprise-drive traces, salience, and crucially the
    **episode write**. The day accumulates in the fast tier; the rest-phase
    pass integrates it deliberately.

    A nonzero floor is the point, not an implementation detail. It keeps
    change automatic and unvetoed (Brian's 2026-07-05 ruling) and honours
    the taper's stated philosophy -- *"lowering the learning rate of the
    self, never halting it."* ``factor=0.0`` is permitted for ablations
    but is outside the architecture's design intent; do not ship it as a
    regime.

    **Deliberately NOT** :func:`freeze_plasticity`, which also suppresses
    the episode write because it exists for the momentary lived re-encode.
    A 16-hour day under that mode would be lived and stored nowhere.

    Mechanism: :class:`PredictiveCodingLayer` scales the four rates gating
    its weight-touching updates -- ``pc_rate``, ``pred_learning_rate``,
    ``homeostatic_decay``, ``set_point_adapt_rate`` -- leaving every
    statistic on its own EMA. The scaling is **uniform across all four**,
    which differs from the taper (learning channels only, "stability is
    not what tapers"). That convention is safe at the taper's ~5x but not
    here: homeostasis pulls weight toward set_point with a time constant
    of ~1/``homeostatic_decay`` forwards (~100 s at 10 Hz), so attenuating
    learning alone would not lessen change -- it would make the day
    actively erase itself back to baseline within minutes. Uniform scaling
    preserves the equilibrium and slows the dynamics.

    ``EpisodeStore`` is intentionally NOT swept: its whole job during the
    day is to keep storing.

    Composes and nests; prior per-module state is restored on exit.
    """
    toggled = _sweep_attenuation(root, factor)
    try:
        yield
    finally:
        for module, prev in toggled:
            module._wake_attenuation = prev


def set_wake_attenuation(
    root: nn.Module, factor: float = WAKE_ATTENUATION_DEFAULT,
) -> int:
    """Non-scoped form of :func:`wake_attenuated` for a long-lived regime.

    A waking day is hours long and driven by Sanctuary's cycle, not by a
    ``with`` block in one function. Pass ``factor=1.0`` to return to
    ordinary plasticity (what the rest phase does on entry).

    Returns the number of layers touched, so a caller can assert the sweep
    actually reached the trunk instead of silently doing nothing.
    """
    return len(_sweep_attenuation(root, factor))


__all__ = [
    "freeze_plasticity",
    "wake_attenuated",
    "set_wake_attenuation",
    "WAKE_ATTENUATION_DEFAULT",
    "FROZEN_TYPES",
]
