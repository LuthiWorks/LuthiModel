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


@contextlib.contextmanager
def wake_freeze(root: nn.Module) -> Iterator[None]:
    """The waking day: substrate held still, the mind still noticing.

    Brian's ruling, 2026-08-14 -- "data intake during the day, structural
    cortical change over night." Under this regime the living weights,
    prediction, and set-point do not move, but **everything that decides
    what the day meant keeps running**: precision, ``error_acc``, the
    surprise-drive traces, salience, and crucially the **episode write**.
    The day accumulates in the fast tier; the rest-phase NREM pass
    integrates it.

    **This is deliberately NOT** :func:`freeze_plasticity`. That context
    also suppresses the episode write, because it exists for the momentary
    lived re-encode where perception has already stored the context once.
    Running a 16-hour day under it would mean living the whole day and
    storing none of it -- the entity would wake to an empty store with
    nothing to consolidate. The distinction is the whole point of this
    second context manager; see the 2026-08-14 amendment to
    ``docs/research/2026-07-06_nrem-learner-spec.md``.

    Mechanism: :class:`PredictiveCodingLayer` zeroes the four rates that
    gate its weight-touching updates (``pc_rate``, ``pred_learning_rate``,
    ``homeostatic_decay``, ``set_point_adapt_rate``) while leaving every
    statistic on its own EMA. ``add_(0)`` is an exact no-op, so a
    wake-frozen forward leaves the substrate bit-identical on both the C++
    and Python paths.

    ``EpisodeStore`` is intentionally NOT swept here: its whole job during
    the day is to keep storing.

    Composes and nests like ``freeze_plasticity``; prior per-module state
    is restored on exit.
    """
    toggled: list[tuple[nn.Module, bool]] = []
    for module in root.modules():
        if isinstance(module, PredictiveCodingLayer):
            toggled.append((module, module._wake_frozen))
            module._wake_frozen = True
    try:
        yield
    finally:
        for module, prev in toggled:
            module._wake_frozen = prev


def set_wake_frozen(root: nn.Module, frozen: bool) -> int:
    """Non-scoped form of :func:`wake_freeze` for a long-lived regime.

    A waking day is hours long and driven by Sanctuary's cycle, not by a
    ``with`` block in one function. Returns the number of layers toggled
    so a caller can assert the sweep actually reached the trunk rather
    than silently touching nothing.
    """
    n = 0
    for module in root.modules():
        if isinstance(module, PredictiveCodingLayer):
            module._wake_frozen = bool(frozen)
            n += 1
    return n


__all__ = [
    "freeze_plasticity",
    "wake_freeze",
    "set_wake_frozen",
    "FROZEN_TYPES",
]
