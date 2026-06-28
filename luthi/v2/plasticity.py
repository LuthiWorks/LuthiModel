"""Frozen-plasticity context for the Item #6 lived JEPA re-encode.

The lived world-model gradient (Item #6) needs a forward pass through the
encoder trunk that produces grad-capable output WITHOUT self-modifying the
living substrate. Perception already self-modified once -- online, during the
cycle's generation forward -- so the learner's offline re-encode must not do
it a second time ("no double-plasticity"), and under the async actor/learner
split (Plan §4) must not race the actor's writes to the same living buffers.

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


__all__ = ["freeze_plasticity", "FROZEN_TYPES"]
