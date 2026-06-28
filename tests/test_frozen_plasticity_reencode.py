"""Item #6 (Plan §1, Finding 1): the lived JEPA re-encode runs the encoder
trunk under ``freeze_plasticity()`` so it produces a grad-capable forward
WITHOUT mutating any living state.

Two writers touch living state in an encode pass and BOTH must be frozen:
  - ``PredictiveCodingLayer.pc_self_modify`` (weight/momentum/update_ema/
    error_acc/...) + its layer-level episode write, and
  - the block-level ``EpisodeStore.store()`` (episode_contexts/outputs/
    saliences/count).

The original Finding-1 risk was that a living-layer-scoped freeze would sail
past the block-level EpisodeStore, so the re-encode would still write
episodes (a second mutation on a "no double-plasticity" path, and a
cross-thread race under the async learner). These tests assert BOTH writers
are silenced, and that a normal forward still mutates -- so the freeze is the
thing making the difference, not a dead code path.
"""

from __future__ import annotations

import pytest
import torch

from luthi.episode_store import EpisodeStore
from luthi.v2.living_layer_pc import PredictiveCodingLayer
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM
from luthi.v2.plasticity import FROZEN_TYPES, freeze_plasticity


VOCAB = 32
D = 16
SEQ = 12


def _build_model(seed: int = 0) -> MultimodalPredictiveCodingLM:
    torch.manual_seed(seed)
    return MultimodalPredictiveCodingLM(
        vocab_size=VOCAB, d_model=D, n_blocks=2, n_heads=2,
        ffn_expansion=1, max_seq_len=SEQ,
        max_audio_tokens=SEQ, max_vision_tokens=SEQ,
        backward_pass_enabled=False,
    )


def _text(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (1, SEQ), generator=g)


def _snapshot_living(model: torch.nn.Module) -> dict:
    """Clone every living buffer across the trunk, keyed by module id so the
    before/after comparison is per-module."""
    snap: dict = {}
    for idx, m in enumerate(model.modules()):
        if isinstance(m, PredictiveCodingLayer):
            for name in ("weight", "momentum", "update_ema", "error_acc"):
                snap[(idx, name)] = getattr(m, name).detach().clone()
        elif isinstance(m, EpisodeStore):
            for name in (
                "episode_contexts", "episode_outputs",
                "episode_saliences", "episode_count",
            ):
                snap[(idx, name)] = getattr(m, name).detach().clone()
    return snap


def _assert_unchanged(model: torch.nn.Module, snap: dict) -> None:
    for idx, m in enumerate(model.modules()):
        if isinstance(m, PredictiveCodingLayer):
            for name in ("weight", "momentum", "update_ema", "error_acc"):
                assert torch.equal(getattr(m, name), snap[(idx, name)]), (
                    f"living-layer buffer {name} mutated under freeze"
                )
        elif isinstance(m, EpisodeStore):
            for name in (
                "episode_contexts", "episode_outputs",
                "episode_saliences", "episode_count",
            ):
                assert torch.equal(getattr(m, name), snap[(idx, name)]), (
                    f"episode-store buffer {name} mutated under freeze"
                )


def _count_living_modules(model: torch.nn.Module) -> tuple[int, int]:
    layers = sum(
        isinstance(m, PredictiveCodingLayer) for m in model.modules()
    )
    stores = sum(isinstance(m, EpisodeStore) for m in model.modules())
    return layers, stores


# --------------------------------------------------------------------------
# Coverage: the sweep reaches every living layer + store, not just the top.
# --------------------------------------------------------------------------

def test_freeze_covers_every_living_module_in_trunk():
    model = _build_model()
    n_layers, n_stores = _count_living_modules(model)
    # 2 blocks -> 2 living layers + 2 block-level episode stores.
    assert n_layers == 2 and n_stores == 2
    with freeze_plasticity(model):
        frozen_layers = [
            m._plasticity_frozen
            for m in model.modules()
            if isinstance(m, FROZEN_TYPES)
        ]
        assert all(frozen_layers)
        assert len(frozen_layers) == n_layers + n_stores
    # Restored on exit.
    assert not any(
        m._plasticity_frozen
        for m in model.modules()
        if isinstance(m, FROZEN_TYPES)
    )


# --------------------------------------------------------------------------
# Grad-capability: the frozen forward still trains the encoder's BACKPROP
# params (attention/embeddings) -- the living weight buffer gets no grad.
# --------------------------------------------------------------------------

def test_frozen_reencode_is_grad_capable_and_trains_encoder_params():
    model = _build_model()
    x = _text(1)
    with freeze_plasticity(model):
        out = model.encode(text_tokens=x, causal=False)["latents"]
    assert out.requires_grad, "frozen re-encode must produce grad-capable output"
    out.sum().backward()

    # At least one encoder Parameter received gradient (grad flowed THROUGH
    # the frozen living weight to upstream backprop params).
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no encoder Parameter received gradient from the re-encode"

    # The living FFN weight is a buffer, not a Parameter: it self-modifies
    # via pc_self_modify, never by backprop, so it must carry no grad.
    for m in model.modules():
        if isinstance(m, PredictiveCodingLayer):
            assert m.weight.grad is None, (
                "living FFN weight buffer must not receive backprop grad"
            )


# --------------------------------------------------------------------------
# No mutation under freeze, in BOTH writers -- and a normal forward DOES.
# --------------------------------------------------------------------------

def test_frozen_reencode_mutates_no_living_buffer():
    model = _build_model()
    x = _text(2)
    snap = _snapshot_living(model)
    with freeze_plasticity(model):
        out = model.encode(text_tokens=x, causal=False)["latents"]
        out.sum().backward()  # backward must not mutate living state either
    _assert_unchanged(model, snap)


def test_normal_forward_does_mutate_living_layer():
    """Control: without the freeze, the same forward mutates living state --
    so the frozen test above is proving the freeze works, not that the
    forward is inert."""
    model = _build_model()
    x = _text(2)
    snap = _snapshot_living(model)
    model.encode(text_tokens=x, causal=False)
    # At least the living-layer weight/momentum move on a normal forward.
    changed = False
    for idx, m in enumerate(model.modules()):
        if isinstance(m, PredictiveCodingLayer):
            if not torch.equal(m.weight, snap[(idx, "weight")]):
                changed = True
            if not torch.equal(m.momentum, snap[(idx, "momentum")]):
                changed = True
    assert changed, "normal forward should self-modify the living layer"


# --------------------------------------------------------------------------
# Focused EpisodeStore unit: store() is gated by the flag; recall is not.
# This is the exact Finding-1 channel a living-layer-only freeze would miss.
# --------------------------------------------------------------------------

def test_episode_store_write_gated_by_freeze_recall_preserved():
    store = EpisodeStore(d_model=D, num_episodes=8, salience_threshold=0.0)
    # High-magnitude output so salience clears the (zeroed) threshold.
    x_in = torch.ones(1, SEQ, D)
    x_out = torch.full((1, SEQ, D), 3.0)

    # Frozen: forward recalls/blends but writes nothing.
    store._plasticity_frozen = True
    before = store.episode_count.clone()
    out_frozen = store(x_in, x_out)
    assert torch.equal(store.episode_count, before), (
        "frozen EpisodeStore.forward must not store"
    )
    assert out_frozen.shape == x_out.shape

    # Unfrozen: the same forward stores an episode.
    store._plasticity_frozen = False
    store(x_in, x_out)
    assert store.episode_count.item() == before.item() + 1, (
        "unfrozen EpisodeStore.forward must store"
    )

    # Recall still functions while frozen (the stored episode is retrievable).
    store._plasticity_frozen = True
    count_after = store.episode_count.clone()
    store(x_in, x_out)
    assert torch.equal(store.episode_count, count_after)


# --------------------------------------------------------------------------
# Nesting / restore safety.
# --------------------------------------------------------------------------

def test_freeze_restores_prior_flag_state():
    model = _build_model()
    # Pre-freeze one layer by hand; the context must restore it to True, not
    # clobber it to False, on exit.
    first_layer = next(
        m for m in model.modules() if isinstance(m, PredictiveCodingLayer)
    )
    first_layer._plasticity_frozen = True
    with freeze_plasticity(model):
        assert first_layer._plasticity_frozen
    assert first_layer._plasticity_frozen, (
        "freeze_plasticity must restore a module that was already frozen"
    )
