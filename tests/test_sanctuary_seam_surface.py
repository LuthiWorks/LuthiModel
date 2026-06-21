"""Tests for the Sanctuary <-> training-seam contract surface.

Phase 1 of the 2026-06-15 seam-integration plan: ``encode_state``,
``select_action``, ``observe_transition``, plus the ``M9Actor`` and
``TransitionSink`` Protocols on which the latter two delegate.

Phase 1 ships the contract. Phase 2 wires ``M9Trainer`` to satisfy the
Protocols; these tests use small fakes to lock the surface shape.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from luthi.sanctuary_interface import (
    ActionSelection,
    M9Actor,
    TransitionSink,
    encode_state,
    observe_transition,
    select_action,
)
from luthi.v2.model_pc import PredictiveCodingLM
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM


# ---------------------------------------------------------------------------
# encode_state -- real model, no fake
# ---------------------------------------------------------------------------


@pytest.fixture
def multimodal_pc_model():
    return MultimodalPredictiveCodingLM(
        vocab_size=64,
        d_model=16,
        n_blocks=2,
        n_heads=2,
        max_seq_len=32,
        pc_rate=0.001,
        pred_learning_rate=0.0001,
        vision_image_size=32,
        vision_patch_size=8,
        max_vision_tokens=16,
        max_audio_tokens=16,
    )


class TestEncodeState:
    def test_text_only_pooled_returns_b_d(self, multimodal_pc_model):
        text = torch.randint(0, 64, (1, 8))
        s_t = encode_state(multimodal_pc_model, text_tokens=text, pool=True)
        assert s_t.shape == (1, 16)  # [B, d_model]
        assert torch.isfinite(s_t).all()
        # Detached -- callers can feed into stop-grad-isolated heads.
        assert not s_t.requires_grad

    def test_pool_false_returns_full_sequence(self, multimodal_pc_model):
        text = torch.randint(0, 64, (1, 8))
        latents = encode_state(
            multimodal_pc_model, text_tokens=text, pool=False,
        )
        assert latents.ndim == 3
        assert latents.shape[0] == 1
        assert latents.shape[2] == 16

    def test_batched_inputs(self, multimodal_pc_model):
        text = torch.randint(0, 64, (4, 8))
        s_t = encode_state(multimodal_pc_model, text_tokens=text)
        assert s_t.shape == (4, 16)

    def test_rejects_model_without_encode(self):
        """v1 text-only model has no encode(); must raise loudly."""
        v2_text_only = PredictiveCodingLM(
            vocab_size=64, d_model=16, n_blocks=2, n_heads=2, max_seq_len=16,
        )
        # PredictiveCodingLM (v2 text-only) doesn't expose .encode either --
        # only the multimodal v2 variant does. Same loud-failure shape.
        with pytest.raises(AttributeError, match="encode"):
            encode_state(v2_text_only, text_tokens=torch.randint(0, 64, (1, 4)))


# ---------------------------------------------------------------------------
# select_action -- delegates to actor, returns ActionSelection
# ---------------------------------------------------------------------------


class _FakeActor:
    """Minimal M9Actor stub. Records the call args and returns a fixed
    ActionSelection so the test asserts the wiring, not the planning."""

    def __init__(self):
        self.last_s_t: torch.Tensor | None = None
        self.last_kwargs: dict[str, Any] | None = None

    def select_action(
        self, s_t: torch.Tensor, **kwargs: Any,
    ) -> ActionSelection:
        self.last_s_t = s_t
        self.last_kwargs = kwargs
        return ActionSelection(
            action=torch.zeros(s_t.shape[-1]),
            readable_summary="hold-still",
            efe_breakdown={"epistemic": 0.0, "pragmatic": 0.1},
        )


class TestSelectAction:
    def test_returns_action_selection(self):
        actor = _FakeActor()
        s_t = torch.randn(16)
        result = select_action(actor, s_t)
        assert isinstance(result, ActionSelection)
        assert result.action.shape == (16,)
        assert result.readable_summary == "hold-still"
        assert result.efe_breakdown == {"epistemic": 0.0, "pragmatic": 0.1}

    def test_delegates_to_actor_with_kwargs(self):
        actor = _FakeActor()
        s_t = torch.randn(16)
        select_action(actor, s_t, budget=4, foo="bar")
        assert actor.last_s_t is s_t
        assert actor.last_kwargs == {"budget": 4, "foo": "bar"}

    def test_fake_satisfies_protocol(self):
        # @runtime_checkable Protocol -- isinstance works without inheritance.
        assert isinstance(_FakeActor(), M9Actor)


# ---------------------------------------------------------------------------
# observe_transition -- delegates to sink, packs ctx, forwards metrics
# ---------------------------------------------------------------------------


class _FakeSink:
    def __init__(self):
        self.last_call: dict[str, Any] | None = None

    def observe_transition(
        self,
        s_t: torch.Tensor,
        a_t: torch.Tensor,
        s_next: torch.Tensor,
        ctx: dict[str, Any],
    ) -> dict[str, float]:
        self.last_call = {
            "s_t": s_t, "a_t": a_t, "s_next": s_next, "ctx": ctx,
        }
        return {"v_loss": 0.5, "habit_loss": 0.3}


class TestObserveTransition:
    def test_forwards_args_and_default_ctx(self):
        sink = _FakeSink()
        s_t = torch.randn(16)
        a_t = torch.randn(16)
        s_next = torch.randn(16)
        metrics = observe_transition(sink, s_t, a_t, s_next)
        assert metrics == {"v_loss": 0.5, "habit_loss": 0.3}
        assert sink.last_call is not None
        assert sink.last_call["s_t"] is s_t
        assert sink.last_call["a_t"] is a_t
        assert sink.last_call["s_next"] is s_next
        # Default ctx contains the documented P3-wiring keys.
        assert sink.last_call["ctx"] == {
            "counterpart_present": False,
            "time_since_emission": 0.0,
        }

    def test_packs_p3_kwargs_into_ctx(self):
        sink = _FakeSink()
        observe_transition(
            sink,
            torch.zeros(4), torch.zeros(4), torch.zeros(4),
            counterpart_present=True,
            time_since_emission=12.5,
        )
        assert sink.last_call["ctx"]["counterpart_present"] is True
        assert sink.last_call["ctx"]["time_since_emission"] == 12.5

    def test_extra_ctx_kwargs_folded_in(self):
        sink = _FakeSink()
        observe_transition(
            sink,
            torch.zeros(4), torch.zeros(4), torch.zeros(4),
            modality="text",
            cycle_id=42,
        )
        ctx = sink.last_call["ctx"]
        assert ctx["modality"] == "text"
        assert ctx["cycle_id"] == 42
        # P3 defaults still present.
        assert ctx["counterpart_present"] is False

    def test_fake_satisfies_protocol(self):
        assert isinstance(_FakeSink(), TransitionSink)
