"""Tests for ``generate_text(return_state=True)`` -- Phase 4a step 4.

The training-seam inference path: instead of running a separate
``encode_state`` (which 4.8's F7a-followup probe showed measurably
perturbs generation via double plasticity), the seam captures
``s_t + ctx_latents`` from generation's own step-0 encode and returns
them alongside the text.
"""

from __future__ import annotations

import pytest
import torch

from luthi.generate import generate_text
from luthi.seam_types import GenerationState
from luthi.tokenizer import BPETokenizer
from luthi.v2.m9.s_t import compute_s_t
from luthi.v2.model_pc import PredictiveCodingLM
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM


SAMPLE_TEXT = (
    "The quiet room. The light through the window. "
    "A soft rain against the glass. Patient mornings."
) * 64


@pytest.fixture
def tokenizer() -> BPETokenizer:
    t = BPETokenizer(target_vocab_size=64)
    t.train(SAMPLE_TEXT)
    return t


@pytest.fixture
def multimodal_pc_model(tokenizer) -> MultimodalPredictiveCodingLM:
    return MultimodalPredictiveCodingLM(
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        n_blocks=2,
        n_heads=2,
        max_seq_len=32,
        pc_rate=0.001,
        pred_learning_rate=0.0001,
        max_audio_tokens=8,
        max_vision_tokens=8,
        vision_image_size=16,
        vision_patch_size=8,
    )


class TestReturnStateOnMultimodalPC:
    def test_returns_tuple_with_generation_state(
        self, multimodal_pc_model, tokenizer,
    ):
        text, state = generate_text(
            model=multimodal_pc_model,
            tokenizer=tokenizer,
            prompt="hello",
            max_tokens=4,
            max_seq_len=32,
            stream=False,
            return_state=True,
        )
        assert isinstance(text, str)
        assert isinstance(state, GenerationState)

    def test_s_t_shape_is_b_d(self, multimodal_pc_model, tokenizer):
        _, state = generate_text(
            model=multimodal_pc_model,
            tokenizer=tokenizer,
            prompt="hello",
            max_tokens=4,
            max_seq_len=32,
            stream=False,
            return_state=True,
        )
        assert state.s_t.dim() == 2
        assert state.s_t.shape[0] == 1
        assert state.s_t.shape[1] == 16  # d_model

    def test_ctx_latents_shape_is_b_t_d(self, multimodal_pc_model, tokenizer):
        _, state = generate_text(
            model=multimodal_pc_model,
            tokenizer=tokenizer,
            prompt="hello",
            max_tokens=4,
            max_seq_len=32,
            stream=False,
            return_state=True,
        )
        assert state.ctx_latents.dim() == 3
        assert state.ctx_latents.shape[0] == 1
        assert state.ctx_latents.shape[2] == 16  # d_model

    def test_s_t_matches_compute_s_t_of_ctx_latents(
        self, multimodal_pc_model, tokenizer,
    ):
        """The captured s_t must equal compute_s_t(ctx_latents). This is
        the no-drift property at the return-shape level: generate_text
        routes through the canonical helper, not inline compute."""
        _, state = generate_text(
            model=multimodal_pc_model,
            tokenizer=tokenizer,
            prompt="hello",
            max_tokens=4,
            max_seq_len=32,
            stream=False,
            return_state=True,
        )
        expected_s_t = compute_s_t(state.ctx_latents)
        assert torch.allclose(state.s_t, expected_s_t)

    def test_s_t_is_detached(self, multimodal_pc_model, tokenizer):
        _, state = generate_text(
            model=multimodal_pc_model,
            tokenizer=tokenizer,
            prompt="hello",
            max_tokens=4,
            max_seq_len=32,
            stream=False,
            return_state=True,
        )
        assert not state.s_t.requires_grad

    def test_text_only_path_when_no_sensory(
        self, multimodal_pc_model, tokenizer,
    ):
        # No audio/vision -- the model takes the legacy recompute-each-step
        # path (multimodal-PC has no kv_cache support). return_state must
        # still capture the encode result at step 0.
        text, state = generate_text(
            model=multimodal_pc_model,
            tokenizer=tokenizer,
            prompt="hello",
            max_tokens=4,
            max_seq_len=32,
            stream=False,
            return_state=True,
        )
        assert isinstance(text, str)
        assert isinstance(state, GenerationState)


class TestReturnStateCapabilityCheck:
    """``return_state=True`` requires a v2 multimodal-PC substrate
    (4.8's 2026-06-16 review: raise eagerly, no ``(text, None)``
    silent-degradation foot-gun)."""

    def test_raises_on_v2_text_only(self, tokenizer):
        v2_text_only = PredictiveCodingLM(
            vocab_size=tokenizer.vocab_size,
            d_model=16, n_blocks=2, n_heads=2, max_seq_len=32,
        )
        with pytest.raises(AttributeError, match="return_state=True"):
            generate_text(
                model=v2_text_only,
                tokenizer=tokenizer,
                prompt="hello",
                max_tokens=4,
                max_seq_len=32,
                stream=False,
                return_state=True,
            )

    def test_raises_on_v1_base(self, tokenizer):
        from luthi.model import LuthiLM
        v1 = LuthiLM(
            vocab_size=tokenizer.vocab_size,
            d_model=16, n_blocks=2, max_seq_len=32,
        )
        with pytest.raises(AttributeError, match="return_state=True"):
            generate_text(
                model=v1,
                tokenizer=tokenizer,
                prompt="hello",
                max_tokens=4,
                max_seq_len=32,
                stream=False,
                return_state=True,
            )

    def test_raises_eagerly_before_any_generation(self, tokenizer):
        """The check fires before the loop starts, not mid-generation,
        so the error names the actual config mismatch rather than a
        late AttributeError on .return_encode_result."""
        from luthi.model import LuthiLM
        v1 = LuthiLM(
            vocab_size=tokenizer.vocab_size,
            d_model=16, n_blocks=2, max_seq_len=32,
        )
        with pytest.raises(AttributeError) as exc_info:
            generate_text(
                model=v1,
                tokenizer=tokenizer,
                prompt="hello",
                max_tokens=4,
                max_seq_len=32,
                stream=False,
                return_state=True,
            )
        msg = str(exc_info.value)
        # Names the requested feature and what's missing.
        assert "return_state=True" in msg
        assert "encode" in msg
        assert "MultimodalPredictiveCodingLM" in msg


class TestReturnStateFalseUnchanged:
    """``return_state=False`` (default) is the existing API; nothing
    about its behavior should change. v1 / v2 text-only must still
    generate successfully through the default path."""

    def test_v2_multimodal_default_returns_str(
        self, multimodal_pc_model, tokenizer,
    ):
        out = generate_text(
            model=multimodal_pc_model,
            tokenizer=tokenizer,
            prompt="hello",
            max_tokens=4,
            max_seq_len=32,
            stream=False,
        )
        assert isinstance(out, str)

    def test_v2_text_only_default_returns_str(self, tokenizer):
        v2_text = PredictiveCodingLM(
            vocab_size=tokenizer.vocab_size,
            d_model=16, n_blocks=2, n_heads=2, max_seq_len=32,
        )
        out = generate_text(
            model=v2_text,
            tokenizer=tokenizer,
            prompt="hello",
            max_tokens=4,
            max_seq_len=32,
            stream=False,
        )
        assert isinstance(out, str)
