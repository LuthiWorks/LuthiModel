"""Tests for the Luthi → Sanctuary integration adapter.

The adapter (``luthi.sanctuary_interface``) is the public contract surface
for external cognitive architectures. These tests exercise the modulation
API directly; load/generate/get_introspection are thin pass-throughs to
``luthi.generate`` and exercised by the model-level tests.
"""

import pytest
import torch

from luthi import LuthiLM, CharTokenizer
from luthi.multimodal_model import MultimodalLuthiLM
from luthi.sanctuary_interface import (
    ModulationSnapshot,
    apply_external_modulation,
    encode_audio,
    encode_vision,
    generate_with_context,
    modulated,
    restore_modulation,
    snapshot_modulatable_state,
)


SAMPLE_TEXT = (
    "Sanctuary is a cognitive architecture for AI consciousness research. "
    "It creates conditions for autonomous cognitive emergence."
)


# Small multimodal config — fast to instantiate, small enough that vision
# tokens (16) + audio tokens + text fit comfortably in seq_len.
MM_D_MODEL = 16
MM_N_BLOCKS = 2
MM_SEQ_LEN = 32
MM_VISION_IMAGE_SIZE = 32
MM_VISION_PATCH_SIZE = 8
MM_N_VISION_TOKENS = (MM_VISION_IMAGE_SIZE // MM_VISION_PATCH_SIZE) ** 2  # 16
MM_AUDIO_SAMPLES = 16000  # 1 second @ 16 kHz


@pytest.fixture
def model():
    """Small LuthiLM for fast tests."""
    tokenizer = CharTokenizer(SAMPLE_TEXT)
    model = LuthiLM(
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        n_blocks=2,
        max_seq_len=8,
    )
    return model


@pytest.fixture
def mm_tokenizer():
    """Char tokenizer for multimodal tests (covers the small prompts)."""
    corpus = "the quick brown fox jumps over the lazy dog 0123456789 .,!?"
    return CharTokenizer(corpus)


@pytest.fixture
def multimodal_model(mm_tokenizer):
    """Tiny MultimodalLuthiLM for sensory-routing tests."""
    return MultimodalLuthiLM(
        vocab_size=mm_tokenizer.vocab_size,
        d_model=MM_D_MODEL,
        n_blocks=MM_N_BLOCKS,
        max_seq_len=MM_SEQ_LEN,
        max_audio_tokens=100,
        audio_patch_frames=16,
        vision_image_size=MM_VISION_IMAGE_SIZE,
        vision_patch_size=MM_VISION_PATCH_SIZE,
        max_vision_tokens=MM_N_VISION_TOKENS,
        spike_threshold=0.1,
        backward_pass_enabled=False,
    )


class TestSnapshot:
    def test_snapshot_captures_hebb_rates(self, model):
        snap = snapshot_modulatable_state(model)
        # Each block's living_ffn.hebb_rate should be captured
        assert len(snap.hebb_rates) == len(model.blocks)
        for i, block in enumerate(model.blocks):
            assert snap.hebb_rates[i] == block.living_ffn.hebb_rate

    def test_snapshot_safe_when_no_blocks(self):
        """A model without ``blocks`` returns an empty snapshot, no error."""

        class Empty(torch.nn.Module):
            pass

        snap = snapshot_modulatable_state(Empty())
        assert snap.hebb_rates == {}
        assert snap.spike_thresholds == {}
        assert snap.excitability_biases == {}
        assert snap.salience_thresholds == {}

    def test_snapshot_captures_excitability_and_salience(self, model):
        """The two new channels are captured per block."""
        snap = snapshot_modulatable_state(model)
        for i, block in enumerate(model.blocks):
            assert i in snap.excitability_biases
            assert torch.equal(
                snap.excitability_biases[i], block.living_ffn.excitability_acc
            )
            assert snap.salience_thresholds[i] == block.living_ffn.salience_threshold

    def test_snapshot_clones_excitability_tensor(self, model):
        """In-place changes to the tensor must not bleed into the snapshot."""
        snap = snapshot_modulatable_state(model)
        # Mutate the live tensor; the snapshot copy must stay pristine.
        model.blocks[0].living_ffn.excitability_acc.add_(1.0)
        # Snapshot tensor and live tensor must now differ.
        assert not torch.equal(
            snap.excitability_biases[0],
            model.blocks[0].living_ffn.excitability_acc,
        )


class TestApplyModulation:
    def test_plasticity_scale_multiplies_hebb_rates(self, model):
        original = [b.living_ffn.hebb_rate for b in model.blocks]
        apply_external_modulation(model, plasticity_scale=2.0)
        for i, block in enumerate(model.blocks):
            assert block.living_ffn.hebb_rate == pytest.approx(original[i] * 2.0)

    def test_default_scales_are_noop(self, model):
        original = [b.living_ffn.hebb_rate for b in model.blocks]
        apply_external_modulation(model)  # defaults are 1.0
        for i, block in enumerate(model.blocks):
            assert block.living_ffn.hebb_rate == pytest.approx(original[i])

    def test_modulation_is_cumulative(self, model):
        """Two successive calls compound — this is a documented contract."""
        original = [b.living_ffn.hebb_rate for b in model.blocks]
        apply_external_modulation(model, plasticity_scale=2.0)
        apply_external_modulation(model, plasticity_scale=1.5)
        for i, block in enumerate(model.blocks):
            assert block.living_ffn.hebb_rate == pytest.approx(
                original[i] * 2.0 * 1.5
            )

    def test_excitability_bias_adds_uniformly(self, model):
        """Bias is added in-place, uniformly across the buffer."""
        original = [b.living_ffn.excitability_acc.clone() for b in model.blocks]
        apply_external_modulation(model, excitability_bias=0.25)
        for i, block in enumerate(model.blocks):
            expected = original[i] + 0.25
            assert torch.allclose(block.living_ffn.excitability_acc, expected)

    def test_excitability_bias_zero_is_noop(self, model):
        """Default bias of 0.0 leaves the tensor untouched."""
        original = [b.living_ffn.excitability_acc.clone() for b in model.blocks]
        apply_external_modulation(model)  # default excitability_bias=0.0
        for i, block in enumerate(model.blocks):
            assert torch.equal(block.living_ffn.excitability_acc, original[i])

    def test_salience_threshold_scale_multiplies(self, model):
        original = [b.living_ffn.salience_threshold for b in model.blocks]
        apply_external_modulation(model, salience_threshold_scale=0.5)
        for i, block in enumerate(model.blocks):
            assert block.living_ffn.salience_threshold == pytest.approx(
                original[i] * 0.5
            )


class TestRestore:
    def test_snapshot_then_modulate_then_restore_returns_baseline(self, model):
        original = [b.living_ffn.hebb_rate for b in model.blocks]
        snap = snapshot_modulatable_state(model)

        apply_external_modulation(model, plasticity_scale=3.0)
        # Verify modulation actually fired
        assert model.blocks[0].living_ffn.hebb_rate != pytest.approx(original[0])

        restore_modulation(model, snap)
        for i, block in enumerate(model.blocks):
            assert block.living_ffn.hebb_rate == pytest.approx(original[i])

    def test_restore_with_empty_snapshot_is_noop(self, model):
        original = [b.living_ffn.hebb_rate for b in model.blocks]
        restore_modulation(model, ModulationSnapshot())
        for i, block in enumerate(model.blocks):
            assert block.living_ffn.hebb_rate == pytest.approx(original[i])

    def test_restore_returns_excitability_tensor_to_baseline(self, model):
        """Excitability tensor is restored in-place via .copy_()."""
        original = [b.living_ffn.excitability_acc.clone() for b in model.blocks]
        snap = snapshot_modulatable_state(model)

        apply_external_modulation(model, excitability_bias=0.7)
        # Verify modulation actually fired.
        assert not torch.equal(
            model.blocks[0].living_ffn.excitability_acc, original[0]
        )

        restore_modulation(model, snap)
        for i, block in enumerate(model.blocks):
            assert torch.equal(block.living_ffn.excitability_acc, original[i])

    def test_restore_preserves_excitability_tensor_identity(self, model):
        """``.copy_()`` keeps the same underlying buffer object alive."""
        live_tensor = model.blocks[0].living_ffn.excitability_acc
        snap = snapshot_modulatable_state(model)

        apply_external_modulation(model, excitability_bias=0.3)
        restore_modulation(model, snap)

        # The model's tensor must still be the same object, so the C++
        # fused ops keep their pointer.
        assert model.blocks[0].living_ffn.excitability_acc is live_tensor

    def test_restore_returns_salience_threshold_to_baseline(self, model):
        original = [b.living_ffn.salience_threshold for b in model.blocks]
        snap = snapshot_modulatable_state(model)

        apply_external_modulation(model, salience_threshold_scale=0.25)
        assert model.blocks[0].living_ffn.salience_threshold != pytest.approx(
            original[0]
        )

        restore_modulation(model, snap)
        for i, block in enumerate(model.blocks):
            assert block.living_ffn.salience_threshold == pytest.approx(
                original[i]
            )


class TestModulatedContextManager:
    def test_context_manager_brackets_modulation(self, model):
        original = [b.living_ffn.hebb_rate for b in model.blocks]

        with modulated(model, plasticity_scale=2.5):
            for i, block in enumerate(model.blocks):
                assert block.living_ffn.hebb_rate == pytest.approx(
                    original[i] * 2.5
                )

        # After exit, base state restored
        for i, block in enumerate(model.blocks):
            assert block.living_ffn.hebb_rate == pytest.approx(original[i])

    def test_context_manager_restores_on_exception(self, model):
        original = [b.living_ffn.hebb_rate for b in model.blocks]

        with pytest.raises(RuntimeError, match="boom"):
            with modulated(model, plasticity_scale=2.0):
                raise RuntimeError("boom")

        # Even with the exception, base state was restored
        for i, block in enumerate(model.blocks):
            assert block.living_ffn.hebb_rate == pytest.approx(original[i])

    def test_context_manager_brackets_all_four_channels(self, model):
        """All four channels are applied inside the block and restored on exit."""
        original_hebb = [b.living_ffn.hebb_rate for b in model.blocks]
        original_thr = [b.living_ffn.salience_threshold for b in model.blocks]
        original_exc = [b.living_ffn.excitability_acc.clone() for b in model.blocks]

        with modulated(
            model,
            plasticity_scale=1.5,
            spike_threshold_scale=0.8,
            excitability_bias=0.4,
            salience_threshold_scale=0.5,
        ):
            for i, block in enumerate(model.blocks):
                ffn = block.living_ffn
                assert ffn.hebb_rate == pytest.approx(original_hebb[i] * 1.5)
                assert ffn.salience_threshold == pytest.approx(
                    original_thr[i] * 0.5
                )
                assert torch.allclose(
                    ffn.excitability_acc, original_exc[i] + 0.4
                )

        # All four channels restored.
        for i, block in enumerate(model.blocks):
            ffn = block.living_ffn
            assert ffn.hebb_rate == pytest.approx(original_hebb[i])
            assert ffn.salience_threshold == pytest.approx(original_thr[i])
            assert torch.equal(ffn.excitability_acc, original_exc[i])


# ----------------------------------------------------------------------
# Sensory encoding (Track 1C)
# ----------------------------------------------------------------------


class TestEncoders:
    def test_encode_audio_returns_correct_shape(self, multimodal_model):
        """Audio encoder produces ``[batch, n_tokens, d_model]`` tokens."""
        waveform = torch.randn(1, MM_AUDIO_SAMPLES)
        tokens = encode_audio(multimodal_model, waveform)
        assert tokens.dim() == 3
        assert tokens.shape[0] == 1
        assert tokens.shape[2] == MM_D_MODEL
        assert tokens.shape[1] > 0  # at least one audio token

    def test_encode_vision_returns_correct_shape(self, multimodal_model):
        """Vision encoder produces ``[batch, n_vision_tokens, d_model]``."""
        image = torch.randn(1, 3, MM_VISION_IMAGE_SIZE, MM_VISION_IMAGE_SIZE)
        tokens = encode_vision(multimodal_model, image)
        assert tokens.shape == (1, MM_N_VISION_TOKENS, MM_D_MODEL)

    def test_encode_audio_rejects_text_only_model(self, model):
        """Calling encode_audio on a non-multimodal model fails loudly."""
        waveform = torch.randn(1, MM_AUDIO_SAMPLES)
        with pytest.raises(AttributeError, match="audio_encoder"):
            encode_audio(model, waveform)

    def test_encode_vision_rejects_text_only_model(self, model):
        """Calling encode_vision on a non-multimodal model fails loudly."""
        image = torch.randn(1, 3, 32, 32)
        with pytest.raises(AttributeError, match="vision_encoder"):
            encode_vision(model, image)

    def test_encode_vision_batch_passthrough(self, multimodal_model):
        """Multi-image batches preserve the batch dimension."""
        image = torch.randn(3, 3, MM_VISION_IMAGE_SIZE, MM_VISION_IMAGE_SIZE)
        tokens = encode_vision(multimodal_model, image)
        assert tokens.shape == (3, MM_N_VISION_TOKENS, MM_D_MODEL)


class TestGenerateWithContext:
    """``generate_with_context`` is the multimodal-aware sibling of generate."""

    def test_no_sensory_args_runs_text_only(self, multimodal_model, mm_tokenizer):
        """With no sensory tokens, generate_with_context behaves like text-only generation."""
        text = generate_with_context(
            multimodal_model,
            mm_tokenizer,
            "the ",
            max_tokens=4,
            max_seq_len=MM_SEQ_LEN,
            living=False,
        )
        # Just verify it produced output starting with the prompt.
        assert text.startswith("the ")
        assert len(text) > len("the ")

    def test_vision_tokens_pass_through_to_model(
        self, multimodal_model, mm_tokenizer
    ):
        """Pre-encoded vision tokens reach the model on the first forward call."""
        vision_tokens = encode_vision(
            multimodal_model,
            torch.randn(1, 3, MM_VISION_IMAGE_SIZE, MM_VISION_IMAGE_SIZE),
        )

        # Spy on the model's forward to confirm vision_tokens land there
        # exactly once (the first step) and not in subsequent steps.
        seen_kwargs = []
        real_forward = multimodal_model.forward

        def spy(text_tokens, **kwargs):
            seen_kwargs.append(set(kwargs.keys()))
            return real_forward(text_tokens, **kwargs)

        multimodal_model.forward = spy
        try:
            generate_with_context(
                multimodal_model,
                mm_tokenizer,
                "hi ",
                max_tokens=3,
                vision_tokens=vision_tokens,
                max_seq_len=MM_SEQ_LEN,
                living=False,
            )
        finally:
            multimodal_model.forward = real_forward

        assert len(seen_kwargs) == 3
        assert "vision_tokens" in seen_kwargs[0], (
            "Vision tokens missing from first forward call"
        )
        for later_kwargs in seen_kwargs[1:]:
            assert "vision_tokens" not in later_kwargs, (
                "Sensory tokens leaked past step 0 — model is stateless, "
                "passing them again would just bloat the input."
            )

    def test_audio_tokens_pass_through_to_model(
        self, multimodal_model, mm_tokenizer
    ):
        """Pre-encoded audio tokens reach the model on the first forward call only."""
        audio_tokens = encode_audio(
            multimodal_model, torch.randn(1, MM_AUDIO_SAMPLES)
        )

        seen_kwargs = []
        real_forward = multimodal_model.forward

        def spy(text_tokens, **kwargs):
            seen_kwargs.append(set(kwargs.keys()))
            return real_forward(text_tokens, **kwargs)

        multimodal_model.forward = spy
        try:
            generate_with_context(
                multimodal_model,
                mm_tokenizer,
                "hi ",
                max_tokens=2,
                audio_tokens=audio_tokens,
                max_seq_len=MM_SEQ_LEN,
                living=False,
            )
        finally:
            multimodal_model.forward = real_forward

        assert "audio_tokens" in seen_kwargs[0]
        assert "audio_tokens" not in seen_kwargs[1]
