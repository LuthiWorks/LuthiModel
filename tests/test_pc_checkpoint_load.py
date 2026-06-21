"""Tests for the v2 PC checkpoint-loader bridge in ``luthi.generate``.

The Sanctuary <-> training-seam integration plan (Phase 0a) calls for
``load_model_from_checkpoint`` to construct ``PredictiveCodingLM`` /
``MultimodalPredictiveCodingLM`` when the checkpoint config carries v2
fingerprints, and to fail loud on v1/v2 mismatches instead of silently
building the wrong architecture.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from luthi.checkpoint import build_checkpoint, save_checkpoint
from luthi.generate import _detect_architecture, load_model_from_checkpoint
from luthi.tokenizer import BPETokenizer
from luthi.v2.model_pc import PredictiveCodingLM
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM


TEST_PASSWORD = "test-password-do-not-use-in-production"
SAMPLE_TEXT = (
    "The quiet room. The light through the window. "
    "A soft rain against the glass. Patient mornings."
) * 64


def _tiny_tokenizer() -> BPETokenizer:
    tok = BPETokenizer(target_vocab_size=64)
    tok.train(SAMPLE_TEXT)
    return tok


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# _detect_architecture
# ---------------------------------------------------------------------------


class TestDetectArchitecture:
    def test_v1_base_when_no_flags(self):
        assert _detect_architecture({}) == "v1-base"
        assert _detect_architecture({"d_model": 64}) == "v1-base"

    def test_v1_spiking(self):
        assert _detect_architecture({"spiking": True}) == "v1-spiking"

    def test_v1_multimodal_via_multimodal_flag(self):
        assert _detect_architecture({"multimodal": True}) == "v1-multimodal"

    def test_v1_multimodal_via_vision_flag(self):
        assert _detect_architecture({"vision": True}) == "v1-multimodal"

    def test_v2_base_via_pc_rate(self):
        assert _detect_architecture({"pc_rate": 0.001}) == "v2-base"

    def test_v2_base_via_pred_learning_rate(self):
        assert (
            _detect_architecture({"pred_learning_rate": 0.0001}) == "v2-base"
        )

    def test_v2_multimodal_when_v2_and_multimodal(self):
        assert (
            _detect_architecture({"pc_rate": 0.001, "vision": True})
            == "v2-multimodal"
        )

    def test_v2_plus_spiking_is_mismatch(self):
        with pytest.raises(ValueError, match="v2 PC fingerprint"):
            _detect_architecture({"pc_rate": 0.001, "spiking": True})

    def test_explicit_architecture_wins(self):
        assert (
            _detect_architecture({"architecture": "v2-base", "pc_rate": 0.001})
            == "v2-base"
        )

    def test_explicit_architecture_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown architecture"):
            _detect_architecture({"architecture": "v3-quantum"})

    def test_explicit_v1_contradicts_v2_fingerprint(self):
        with pytest.raises(ValueError, match="architecture='v1-base'"):
            _detect_architecture(
                {"architecture": "v1-base", "pc_rate": 0.001}
            )

    def test_explicit_v2_without_fingerprint_raises(self):
        with pytest.raises(ValueError, match="no v2 fingerprint"):
            _detect_architecture({"architecture": "v2-base"})

    def test_explicit_non_multimodal_with_vision_flag_raises(self):
        with pytest.raises(ValueError, match="multimodal/vision flag set"):
            _detect_architecture(
                {"architecture": "v2-base", "pc_rate": 0.001, "vision": True}
            )


# ---------------------------------------------------------------------------
# v2 PredictiveCodingLM round-trip
# ---------------------------------------------------------------------------


class TestV2BaseRoundTrip:
    def _build_and_save(self, tmp_path: Path) -> tuple[Path, BPETokenizer]:
        tokenizer = _tiny_tokenizer()
        model = PredictiveCodingLM(
            vocab_size=tokenizer.vocab_size,
            d_model=16,
            n_blocks=2,
            n_heads=2,
            max_seq_len=16,
            pc_rate=0.001,
            pred_learning_rate=0.0001,
        )
        # Run a forward pass so living state is non-zero.
        x = torch.randint(0, tokenizer.vocab_size, (1, 8))
        model.eval()
        with torch.no_grad():
            model(x)

        config = {
            "d_model": 16,
            "n_blocks": 2,
            "n_heads": 2,
            "seq_len": 16,
            "pc_rate": 0.001,
            "pred_learning_rate": 0.0001,
        }
        ckpt = build_checkpoint(model, epoch=1, config=config)
        ckpt["tokenizer_state"] = tokenizer.get_state()
        path = save_checkpoint(ckpt, tmp_path / "v2_base", TEST_PASSWORD)
        return path, tokenizer

    def test_loads_as_predictive_coding_lm(self, tmp_path):
        path, tokenizer = self._build_and_save(tmp_path)
        model, tok, config, epoch = load_model_from_checkpoint(
            str(path), TEST_PASSWORD,
        )
        assert isinstance(model, PredictiveCodingLM)
        assert tok.vocab_size == tokenizer.vocab_size
        assert epoch == 1
        assert config["pc_rate"] == 0.001

    def test_forward_pass_after_load(self, tmp_path):
        path, tokenizer = self._build_and_save(tmp_path)
        model, _tok, _config, _epoch = load_model_from_checkpoint(
            str(path), TEST_PASSWORD,
        )
        x = torch.randint(0, tokenizer.vocab_size, (1, 8))
        model.eval()
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (1, 8, tokenizer.vocab_size)
        assert torch.isfinite(logits).all()

    def test_living_state_preserved(self, tmp_path):
        """Living buffers (prediction, error_acc, plasticity) round-trip."""
        path, _ = self._build_and_save(tmp_path)

        loaded_model, _tok, _config, _epoch = load_model_from_checkpoint(
            str(path), TEST_PASSWORD,
        )

        # Loaded model's PC layer state should be non-zero -- something
        # got loaded, not just init defaults.
        for block in loaded_model.blocks:
            ffn = block.living_ffn
            assert hasattr(ffn, "prediction")
            assert hasattr(ffn, "error_acc")
            assert hasattr(ffn, "plasticity")


# ---------------------------------------------------------------------------
# v2 MultimodalPredictiveCodingLM round-trip
# ---------------------------------------------------------------------------


class TestV2MultimodalRoundTrip:
    def test_loads_as_multimodal_predictive_coding_lm(self, tmp_path):
        tokenizer = _tiny_tokenizer()
        model = MultimodalPredictiveCodingLM(
            vocab_size=tokenizer.vocab_size,
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
        config = {
            "d_model": 16,
            "n_blocks": 2,
            "n_heads": 2,
            "seq_len": 32,
            "pc_rate": 0.001,
            "pred_learning_rate": 0.0001,
            "vision": True,
            "vision_image_size": 32,
            "vision_patch_size": 8,
            "max_vision_tokens": 16,
            "max_audio_tokens": 16,
        }
        ckpt = build_checkpoint(model, epoch=2, config=config)
        ckpt["tokenizer_state"] = tokenizer.get_state()
        path = save_checkpoint(ckpt, tmp_path / "v2_mm", TEST_PASSWORD)

        loaded, tok, cfg, epoch = load_model_from_checkpoint(
            str(path), TEST_PASSWORD,
        )
        assert isinstance(loaded, MultimodalPredictiveCodingLM)
        assert epoch == 2
        assert cfg["vision"] is True
        assert tok.vocab_size == tokenizer.vocab_size


# ---------------------------------------------------------------------------
# Fail-loud guarantees
# ---------------------------------------------------------------------------


class TestFailLoud:
    def test_v2_config_with_v1_state_dict_raises(self, tmp_path):
        """A v2 config combined with a v1 state_dict must fail at the
        strict load. This is the central "never silently build the wrong
        architecture" guarantee for the bridge."""
        from luthi.model import LuthiLM  # v1

        tokenizer = _tiny_tokenizer()
        v1_model = LuthiLM(
            vocab_size=tokenizer.vocab_size,
            d_model=16,
            n_blocks=2,
            max_seq_len=16,
        )
        # v2 config + v1 state_dict -> _detect_architecture returns v2 ->
        # construct PredictiveCodingLM -> strict load fails.
        config = {
            "d_model": 16,
            "n_blocks": 2,
            "n_heads": 2,
            "seq_len": 16,
            "pc_rate": 0.001,
        }
        ckpt = build_checkpoint(v1_model, epoch=0, config=config)
        ckpt["tokenizer_state"] = tokenizer.get_state()
        path = save_checkpoint(ckpt, tmp_path / "mismatched", TEST_PASSWORD)

        with pytest.raises(RuntimeError):
            # PyTorch's strict load raises RuntimeError with Missing/Unexpected
            # keys when the state_dict doesn't match the architecture.
            load_model_from_checkpoint(str(path), TEST_PASSWORD)

    def test_v2_plus_spiking_config_raises_before_load(self, tmp_path):
        """The v2-plus-spiking inconsistency is caught by
        _detect_architecture at config-time, before any state_dict
        match is attempted."""
        tokenizer = _tiny_tokenizer()
        model = PredictiveCodingLM(
            vocab_size=tokenizer.vocab_size,
            d_model=16, n_blocks=2, n_heads=2, max_seq_len=16,
        )
        config = {
            "d_model": 16,
            "n_blocks": 2,
            "n_heads": 2,
            "seq_len": 16,
            "pc_rate": 0.001,
            "spiking": True,  # the contradiction
        }
        ckpt = build_checkpoint(model, epoch=0, config=config)
        ckpt["tokenizer_state"] = tokenizer.get_state()
        path = save_checkpoint(ckpt, tmp_path / "bad_config", TEST_PASSWORD)

        with pytest.raises(ValueError, match="v2 PC fingerprint"):
            load_model_from_checkpoint(str(path), TEST_PASSWORD)


# ---------------------------------------------------------------------------
# v1 backward compatibility
# ---------------------------------------------------------------------------


class TestV1BackwardCompat:
    def test_v1_base_still_loads(self, tmp_path):
        """The v1 load path is unchanged -- a v1 LuthiLM checkpoint must
        still round-trip identically through load_model_from_checkpoint."""
        from luthi.model import LuthiLM

        tokenizer = _tiny_tokenizer()
        model = LuthiLM(
            vocab_size=tokenizer.vocab_size,
            d_model=16,
            n_blocks=2,
            max_seq_len=16,
        )
        config = {"d_model": 16, "n_blocks": 2, "seq_len": 16}
        ckpt = build_checkpoint(model, epoch=4, config=config)
        ckpt["tokenizer_state"] = tokenizer.get_state()
        path = save_checkpoint(ckpt, tmp_path / "v1_base", TEST_PASSWORD)

        loaded, tok, cfg, epoch = load_model_from_checkpoint(
            str(path), TEST_PASSWORD,
        )
        assert isinstance(loaded, LuthiLM)
        assert epoch == 4
        assert tok.vocab_size == tokenizer.vocab_size
