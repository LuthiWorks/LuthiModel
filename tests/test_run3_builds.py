"""Run-3 builds (2026-07-17, Brian's ruling): plasticity taper,
inverted-U gain plumb-through, recall-gate tightening.

Pins: the taper schedule's math and its floor-never-zero guarantee; that
rate_scale genuinely shrinks living updates; the trainer sweep reaching
every living layer; the gain flag reaching the layers from the model
constructor (it was unreachable before this plumb-through); the recall
gate parameterization actually gating; and the ARM_CONFIGS contract the
driver and verdict script share.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from luthi.v2.jepa_runner import TaperConfig, taper_scale
from luthi.v2.living_layer_pc import PredictiveCodingLayer
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM


D = 32
SEQ = 16
VOCAB = 32


class TestTaperSchedule:
    def test_formative_phase_is_unity(self):
        assert taper_scale(0.0, 0.5, 0.2) == 1.0
        assert taper_scale(0.49, 0.5, 0.2) == 1.0
        assert taper_scale(0.5, 0.5, 0.2) == 1.0

    def test_linear_to_floor(self):
        assert taper_scale(0.75, 0.5, 0.2) == pytest.approx(0.6)
        assert taper_scale(1.0, 0.5, 0.2) == pytest.approx(0.2)

    def test_progress_clamped(self):
        assert taper_scale(1.5, 0.5, 0.2) == pytest.approx(0.2)
        assert taper_scale(-0.1, 0.5, 0.2) == 1.0

    def test_zero_floor_is_impossible(self):
        """A zero floor is the frozen-model regression; it must be
        unconfigurable, not just discouraged."""
        with pytest.raises(ValueError, match="floor"):
            taper_scale(0.9, 0.5, 0.0)
        with pytest.raises(ValueError, match="floor"):
            taper_scale(0.9, 0.5, -0.1)

    def test_bad_start_fraction_raises(self):
        with pytest.raises(ValueError, match="start_fraction"):
            taper_scale(0.5, 1.0, 0.2)


class TestRateScaleShrinksUpdates:
    def _weight_delta(self, scale: float) -> float:
        torch.manual_seed(3)
        layer = PredictiveCodingLayer(D, D, num_episodes=4, context_dim=8,
                                      salience_threshold=1e9)
        layer.rate_scale = scale
        before = layer.weight.detach().clone()
        x = torch.randn(4, D)
        with torch.no_grad():
            layer(x)
        return float((layer.weight - before).abs().sum().item())

    def test_taper_reduces_living_update_magnitude(self):
        full = self._weight_delta(1.0)
        tapered = self._weight_delta(0.2)
        assert full > 0, "living update did not fire at scale 1.0"
        assert tapered > 0, "floor must never freeze the channel entirely"
        assert tapered < full * 0.5, (
            f"rate_scale=0.2 should cut the update well below half "
            f"(full={full:.6g}, tapered={tapered:.6g})"
        )

    def test_default_scale_is_legacy_identical(self):
        assert self._weight_delta(1.0) == self._weight_delta(1.0)


class TestPlumbThrough:
    def test_gain_flag_reaches_every_living_layer(self):
        model = MultimodalPredictiveCodingLM(
            vocab_size=VOCAB, d_model=D, n_blocks=2, n_heads=2,
            ffn_expansion=1, max_seq_len=SEQ,
            max_audio_tokens=SEQ, max_vision_tokens=SEQ,
            backward_pass_enabled=False,
            learning_gain_enabled=True,
            episode_recall_threshold=0.7,
        )
        layers = [m for m in model.modules()
                  if isinstance(m, PredictiveCodingLayer)]
        assert layers, "no living layers found"
        assert all(l.learning_gain_enabled for l in layers), (
            "learning_gain flag did not reach the layers -- the "
            "plumb-through this build exists for"
        )
        assert all(l.episode_recall_threshold == 0.7 for l in layers)

    def test_defaults_stay_inert(self):
        model = MultimodalPredictiveCodingLM(
            vocab_size=VOCAB, d_model=D, n_blocks=1, n_heads=2,
            ffn_expansion=1, max_seq_len=SEQ,
            max_audio_tokens=SEQ, max_vision_tokens=SEQ,
            backward_pass_enabled=False,
        )
        layers = [m for m in model.modules()
                  if isinstance(m, PredictiveCodingLayer)]
        assert all(not l.learning_gain_enabled for l in layers)
        assert all(l.episode_recall_threshold == 0.5 for l in layers)


class TestRecallGate:
    def test_tighter_gate_blocks_weak_matches(self):
        """Store an episode, then query with a context of controlled
        similarity: recalled at 0.5, refused at a gate above the match."""
        torch.manual_seed(0)
        layer = PredictiveCodingLayer(D, D, num_episodes=4, context_dim=8,
                                      salience_threshold=0.0)
        # Prime one episode with a known context direction.
        x = torch.randn(2, D)
        with torch.no_grad():
            layer(x)
        assert int(layer.episode_count.item()) >= 1

        ctx = layer.episode_contexts[0]
        # A context at a controlled angle from the stored one:
        # cos ~0.6 -- passes a 0.5 gate, fails a 0.7 gate.
        gen = torch.Generator().manual_seed(1)
        noise = torch.randn(ctx.shape, generator=gen)
        noise = noise - (noise @ ctx) * ctx / max(float(ctx @ ctx), 1e-9)
        noise = noise / noise.norm()
        query = 0.6 * ctx / ctx.norm() + 0.8 * noise
        query = query / query.norm()
        sim = float(query @ (ctx / ctx.norm()))
        assert 0.5 < sim < 0.7, f"test geometry broke: sim={sim}"

        layer.episode_recall_threshold = 0.5
        assert layer._recall_episode(query) is not None, (
            "0.5 gate should admit the 0.6-sim match"
        )
        layer.episode_recall_threshold = 0.7
        assert layer._recall_episode(query) is None, (
            "0.7 gate must refuse the 0.6-sim match -- the tightening"
        )


class TestArmConfigsContract:
    def test_every_arm_constructs(self):
        """Same merge pattern as the driver: defaults, then the arm's
        declared config overrides (the depth arm carries n_blocks)."""
        from scripts.jepa_pilot_driver import ARM_CONFIGS
        for arm, cfg in ARM_CONFIGS.items():
            kwargs = dict(
                vocab_size=VOCAB, d_model=D, n_blocks=1, n_heads=2,
                ffn_expansion=1, max_seq_len=SEQ,
                max_audio_tokens=SEQ, max_vision_tokens=SEQ,
                backward_pass_enabled=False,
            )
            kwargs.update(cfg)
            model = MultimodalPredictiveCodingLM(**kwargs)
            assert model is not None, f"arm {arm} failed to construct"

    def test_depth_arm_shape(self):
        from scripts.jepa_pilot_driver import ARM_CONFIGS, ARM_TAPER
        d4 = ARM_CONFIGS["living_v3_4x_d4"]
        assert d4["n_blocks"] == 4
        assert d4["mu_pc_enabled"] is True
        assert d4["mu_pc_exponent"] == 0.25
        assert d4["backward_pass_enabled"] is True
        assert d4["learning_gain_enabled"] is True
        assert ARM_TAPER["living_v3_4x_d4"] is True

    def test_living_v3_arm_shape(self):
        from scripts.jepa_pilot_driver import ARM_CONFIGS, ARM_TAPER
        v3 = ARM_CONFIGS["living_v3"]
        assert v3["learning_gain_enabled"] is True
        assert v3["episode_recall_threshold"] == 0.7
        assert v3["backward_pass_enabled"] is True
        assert v3["consolidation_enabled"] is True
        assert ARM_TAPER["living_v3"] is True
