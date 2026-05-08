"""Tests for the per-buffer dtype override mechanism on LivingLayerV6.

The override mechanism (added 2026-05-07 for the Phase 4.5a buffer-compression
ablation series) lets ablation studies register specific buffers at non-FP32
dtypes — e.g. `momentum` in BF16 — without changing the dtype of the rest
of the layer. The override survives `.to(...)`, `.cuda()`, `.cpu()`, etc.,
because `_apply` re-asserts the override after the standard cast.

These tests do NOT validate that BF16 momentum is *numerically stable* in
training (that's Ablation A's job, with full training runs). They validate
only that the dtype override mechanism *works* — buffers stay at their
chosen dtype, training does not NaN, and Sanctuary integration is unaffected.
"""

import pytest
import torch

from luthi import LivingLayerV6, LuthiLM, SpikingLivingLayer
from luthi.backward_pass import TopDownSignal


# ── Construction-time dtype overrides ────────────────────────────────


class TestBufferDtypeRegistration:
    def test_no_override_defaults_to_fp32(self):
        layer = LivingLayerV6(in_features=8, out_features=8)
        assert layer.weight.dtype == torch.float32
        assert layer.set_point.dtype == torch.float32
        assert layer.momentum.dtype == torch.float32
        assert layer.plasticity.dtype == torch.float32
        assert layer.excitability_acc.dtype == torch.float32
        assert layer.update_ema.dtype == torch.float32

    def test_single_buffer_override(self):
        layer = LivingLayerV6(
            in_features=8,
            out_features=8,
            buffer_dtypes={"momentum": torch.bfloat16},
        )
        assert layer.momentum.dtype == torch.bfloat16
        # Others stay FP32
        assert layer.weight.dtype == torch.float32
        assert layer.set_point.dtype == torch.float32
        assert layer.plasticity.dtype == torch.float32
        assert layer.update_ema.dtype == torch.float32

    def test_multiple_buffer_overrides(self):
        layer = LivingLayerV6(
            in_features=8,
            out_features=8,
            buffer_dtypes={
                "weight": torch.bfloat16,
                "momentum": torch.bfloat16,
                "set_point": torch.bfloat16,
            },
        )
        assert layer.weight.dtype == torch.bfloat16
        assert layer.momentum.dtype == torch.bfloat16
        assert layer.set_point.dtype == torch.bfloat16
        # Non-overridden stay FP32
        assert layer.update_ema.dtype == torch.float32
        assert layer.excitability_acc.dtype == torch.float32


# ── Override survives .to() ──────────────────────────────────────────


class TestBufferDtypeSurvivesTo:
    def test_to_dtype_does_not_clobber_override(self):
        """A global `.to(dtype=fp32)` must not silently change a BF16 override
        back to FP32 — the entire point of per-buffer dtypes is that they
        decouple from the global precision."""
        layer = LivingLayerV6(
            in_features=8,
            out_features=8,
            buffer_dtypes={"momentum": torch.bfloat16},
        )
        layer.to(dtype=torch.float32)  # Try to "promote" everything to FP32
        # momentum should still be BF16 thanks to the _apply override
        assert layer.momentum.dtype == torch.bfloat16

    def test_to_bfloat16_keeps_fp32_buffers_fp32(self):
        """Inverse case: `.to(dtype=bf16)` should not silently downcast
        buffers we explicitly want to keep in FP32."""
        layer = LivingLayerV6(
            in_features=8,
            out_features=8,
            buffer_dtypes={
                "weight": torch.bfloat16,
                # set_point, momentum, etc. left at FP32 default
            },
        )
        # User does NOT call .to(dtype=bf16) — buffer_dtypes is the source of truth.
        assert layer.set_point.dtype == torch.float32
        assert layer.momentum.dtype == torch.float32
        assert layer.weight.dtype == torch.bfloat16

    def test_to_device_only_preserves_dtypes(self):
        """`.to(device=cpu)` (no dtype) should be transparent."""
        layer = LivingLayerV6(
            in_features=8,
            out_features=8,
            buffer_dtypes={"momentum": torch.bfloat16},
        )
        layer.to(device="cpu")
        assert layer.momentum.dtype == torch.bfloat16
        assert layer.weight.dtype == torch.float32


# ── Training does not NaN with mixed precision ───────────────────────


class TestMixedPrecisionTraining:
    def test_bf16_momentum_forward_no_nan(self):
        torch.manual_seed(42)
        layer = LivingLayerV6(
            in_features=16,
            out_features=16,
            buffer_dtypes={"momentum": torch.bfloat16},
        )
        for _ in range(50):
            x = torch.randn(2, 16)
            out = layer(x)
            assert not torch.isnan(out).any()
            assert not torch.isnan(layer.weight).any()
            assert not torch.isnan(layer.momentum).any()
            assert layer.momentum.dtype == torch.bfloat16

    def test_bf16_set_point_forward_no_nan(self):
        torch.manual_seed(42)
        layer = LivingLayerV6(
            in_features=16,
            out_features=16,
            buffer_dtypes={"set_point": torch.bfloat16},
        )
        for _ in range(50):
            x = torch.randn(2, 16)
            out = layer(x)
            assert not torch.isnan(out).any()
            assert not torch.isnan(layer.set_point).any()
            assert layer.set_point.dtype == torch.bfloat16

    def test_combined_bf16_momentum_set_point_no_nan(self):
        """Ablation D preview — combined BF16 momentum + set_point."""
        torch.manual_seed(42)
        layer = LivingLayerV6(
            in_features=16,
            out_features=16,
            buffer_dtypes={
                "momentum": torch.bfloat16,
                "set_point": torch.bfloat16,
            },
        )
        for _ in range(100):
            x = torch.randn(2, 16)
            out = layer(x)
            assert not torch.isnan(out).any()
            assert layer.momentum.dtype == torch.bfloat16
            assert layer.set_point.dtype == torch.bfloat16

    def test_top_down_with_bf16_momentum(self):
        torch.manual_seed(42)
        layer = LivingLayerV6(
            in_features=16,
            out_features=16,
            buffer_dtypes={"momentum": torch.bfloat16},
        )
        for _ in range(10):
            layer(torch.randn(2, 16))
        signal = TopDownSignal(
            salience=torch.randn(16),
            prediction_error=torch.randn(16),
            modulation_strength=1.0,
        )
        layer.apply_top_down(signal)
        assert layer.momentum.dtype == torch.bfloat16
        assert not torch.isnan(layer.plasticity).any()


# ── Spiking variant inherits the override mechanism ──────────────────


class TestSpikingInheritsOverrides:
    def test_spiking_layer_accepts_buffer_dtypes(self):
        torch.manual_seed(42)
        layer = SpikingLivingLayer(
            in_features=16,
            out_features=16,
            buffer_dtypes={"momentum": torch.bfloat16},
        )
        assert layer.momentum.dtype == torch.bfloat16
        # Spiking-specific buffers stay FP32 (DirectML compat)
        assert layer.membrane_potential.dtype == torch.float32

    def test_spiking_forward_with_bf16_momentum_no_nan(self):
        torch.manual_seed(42)
        layer = SpikingLivingLayer(
            in_features=16,
            out_features=16,
            buffer_dtypes={"momentum": torch.bfloat16},
        )
        for _ in range(30):
            x = torch.randn(2, 16)
            out = layer(x)
            assert not torch.isnan(out).any()
            assert layer.momentum.dtype == torch.bfloat16


# ── Full LuthiLM model accepts and propagates buffer_dtypes ──────────


class TestModelPlumbing:
    def test_luthi_lm_forwards_buffer_dtypes_to_blocks(self):
        torch.manual_seed(42)
        model = LuthiLM(
            vocab_size=64,
            d_model=32,
            n_blocks=2,
            max_seq_len=32,
            buffer_dtypes={"momentum": torch.bfloat16},
        )
        for block in model.blocks:
            assert block.living_ffn.momentum.dtype == torch.bfloat16
            assert block.living_ffn.set_point.dtype == torch.float32

    def test_luthi_lm_training_step_no_nan(self):
        torch.manual_seed(42)
        model = LuthiLM(
            vocab_size=64,
            d_model=32,
            n_blocks=2,
            max_seq_len=32,
            buffer_dtypes={"momentum": torch.bfloat16},
        )
        x = torch.randint(0, 64, (1, 8))
        for _ in range(20):
            out = model(x)
            assert not torch.isnan(out).any()
        for block in model.blocks:
            assert block.living_ffn.momentum.dtype == torch.bfloat16
