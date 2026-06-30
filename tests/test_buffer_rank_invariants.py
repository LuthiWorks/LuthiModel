"""Empirical validation that per-weight buffers `plasticity` and
`excitability_acc` are mathematically rank-1 by the structure of their
update rules.

If these invariants hold across all current code paths, the buffers can be
stored as 1D vectors (`[in_features]` for plasticity, `[out_features]` for
excitability_acc) without changing behavior — a free ~16 bytes/param memory
reduction in the living FFN.

This test runs on the current per-weight storage and confirms that no code
path breaks rank-1. If any test here fails, there is a modification site
not accounted for in `docs/PER_CHANNEL_ABLATION_PROTOCOL.md`, and the
per-channel refactor must be revisited before proceeding.

The invariants tested:
- `plasticity[i, :] == plasticity[j, :]` for all output rows i, j
  (every row is identical — buffer is rank-1 along the input axis)
- `excitability_acc[:, i] == excitability_acc[:, j]` for all input cols i, j
  (every column is identical — buffer is rank-1 along the output axis)
"""

import pytest
import torch

from luthi import LivingLayerV6, SpikingLivingLayer, LuthiLM
from luthi.backward_pass import TopDownSignal
from luthi.sanctuary_interface import modulated


# ── Invariant predicates ─────────────────────────────────────────────


def is_rank1_along_input_axis(buffer: torch.Tensor) -> bool:
    """True iff `buffer` is rank-1 along the input axis.

    A 1D buffer of shape `[in]` is the canonical (and most compact) rank-1
    form — there is no output axis to vary along. A 2D buffer of shape
    `[out, in]` is rank-1 along input iff every output row is identical.
    """
    if buffer.dim() == 1:
        return True
    if buffer.dim() != 2:
        return False
    first_row = buffer[0:1, :]  # [1, in]
    return torch.allclose(first_row.expand_as(buffer), buffer, atol=1e-6)


def is_rank1_along_output_axis(buffer: torch.Tensor) -> bool:
    """True iff `buffer` is rank-1 along the output axis.

    A 1D buffer of shape `[out]` is the canonical rank-1 form. A 2D buffer
    of shape `[out, in]` is rank-1 along output iff every input column is
    identical.
    """
    if buffer.dim() == 1:
        return True
    if buffer.dim() != 2:
        return False
    first_col = buffer[:, 0:1]  # [out, 1]
    return torch.allclose(first_col.expand_as(buffer), buffer, atol=1e-6)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def layer():
    torch.manual_seed(42)
    return LivingLayerV6(in_features=16, out_features=16)


@pytest.fixture
def spiking_layer():
    torch.manual_seed(42)
    return SpikingLivingLayer(in_features=16, out_features=16)


@pytest.fixture
def small_model():
    torch.manual_seed(42)
    return LuthiLM(
        vocab_size=64,
        d_model=32,
        n_blocks=2,
        max_seq_len=32,
        backward_pass_enabled=True,
    )


def _make_top_down_signal(in_features: int, seed: int = 0) -> TopDownSignal:
    g = torch.Generator().manual_seed(seed)
    return TopDownSignal(
        salience=torch.randn(in_features, generator=g),
        prediction_error=torch.randn(in_features, generator=g),
        modulation_strength=1.0,
    )


# ── Plasticity: rank-1 along input axis ──────────────────────────────


class TestPlasticityRank1AlongInput:
    """plasticity[i, :] is identical for every output row i."""

    def test_initial_state_is_rank1(self, layer):
        """torch.ones initialization is trivially rank-1."""
        assert is_rank1_along_input_axis(layer.plasticity)

    def test_one_top_down_preserves_rank1(self, layer):
        signal = _make_top_down_signal(layer.in_features, seed=1)
        layer.apply_top_down(signal)
        assert is_rank1_along_input_axis(layer.plasticity)

    def test_many_top_downs_preserve_rank1(self, layer):
        for s in range(20):
            signal = _make_top_down_signal(layer.in_features, seed=s)
            layer.apply_top_down(signal)
            assert is_rank1_along_input_axis(
                layer.plasticity
            ), f"Broken at iteration {s}"

    def test_clamp_preserves_rank1(self, layer):
        """The clamp_(0.01, 10.0) in apply_top_down is element-wise; rank-1
        input must yield rank-1 output. (Lower bound harmonized 0.1 -> 0.01
        on 2026-06-2x to match the v2 substrate, living_layer_pc.py:662;
        this test's constant was stale until 2026-06-29.)"""
        # Force plasticity values to span the clamp boundaries
        for s in range(50):
            signal = _make_top_down_signal(layer.in_features, seed=100 + s)
            # Amplify to push values outside [0.01, 10.0]
            signal = TopDownSignal(
                salience=signal.salience * 50.0,
                prediction_error=signal.prediction_error,
                modulation_strength=10.0,
            )
            layer.apply_top_down(signal)
        # Should have hit the clamp by now
        assert layer.plasticity.min() >= 0.01
        assert layer.plasticity.max() <= 10.0
        assert is_rank1_along_input_axis(layer.plasticity)

    def test_forward_pass_does_not_modify_plasticity(self, layer):
        """Confirm plasticity is not written to during forward."""
        before = layer.plasticity.clone()
        x = torch.randn(2, 16)
        for _ in range(10):
            layer(x)
        assert torch.equal(before, layer.plasticity)
        assert is_rank1_along_input_axis(layer.plasticity)


# ── Excitability_acc: rank-1 along output axis ───────────────────────


class TestExcitabilityRank1AlongOutput:
    """excitability_acc[:, j] is identical for every input column j."""

    def test_initial_state_is_rank1(self, layer):
        """torch.zeros initialization is trivially rank-1."""
        assert is_rank1_along_output_axis(layer.excitability_acc)

    def test_one_forward_preserves_rank1(self, layer):
        x = torch.randn(2, 16)
        layer(x)
        assert is_rank1_along_output_axis(layer.excitability_acc)

    def test_many_forwards_preserve_rank1(self, layer):
        for i in range(50):
            x = torch.randn(2, 16)
            layer(x)
            assert is_rank1_along_output_axis(
                layer.excitability_acc
            ), f"Broken at iteration {i}"

    def test_sanctuary_scalar_bias_preserves_rank1(self, layer):
        """sanctuary_interface.apply_external_modulation adds a scalar."""
        # Run some forwards to get a non-zero starting state
        for _ in range(10):
            layer(torch.randn(2, 16))
        layer.excitability_acc.add_(0.5)
        assert is_rank1_along_output_axis(layer.excitability_acc)

    def test_modulated_context_preserves_rank1(self, small_model):
        """End-to-end: Sanctuary's modulated() context manager."""
        x = torch.randint(0, 64, (1, 8))
        # Bake in some history first
        for _ in range(5):
            small_model(x)
        with modulated(small_model, excitability_bias=0.3):
            for _ in range(10):
                small_model(x)
        for block in small_model.blocks:
            assert is_rank1_along_output_axis(block.living_ffn.excitability_acc)


# ── Combined: invariant holds during realistic training-style loop ───


class TestRankInvariantsUnderTraining:
    """A 200-step forward+top-down loop must preserve both invariants."""

    def test_invariants_hold_in_full_model(self, small_model):
        x = torch.randint(0, 64, (1, 8))
        for step in range(200):
            small_model(x)
            for i, block in enumerate(small_model.blocks):
                ffn = block.living_ffn
                assert is_rank1_along_input_axis(
                    ffn.plasticity
                ), f"plasticity rank-1 broken at step {step}, block {i}"
                assert is_rank1_along_output_axis(
                    ffn.excitability_acc
                ), f"excitability_acc rank-1 broken at step {step}, block {i}"


# ── Spiking variant: same invariants must hold ───────────────────────


class TestSpikingRankInvariants:
    """SpikingLivingLayer inherits the buffer structure; invariants must
    survive the spike-gated update path too."""

    def test_initial_state(self, spiking_layer):
        assert is_rank1_along_input_axis(spiking_layer.plasticity)
        assert is_rank1_along_output_axis(spiking_layer.excitability_acc)

    def test_forward_passes(self, spiking_layer):
        for i in range(50):
            x = torch.randn(2, 16)
            spiking_layer(x)
            assert is_rank1_along_input_axis(
                spiking_layer.plasticity
            ), f"plasticity broken at step {i}"
            assert is_rank1_along_output_axis(
                spiking_layer.excitability_acc
            ), f"excitability_acc broken at step {i}"

    def test_top_down_with_spikes(self, spiking_layer):
        """apply_top_down on the spiking variant should still preserve
        plasticity rank-1."""
        # Run forwards first to seed spike history
        for _ in range(10):
            spiking_layer(torch.randn(2, 16))
        for s in range(20):
            signal = _make_top_down_signal(spiking_layer.in_features, seed=s)
            spiking_layer.apply_top_down(signal)
            assert is_rank1_along_input_axis(
                spiking_layer.plasticity
            ), f"Broken at iteration {s}"
            assert is_rank1_along_output_axis(
                spiking_layer.excitability_acc
            ), f"Broken at iteration {s}"
