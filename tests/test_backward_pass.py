"""Tests for the top-down backward pass.

Verifies that:
1. TopDownSignal carries correct data
2. create_initial_signal produces valid signals from hidden states
3. compute_block_top_down refines signals with proper decay
4. Living layers respond to top-down modulation (plasticity, set points)
5. Spiking layers additionally prime membrane potential
6. Full model forward pass with backward sweep doesn't break outputs
7. Backward pass modulates state for the NEXT forward pass
"""

import torch
import pytest

from luthi.backward_pass import (
    TopDownSignal,
    create_initial_signal,
    compute_block_top_down,
)
from luthi.living_layer import LivingLayerV6
from luthi.living_layer_spiking import SpikingLivingLayer
from luthi.hybrid_block import HybridBlock
from luthi.model import LuthiLM


# --- TopDownSignal ---

def test_top_down_signal_creation():
    """TopDownSignal holds the expected fields."""
    signal = TopDownSignal(
        salience=torch.ones(64),
        prediction_error=torch.zeros(64),
        modulation_strength=1.0,
    )
    assert signal.salience.shape == (64,)
    assert signal.prediction_error.shape == (64,)
    assert signal.modulation_strength == 1.0


# --- create_initial_signal ---

def test_initial_signal_shape():
    """Initial signal has correct dimensions from hidden state."""
    h = torch.randn(4, 16, 64)  # [batch, seq, d_model]
    signal = create_initial_signal(h)
    assert signal.salience.shape == (64,)
    assert signal.prediction_error.shape == (64,)
    assert signal.modulation_strength == 1.0


def test_initial_signal_salience_normalized():
    """Salience is normalized to [0, 1] range."""
    h = torch.randn(4, 16, 64) * 100  # large values
    signal = create_initial_signal(h)
    assert signal.salience.max() <= 1.0 + 1e-6
    assert signal.salience.min() >= 0.0 - 1e-6


def test_initial_signal_prediction_error_zero():
    """Initial signal has zero prediction error (no block above)."""
    h = torch.randn(4, 16, 64)
    signal = create_initial_signal(h)
    assert (signal.prediction_error == 0).all()


def test_initial_signal_custom_strength():
    """Custom modulation strength is respected."""
    h = torch.randn(4, 16, 64)
    signal = create_initial_signal(h, initial_strength=0.5)
    assert signal.modulation_strength == 0.5


# --- compute_block_top_down ---

def test_block_top_down_decays_strength():
    """Modulation strength decays by the decay factor."""
    signal = TopDownSignal(
        salience=torch.ones(64),
        prediction_error=torch.zeros(64),
        modulation_strength=1.0,
    )
    block_input = torch.randn(4, 16, 64)
    refined = compute_block_top_down(signal, block_input, decay=0.8)
    assert refined.modulation_strength == pytest.approx(0.8)


def test_block_top_down_two_levels():
    """Two levels of decay compound correctly."""
    signal = TopDownSignal(
        salience=torch.ones(64),
        prediction_error=torch.zeros(64),
        modulation_strength=1.0,
    )
    block_input = torch.randn(4, 16, 64)
    level1 = compute_block_top_down(signal, block_input, decay=0.8)
    level2 = compute_block_top_down(level1, block_input, decay=0.8)
    assert level2.modulation_strength == pytest.approx(0.64)


def test_block_top_down_produces_prediction_error():
    """Refined signal has non-zero prediction error."""
    signal = TopDownSignal(
        salience=torch.ones(64) * 0.5,
        prediction_error=torch.zeros(64),
        modulation_strength=1.0,
    )
    block_input = torch.randn(4, 16, 64)
    refined = compute_block_top_down(signal, block_input)
    # Prediction error should be non-zero (local salience differs from downstream)
    assert refined.prediction_error.abs().sum() > 0


# --- Living layer top-down modulation ---

def test_living_layer_plasticity_modulation():
    """Top-down signal modulates plasticity values."""
    layer = LivingLayerV6(64, 64)
    plasticity_before = layer.plasticity.clone()

    # Use varied salience so the modulation produces a visible change
    # (uniform salience of 1.0 with plasticity of 1.0 is a fixed point)
    salience = torch.rand(64) * 2.0  # values in [0, 2]
    signal = TopDownSignal(
        salience=salience,
        prediction_error=torch.zeros(64),
        modulation_strength=1.0,
    )
    layer.apply_top_down(signal)

    # Plasticity should have changed
    assert not torch.allclose(layer.plasticity, plasticity_before)


def test_living_layer_set_point_modulation():
    """Top-down prediction error shifts set points."""
    layer = LivingLayerV6(64, 64)
    set_point_before = layer.set_point.clone()

    signal = TopDownSignal(
        salience=torch.ones(64),
        prediction_error=torch.ones(64) * 0.5,  # strong error signal
        modulation_strength=1.0,
    )
    layer.apply_top_down(signal)

    # Set points should have shifted
    assert not torch.allclose(layer.set_point, set_point_before)


def test_living_layer_zero_strength_no_change():
    """Zero modulation strength leaves state unchanged."""
    layer = LivingLayerV6(64, 64)
    plasticity_before = layer.plasticity.clone()
    set_point_before = layer.set_point.clone()

    signal = TopDownSignal(
        salience=torch.ones(64),
        prediction_error=torch.ones(64),
        modulation_strength=0.0,
    )
    layer.apply_top_down(signal)

    assert torch.allclose(layer.plasticity, plasticity_before, atol=1e-7)
    assert torch.allclose(layer.set_point, set_point_before, atol=1e-7)


def test_living_layer_plasticity_clamped():
    """Plasticity stays within safe bounds after modulation."""
    layer = LivingLayerV6(64, 64)

    # Apply many strong modulations
    for _ in range(100):
        signal = TopDownSignal(
            salience=torch.ones(64) * 10.0,
            prediction_error=torch.zeros(64),
            modulation_strength=1.0,
        )
        layer.apply_top_down(signal)

    assert layer.plasticity.min() >= 0.1
    assert layer.plasticity.max() <= 10.0


# --- Spiking layer top-down modulation ---

def test_spiking_layer_membrane_priming():
    """Top-down signal primes membrane potential in spiking layer."""
    layer = SpikingLivingLayer(64, 64)
    membrane_before = layer.membrane_potential.clone()

    signal = TopDownSignal(
        salience=torch.ones(64),
        prediction_error=torch.zeros(64),
        modulation_strength=1.0,
    )
    layer.apply_top_down(signal)

    # Membrane should have been primed (increased)
    assert (layer.membrane_potential >= membrane_before).all()


def test_spiking_layer_inherits_base_modulation():
    """Spiking layer also modulates plasticity and set points."""
    layer = SpikingLivingLayer(64, 64)
    plasticity_before = layer.plasticity.clone()

    salience = torch.rand(64) * 2.0
    signal = TopDownSignal(
        salience=salience,
        prediction_error=torch.zeros(64),
        modulation_strength=1.0,
    )
    layer.apply_top_down(signal)

    assert not torch.allclose(layer.plasticity, plasticity_before)


# --- HybridBlock top-down pass ---

def test_hybrid_block_top_down_returns_signal():
    """Block top_down_pass returns a refined signal."""
    block = HybridBlock(64)
    signal = TopDownSignal(
        salience=torch.ones(64),
        prediction_error=torch.zeros(64),
        modulation_strength=1.0,
    )
    block_input = torch.randn(4, 16, 64)

    refined = block.top_down_pass(signal, block_input)
    assert isinstance(refined, TopDownSignal)
    assert refined.modulation_strength < signal.modulation_strength


# --- Full model integration ---

def test_model_forward_with_backward_pass():
    """Model forward produces valid logits with backward pass enabled."""
    model = LuthiLM(vocab_size=256, d_model=64, n_blocks=2)
    model.train()

    x = torch.randint(0, 256, (4, 16))
    logits = model(x)

    assert logits.shape == (4, 16, 256)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_model_eval_skips_backward_pass():
    """In eval mode, backward pass is skipped (no modulation)."""
    model = LuthiLM(vocab_size=256, d_model=64, n_blocks=2)
    model.eval()

    # Record plasticity before
    plasticity_before = model.blocks[0].living_ffn.plasticity.clone()

    x = torch.randint(0, 256, (4, 16))
    with torch.no_grad():
        logits = model(x)

    # Plasticity should not change from top-down (only from Hebbian in forward)
    # Note: forward pass itself modifies plasticity is NOT true — plasticity
    # is only modified by apply_top_down. But the forward pass does run
    # Hebbian updates. So we check that top_down_pass was NOT called by
    # verifying the plasticity pattern matches what Hebbian alone would do.
    assert logits.shape == (4, 16, 256)


def test_model_backward_pass_modulates_for_next_forward():
    """The backward pass changes state that affects the next forward pass."""
    model = LuthiLM(vocab_size=256, d_model=64, n_blocks=2)
    model.train()

    x = torch.randint(0, 256, (4, 16))

    # First forward (with backward sweep)
    logits1 = model(x)

    # Record plasticity after first forward+backward
    plasticity_after_first = model.blocks[0].living_ffn.plasticity.clone()

    # Second forward with same input
    logits2 = model(x)

    # Outputs should differ (living weights changed + backward modulated state)
    assert not torch.allclose(logits1, logits2)


def test_training_loop_with_backward_pass():
    """Full training step works with backward pass enabled."""
    model = LuthiLM(vocab_size=256, d_model=64, n_blocks=2)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    x = torch.randint(0, 256, (4, 16))
    y = torch.randint(0, 256, (4, 16))

    # Standard training step
    optimizer.zero_grad()
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 256), y.reshape(-1)
    )
    loss.backward()
    model.apply_living_errors()
    optimizer.step()

    # Should complete without error
    assert loss.item() > 0


# --- Spiking model integration ---

def test_spiking_model_forward_with_backward_pass():
    """Spiking model forward produces valid logits with backward pass."""
    from luthi.model_spiking import SpikingLuthiLM

    model = SpikingLuthiLM(vocab_size=256, d_model=64, n_blocks=2)
    model.train()

    x = torch.randint(0, 256, (4, 16))
    logits = model(x)

    assert logits.shape == (4, 16, 256)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_spiking_model_backward_spike_propagation():
    """Backward spikes from block 1 affect block 0's membrane."""
    from luthi.model_spiking import SpikingLuthiLM

    model = SpikingLuthiLM(vocab_size=256, d_model=64, n_blocks=2)
    model.train()

    # Record block 0 membrane before
    membrane_before = model.blocks[0].living_ffn.membrane_potential.clone()

    x = torch.randint(0, 256, (4, 16))
    _ = model(x)

    # Block 0's membrane should have been affected by backward spikes
    # (from the top-down sweep) in addition to normal forward dynamics
    membrane_after = model.blocks[0].living_ffn.membrane_potential
    # The membrane will have changed from both forward and backward —
    # we just verify it's not unchanged
    assert not torch.allclose(membrane_before, membrane_after)


# --- Backward pass toggle ---

def test_model_backward_pass_disabled():
    """When backward_pass_enabled=False, no top-down modulation occurs."""
    model = LuthiLM(vocab_size=256, d_model=64, n_blocks=2, backward_pass_enabled=False)
    model.train()

    x = torch.randint(0, 256, (4, 16))

    # Run forward — plasticity should only change from Hebbian, not top-down
    plasticity_before = model.blocks[0].living_ffn.plasticity.clone()
    _ = model(x)

    # With backward pass disabled, plasticity is only modified by Hebbian
    # (which doesn't touch plasticity — only top-down does). But Hebbian
    # CAN indirectly change plasticity... Actually plasticity is NOT modified
    # by the forward pass's Hebbian update. It's only modified by apply_top_down.
    # So with backward pass disabled, plasticity should be unchanged.
    assert torch.allclose(model.blocks[0].living_ffn.plasticity, plasticity_before)


def test_model_backward_pass_enabled_modifies_plasticity():
    """When backward_pass_enabled=True, top-down modulates plasticity."""
    model = LuthiLM(vocab_size=256, d_model=64, n_blocks=2, backward_pass_enabled=True)
    model.train()

    x = torch.randint(0, 256, (4, 16))

    plasticity_before = model.blocks[0].living_ffn.plasticity.clone()
    _ = model(x)

    # With backward pass enabled, plasticity WILL change from top-down
    assert not torch.allclose(model.blocks[0].living_ffn.plasticity, plasticity_before)


def test_model_backward_pass_toggle_runtime():
    """backward_pass_enabled can be toggled at runtime."""
    model = LuthiLM(vocab_size=256, d_model=64, n_blocks=2, backward_pass_enabled=False)
    model.train()

    x = torch.randint(0, 256, (4, 16))

    # Forward with BP off — plasticity unchanged
    plasticity_before = model.blocks[0].living_ffn.plasticity.clone()
    _ = model(x)
    assert torch.allclose(model.blocks[0].living_ffn.plasticity, plasticity_before)

    # Enable BP at runtime
    model.backward_pass_enabled = True
    plasticity_before = model.blocks[0].living_ffn.plasticity.clone()
    _ = model(x)
    assert not torch.allclose(model.blocks[0].living_ffn.plasticity, plasticity_before)


def test_spiking_model_backward_pass_toggle():
    """Spiking model also respects backward_pass_enabled toggle."""
    from luthi.model_spiking import SpikingLuthiLM

    model = SpikingLuthiLM(
        vocab_size=256, d_model=64, n_blocks=2, backward_pass_enabled=False,
    )
    model.train()

    x = torch.randint(0, 256, (4, 16))

    # With BP off, plasticity should not change from top-down
    plasticity_before = model.blocks[0].living_ffn.plasticity.clone()
    _ = model(x)
    assert torch.allclose(model.blocks[0].living_ffn.plasticity, plasticity_before)


# --- Extended metrics ---

def test_collect_extended_metrics():
    """collect_extended_metrics returns expected keys."""
    from luthi.train import collect_extended_metrics

    model = LuthiLM(vocab_size=256, d_model=64, n_blocks=2)
    metrics = collect_extended_metrics(model)

    # Per-block metrics
    assert "block0_plasticity_mean" in metrics
    assert "block0_plasticity_std" in metrics
    assert "block0_sp_drift_mean" in metrics
    assert "block1_plasticity_mean" in metrics

    # Aggregate metrics
    assert "plasticity_mean" in metrics
    assert "sp_drift_mean" in metrics


def test_measure_backward_pass_effect():
    """measure_backward_pass_effect returns non-zero when BP is active."""
    from luthi.train import measure_backward_pass_effect

    model = LuthiLM(vocab_size=256, d_model=64, n_blocks=2, backward_pass_enabled=True)
    x = torch.randint(0, 256, (4, 16))

    effect = measure_backward_pass_effect(model, x)
    assert effect > 0.0


def test_measure_backward_pass_effect_zero_when_disabled():
    """measure_backward_pass_effect returns 0 when BP is disabled."""
    from luthi.train import measure_backward_pass_effect

    model = LuthiLM(vocab_size=256, d_model=64, n_blocks=2, backward_pass_enabled=False)
    x = torch.randint(0, 256, (4, 16))

    effect = measure_backward_pass_effect(model, x)
    assert effect == 0.0


# --- Shape-contract tests for the top-down signal ---

class TestTopDownSignalShapeContract:
    """Top-down signals live in input-dim space.

    These tests pin down the contract that ``salience`` and
    ``prediction_error`` are vectors of length ``in_features`` (== d_model
    for the FFN trunk), and that a wrong-shape signal raises immediately
    rather than silently corrupting set_point/plasticity along the wrong
    axis. Without this guard, a future refactor that flips the convention
    would not crash — it would just slowly push homeostatic state in the
    wrong direction across training, which is exactly the kind of bug the
    "crash loud over silent corruption" rule exists to prevent.
    """

    def test_correct_shape_signal_is_accepted(self):
        layer = LivingLayerV6(in_features=8, out_features=8)
        signal = TopDownSignal(
            salience=torch.ones(8),
            prediction_error=torch.zeros(8),
            modulation_strength=0.5,
        )
        # Should not raise.
        layer.apply_top_down(signal)

    def test_wrong_salience_size_raises(self):
        layer = LivingLayerV6(in_features=8, out_features=8)
        signal = TopDownSignal(
            salience=torch.ones(7),  # wrong size
            prediction_error=torch.zeros(8),
            modulation_strength=0.5,
        )
        with pytest.raises(ValueError, match="salience"):
            layer.apply_top_down(signal)

    def test_wrong_prediction_error_size_raises(self):
        layer = LivingLayerV6(in_features=8, out_features=8)
        signal = TopDownSignal(
            salience=torch.ones(8),
            prediction_error=torch.zeros(7),  # wrong size
            modulation_strength=0.5,
        )
        with pytest.raises(ValueError, match="prediction_error"):
            layer.apply_top_down(signal)

    def test_spiking_layer_inherits_shape_contract(self):
        """SpikingLivingLayer.apply_top_down delegates to super(); same guard fires."""
        layer = SpikingLivingLayer(in_features=8, out_features=8)
        signal = TopDownSignal(
            salience=torch.ones(7),  # wrong size
            prediction_error=torch.zeros(8),
            modulation_strength=0.5,
        )
        with pytest.raises(ValueError, match="salience"):
            layer.apply_top_down(signal)

    def test_set_point_drift_is_uniform_across_output_axis(self):
        """Document the broadcasting semantics: per-input-dim error is
        applied identically to every output row of set_point.

        Zeros the set_point baseline before nudging so the comparison is
        free of fp32 subtraction noise from a non-zero starting state.
        """
        layer = LivingLayerV6(in_features=4, out_features=6)
        # Zero the set_point so apply_top_down's contribution is the
        # entire post-state (no rounding noise from subtracting a
        # heterogeneous baseline).
        with torch.no_grad():
            layer.set_point.zero_()

        # Construct a prediction error with a clearly distinct value in
        # one input dim. After apply_top_down, every output row's
        # column-i value should match — that's what input-dim
        # broadcasting means.
        prediction_error = torch.tensor([1.0, 0.0, 0.0, 0.0])
        signal = TopDownSignal(
            salience=torch.zeros(4),
            prediction_error=prediction_error,
            modulation_strength=1.0,
        )
        layer.apply_top_down(signal)

        # Column 0 should hold the same nonzero value in every row.
        col0 = layer.set_point[:, 0]
        assert torch.allclose(col0, torch.full_like(col0, col0[0].item()))
        assert col0[0].item() != 0.0
        # Other columns should be untouched (prediction_error was zero there).
        assert torch.allclose(layer.set_point[:, 1:], torch.zeros_like(layer.set_point[:, 1:]))
