"""M2 unit tests for PredictiveCodingBlock + two-channel top-down sweep.

Per `docs/V2_IMPLEMENTATION_PLAN.md` M2 (Days 4-5) and the 2026-05-08
refinement 3 isolation suite.

Tests:
  1. Forward+backward decreasing error on a fixed input pattern.
  2. Prediction-only sweep (apply_modulation=False): pred_error still
     decreases via intrinsic PC dynamics, weight updates still flow.
  3. Modulation-only sweep (synthetic external signal, no real PC
     prediction): plasticity/set_point modulate as v1 backward pass.
  4. Joint non-interference: drift in the joint case is within 15% of
     the linear sum of isolated-channel drifts.
"""

import pytest
import torch

from luthi.v2 import (
    PredictiveCodingBlock,
    PredictiveCodingLayer,
    TopDownSignal,
    create_initial_signal,
)


def _make_block(
    d_model: int = 16,
    pred_learning_rate: float = 0.0001,
    ffn_expansion: int = 1,
) -> PredictiveCodingBlock:
    return PredictiveCodingBlock(
        d_model=d_model,
        num_episodes=4,
        pred_learning_rate=pred_learning_rate,
        ffn_expansion=ffn_expansion,
    )


def _make_layer(d: int = 16) -> PredictiveCodingLayer:
    return PredictiveCodingLayer(in_features=d, out_features=d, num_episodes=4)


def _measure_layer_pred_error(
    block: PredictiveCodingBlock, x: torch.Tensor
) -> float:
    """Probe the block's PC layer pred_error on a fixed input without
    running the full block forward (which would mutate state).
    """
    with torch.no_grad():
        layer = block.living_ffn
        ffn_in = block.norm2(x + block.attention(block.norm1(x)))
        x_flat = ffn_in.reshape(-1, layer.in_features)
        output = x_flat @ layer.weight.T
        output_mean = output.mean(dim=0)
        actual = x_flat.mean(dim=0)
        predicted = output_mean @ layer.prediction
        return (actual - predicted).abs().mean().item()


# 1. Baseline: forward+backward decreasing error -----------------------------

def test_forward_backward_decreasing_error():
    """Repeated forward+top_down on a fixed input pattern: the PC layer's
    pred_error decreases over 200 steps. This is the M2 baseline gate.

    Uses pred_learning_rate=0.01 (100x default) to make convergence visible
    in 200 steps — same M1-test-3 rationale (refinement 1's M3 grid search
    is the right place to tune production HPs).
    """
    torch.manual_seed(0)
    block = _make_block(pred_learning_rate=0.01)

    x_fixed = torch.randn(2, 8, 16) * 0.5
    initial_err = _measure_layer_pred_error(block, x_fixed)

    for _ in range(200):
        out = block(x_fixed)
        signal = create_initial_signal(out)
        block.top_down_pass(signal)

    final_err = _measure_layer_pred_error(block, x_fixed)
    assert final_err < initial_err * 0.9, (
        f"Block pred_error did not decrease after 200 steps: "
        f"{initial_err:.4f} -> {final_err:.4f}"
    )


# 2. Prediction-only (no top-down modulation) --------------------------------

def test_prediction_only_sweep():
    """Refinement 3: with apply_modulation=False the top-down signal is
    not applied to the layer, but the layer's intrinsic PC dynamics still
    drive pred_error down and weight still updates.

    Same pred_learning_rate=0.01 override as test 1 — both are mechanism
    checks, not production-HP checks.
    """
    torch.manual_seed(0)
    block = _make_block(pred_learning_rate=0.01)

    x_fixed = torch.randn(2, 8, 16) * 0.5
    initial_err = _measure_layer_pred_error(block, x_fixed)
    initial_weight = block.living_ffn.weight.clone()

    for _ in range(200):
        out = block(x_fixed)
        signal = create_initial_signal(out)
        block.top_down_pass(signal, apply_modulation=False)

    final_err = _measure_layer_pred_error(block, x_fixed)
    final_weight = block.living_ffn.weight

    assert final_err < initial_err * 0.9, (
        f"Intrinsic PC convergence broke when modulation disabled: "
        f"{initial_err:.4f} -> {final_err:.4f}"
    )

    weight_drift = (final_weight - initial_weight).abs().mean().item()
    assert weight_drift > 1e-5, (
        f"Weight did not update without modulation: drift={weight_drift:.6f}"
    )


# 3. Modulation-only (synthetic external signal) -----------------------------

def test_modulation_only_sweep():
    """Refinement 3: calling apply_top_down directly with a fixed external
    signal modulates plasticity (from salience) and set_point (from
    prediction_error) as v1's backward pass would, independent of any PC
    forward dynamics.
    """
    torch.manual_seed(0)
    layer = _make_layer()

    initial_plasticity = layer.plasticity.clone()
    initial_set_point = layer.set_point.clone()

    salience = torch.rand(16) * 5.0
    pred_error = torch.randn(16) * 1.0
    signal = TopDownSignal(
        salience=salience,
        prediction_error=pred_error,
        modulation_strength=1.0,
    )

    for _ in range(50):
        layer.apply_top_down(signal)

    plasticity_drift = (layer.plasticity - initial_plasticity).abs().mean().item()
    set_point_drift = (layer.set_point - initial_set_point).abs().mean().item()

    assert plasticity_drift > 0.01, (
        f"Plasticity did not modulate from salience signal: "
        f"drift={plasticity_drift:.6f}"
    )
    assert set_point_drift > 0.0, (
        f"set_point did not modulate from prediction_error signal: "
        f"drift={set_point_drift:.6f}"
    )

    # Verify direction: plasticity should pull toward salience-shaped pattern.
    # With strength=1.0 over 50 steps, plasticity converges toward salience
    # values per the apply_top_down formula (salience * 0.01 / 0.01 = salience).
    final_correlation = torch.corrcoef(
        torch.stack([layer.plasticity, salience])
    )[0, 1].item()
    assert final_correlation > 0.5, (
        f"Plasticity did not correlate with salience pattern: "
        f"corr={final_correlation:.4f}"
    )


# FFN expansion (2026-05-10 audit) ------------------------------------------

def test_block_with_ffn_expansion_forward_shape():
    """Block with ffn_expansion=4 should produce same-shape output as
    expansion=1 — expansion changes internal dim, not block I/O."""
    torch.manual_seed(0)
    block_plain = _make_block(d_model=16, ffn_expansion=1)
    torch.manual_seed(0)
    block_expanded = _make_block(d_model=16, ffn_expansion=4)
    x = torch.randn(2, 8, 16)
    out_plain = block_plain(x)
    out_expanded = block_expanded(x)
    assert out_plain.shape == out_expanded.shape == x.shape


def test_block_with_ffn_expansion_has_more_capacity():
    """Verify the expanded block actually has more trainable parameters
    (up_proj + down_proj add ~2*d_model*inner_dim trainable params)."""
    block_plain = _make_block(d_model=16, ffn_expansion=1)
    block_expanded = _make_block(d_model=16, ffn_expansion=4)

    n_plain = sum(p.numel() for p in block_plain.parameters() if p.requires_grad)
    n_expanded = sum(p.numel() for p in block_expanded.parameters() if p.requires_grad)
    assert n_expanded > n_plain, (
        f"Expanded block should have more trainable params; "
        f"plain={n_plain}, expanded={n_expanded}"
    )


def test_block_top_down_with_expansion_runs():
    """Top-down sweep through an expanded block runs without error.
    With ffn_expansion > 1 the PC layer is in expanded space, so the
    inter-block prediction signal falls back to v1's heuristic — verify
    that path executes cleanly and the block still returns a valid signal.
    """
    torch.manual_seed(0)
    block = _make_block(d_model=16, ffn_expansion=2)
    x = torch.randn(2, 8, 16)
    out = block(x)
    signal = create_initial_signal(out)
    refined = block.top_down_pass(signal)
    assert refined.salience.shape == (16,)
    assert refined.prediction_error.shape == (16,)


# 4. Joint non-interference --------------------------------------------------

def test_joint_non_interference():
    """Refinement 3: drift in the joint (both channels) case is within 15%
    of the linear sum of isolated-channel drifts. Coupling between channels
    flows indirectly through the layer state — plasticity changes affect
    pc_self_modify's delta_w, which affects weight, which affects set_point's
    homeostatic adaptation. The test bounds this coupling.
    """
    d = 16

    def make_layer_with_seed():
        torch.manual_seed(0)
        return _make_layer(d)

    layer_n = make_layer_with_seed()  # no top-down baseline
    layer_a = make_layer_with_seed()  # salience-only top-down
    layer_b = make_layer_with_seed()  # pred_error-only top-down
    layer_c = make_layer_with_seed()  # joint top-down

    initial_plasticity = layer_n.plasticity.clone()
    initial_set_point = layer_n.set_point.clone()

    torch.manual_seed(42)
    inputs = [torch.randn(4, d) for _ in range(100)]
    salience_signals = [torch.rand(d) for _ in range(100)]
    pred_err_signals = [torch.randn(d) * 0.1 for _ in range(100)]

    for x, sal, perr in zip(inputs, salience_signals, pred_err_signals):
        zero = torch.zeros_like(sal)

        # Baseline calls apply_top_down with an all-zero signal so the
        # passive-modulation effects (multiplicative decay on plasticity
        # in apply_top_down) are shared across all four layers. Without
        # this baseline parity, plasticity in B drifts via decay alone
        # while plasticity in N stays at 1.0, breaking the linearity
        # decomposition.
        layer_n(x)
        layer_n.apply_top_down(TopDownSignal(
            salience=zero, prediction_error=zero,
            modulation_strength=1.0,
        ))

        layer_a(x)
        layer_a.apply_top_down(TopDownSignal(
            salience=sal, prediction_error=zero,
            modulation_strength=1.0,
        ))

        layer_b(x)
        layer_b.apply_top_down(TopDownSignal(
            salience=zero, prediction_error=perr,
            modulation_strength=1.0,
        ))

        layer_c(x)
        layer_c.apply_top_down(TopDownSignal(
            salience=sal, prediction_error=perr,
            modulation_strength=1.0,
        ))

    # Drift attributable to top-down = layer's drift minus baseline's drift.
    def attributable(layer_x, layer_n, attr_name, initial):
        x_drift = getattr(layer_x, attr_name) - initial
        n_drift = getattr(layer_n, attr_name) - initial
        return x_drift - n_drift

    p_a = attributable(layer_a, layer_n, "plasticity", initial_plasticity)
    p_b = attributable(layer_b, layer_n, "plasticity", initial_plasticity)
    p_c = attributable(layer_c, layer_n, "plasticity", initial_plasticity)

    sp_a = attributable(layer_a, layer_n, "set_point", initial_set_point)
    sp_b = attributable(layer_b, layer_n, "set_point", initial_set_point)
    sp_c = attributable(layer_c, layer_n, "set_point", initial_set_point)

    # Linear-sum prediction vs joint reality.
    p_linear = p_a + p_b
    sp_linear = sp_a + sp_b

    p_interference = (p_c - p_linear).norm() / max(p_linear.norm().item(), 1e-8)
    sp_interference = (sp_c - sp_linear).norm() / max(sp_linear.norm().item(), 1e-8)

    assert p_interference < 0.15, (
        f"Plasticity channel interference exceeded 15%: "
        f"{p_interference.item():.4f}"
    )
    assert sp_interference < 0.15, (
        f"Set-point channel interference exceeded 15%: "
        f"{sp_interference.item():.4f}"
    )
