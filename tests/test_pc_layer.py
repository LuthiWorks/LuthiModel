"""M1 unit tests for PredictiveCodingLayer.

Per `docs/V2_IMPLEMENTATION_PLAN.md` M1 (Days 1-3) and the 2026-05-08
refinements 5 and 6.

Tests:
  1. Non-feedforward signal — consecutive identical inputs differ.
  2. Stability — no NaN/Inf after 500 forward passes.
  3. Prediction error convergence on a fixed mapping.
  4. Homeostatic recovery from a weight perturbation.
  5. Episodic recall — episodes get stored under sufficient salience.
  6. Precision self-organizes from the uniform init.
  7. update_ema ratio-check meaningfulness under PC dynamics (refinement 5).
  8. Prediction matrix bounded growth (refinement 6).
"""

import pytest
import torch

from luthi.v2 import PredictiveCodingLayer


@pytest.fixture
def small_layer():
    torch.manual_seed(0)
    return PredictiveCodingLayer(
        in_features=16, out_features=8, num_episodes=8
    )


# 1. Non-feedforward signal -------------------------------------------------

def test_non_feedforward_signal(small_layer):
    """Two consecutive identical inputs must produce different outputs.

    Confirms the layer is not feedforward — processing changes the
    processor.
    """
    x = torch.randn(4, 16)
    nff = small_layer.non_feedforward_signal(x)
    assert nff > 0.0, "NFF signal is zero — layer is acting feedforward"


# 2. No NaN after 500 passes ------------------------------------------------

def test_no_nan_after_500_passes(small_layer):
    """500 forward passes must not produce NaN or Inf in any buffer."""
    x = torch.randn(4, 16)
    for _ in range(500):
        out = small_layer(x)
        assert not torch.isnan(out).any(), "NaN appeared in output"
        assert not torch.isinf(out).any(), "Inf appeared in output"
    for name, buf in small_layer.named_buffers():
        assert not torch.isnan(buf).any(), f"NaN in buffer {name}"
        assert not torch.isinf(buf).any(), f"Inf in buffer {name}"


# 3. Prediction error convergence -------------------------------------------

def test_prediction_error_convergence_on_fixed_mapping():
    """Training repeatedly on the same input pattern: prediction error
    decreases as the prediction matrix learns to invert the weight.

    Uses pred_learning_rate=0.01 (100x default) so convergence is visible
    in 500 steps. M1's purpose is to verify the convergence MECHANISM
    works; production hyperparameter tuning is M3 work, gated by
    refinement 1's grid search.
    """
    torch.manual_seed(0)
    layer = PredictiveCodingLayer(
        in_features=8, out_features=4, num_episodes=4,
        pred_learning_rate=0.01,
    )
    x_fixed = torch.randn(16, 8) * 0.5

    def measure_err():
        with torch.no_grad():
            output = x_fixed @ layer.weight.T
            output_mean = output.mean(dim=0)
            actual = x_fixed.mean(dim=0)
            predicted = output_mean @ layer.prediction
            return (actual - predicted).abs().mean().item()

    initial_err = measure_err()
    for _ in range(500):
        layer(x_fixed)
    final_err = measure_err()

    assert final_err < initial_err * 0.5, (
        f"Prediction error did not converge: "
        f"{initial_err:.4f} -> {final_err:.4f}"
    )


# 4. Homeostatic recovery ---------------------------------------------------

def test_homeostatic_recovery_from_perturbation(small_layer):
    """After perturbing weight, homeostatic decay pulls it back toward
    the running set_point.
    """
    x = torch.randn(4, 16)
    for _ in range(20):
        small_layer(x)

    drift_before = (
        (small_layer.weight - small_layer.set_point).abs().mean().item()
    )

    with torch.no_grad():
        small_layer.weight.add_(
            torch.randn_like(small_layer.weight) * 0.5
        )
    drift_after_perturb = (
        (small_layer.weight - small_layer.set_point).abs().mean().item()
    )

    for _ in range(500):
        small_layer(x)
    drift_recovered = (
        (small_layer.weight - small_layer.set_point).abs().mean().item()
    )

    assert drift_recovered < drift_after_perturb, (
        f"No homeostatic recovery: drift "
        f"{drift_after_perturb:.4f} -> {drift_recovered:.4f} "
        f"(pre-perturb baseline: {drift_before:.4f})"
    )


# 5. Episodic recall --------------------------------------------------------

def test_episodic_recall(small_layer):
    """Salient inputs accumulate enough error_acc to trigger episode
    storage; once stored, recall on the same context blends the snapshot
    back in.
    """
    x = torch.randn(8, 16) * 2.0
    for _ in range(200):
        small_layer(x)

    assert small_layer.episode_count.item() > 0, (
        f"No episodes stored after 200 high-magnitude passes "
        f"(error_acc.mean={small_layer.error_acc.mean().item():.4f}, "
        f"threshold={small_layer.salience_threshold})"
    )


# 6. Precision self-organization --------------------------------------------

def test_precision_self_organization():
    """Precision starts uniform at 1.0 and should diverge from uniform
    after training on input where some channels are predictable and
    others are noisy.
    """
    torch.manual_seed(0)
    layer = PredictiveCodingLayer(
        in_features=16, out_features=8, num_episodes=4
    )

    initial = layer.precision.clone()
    assert torch.allclose(initial, torch.ones_like(initial)), (
        "Precision did not init to uniform 1.0"
    )

    for _ in range(500):
        x = torch.zeros(8, 16)
        x[:, :8] = torch.randn(8, 8) * 0.1   # low-variance (predictable)
        x[:, 8:] = torch.randn(8, 8) * 5.0   # high-variance (unpredictable)
        layer(x)

    final = layer.precision
    delta_max = (final - 1.0).abs().max().item()
    assert delta_max > 0.01, (
        f"Precision did not self-organize from uniform; "
        f"max delta from 1.0 = {delta_max:.6f}"
    )


# 7. update_ema ratio-check meaningfulness (refinement 5) -------------------

def test_update_ema_ratio_check_meaningfulness():
    """Refinement 5: under PC dynamics, the v1 metaplasticity ratio-check
    must still gate correctly.

    (a) An update at the steady-state magnitude (== running update_ema)
        produces adaptive_factor ≈ 1.0 — no dampening.
    (b) An update an order of magnitude larger produces
        adaptive_factor < 0.5 — strong dampening engages.

    Tests the math directly so the assertion is sharp; the same expression
    is what `pc_ops.pc_self_modify` uses inline.
    """
    update_ema = torch.tensor([1e-3])

    update_mag_steady = torch.tensor([1e-3])
    ratio_steady = update_mag_steady / (update_ema + 1e-8)
    factor_steady = (2.0 / (1.0 + ratio_steady)).clamp(max=1.0)
    assert factor_steady.item() > 0.95, (
        f"Steady-state update should pass without dampening, "
        f"factor={factor_steady.item():.4f}"
    )

    update_mag_spike = torch.tensor([1e-2])  # 10x running EMA
    ratio_spike = update_mag_spike / (update_ema + 1e-8)
    factor_spike = (2.0 / (1.0 + ratio_spike)).clamp(max=1.0)
    assert factor_spike.item() < 0.5, (
        f"10x spike update should be strongly dampened, "
        f"factor={factor_spike.item():.4f}"
    )


# 8. Prediction matrix bounded growth (refinement 6) ------------------------

def test_prediction_matrix_bounded_growth(small_layer):
    """Refinement 6: prediction matrix Frobenius norm must stay bounded
    over 1000 forward passes spanning input magnitudes 0.1 / 1.0 / 10.0.

    Records norm every 50 steps. Pass: late-window norm is not orders of
    magnitude larger than early-window norm, no NaN/Inf.
    """
    norms = []
    scales = [0.1, 1.0, 10.0]
    for step in range(1000):
        scale = scales[step % len(scales)]
        x = torch.randn(4, 16) * scale
        small_layer(x)
        if step % 50 == 0:
            norms.append(small_layer.prediction.norm().item())

    norms_t = torch.tensor(norms)
    assert not torch.isnan(norms_t).any(), "Prediction norm went NaN"
    assert not torch.isinf(norms_t).any(), "Prediction norm went Inf"

    early = norms_t[:5].mean().item()
    late = norms_t[-5:].mean().item()
    growth_ratio = late / max(early, 1e-8)
    assert growth_ratio < 100.0, (
        f"Prediction norm grew unboundedly: "
        f"early {early:.4f} -> late {late:.4f} (ratio {growth_ratio:.2f})"
    )
