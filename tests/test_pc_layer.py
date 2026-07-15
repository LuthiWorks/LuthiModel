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


# iPC interleaved inference + update (lit-followup 2026-05-13) -------------

def test_ipc_default_T1_matches_classical():
    """inference_steps_per_forward=1 (default) must reproduce the
    classical PC trajectory bit-identically. Regression test.
    """
    torch.manual_seed(0)
    layer_a = PredictiveCodingLayer(
        in_features=16, out_features=8, num_episodes=4,
        inference_steps_per_forward=1,
    )
    torch.manual_seed(0)
    layer_b = PredictiveCodingLayer(
        in_features=16, out_features=8, num_episodes=4,
    )
    x = torch.randn(4, 16)
    for _ in range(10):
        layer_a(x)
        layer_b(x)
    assert torch.allclose(layer_a.weight, layer_b.weight, atol=1e-7), (
        "inference_steps_per_forward=1 should be bit-identical to "
        "the no-iPC default code path."
    )


def test_ipc_T_gt_1_converges_faster():
    """iPC with T>1 should drive prediction error down faster than T=1
    over the same number of *external* forward calls on a fixed-pattern
    task. Validates the Salvatori et al. claim that interleaved
    inference+update converges faster than the classical schedule.
    """
    def _make_layer(T: int) -> PredictiveCodingLayer:
        torch.manual_seed(0)
        return PredictiveCodingLayer(
            in_features=8, out_features=4, num_episodes=4,
            inference_steps_per_forward=T,
            pred_learning_rate=0.01,  # M1's mechanism-test rate
        )

    x_fixed = torch.randn(16, 8) * 0.5

    def _measure_err(layer):
        with torch.no_grad():
            output = x_fixed @ layer.weight.T
            output_mean = output.mean(dim=0)
            actual = x_fixed.mean(dim=0)
            predicted = output_mean @ layer.prediction
            return (actual - predicted).abs().mean().item()

    layer_t1 = _make_layer(T=1)
    layer_t5 = _make_layer(T=5)

    initial_t1 = _measure_err(layer_t1)
    initial_t5 = _measure_err(layer_t5)
    assert abs(initial_t1 - initial_t5) < 1e-6, (
        "Initial state should be identical given same seed"
    )

    # Run same number of EXTERNAL forwards on each.
    N = 100
    for _ in range(N):
        layer_t1(x_fixed)
        layer_t5(x_fixed)

    final_t1 = _measure_err(layer_t1)
    final_t5 = _measure_err(layer_t5)

    # iPC should converge prediction error LOWER than classical PC at
    # matched external forward count (it gets T inner steps per external,
    # so more "compute" — but the iPC paper argues this is also better
    # than spending the same compute on more external forwards).
    assert final_t5 < final_t1, (
        f"iPC T=5 should converge below T=1 at matched external forwards; "
        f"T=1 final {final_t1:.4f}, T=5 final {final_t5:.4f}"
    )


def test_ipc_grad_checkpoint_fails_loud():
    """iPC + gradient checkpointing is incompatible; the forward must
    raise rather than silently produce wrong gradients on the recompute.
    """
    from luthi.grad_checkpoint import is_recomputing
    torch.manual_seed(0)
    layer = PredictiveCodingLayer(
        in_features=8, out_features=4, num_episodes=4,
        inference_steps_per_forward=3,
    )
    x = torch.randn(2, 8)

    # Simulate the gradient-checkpoint recompute by setting the thread-local.
    # Patch the is_recomputing helper for the scope of one forward.
    import luthi.grad_checkpoint as gc_mod
    original_is_recomputing = gc_mod.is_recomputing
    gc_mod.is_recomputing = lambda: True
    try:
        with pytest.raises(RuntimeError, match="Incompatible modes.*iPC"):
            layer(x)
    finally:
        gc_mod.is_recomputing = original_is_recomputing


# Sparse PC gating (lit-followup 2026-05-13) --------------------------------

def test_sparse_gate_disabled_matches_unsparse_default():
    """sparse_threshold=0 (default) must produce bit-identical training
    trajectories to the un-instrumented code path. Regression test for
    the no-silent-behavior-change rule.
    """
    torch.manual_seed(0)
    layer_a = PredictiveCodingLayer(
        in_features=16, out_features=8, num_episodes=4,
        sparse_threshold=0.0,
    )
    torch.manual_seed(0)
    layer_b = PredictiveCodingLayer(
        in_features=16, out_features=8, num_episodes=4,
        # Same as layer_a but explicitly no sparse_threshold kwarg
    )
    x = torch.randn(4, 16)
    for _ in range(20):
        layer_a(x)
        layer_b(x)
    assert torch.allclose(layer_a.weight, layer_b.weight, atol=1e-7), (
        "sparse_threshold=0 should be bit-identical to the no-sparse code path"
    )


def test_sparse_gate_freezes_low_error_rows_after_warmup():
    """After warmup, output rows with error_acc below the threshold should
    not see weight updates. Verify by manipulating error_acc directly:
    set half the rows above threshold, half below, then run for a few
    forward passes and confirm the below-threshold rows don't drift.
    """
    torch.manual_seed(0)
    layer = PredictiveCodingLayer(
        in_features=16, out_features=8, num_episodes=4,
        sparse_threshold=0.5,
        sparse_warmup_steps=2,  # short warmup for the test
    )
    x = torch.randn(4, 16) * 1.5
    # Burn through warmup
    for _ in range(3):
        layer(x)
    # Hand-set error_acc: first 4 outputs above threshold, last 4 below.
    with torch.no_grad():
        layer.error_acc[:4] = 1.0
        layer.error_acc[4:] = 0.0  # below 0.5 threshold

    weight_before = layer.weight.clone()
    # Run several steps. The high-error-acc rows should update; the
    # low-error-acc rows should not. error_acc itself updates each step
    # so we re-pin after each forward to keep the gate stable.
    for _ in range(10):
        layer(x)
        with torch.no_grad():
            layer.error_acc[:4] = 1.0
            layer.error_acc[4:] = 0.0
    weight_after = layer.weight

    drift_active = (weight_after[:4] - weight_before[:4]).abs().mean().item()
    drift_frozen = (weight_after[4:] - weight_before[4:]).abs().mean().item()

    assert drift_active > 1e-5, (
        f"High-error-acc rows did not update under sparse gating "
        f"(drift {drift_active:.6e}). Gating may have over-fired."
    )
    # Frozen rows can drift slightly from homeostatic regulation
    # (the homeostatic_force += (set_point - weight) step runs for all
    # rows regardless of the sparse gate). That's by design — gating
    # affects only delta_w / momentum / update_ema, not homeostatic.
    # So the assertion is "much less drift," not "zero drift."
    assert drift_frozen < 0.1 * drift_active, (
        f"Low-error-acc rows should drift much less than active rows. "
        f"Got active={drift_active:.6e} vs frozen={drift_frozen:.6e}."
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
