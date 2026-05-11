"""M4 unit tests for consolidation — the v2 STOP GATE milestone.

Per `docs/V2_IMPLEMENTATION_PLAN.md` M4 (Days 9-11) and refinement 4
(rolling 1000-step bootstrap window).

Tests:
  Tracker:
    1. No fire during warmup window
    2. No fire under stable high variance
    3. Fires after N consecutive sub-threshold steps
    4. Resets sub-threshold counter after firing
  Consolidation function:
    5. No-op on empty episode store
    6. Pulls weight toward stored snapshot
    7. Nudges prediction matrix by consolidation_error
  End-to-end:
    8. Layer with consolidation_enabled fires under low-variance conditions
  M4 STOP GATE:
    9. Episodes measurably shape prediction post-consolidation
       (if this fails, v2 has no architectural novelty over vanilla
       transformer + episode store; abandon v2 per plan)
"""

import pytest
import torch

from luthi.v2 import (
    PredictiveCodingLayer,
    ConsolidationTracker,
    consolidate_layer,
)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

def test_tracker_does_not_fire_during_warmup():
    """Refinement 4: triggers can never fire while the rolling window is
    still filling. There's no consolidation target before the model has
    any predictive structure to consolidate against.
    """
    tracker = ConsolidationTracker(
        window_size=100, trigger_window=10, threshold_factor=0.5
    )
    # Feed 99 zero-variance values — would trivially exceed any fire
    # condition if the warmup gate weren't present.
    fired = False
    for _ in range(99):
        if tracker.step(0.0):
            fired = True
    assert not fired, "Tracker fired during warmup window"
    assert not tracker.is_warmed_up


def test_tracker_does_not_fire_under_high_variance():
    """When pred_error variance stays above threshold, no consolidation."""
    tracker = ConsolidationTracker(
        window_size=100, trigger_window=10, threshold_factor=0.5
    )
    # Warm up with values around 1.0 (baseline = 1.0, threshold = 0.5)
    for _ in range(100):
        tracker.step(1.0)
    # Now feed values at 1.5 — well above 0.5 threshold
    fired = False
    for _ in range(50):
        if tracker.step(1.5):
            fired = True
    assert not fired


def test_tracker_fires_after_sub_threshold_window():
    """Once warmed up and N consecutive variances are below threshold,
    the tracker fires."""
    tracker = ConsolidationTracker(
        window_size=100, trigger_window=10, threshold_factor=0.5
    )
    for _ in range(100):
        tracker.step(1.0)
    # baseline ≈ 1.0, threshold = 0.5 — feed sub-threshold values
    fired_at = None
    for i in range(20):
        if tracker.step(0.1):
            fired_at = i
            break
    assert fired_at is not None and fired_at < 15, (
        f"Tracker did not fire within expected window; fired_at={fired_at}"
    )


def test_tracker_resets_after_firing():
    """After firing, the sub-threshold counter resets — the next
    consolidation event requires a fresh full sub-threshold window."""
    tracker = ConsolidationTracker(
        window_size=100, trigger_window=10, threshold_factor=0.5
    )
    for _ in range(100):
        tracker.step(1.0)
    # Fire once
    fires = 0
    for _ in range(10):
        if tracker.step(0.1):
            fires += 1
    assert fires == 1, f"Expected exactly one fire, got {fires}"
    # Now feed a single high-variance step and N-1 low ones — should NOT fire,
    # because the counter reset and we need a fresh trigger_window of lows.
    tracker.step(1.5)
    fires_after = 0
    for _ in range(9):  # one short of the trigger window
        if tracker.step(0.1):
            fires_after += 1
    assert fires_after == 0


# ---------------------------------------------------------------------------
# consolidate_layer function
# ---------------------------------------------------------------------------

def test_consolidate_layer_noop_on_empty_store():
    """No episodes stored → consolidate_layer returns 0, no state changes."""
    torch.manual_seed(0)
    layer = PredictiveCodingLayer(in_features=8, out_features=4, num_episodes=4)
    weight_before = layer.weight.clone()
    prediction_before = layer.prediction.clone()
    n_replayed = consolidate_layer(layer)
    assert n_replayed == 0
    assert torch.allclose(layer.weight, weight_before)
    assert torch.allclose(layer.prediction, prediction_before)


def test_consolidate_layer_pulls_weight_toward_snapshot():
    """consolidate_layer with one stored episode moves weight toward
    the stored snapshot."""
    torch.manual_seed(0)
    layer = PredictiveCodingLayer(in_features=8, out_features=4, num_episodes=4)
    # Force-store a known snapshot
    target_snapshot = torch.randn(4, 8) * 2.0
    layer.episode_values[0] = target_snapshot
    layer.episode_contexts[0] = torch.randn(layer.context_dim)
    layer.episode_saliences[0] = 1.0
    layer.episode_count.fill_(1)

    distance_before = (layer.weight - target_snapshot).abs().mean().item()
    consolidate_layer(layer, consolidation_rate_factor=10.0)  # exaggerate for visibility
    distance_after = (layer.weight - target_snapshot).abs().mean().item()

    assert distance_after < distance_before, (
        f"Consolidation did not pull weight toward snapshot: "
        f"distance {distance_before:.4f} -> {distance_after:.4f}"
    )


def test_consolidate_layer_does_not_touch_prediction_matrix():
    """Audit 2026-05-11: consolidation must NOT directly update the
    prediction matrix. weight and prediction live in different
    mathematical spaces; applying a weight-space delta to prediction was
    semantically wrong. The natural PC forward loop adapts prediction to
    the consolidated weights via its usual update rule.
    """
    torch.manual_seed(0)
    layer = PredictiveCodingLayer(in_features=8, out_features=4, num_episodes=4)
    target_snapshot = torch.randn(4, 8) * 2.0
    layer.episode_values[0] = target_snapshot
    layer.episode_contexts[0] = torch.randn(layer.context_dim)
    layer.episode_saliences[0] = 1.0
    layer.episode_count.fill_(1)

    pred_before = layer.prediction.clone()
    consolidate_layer(layer, consolidation_rate_factor=10.0)
    pred_after = layer.prediction

    assert torch.equal(pred_before, pred_after), (
        "Consolidation should not modify the prediction matrix directly "
        "— prediction lives in a different mathematical space from weight."
    )


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def test_layer_with_consolidation_enabled_fires():
    """A layer with consolidation_enabled=True and a small enough window
    fires consolidation events under stable training input."""
    torch.manual_seed(0)
    layer = PredictiveCodingLayer(
        in_features=16, out_features=8, num_episodes=8,
        consolidation_enabled=True,
        consolidation_window=50,
        consolidation_trigger_window=10,
    )

    # Warm up with varied input to seed the variance baseline.
    for _ in range(60):
        x = torch.randn(4, 16) * 2.0
        layer(x)

    # Now feed near-constant input — pred_error variance should drop.
    constant_x = torch.ones(4, 16) * 0.1
    for _ in range(100):
        layer(constant_x)

    assert layer._consolidation_fire_count > 0, (
        f"Consolidation never fired under stable input "
        f"(fire_count={layer._consolidation_fire_count})"
    )


# ---------------------------------------------------------------------------
# M4 STOP GATE — the project-pivotal test
# ---------------------------------------------------------------------------

def test_consolidation_shapes_prediction_post_drift():
    """M4 STOP GATE: episodes measurably shape prediction post-consolidation.

    Setup:
      Phase 1 — train on pattern A (drives episode storage with stable
                A-like predictive structure).
      Phase 2 — train on pattern B with different statistics. Both layers'
                predictions drift away from pattern A.
      Phase 3 — only the consolidate-arm runs explicit consolidation passes.
      Phase 4 — measure pred_error on pattern A. The consolidate-arm
                should be measurably closer to its stored A-snapshot than
                the control (which only has the drifted weights).

    If this fails, v2 has no architectural novelty over a vanilla
    transformer + episode store, and the strategic shift (committing to
    PC as primary substrate) needs to be revisited.
    """
    d = 16
    torch.manual_seed(0)
    layer_consol = PredictiveCodingLayer(
        in_features=d, out_features=8, num_episodes=8,
        salience_threshold=0.001,  # ensure episode storage triggers
    )
    torch.manual_seed(0)
    layer_control = PredictiveCodingLayer(
        in_features=d, out_features=8, num_episodes=8,
        salience_threshold=0.001,
    )

    # Phase 1: train on pattern A (high magnitude)
    torch.manual_seed(11)
    pattern_a = torch.randn(8, d) * 2.0
    for _ in range(200):
        layer_consol(pattern_a)
        layer_control(pattern_a)

    # Both layers should now have stored episodes capturing pattern A.
    assert layer_consol.episode_count.item() > 0
    assert layer_control.episode_count.item() > 0

    def measure_pred_error_on(layer, x):
        with torch.no_grad():
            output = x @ layer.weight.T
            output_mean = output.mean(dim=0)
            actual = x.mean(dim=0)
            predicted = output_mean @ layer.prediction
            return (actual - predicted).abs().mean().item()

    err_a_after_phase1 = measure_pred_error_on(layer_consol, pattern_a)

    # Phase 2: train on pattern B (different statistics). Predictions drift
    # away from pattern A.
    torch.manual_seed(22)
    pattern_b = torch.randn(8, d) * 0.1
    for _ in range(500):
        layer_consol(pattern_b)
        layer_control(pattern_b)

    err_a_drifted_consol = measure_pred_error_on(layer_consol, pattern_a)
    err_a_drifted_control = measure_pred_error_on(layer_control, pattern_a)

    # Sanity check: drift moved err_a away from the post-Phase-1 baseline
    # in both arms (otherwise pattern_b wasn't different enough to matter).
    assert err_a_drifted_consol > err_a_after_phase1, (
        "Pattern B did not cause pred_error drift on pattern A; "
        "test setup is inert"
    )

    # Phase 3: consolidate the consol-arm only.
    for _ in range(20):
        consolidate_layer(layer_consol, consolidation_rate_factor=5.0)

    # Phase 4: measure pred_error on pattern A after consolidation.
    err_a_consolidated = measure_pred_error_on(layer_consol, pattern_a)
    err_a_control = err_a_drifted_control

    # The STOP GATE: consolidated layer's pred_error on pattern A is
    # measurably lower than control's — episodes have shaped predictive
    # structure beyond what the episode store alone provides.
    assert err_a_consolidated < err_a_control, (
        f"M4 STOP GATE FAILED: consolidation did not measurably shape "
        f"prediction. Consolidated err on pattern A: {err_a_consolidated:.4f}, "
        f"control err: {err_a_control:.4f}. v2 has no architectural novelty "
        f"over vanilla transformer + episode store. Strategic shift "
        f"(2026-05-09 commitment to v2 as primary substrate) needs revisiting."
    )

    # Stronger soft-pass: improvement should be at least 1% to count as
    # a "measurable" effect.
    relative_improvement = (
        (err_a_control - err_a_consolidated) / max(err_a_control, 1e-8)
    )
    assert relative_improvement > 0.01, (
        f"M4 STOP GATE soft-pass: improvement is below 1% threshold "
        f"({relative_improvement:.4%}). Consolidation effect exists but "
        f"may be too small to matter at scale."
    )


# ---------------------------------------------------------------------------
# Compressed (INT8) episode storage — 2026-05-10 audit follow-up
# ---------------------------------------------------------------------------

def test_consolidation_with_compressed_episodes():
    """`compressed_episodes=True` stores episode snapshots as INT8 to
    reduce the 150 GB Phase-5 budget. Verify the M4 STOP GATE still
    passes — consolidation should shape prediction post-drift even when
    snapshots have quantization noise.
    """
    d = 16
    torch.manual_seed(0)
    layer_consol = PredictiveCodingLayer(
        in_features=d, out_features=8, num_episodes=8,
        salience_threshold=0.001,
        compressed_episodes=True,  # INT8 episode_values
    )
    torch.manual_seed(0)
    layer_control = PredictiveCodingLayer(
        in_features=d, out_features=8, num_episodes=8,
        salience_threshold=0.001,
        compressed_episodes=True,
    )

    # Verify the underlying buffer is INT8 (this is the contract we're
    # testing — the architecture supports compression).
    assert layer_consol.episode_values.dtype == torch.int8, (
        f"Expected INT8 episode_values, got {layer_consol.episode_values.dtype}"
    )

    torch.manual_seed(11)
    pattern_a = torch.randn(8, d) * 2.0
    for _ in range(200):
        layer_consol(pattern_a)
        layer_control(pattern_a)

    assert layer_consol.episode_count.item() > 0

    def measure_pred_error_on(layer, x):
        with torch.no_grad():
            output = x @ layer.weight.T
            output_mean = output.mean(dim=0)
            actual = x.mean(dim=0)
            predicted = output_mean @ layer.prediction
            return (actual - predicted).abs().mean().item()

    err_a_after_phase1 = measure_pred_error_on(layer_consol, pattern_a)

    torch.manual_seed(22)
    pattern_b = torch.randn(8, d) * 0.1
    for _ in range(500):
        layer_consol(pattern_b)
        layer_control(pattern_b)

    err_a_drifted_consol = measure_pred_error_on(layer_consol, pattern_a)
    err_a_drifted_control = measure_pred_error_on(layer_control, pattern_a)

    assert err_a_drifted_consol > err_a_after_phase1, (
        "Pattern B did not cause pred_error drift; setup inert"
    )

    for _ in range(20):
        consolidate_layer(layer_consol, consolidation_rate_factor=5.0)

    err_a_consolidated = measure_pred_error_on(layer_consol, pattern_a)
    err_a_control_final = err_a_drifted_control

    # Compressed episodes should still produce measurable shaping (the
    # INT8 quantization noise per-snapshot is much smaller than the
    # drift the consolidation is correcting for).
    assert err_a_consolidated < err_a_control_final, (
        f"INT8 compression broke consolidation effect: "
        f"consolidated err {err_a_consolidated:.4f} vs control "
        f"{err_a_control_final:.4f}. Phase 5 compression strategy needs "
        f"revisiting before scaling."
    )
