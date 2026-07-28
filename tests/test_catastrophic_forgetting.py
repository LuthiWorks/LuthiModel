"""Catastrophic-forgetting harness for v2 PC consolidation pathways.

The behavioral test that should have existed before yesterday's Salvatori
attractor consolidation landed. Direct empirical answer to the question:
does the consolidation pathway preserve old learning when new (distractor)
training arrives?

> **Read `docs/research/2026-05-16_catastrophic-forgetting-harness.md`
> before changing this file.** The harness went through three
> iterations before producing data the assertions could honestly test
> against — including a metric reframe (pred_err → weight_drift)
> driven by the 2026-05-11 audit's design choice that gradient
> consolidation deliberately only restores weight, not the prediction
> matrix. The two `xfail(strict=True)` markers in this file are
> empirical observations pinned to code, not bugs awaiting fix. The
> research log entry documents what each iteration tried, what the
> data revealed, and why the design changed at each step.

Pattern (standard continual-learning evaluation):
  Phase 1 — train layer on pattern set A until PC dynamics settle
  Phase 2 — snapshot prediction error on A
  Phase 3 — train on a distractor set B (different distribution)
  Phase 4 — re-measure prediction error on A
  Phase 5 — compute the "forgetting metric" = post-distractor error minus
            pre-distractor error on the A patterns

A consolidation pathway "preserves A" if it lowers the forgetting metric
relative to the no-consolidation baseline. The four conditions exercised
here:
  - none      : no consolidation called during distractor phase (baseline
                forgetting curve)
  - gradient  : `consolidate_layer` called at scheduled points during
                distractor phase
  - attractor : `consolidate_layer_attractor` called at the same points
  - both      : gradient first then attractor at each point

Consolidation is invoked directly here, not via the ConsolidationTracker.
The tracker's trigger dynamics are validated separately in
`test_pc_consolidation.py`; this harness isolates the consolidation
*update rule* effect from the trigger logic so the test asserts what
the rule does, not when it fires.

Notes on what this harness does NOT yet measure:
  - **Compositional cued retrieval** (the peer 4.7 review point — taffy
    activates fudge from shared feature bindings). That test belongs
    here too once HDC memory (Direction A in
    `docs/RESEARCH_HDC_VSA_INTEGRATION.md`) is implemented; the helpers
    below are structured so a `test_compositional_cued_retrieval`
    function can be added without restructuring the file.
  - **Full-model effects.** This harness operates on a single
    `PredictiveCodingLayer` for isolation and speed. A model-level
    harness running across the full v2 trunk is a future extension —
    same harness shape, different invocation.
"""

from __future__ import annotations

import pytest
import torch

from luthi.v2 import (
    PredictiveCodingLayer,
    consolidate_layer,
    consolidate_layer_attractor,
)


# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------

# Pattern set sizes and training durations chosen so a single pytest run
# completes in a few seconds on CPU while still producing distinguishable
# forgetting curves. The qualitative result (whether consolidation
# reduces forgetting) is robust to these knobs; the *magnitude* would
# change with longer training, but the *direction* is stable.

IN_FEATURES = 16
OUT_FEATURES = 8
N_PATTERNS_PER_SET = 6
BATCH_PER_STEP = 8

# Aggressive training budgets calibrated to actually produce forgetting in
# v2's bounded PC substrate. Initial calibration at pc_rate=0.01,
# PHASE_A=60, DISTRACTOR=120 produced *negative* forgetting (the layer
# got better at A after B training) because v2's homeostatic regulation
# and per-step update clamps resist drift by construction. That's a real
# property of the substrate; the test budget has to push past it for the
# forgetting comparison to be meaningful.
PHASE_A_STEPS = 200         # initial training on A
DISTRACTOR_STEPS = 500      # distractor training on B
# Production consolidation fires near-continuously once warmup is past
# (low-variance windows in `ConsolidationTracker`). The first test
# iteration used CONSOLIDATION_EVERY=25 to keep the harness fast but
# that pushed the consolidation pathways into a regime where the per-
# event correction was large enough to over-shoot and destabilize. A
# cadence of every 5 steps with single-pass attractor matches the
# production "consolidation is a small frequent nudge, not a sparse
# large jump" regime much more honestly.
CONSOLIDATION_EVERY = 5     # how often consolidate_* is called during B
N_REPLAY_PASSES = 1         # attractor passes per event (production default)

# Pattern-set bias: A and B drawn from offset Gaussians so the optimal
# weight for predicting one is different from the optimal for the other.
PATTERN_SET_A_OFFSET = -0.8
PATTERN_SET_B_OFFSET = +0.8


def _make_pattern_set(
    n_patterns: int, in_features: int, seed: int, offset: float = 0.0,
) -> torch.Tensor:
    """Generate a fixed, reproducible pattern set as [n_patterns, in_features].

    `offset` shifts the whole set by a constant in input-space. Using
    opposite offsets for A and B makes them genuinely distinct
    distributions, not just different samples from the same one — which
    is needed to produce measurable forgetting in v2's bounded substrate.
    """
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(n_patterns, in_features, generator=gen) * 0.5) + offset


def _make_layer(*, pc_rate: float = 0.05, num_episodes: int = 32) -> PredictiveCodingLayer:
    """Build a PC layer sized for fast unit-test convergence.

    `num_episodes` is generous (32 vs the production-default 32 to 64) so
    the storage policy doesn't evict A's episodes during B's training —
    this isolates the consolidation effect from the eviction policy.
    `pc_rate` is bumped from the M5 default 0.001 to 0.01 for visible
    learning over the short training budgets the test uses.
    """
    torch.manual_seed(0)
    return PredictiveCodingLayer(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        pc_rate=pc_rate,
        pred_learning_rate=0.001,
        num_episodes=num_episodes,
        salience_threshold=0.0,  # store eagerly (legacy admission path)
        # Tracker-driven consolidation off here; we invoke directly.
        consolidation_enabled=False,
    )


def _train_phase(
    layer: PredictiveCodingLayer,
    pattern_set: torch.Tensor,
    n_steps: int,
    *,
    consolidation_style: str = "none",
    consolidation_period: int = CONSOLIDATION_EVERY,
    n_replay_passes: int = N_REPLAY_PASSES,
) -> None:
    """Feed `pattern_set` through the layer for `n_steps`, optionally
    invoking consolidation every `consolidation_period` steps.

    `consolidation_style` controls which pathway fires:
      - "none"      : no consolidation
      - "gradient"  : consolidate_layer
      - "attractor" : consolidate_layer_attractor (n_replay_passes per event)
      - "both"      : gradient first, then attractor
    """
    n_patterns = pattern_set.shape[0]
    for step in range(n_steps):
        # Sample a small batch from the pattern set.
        idx = torch.randint(0, n_patterns, (BATCH_PER_STEP,))
        x = pattern_set[idx]
        layer(x)

        # Fire consolidation at the scheduled cadence (skip step 0 so the
        # store has something to consolidate from).
        if (
            consolidation_style != "none"
            and step > 0
            and step % consolidation_period == 0
        ):
            if consolidation_style in ("gradient", "both"):
                consolidate_layer(layer, consolidation_rate_factor=0.5)
            if consolidation_style in ("attractor", "both"):
                consolidate_layer_attractor(
                    layer,
                    consolidation_rate_factor=0.5,
                    n_replay_passes=n_replay_passes,
                )


def _weight_drift_from(layer: PredictiveCodingLayer, reference_weight: torch.Tensor) -> float:
    """L2 distance from `reference_weight` (the post-phase-1 snapshot)
    to the layer's current weight, normalized by reference norm.

    This is the **primary forgetting metric** for v2's consolidation
    pathways: consolidation is explicitly designed to preserve the
    weight regime that was reached during phase 1 (the M5 STOP GATE
    test in `test_pc_consolidation.py` validates this). A pathway that
    successfully resists drift produces a smaller drift value than the
    no-consolidation baseline.

    Note: this is NOT the same as `pred_error` on the A patterns. The
    2026-05-11 audit removed prediction-matrix consolidation from
    `consolidate_layer` (gradient-replay only modifies weight; prediction
    is re-adapted via normal forward updates). So a layer can have its
    weight pulled back toward the A snapshot while its prediction matrix
    is still B-shaped, producing a weight-similar-to-A but
    prediction-error-on-A-is-bad state. Weight drift is the metric that
    isolates what gradient consolidation actually does.
    """
    with torch.no_grad():
        diff = (layer.weight - reference_weight).norm().item()
        ref_norm = reference_weight.norm().item()
        return diff / max(ref_norm, 1e-8)


def _measure_pred_error_on_set(
    layer: PredictiveCodingLayer,
    pattern_set: torch.Tensor,
) -> float:
    """Mean absolute prediction error for the layer on the pattern set.

    Uses the same prediction-error definition the layer's PC update uses:
    `pred_error = actual_input - (output_mean @ prediction)` per pattern,
    averaged over the set. No mutation of layer state.

    Kept as a **secondary diagnostic** rather than a pass/fail metric.
    Pred-error depends on BOTH weight and prediction matrix; v2's
    consolidation pathways operate primarily on weight. The weight-drift
    metric isolates the consolidation effect more cleanly. Pred-error is
    still useful for surfacing pathological prediction-matrix drift if
    it ever appears.
    """
    with torch.no_grad():
        total = 0.0
        for i in range(pattern_set.shape[0]):
            x = pattern_set[i].unsqueeze(0)
            output = x @ layer.weight.T
            output_mean = output.mean(dim=0)
            predicted_input = output_mean @ layer.prediction
            actual_input = x.mean(dim=0)
            total += (actual_input - predicted_input).abs().mean().item()
        return total / pattern_set.shape[0]


def _run_forgetting_experiment(consolidation_style: str) -> dict:
    """One run of the canonical pattern. Returns a diagnostic dict.

    Keys:
      weight_drift_baseline_to_current : L2 drift of weight from end of
                                          phase 1 to end of phase 3,
                                          normalized
      err_A_after_phase1               : pred error on A at end of phase 1
      err_A_after_distractor           : pred error on A at end of phase 3
    """
    # ---- Phase 1: train on A (no consolidation; clean A baseline) -----
    layer = _make_layer()
    set_A = _make_pattern_set(
        N_PATTERNS_PER_SET, IN_FEATURES,
        seed=42, offset=PATTERN_SET_A_OFFSET,
    )
    _train_phase(layer, set_A, PHASE_A_STEPS, consolidation_style="none")

    # Snapshot the layer's weight at the end of phase 1. This is the
    # "what A learning produced" reference; the primary metric below
    # measures how far the weight drifts from this snapshot during
    # phase 3.
    weight_snapshot_after_phase1 = layer.weight.clone()
    err_A_after_phase1 = _measure_pred_error_on_set(layer, set_A)

    # ---- Phase 3: distractor B with optional consolidation -----------
    # B is offset in the opposite direction from A so the two pattern
    # sets pull the weight toward different regions of weight space.
    set_B = _make_pattern_set(
        N_PATTERNS_PER_SET, IN_FEATURES,
        seed=1337, offset=PATTERN_SET_B_OFFSET,
    )
    _train_phase(
        layer, set_B, DISTRACTOR_STEPS,
        consolidation_style=consolidation_style,
    )

    # ---- Phase 5: measurement ----------------------------------------
    weight_drift = _weight_drift_from(layer, weight_snapshot_after_phase1)
    err_A_after_distractor = _measure_pred_error_on_set(layer, set_A)

    return {
        "weight_drift": weight_drift,
        "err_A_after_phase1": err_A_after_phase1,
        "err_A_after_distractor": err_A_after_distractor,
        "pred_err_delta": err_A_after_distractor - err_A_after_phase1,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_baseline_shows_measurable_weight_drift():
    """Without consolidation, training on B after A drifts the weight
    away from the post-phase-1 snapshot. If this test fails, the
    experimental setup isn't producing the drift the downstream
    consolidation comparisons are supposed to push back against.
    """
    result = _run_forgetting_experiment("none")
    assert result["weight_drift"] > 0.01, (
        f"Baseline weight drift too small to measure consolidation "
        f"effect: drift={result['weight_drift']:.4f}. Distractor phase "
        f"may not be long enough or pc_rate may be too small."
    )


def test_gradient_consolidation_reduces_weight_drift_vs_baseline():
    """Gradient-replay consolidation during the distractor phase should
    keep the weight closer to the post-phase-1 snapshot than no
    consolidation. This is the behavioral check for the M4 pathway —
    gradient-replay pulls weight linearly toward stored A snapshots,
    so the layer's weight at end of phase 3 should drift less.

    NOTE: this test measures WEIGHT drift, not pred_error on A patterns.
    The 2026-05-11 audit removed prediction-matrix consolidation from
    `consolidate_layer` (gradient pulls only weight, not prediction).
    A pred-error-on-A metric would conflate "weight preserved" with
    "prediction matrix preserved" — and gradient consolidation only
    does the first by design. The pred-error trajectory is captured
    in the result dict as a secondary diagnostic for future analysis.
    """
    baseline = _run_forgetting_experiment("none")
    gradient = _run_forgetting_experiment("gradient")
    assert gradient["weight_drift"] < baseline["weight_drift"], (
        f"Gradient consolidation did not reduce weight drift: "
        f"baseline_drift={baseline['weight_drift']:.4f}, "
        f"gradient_drift={gradient['weight_drift']:.4f}. "
        f"Secondary diagnostic — baseline pred-err delta on A: "
        f"{baseline['pred_err_delta']:.4f}, gradient: "
        f"{gradient['pred_err_delta']:.4f}. "
        f"This is a regression on the M4 pathway."
    )


@pytest.mark.xfail(
    strict=False,   # see the 2026-07-27 note in the test body
    reason=(
        "EMPIRICAL OBSERVATION (2026-05-16): the Salvatori attractor "
        "pathway INCREASES weight drift relative to the baseline (~5.5× "
        "the gradient pathway's drift at this test budget), which is "
        "structurally consistent with how attractor consolidation works — "
        "it makes stored patterns local minima of the layer's *dynamics*, "
        "not local fixed points of the layer's *weight*. The layer can "
        "produce low-prediction-error responses for stored patterns from "
        "many different weight configurations. Weight-drift is therefore "
        "the wrong pass/fail axis for the Salvatori pathway. A separate "
        "behavioral-preservation metric (does the layer still encode "
        "stored patterns well, regardless of where weight ended up?) is "
        "the right falsifier. That metric is captured as `pred_err_delta` "
        "in the result dict for future test design but requires a more "
        "careful comparison (currently the prediction matrix is also "
        "drifting, so pred_err alone conflates two effects). See "
        "`docs/CATASTROPHIC_FORGETTING_HARNESS_BUILD_LOG.md` for the full "
        "iteration trail that landed on this observation."
    ),
)
def test_attractor_consolidation_reduces_weight_drift_vs_baseline():
    """Empirically the attractor pathway moves weight further than baseline
    (it preserves dynamics, not weight). Kept as a forcing function — if
    anyone fixes the attractor pathway to preserve weight, this going green
    will alert them.

    2026-07-27 — THE ALARM FIRED, and it was doing its job. With the adaptive
    episode store enabled (`adaptive_episodes=True`, the fix for the frozen-
    store defect), this test PASSES: attractor consolidation reduces weight
    drift below baseline. The old observation was made against a store that
    was a fossil of the initialization transient — replaying a stale,
    monocultural pattern dragged the weight away from the manifold that
    supported it. Replaying diverse, current episodes keeps it near. So the
    2026-05-16 conclusion was true of the store, not of the pathway.

    strict=False because the outcome is now configuration-dependent: xfail on
    the legacy store (the shipped default), xpass with the adaptive one. If
    the adaptive store becomes the default, invert this marker and assert the
    pass outright. See docs/research/2026-07-27_episode-store-frozen-defect.md.
    """
    baseline = _run_forgetting_experiment("none")
    attractor = _run_forgetting_experiment("attractor")
    assert attractor["weight_drift"] < baseline["weight_drift"], (
        f"baseline_drift={baseline['weight_drift']:.4f}, "
        f"attractor_drift={attractor['weight_drift']:.4f}"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "EMPIRICAL OBSERVATION (2026-05-16): the 'both' pathway shows "
        "weight drift intermediate between gradient (lowest) and attractor "
        "(highest), confirming the two pathways' update rules drag the "
        "weight in different directions. This isn't necessarily "
        "destructive interaction — it could be that gradient pulls toward "
        "the stored snapshot while attractor lets the weight find a "
        "different low-energy configuration, and 'both' splits the "
        "difference. Whether 'both' is a useful production default "
        "depends on which axis (weight preservation vs dynamics "
        "preservation) matters more for the eventual use case. See "
        "`docs/CATASTROPHIC_FORGETTING_HARNESS_BUILD_LOG.md`."
    ),
)
def test_both_pathway_at_least_matches_either_alone():
    """Marked xfail-strict: empirically 'both' drifts more than gradient
    alone because attractor's pull dominates the combined update.
    """
    gradient = _run_forgetting_experiment("gradient")
    attractor = _run_forgetting_experiment("attractor")
    both = _run_forgetting_experiment("both")
    best_single = min(gradient["weight_drift"], attractor["weight_drift"])
    tolerance = 0.15 * best_single + 1e-3
    assert both["weight_drift"] <= best_single + tolerance, (
        f"gradient_drift={gradient['weight_drift']:.4f}, "
        f"attractor_drift={attractor['weight_drift']:.4f}, "
        f"both_drift={both['weight_drift']:.4f}"
    )


def test_all_pathways_produce_finite_states():
    """Whatever the relative ordering, all four conditions must finish
    with finite weights, finite prediction matrices, finite buffers, and
    zero NaN/Inf. Catches catastrophic numerical instability that
    val-loss tests at training scale might miss.
    """
    for style in ("none", "gradient", "attractor", "both"):
        layer = _make_layer()
        set_A = _make_pattern_set(
            N_PATTERNS_PER_SET, IN_FEATURES,
            seed=42, offset=PATTERN_SET_A_OFFSET,
        )
        set_B = _make_pattern_set(
            N_PATTERNS_PER_SET, IN_FEATURES,
            seed=1337, offset=PATTERN_SET_B_OFFSET,
        )
        _train_phase(layer, set_A, PHASE_A_STEPS, consolidation_style="none")
        _train_phase(layer, set_B, DISTRACTOR_STEPS, consolidation_style=style)
        for name in ("weight", "prediction", "set_point", "momentum",
                     "update_ema", "precision", "error_acc", "plasticity"):
            buf = getattr(layer, name)
            assert torch.isfinite(buf).all(), (
                f"style={style!r}: buffer {name!r} contains non-finite values"
            )
