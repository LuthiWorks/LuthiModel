"""M5 falsification suite: v2 PC vs DeadLM head-to-head.

Two kinds of tests live here:

1. **Attractor dynamics test (structural, CPU-only).** Verifies that v2's
   PC layer shows basin-attractor behavior under weight perturbation —
   measurable recovery toward the unperturbed trajectory — that is
   distinguishable from a non-living random control. Per the V2 plan,
   if attractor dynamics are indistinguishable from random modulation,
   v2's "temporal existence" claim is unfalsifiable and v2 should be
   abandoned.

2. **Head-to-head comparison test (loads m5_runner results).** Asserts
   the falsification criteria against trained runs: convergence penalty
   within 20% of DeadLM at matched compute, no NaN events, etc. Skipped
   when results don't exist (so unit-test runs don't require the GPU runs).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from luthi.v2 import PredictiveCodingLayer


# ---------------------------------------------------------------------------
# 1. Attractor dynamics — structural test
# ---------------------------------------------------------------------------

def _capture_trajectory(
    layer: PredictiveCodingLayer,
    x: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    """Run `n_steps` forward passes on `x`, return [n_steps, ...] outputs.

    Each forward call mutates the layer state (self-modification), so the
    output drifts even without perturbation. That drift IS the trajectory.
    """
    outputs = []
    with torch.no_grad():
        for _ in range(n_steps):
            out = layer(x)
            outputs.append(out.detach().clone())
    return torch.stack(outputs)


def _snapshot_living_state(layer: PredictiveCodingLayer) -> dict:
    """Snapshot all living buffers so the layer can be restored exactly."""
    return {
        name: buf.detach().clone()
        for name, buf in layer.named_buffers()
    }


def _restore_living_state(layer: PredictiveCodingLayer, snapshot: dict) -> None:
    """Restore living buffers from a snapshot in-place."""
    for name, value in snapshot.items():
        getattr(layer, name).copy_(value)


def test_attractor_recovery_distinguishable_from_random():
    """M5 falsification: v2's perturbation recovery must be measurably
    stronger than a non-living random control's.

    Protocol:
      1. Build a v2 layer, drive it briefly to settle into a basin.
      2. Snapshot all living state.
      3. Baseline trajectory: continue running on a fixed input.
      4. Restore snapshot, perturb weight at 25% of weight std, run again.
      5. Measure trajectory deviation from baseline at each step.
      6. Same protocol with a "frozen v2" — perturbed but with no
         subsequent self-modification (we manually freeze by setting
         pc_rate to 0 in a controlled comparison layer).
      7. Compare recovery profiles. v2 should show measurable narrowing of
         deviation over time (homeostatic + PC pull toward set_point);
         frozen control should not.
    """
    d = 32
    n_settle_steps = 100   # let the layer establish a basin
    n_traj_steps = 60      # how many steps to record after perturbation
    perturbation_strength = 0.25

    torch.manual_seed(0)
    layer = PredictiveCodingLayer(in_features=d, out_features=d, num_episodes=4)

    torch.manual_seed(42)
    fixed_input = torch.randn(8, d) * 0.5

    # Settle into a basin
    for _ in range(n_settle_steps):
        layer(fixed_input)

    # Snapshot living state for restoration
    snapshot = _snapshot_living_state(layer)

    # Baseline trajectory (no perturbation)
    baseline = _capture_trajectory(layer, fixed_input, n_traj_steps)

    # Restore and perturb the weight
    _restore_living_state(layer, snapshot)
    weight_std = layer.weight.std().item()
    torch.manual_seed(123)
    perturb_v2 = torch.randn_like(layer.weight) * (perturbation_strength * weight_std)
    layer.weight.add_(perturb_v2)
    perturbed_v2 = _capture_trajectory(layer, fixed_input, n_traj_steps)

    # Build a frozen control: same starting state, same perturbation, but
    # PC self-modification is essentially disabled (rates set very low).
    # Without self-mod, the layer has no dynamics that could pull the
    # trajectory back toward baseline — recovery should be flat/none.
    torch.manual_seed(0)
    control = PredictiveCodingLayer(
        in_features=d, out_features=d, num_episodes=4,
        pc_rate=1e-12, pred_learning_rate=1e-12,
        homeostatic_decay=1e-12, set_point_adapt_rate=1e-12,
    )
    for _ in range(n_settle_steps):
        control(fixed_input)
    # Match the same weight as the v2 layer at snapshot time so both start
    # from the SAME basin. Otherwise we'd compare different starting points.
    with torch.no_grad():
        control.weight.copy_(snapshot["weight"])
        control.set_point.copy_(snapshot["set_point"])
    baseline_control = _capture_trajectory(control, fixed_input, n_traj_steps)
    with torch.no_grad():
        control.weight.copy_(snapshot["weight"])
        control.set_point.copy_(snapshot["set_point"])
        control.weight.add_(perturb_v2)
    perturbed_control = _capture_trajectory(control, fixed_input, n_traj_steps)

    # Measure trajectory deviation per step (L2 norm over batch+feature dims)
    def deviation(perturbed, baseline):
        diff = (perturbed - baseline).reshape(perturbed.shape[0], -1)
        return diff.norm(dim=-1)

    dev_v2 = deviation(perturbed_v2, baseline)            # [n_traj_steps]
    dev_control = deviation(perturbed_control, baseline_control)

    # Recovery = how much deviation shrinks from early to late window
    def recovery_ratio(dev):
        early = dev[:5].mean().item() + 1e-8
        late = dev[-5:].mean().item()
        return (early - late) / early  # positive = recovered toward baseline

    v2_recovery = recovery_ratio(dev_v2)
    control_recovery = recovery_ratio(dev_control)

    # Falsification check: v2 must recover meaningfully more than control.
    # Either v2 actively recovers (positive) AND control stays flat or
    # diverges (near-zero or negative), or v2 simply recovers far more.
    assert v2_recovery > control_recovery + 0.02, (
        f"M5 attractor falsification: v2 recovery ({v2_recovery:.4f}) is "
        f"not distinguishable from frozen control ({control_recovery:.4f}). "
        f"PC dynamics don't show basin-attractor behavior; 'temporal "
        f"existence' is unfalsifiable in v2. Strategic shift to v2-primary "
        f"(2026-05-09) needs revisiting."
    )


# ---------------------------------------------------------------------------
# 2. Head-to-head comparison — loads m5_runner results, skips if absent
# ---------------------------------------------------------------------------

M5_RUNS_DIR = Path("E:/runs/m5")  # pre-JEPA runs moved to E: 2026-07-22


def _collect_results(arch: str, expected_seeds: list[int]) -> list[dict]:
    """Load results.json for all expected (arch, seed) combinations.

    Returns the list of result dicts. Skips the test if any are missing —
    the head-to-head suite needs all data to be meaningful.
    """
    results = []
    for seed in expected_seeds:
        path = M5_RUNS_DIR / f"{arch}_seed{seed}" / "results.json"
        if not path.exists():
            pytest.skip(
                f"M5 head-to-head needs {path} (run "
                f"`python -m luthi.v2.m5_runner --arch {arch} --seed {seed} ...`)"
            )
        results.append(json.loads(path.read_text()))
    return results


def _agg_best_val(results: list[dict]) -> tuple[float, float]:
    """Mean ± std of best_val across seeds."""
    vals = [r["best_val"] for r in results]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
    return mean, var ** 0.5


def test_m5_v2_convergence_within_20pct_of_dead():
    """v2's mean best val loss must be within 20% of DeadLM's at matched
    compute. >20% worse = v2 has a fundamental efficiency problem and
    fails M5 falsification.
    """
    seeds = [42, 1337, 2026]
    v2_results = _collect_results("v2", seeds)
    dead_results = _collect_results("dead", seeds)

    v2_mean, v2_std = _agg_best_val(v2_results)
    dead_mean, dead_std = _agg_best_val(dead_results)

    relative = (v2_mean - dead_mean) / max(dead_mean, 1e-8)
    print(
        f"\n[M5] v2: {v2_mean:.4f} ± {v2_std:.4f}\n"
        f"[M5] dead: {dead_mean:.4f} ± {dead_std:.4f}\n"
        f"[M5] v2-vs-dead relative: {relative:+.4%}"
    )

    assert relative < 0.20, (
        f"M5 falsification: v2 best val ({v2_mean:.4f}) is more than 20% "
        f"worse than DeadLM ({dead_mean:.4f}). Relative gap "
        f"{relative:.2%}. v2 has a fundamental efficiency problem at "
        f"matched compute."
    )


def test_m5_no_nan_events_in_either_arch():
    """Both architectures should train without NaN events at this scale."""
    seeds = [42, 1337, 2026]
    for arch in ["v2", "dead"]:
        results = _collect_results(arch, seeds)
        for r in results:
            assert r["nan_events"] == 0, (
                f"M5 stability: {arch} seed={r['seed']} hit "
                f"{r['nan_events']} NaN events during training. v2 should "
                f"have its NaN-race guard active; dead has no living state "
                f"to corrupt — any NaN here points at a deeper issue."
            )


def test_m5_v2_train_val_gap_not_worse_than_dead():
    """v2's overfit gap (train - val) should not be dramatically larger
    than DeadLM's. v2's PC dynamics + episode store are claimed to
    generalize better; this test catches the failure mode where v2
    memorizes train more aggressively than DeadLM.
    """
    seeds = [42, 1337, 2026]
    v2_results = _collect_results("v2", seeds)
    dead_results = _collect_results("dead", seeds)

    def gap(r: dict) -> float:
        return r["val_losses"][-1] - r["train_losses"][-1]

    v2_gap = sum(gap(r) for r in v2_results) / len(v2_results)
    dead_gap = sum(gap(r) for r in dead_results) / len(dead_results)

    # Allow v2 up to 2x DeadLM's gap (PC dynamics can shape representations
    # differently). Worse than 2x is a generalization failure signal.
    assert v2_gap <= 2.0 * dead_gap + 0.5, (
        f"M5 generalization: v2 train-val gap ({v2_gap:.4f}) is far worse "
        f"than DeadLM's ({dead_gap:.4f}). v2 may be overfitting through "
        f"its episode store or self-modification dynamics."
    )
