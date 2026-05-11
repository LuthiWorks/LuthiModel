"""Consolidation: low-variance trigger + gradient-based replay.

The two-tier memory architecture (the v2 distinguishing feature):

    Slow PC weights  ←─ consolidation ──  Fast episode store

Episodes are layer-level weight snapshots stored during forward when the
PC update is salient. Consolidation periodically replays them into the PC
weights so accumulated history shapes the model's predictive structure,
not just its retrieval store.

Per `docs/V2_IMPLEMENTATION_PLAN.md` M4:
- Trigger: rolling 1000-step window of per-step prediction error variance.
  Triggers begin only after the window is full. Baseline = mean of window.
  Threshold = 0.5 × baseline. N=100 consecutive sub-threshold steps fire.
- Replay: for each stored episode (by salience, highest first), pull
  current weight toward the stored snapshot at consolidation_rate
  (= 10% of pc_rate by default). Prediction matrix nudges by the same
  consolidation_error so it stays consistent with the consolidated weight.

The M4 STOP GATE: if consolidated layer's behavior on the episode's
context is not measurably closer to the stored snapshot than a control
without consolidation, v2 has no architectural novelty over a vanilla
transformer + episode store and should be abandoned (per the brief).
Refinement 4's bootstrap window prevents premature triggering before
the model has any predictive structure to consolidate against.
"""

from __future__ import annotations

from collections import deque

import torch


class ConsolidationTracker:
    """Variance trigger with a frozen warmup-baseline.

    Maintains a deque of the last `window_size` per-step prediction-error
    variance values. The baseline is **snapshot once at the end of warmup**
    (when the window first fills) and never updated again. Threshold =
    `threshold_factor × frozen_baseline`. Fires when the most recent
    `trigger_window` steps have all been below threshold; resets the
    sub-threshold counter after firing.

    Refinement 4 (2026-05-08): the rolling-window bootstrap means triggers
    can never fire in the first `window_size` steps. Intentional — there's
    no consolidation target before the model has any predictive structure.

    2026-05-10 audit fix: the baseline used to be `mean(self._history)` re-
    computed each step. As training stabilized, variance dropped, the
    rolling mean dropped with it, and the threshold dropped — firing
    consolidation increasingly often, even when there was nothing meaningful
    to consolidate. Freezing the baseline at the end of warmup eliminates
    that positive-feedback loop. If the training distribution shifts later
    (e.g., curriculum stage transitions), call `reset()` to re-warm up.
    """

    def __init__(
        self,
        window_size: int = 1000,
        trigger_window: int = 100,
        threshold_factor: float = 0.5,
    ):
        self.window_size = window_size
        self.trigger_window = trigger_window
        self.threshold_factor = threshold_factor
        self._history: deque[float] = deque(maxlen=window_size)
        self._below_threshold_count: int = 0
        self._baseline: float | None = None  # set once at end of warmup

    def step(self, pred_error_variance: float) -> bool:
        """Record one variance value; return True if consolidation should fire."""
        self._history.append(float(pred_error_variance))

        if len(self._history) < self.window_size:
            return False

        # Freeze the baseline the first time the warmup window completes.
        if self._baseline is None:
            self._baseline = sum(self._history) / len(self._history)

        threshold = self._baseline * self.threshold_factor

        if pred_error_variance < threshold:
            self._below_threshold_count += 1
        else:
            self._below_threshold_count = 0

        if self._below_threshold_count >= self.trigger_window:
            self._below_threshold_count = 0
            return True
        return False

    def reset(self) -> None:
        """Clear history and baseline. Use this on curriculum stage
        transitions or any time the training distribution changes — the
        next warmup will snapshot a fresh baseline for the new regime.
        """
        self._history.clear()
        self._below_threshold_count = 0
        self._baseline = None

    @property
    def is_warmed_up(self) -> bool:
        return self._baseline is not None

    @property
    def baseline(self) -> float | None:
        """The frozen baseline variance, or None if still warming up."""
        return self._baseline


def consolidate_layer(
    layer,
    consolidation_rate_factor: float = 0.1,
) -> int:
    """Replay all stored episodes by pulling current weight toward each
    stored snapshot.

    For each episode (sorted by salience, highest first):
      consolidation_error = stored_snapshot - current_weight
      weight += consolidation_error * (pc_rate * factor)

    The factor (default 0.1) is "10% of pc_rate" per the plan's hyperparameter
    table — gentle enough that no single replay pass overwrites recent
    learning, persistent enough across many low-variance windows to let
    accumulated history shape the predictive structure.

    Audit 2026-05-11 fix: the previous implementation also added
    `consolidation_error` to `layer.prediction`. That's semantically wrong
    — `prediction` maps outputs to predicted inputs (a different
    mathematical space than `weight`). Applying the weight-delta to
    prediction was nudging it toward weight magnitudes rather than toward
    input-prediction accuracy. Removed; the natural PC forward loop will
    re-adapt prediction to the consolidated weights via its usual update
    rule. If a prediction-specific consolidation target is needed later,
    it should come from replaying stored context through the layer and
    computing a prediction-specific error — not from the weight delta.

    Returns the number of episodes replayed (for logging/diagnostics).

    All updates wrapped in `torch.no_grad()`. Caller (typically the layer's
    own forward when triggered) is responsible for guarding against being
    called when the episode store is empty — this function returns 0
    cleanly in that case.
    """
    if layer.episode_count.item() == 0:
        return 0

    n = layer.episode_count.item()
    saliences = layer.episode_saliences[:n]
    order = torch.argsort(saliences, descending=True)

    consolidation_rate = layer.pc_rate * consolidation_rate_factor

    with torch.no_grad():
        for idx in order:
            if layer.episode_values.dtype == torch.int8:
                snapshot = (
                    layer.episode_values[idx].to(layer.weight.dtype)
                    * layer.episode_scales[idx]
                )
            else:
                snapshot = layer.episode_values[idx]

            consolidation_error = snapshot - layer.weight
            layer.weight.add_(
                consolidation_error, alpha=consolidation_rate
            )

    return int(n)
