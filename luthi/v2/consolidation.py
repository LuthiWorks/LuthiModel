"""Consolidation: low-variance trigger + two replay pathways.

The two-tier memory architecture (the v2 distinguishing feature):

    Slow PC weights  ←─ consolidation ──  Fast episode store

Episodes are layer-level weight snapshots + input patterns stored during
forward when the PC update is salient. Consolidation periodically
replays them into the PC weights so accumulated history shapes the
model's predictive structure, not just its retrieval store.

Per `docs/V2_IMPLEMENTATION_PLAN.md` M4:
- Trigger: rolling 1000-step window of per-step prediction error variance.
  Triggers begin only after the window is full. Baseline = mean of window.
  Threshold = 0.5 × baseline. N=100 consecutive sub-threshold steps fire.

Two replay pathways (selectable by `consolidation_style`):

1. **Gradient-replay** (`consolidate_layer`, default — M4 original).
   For each stored snapshot W_stored, pull current weight toward the
   snapshot at consolidation_rate (= 10% of pc_rate by default).
   Simple linear interpolation in weight space.

2. **Attractor-style** (`consolidate_layer_attractor`, added 2026-05-14).
   Salvatori et al. (2023), "Associative Memories via Predictive
   Coding." For each stored input pattern, re-present it through the
   layer and run pc_self_modify at consolidation rate. The stored
   patterns become local minima of the prediction-error energy —
   future inputs near a stored pattern are pulled toward it by the
   forward dynamics. Engineered attractors on the slow path, on top of
   the basin-attractor dynamics already emergent on the fast path.

The two pathways are additive, not mutually exclusive (`consolidation_style="both"`
runs both). Choice of which becomes the production default is an empirical
question — Phase 3G has the validation ablation in the To-Do.

The M4 STOP GATE: if consolidation has no measurable effect on
prediction quality post-replay, v2 has no architectural novelty over a
vanilla transformer + episode store and should be abandoned (per the
brief). Refinement 4's bootstrap window prevents premature triggering
before the model has any predictive structure to consolidate against.

Wording corrected 2026-07-29. This docstring previously stated the gate
as "the consolidated layer's behavior on the episode's context is
measurably closer to the stored snapshot than a control" -- the
ML_GLOSSARY prediction-quality formulation above is the correct one, and
the difference is not cosmetic. The closer-to-snapshot test only ever
made sense for gradient-replay. Attractor consolidation moves weights
*away* from the stored snapshots (measured: +3.6e-3 to +3.8e-3 relative
across all six v5 runs) while reducing prediction error on the stored
inputs by 12.9-16.1%, because it does not pull toward a past weight
state -- it makes stored inputs fixed points of the layer's own
dynamics. Read literally, the old wording would have failed the one
pathway that demonstrably works. See
docs/research/2026-07-29_m4-stop-gate-rerun.md.
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


def consolidate_layer_attractor(
    layer,
    consolidation_rate_factor: float = 0.1,
    n_replay_passes: int = 1,
) -> int:
    """Salvatori-style attractor consolidation.

    For each stored episode (highest-salience first), re-present the
    stored input pattern through the layer's PC dynamics at a reduced
    rate. The effect: stored patterns become local minima of the
    layer's prediction-error energy, so future inputs near a stored
    pattern are pulled toward it by the forward dynamics. This makes
    basin-attractor structure an engineered property of the slow
    consolidation pathway rather than an emergent property of the fast
    inference dynamics alone.

    Reference: Salvatori, T., Mali, A., Buckley, C., Tschantz, A.,
    Friston, K., Bogacz, R., Lukasiewicz, T. (2023). "Associative
    Memories via Predictive Coding."

    Math contrast with `consolidate_layer` (gradient-replay):
      gradient-replay: W += α * (W_stored - W)            -- linear pull
                                                              in weight space
      attractor:       run pc_self_modify(x=stored_input, ...)
                                                           -- runs the
                       layer's own update rule on the stored pattern,
                       reinforcing the input as a fixed point.

    The two pathways are complementary. Gradient-replay says "be more
    like you were when this happened." Attractor says "this input should
    resolve to a stable state." A future input that matches a stored
    pattern benefits from both; a future input that's just structurally
    similar to a stored context benefits only from gradient-replay.

    Args:
        layer: A PredictiveCodingLayer instance. Must have an
            `episode_inputs` buffer populated by `_store_episode`.
        consolidation_rate_factor: Scales both pc_rate and
            pred_learning_rate down for replay. Default 0.1 matches the
            gradient-replay path's "10% of pc_rate" convention so a
            single consolidation event has comparable magnitude across
            the two pathways.
        n_replay_passes: Number of times to iterate over all stored
            episodes per consolidation event. Default 1 mirrors
            gradient-replay's single-pass semantics. Salvatori's paper
            uses many passes for full convergence; we leave that to the
            caller because consolidation events themselves recur.

    Returns:
        Number of stored episodes replayed (per single pass, not
        n_replay_passes × that). 0 if the store is empty.
    """
    if not hasattr(layer, "episode_inputs"):
        raise RuntimeError(
            "consolidate_layer_attractor requires the `episode_inputs` "
            "buffer on the layer. Either the layer pre-dates the 2026-05-14 "
            "Salvatori implementation, or the buffer was not registered. "
            "Use consolidate_layer (gradient-replay) instead, or rebuild "
            "the layer."
        )
    if layer.episode_count.item() == 0:
        return 0

    n = layer.episode_count.item()
    saliences = layer.episode_saliences[:n]
    order = torch.argsort(saliences, descending=True)

    consolidation_pc_rate = layer.pc_rate * consolidation_rate_factor
    consolidation_pred_rate = (
        layer.pred_learning_rate * consolidation_rate_factor
    )

    from luthi.v2.pc_ops import pc_self_modify

    with torch.no_grad():
        for _ in range(n_replay_passes):
            for idx in order:
                # Recover the stored input pattern as a batch-1 tensor.
                # pc_self_modify expects [batch, in_features] for x_flat
                # and [batch, out_features] for output; batch=1 is fine
                # because the math takes mean(dim=0) of both.
                stored_input = (
                    layer.episode_inputs[idx]
                    .to(layer.weight.dtype)
                    .unsqueeze(0)
                )
                # Re-present through the *current* weight, not the
                # stored snapshot. The point of attractor consolidation
                # is to reshape the current weight so the stored input
                # is a fixed point of its dynamics — not to revert the
                # weight to its state when this episode was stored.
                output = stored_input @ layer.weight.T

                # NaN safety mirrors the forward path's guard.
                if not torch.isfinite(output).all():
                    continue

                pc_self_modify(
                    layer.weight, layer.prediction, layer.set_point,
                    layer.momentum, layer.update_ema, layer.precision,
                    layer.error_acc, layer.plasticity,
                    stored_input, output,
                    consolidation_pc_rate, consolidation_pred_rate,
                    layer.homeostatic_decay, layer.set_point_adapt_rate,
                    layer.momentum_decay, layer.update_ema_decay,
                    layer.precision_ema_decay,
                    layer.precision_min, layer.precision_max,
                    layer.prediction_clamp,
                )

    return int(n)
