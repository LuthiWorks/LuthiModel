"""Top-down backward sweep for v2 (predictive coding).

Replaces v1's `backward_pass.py`. Carries TWO channels in one sweep
per V2_IMPLEMENTATION_PLAN.md decision 3:

- **Prediction signal** — the upper block's PC pred_error flows down as
  the prediction_error component of the next signal. This is "what the
  upper block expected to receive that it didn't" — propagated as a
  modulation cue for the block below.
- **Modulation signal** — the salience component (which input dimensions
  carried useful information) modulates the lower block's plasticity;
  the prediction_error component nudges the lower block's set_point.

Refinement 3's M2 isolation tests verify each channel does its job
independently without destructive compounding when joint.
"""

from dataclasses import dataclass

import torch


@dataclass
class TopDownSignal:
    """Signal flowing from a higher block to a lower one in the v2 sweep.

    Same shape as v1's TopDownSignal (`luthi.backward_pass.TopDownSignal`)
    so the apply_top_down contract is identical. Both vectors live in
    **input-dimension space** for the receiving PC layer — the same input-
    dim adjustment broadcasts uniformly across the layer's output rows.

    Attributes:
        salience: [d_model] which input dimensions carried useful
            information. Modulates the lower block's plasticity.
        prediction_error: [d_model] per-input-dim mismatch between actual
            and predicted block input — sourced directly from the upper
            block's PC layer pred_error rather than a salience heuristic.
            Modulates the lower block's set_point.
        modulation_strength: Scalar, decays 0.8x per block downward.
    """

    salience: torch.Tensor
    prediction_error: torch.Tensor
    modulation_strength: float


def create_initial_signal(
    final_hidden: torch.Tensor,
    initial_strength: float = 1.0,
) -> TopDownSignal:
    """Create the initial top-down signal from the model's final hidden state.

    The top of the stack has no "block above" with PC pred_error to forward,
    so the initial prediction_error is zero. Salience is the per-dim
    activation magnitude of the final hidden state, normalized to [0, 1].
    """
    with torch.no_grad():
        salience = final_hidden.abs().mean(dim=(0, 1))
        salience = salience / (salience.max() + 1e-8)
        prediction_error = torch.zeros_like(salience)

    return TopDownSignal(
        salience=salience,
        prediction_error=prediction_error,
        modulation_strength=initial_strength,
    )


def compute_block_top_down(
    signal: TopDownSignal,
    block_input: torch.Tensor,
    pc_pred_error: torch.Tensor | None = None,
    decay: float = 0.8,
) -> TopDownSignal:
    """Compute the refined top-down signal to pass to the next block below.

    Args:
        signal: Top-down signal received from the block above.
        block_input: [batch, seq_len, d_model] this block's input from
            the forward pass — used to compute local salience.
        pc_pred_error: [d_model] the upper block's PC layer pred_error
            from its most recent forward pass. When provided, this becomes
            the prediction_error of the signal flowing down — the actual
            PC error, not a salience heuristic. When None (e.g., for v1
            compatibility or testing), falls back to v1's heuristic
            (local_salience - downstream_salience).
        decay: Modulation strength decay per block. 0.8 means each block
            below gets 80% of the previous strength.
    """
    with torch.no_grad():
        local_salience = block_input.abs().mean(dim=(0, 1))
        local_salience = local_salience / (local_salience.max() + 1e-8)

        blended_salience = signal.salience * 0.9 + local_salience * 0.1

        if pc_pred_error is not None:
            prediction_error = pc_pred_error
        else:
            prediction_error = local_salience - signal.salience

    return TopDownSignal(
        salience=blended_salience,
        prediction_error=prediction_error,
        modulation_strength=signal.modulation_strength * decay,
    )


def mask_signal(
    signal: TopDownSignal,
    enable_salience: bool = True,
    enable_prediction_error: bool = True,
) -> TopDownSignal:
    """Return a copy of `signal` with selected channels zeroed.

    Used by the M2 isolation tests (refinement 3) to disable one channel
    while leaving the other live. Zeroing rather than removing keeps the
    apply_top_down contract intact — the receiver sees a valid signal whose
    affected channel does nothing.
    """
    return TopDownSignal(
        salience=(
            signal.salience
            if enable_salience
            else torch.zeros_like(signal.salience)
        ),
        prediction_error=(
            signal.prediction_error
            if enable_prediction_error
            else torch.zeros_like(signal.prediction_error)
        ),
        modulation_strength=signal.modulation_strength,
    )
