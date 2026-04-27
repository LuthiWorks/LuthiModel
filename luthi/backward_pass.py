"""Top-down backward pass for bidirectional information flow.

Implements predictive processing in the LuthiModel architecture:
- Bottom-up (forward pass): input signals flow through blocks, carrying
  sensory information upward
- Top-down (backward pass): higher blocks send modulation signals downward,
  telling earlier blocks what was important and what was unexpected

This is NOT gradient backpropagation. It is a functional second pass where
later blocks inform earlier blocks about what mattered downstream, modulating
their living weight dynamics for the NEXT forward pass.

The biological analogy: cortical feedback connections. Higher visual areas
tell V1 "pay attention to edges in this region" — they don't send gradients,
they send predictions and salience signals.
"""

from dataclasses import dataclass

import torch


@dataclass
class TopDownSignal:
    """Signal flowing from a higher block to a lower one.

    Both ``salience`` and ``prediction_error`` are vectors of length
    ``d_model``, and they live in **input-dimension space** for the
    receiving living layer. They describe properties of the dimensions
    that *flowed into* the block (its input axis), not properties of
    the dimensions it produced. Living layers therefore broadcast these
    signals across their ``out_features`` axis when applying them — the
    same input-dim adjustment is uniformly applied to every output row.

    This convention matters because a living layer's ``set_point`` and
    ``plasticity`` are 2D ``[out, in]`` matrices: indexing the signal by
    the wrong axis would push the homeostatic state in the wrong
    direction. See ``LivingLayerV6.apply_top_down`` for how the broadcast
    is performed.

    Attributes:
        salience: [d_model] (= [in_features]) which input dimensions
            carried useful information for the task. High values mean
            "this input mattered."
        prediction_error: [d_model] (= [in_features]) per-input-dim
            mismatch between this block's actual input importance and
            what downstream blocks expected. Computed in
            ``compute_block_top_down`` as
            ``local_input_salience - downstream_salience``.
        modulation_strength: Scalar controlling how strongly this signal
            modulates the receiving block. Decays with distance — nearby
            blocks get strong modulation, distant blocks get gentle guidance.
    """

    salience: torch.Tensor
    prediction_error: torch.Tensor
    modulation_strength: float


def create_initial_signal(
    final_hidden: torch.Tensor,
    initial_strength: float = 1.0,
) -> TopDownSignal:
    """Create the initial top-down signal from the model's final hidden state.

    The initial signal captures what the output layer found important
    (high-magnitude activations) across all dimensions.

    Args:
        final_hidden: [batch, seq_len, d_model] final hidden state before
            the output projection.
        initial_strength: Starting modulation strength. Default 1.0.

    Returns:
        TopDownSignal to feed into the last block's top_down_pass.
    """
    with torch.no_grad():
        # Salience: which dimensions had the strongest activations?
        salience = final_hidden.abs().mean(dim=(0, 1))  # [d_model]

        # Normalize salience to [0, 1] range for stable modulation
        salience = salience / (salience.max() + 1e-8)

        # No prediction error for the initial signal — there's no
        # "block above" to have expectations
        prediction_error = torch.zeros_like(salience)

    return TopDownSignal(
        salience=salience,
        prediction_error=prediction_error,
        modulation_strength=initial_strength,
    )


def compute_block_top_down(
    signal: TopDownSignal,
    block_input: torch.Tensor,
    decay: float = 0.8,
) -> TopDownSignal:
    """Compute the refined top-down signal to pass to the next block below.

    Each block:
    1. Blends incoming salience with its own input's salience
    2. Computes prediction error (what it sent up vs what was expected)
    3. Decays modulation strength

    Args:
        signal: Top-down signal received from the block above.
        block_input: [batch, seq_len, d_model] this block's input from
            the forward pass.
        decay: Modulation strength decay factor per block. Default 0.8
            means each block backward gets 80% of the previous strength.

    Returns:
        Refined TopDownSignal for the block below.
    """
    with torch.no_grad():
        # This block's own salience: what was active in its input?
        local_salience = block_input.abs().mean(dim=(0, 1))  # [d_model]
        local_salience = local_salience / (local_salience.max() + 1e-8)

        # Blend: mostly downstream salience, with local context mixed in
        blended_salience = signal.salience * 0.9 + local_salience * 0.1

        # Prediction error: difference between what this block produced
        # (local_salience) and what downstream found important (signal.salience)
        prediction_error = local_salience - signal.salience

    return TopDownSignal(
        salience=blended_salience,
        prediction_error=prediction_error,
        modulation_strength=signal.modulation_strength * decay,
    )
