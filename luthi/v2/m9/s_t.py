"""Canonical s_t definition for the training seam.

Single source of truth for "given a sequence of post-trunk latents,
what is the cycle's state vector?". Phase 4a of the 2026-06-15
sanctuary-training-seam-integration-plan; written to close the
no-drift gap 4.8's review flagged: training (``_m9_head_step``) and
inference (``generate_text(return_state=True)``) must compute s_t via
the same function, or the V-head silently sees two different
distributions.

The companion no-drift test in ``tests/test_no_inline_s_t_pool.py``
greps the seam codepath for any inline ``.mean(dim=1)`` /
``.mean(1)`` over encoder latents and fails if anything other than
this file does that compute. That's what turns "shared helper" from
a naming convention into an actual guarantee.

**This file is the only place in the seam where the canonical
state-pool rule lives.** Inlining the same compute elsewhere makes
the s_t definition drift; the test will bite. If a future use case
needs a *different* pool rule (median, attention-weighted, etc.),
add a new named function here -- don't open-code the compute at the
call site.

The pool rule (mean over the time dimension) matches what the M9
trainer's ``_m9_head_step`` has historically done since M9 step 1:
``raw["online_context_latents"].detach().mean(dim=1)``. Phase 4a
just centralizes it.
"""

from __future__ import annotations

import torch

__all__ = ["compute_s_t", "pool_state_grad"]


def pool_state_grad(latents: torch.Tensor) -> torch.Tensor:
    """Pool ``[B, T, D]`` latents into a ``[B, D]`` state, KEEPING gradient.

    Identical pool rule to :func:`compute_s_t` (mean over the time
    dimension) but WITHOUT the ``detach()`` -- so the gradient flows back
    through the pooled state. This is the lived-JEPA prediction pooler
    (Item #6): the predictor's forecast must stay grad-connected so lived
    prediction error can train the encoder + predictor. Its detached twin
    ``compute_s_t`` is for the *target* and the M9 head, where the
    stop-grad discipline is required.

    Co-located here on purpose: keeping both poolers in one file means a
    future change to the pool rule (median, attention-weighted, ...)
    updates the grad and stop-grad paths together instead of letting them
    drift. ``compute_s_t`` is conceptually ``pool_state_grad(...).detach()``
    -- it is kept as a separate explicit line only so the
    ``test_no_inline_s_t_pool`` AST guard can pin the one canonical
    ``.detach().mean(dim=1)`` site.

    Args:
        latents: ``[B, T, D]`` post-trunk latents (e.g. the predictor's
            forecast over the continuation positions).

    Returns:
        ``[B, D]`` grad-connected state vector.

    Raises:
        ValueError: if ``latents`` isn't ``[B, T, D]``.
    """
    if latents.dim() != 3:
        raise ValueError(
            f"pool_state_grad expects [B, T, D] latents; got shape "
            f"{tuple(latents.shape)}. Add the batch dim explicitly at the "
            f"call site rather than passing a 2D tensor."
        )
    return latents.mean(dim=1)


def compute_s_t(latents: torch.Tensor) -> torch.Tensor:  # noqa: ANN -- explicit
    """Pool a ``[B, T, D]`` latent sequence into the cycle's ``[B, D]`` state.

    Detaches first, then mean-pools over the time dimension. Detach
    preserves the stop-grad discipline both the training-time
    ``_m9_head_step`` and the inference-time seam require (the M9
    head gradients must not flow back into the JEPA encoder).

    Args:
        latents: ``[B, T, D]`` post-trunk encoder output for a single
            modality (typically ``encode_result["per_modality"]["text"]``
            at inference, or ``raw["online_context_latents"]`` at
            training). Must be 3D; a 2D ``[T, D]`` input is rejected
            here so callers are forced to be explicit about the batch
            dim and a stale shape doesn't silently slip through.

    Returns:
        ``[B, D]`` detached state vector.

    Raises:
        ValueError: if ``latents`` isn't ``[B, T, D]``.
    """
    if latents.dim() != 3:
        raise ValueError(
            f"compute_s_t expects [B, T, D] latents; got shape "
            f"{tuple(latents.shape)}. Add the batch dim explicitly at "
            f"the call site rather than passing a 2D tensor."
        )
    return latents.detach().mean(dim=1)
