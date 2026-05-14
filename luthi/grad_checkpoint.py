"""Gradient checkpointing with side-effect awareness for living layers.

torch.utils.checkpoint.checkpoint runs the wrapped function twice: once
during the original forward (under torch.no_grad()) and once during
backward to recompute activations (under torch.enable_grad()). For a
plain compute layer that's fine. Living layers self-modify on every
forward call, so without recompute detection the self-modification update
would fire twice per step and gradients would drift because the second
pass sees the post-modification weight.

This module exposes `is_recomputing()`, a thread-local flag toggled by
`luthi_context_fn()`. Pass `context_fn=luthi_context_fn` to
`torch.utils.checkpoint.checkpoint(..., use_reentrant=False, ...)` and
the flag will be True only during the recompute phase. Living layers
check the flag and:

  - Skip self-modification / homeostatic / episode-store updates on recompute.
  - Reuse the weight snapshot from the original forward so the recomputed
    activations are bit-identical to the originals.

This keeps `gradient checkpointing == no checkpointing` at the gradient
and living-state level, which the equivalence test verifies.
"""

from __future__ import annotations

import contextlib
import threading


_state = threading.local()


def is_recomputing() -> bool:
    """True while the gradient checkpoint mechanism is replaying forward."""
    return getattr(_state, "recomputing", False)


@contextlib.contextmanager
def _recompute_active():
    """Context manager that toggles the recompute flag on entry/exit."""
    prev = getattr(_state, "recomputing", False)
    _state.recomputing = True
    try:
        yield
    finally:
        _state.recomputing = prev


def luthi_context_fn():
    """Return (forward_ctx, recompute_ctx) tuple for torch.utils.checkpoint.

    Pass this as the `context_fn` argument to
    `torch.utils.checkpoint.checkpoint(..., use_reentrant=False,
    context_fn=luthi_context_fn)`.
    """
    return contextlib.nullcontext(), _recompute_active()
