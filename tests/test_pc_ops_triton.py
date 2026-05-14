"""Invariant tests for the Triton pc_ops skeleton.

These are FORCING-FUNCTION tests for the kernel implementer:

  - `test_triton_not_implemented_raises_loud` runs everywhere and locks
    in the no-silent-fallback contract while the skeleton is unfilled.
  - `test_triton_matches_python` is the bit-identity gate. Skipped on
    hardware without Triton (DirectML / Brian's 7800 XT). Once the
    kernel is filled in, this test going green is the proof of
    correctness. While the skeleton's kernel body is a no-op, this
    test is `xfail` on Triton-equipped hardware so CI surfaces the
    moment someone wires it up without finishing the math.
"""

import pytest
import torch

from luthi.v2.pc_ops_triton import (
    is_triton_available,
    pc_self_modify_triton,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_buffers(out_features: int = 8, in_features: int = 16,
                  batch_size: int = 4, device: str = "cpu"):
    torch.manual_seed(0)
    return dict(
        weight=torch.randn(out_features, in_features, device=device) * 0.1,
        prediction=torch.randn(out_features, in_features, device=device) * 0.1,
        set_point=torch.randn(out_features, in_features, device=device) * 0.1,
        momentum=torch.zeros(out_features, in_features, device=device),
        update_ema=torch.zeros(out_features, in_features, device=device),
        precision=torch.ones(in_features, device=device),
        error_acc=torch.zeros(out_features, device=device),
        plasticity=torch.ones(in_features, device=device),
        x_flat=torch.randn(batch_size, in_features, device=device),
        output=torch.randn(batch_size, out_features, device=device),
    )


def _default_scalars():
    return dict(
        pc_rate=0.001,
        pred_learning_rate=0.0001,
        homeostatic_decay=0.001,
        set_point_adapt_rate=1e-6,
        momentum_decay=0.9,
        update_ema_decay=0.99,
        precision_ema_decay=0.99,
        precision_min=0.1,
        precision_max=10.0,
        prediction_clamp=1.0,
    )


# ---------------------------------------------------------------------------
# Always-on contract tests
# ---------------------------------------------------------------------------

def test_triton_not_implemented_raises_loud():
    """While the skeleton is unfilled, calling the Triton path must raise
    a NotImplementedError. No silent fallback to the Python path here —
    the dispatcher in `pc_ops.py` owns that decision.
    """
    bufs = _make_buffers()
    scalars = _default_scalars()
    with pytest.raises(NotImplementedError):
        pc_self_modify_triton(**bufs, **scalars)


# ---------------------------------------------------------------------------
# GPU-only invariant test (bit-identity with Python reference)
# ---------------------------------------------------------------------------

triton_unavailable = not (is_triton_available() and torch.cuda.is_available())


@pytest.mark.skipif(
    triton_unavailable,
    reason="Triton + CUDA required; DirectML/CPU box has neither.",
)
@pytest.mark.xfail(
    reason="Triton kernel body is a skeleton (no-op). This test goes "
           "green only after the kernel math is filled in to bit-match "
           "_pc_self_modify_python.",
    strict=True,
)
def test_triton_matches_python():
    """Triton output must match `_pc_self_modify_python` bit-identically.

    Run on the same inputs through both paths; compare every modified
    buffer with `torch.equal` (not `allclose` — Triton kernels should
    produce the same float results as the Python reference modulo
    associativity, which is tightly controlled here).
    """
    from luthi.v2.pc_ops import _pc_self_modify_python

    py_bufs = _make_buffers(device="cuda")
    triton_bufs = {k: v.clone() for k, v in py_bufs.items()}
    scalars = _default_scalars()

    py_salience, py_pred_err = _pc_self_modify_python(**py_bufs, **scalars)
    triton_salience, triton_pred_err = pc_self_modify_triton(
        **triton_bufs, **scalars,
    )

    assert py_salience == triton_salience
    assert torch.equal(py_pred_err, triton_pred_err)
    for k in py_bufs:
        if k in ("x_flat", "output"):
            continue  # inputs only, not modified
        assert torch.equal(py_bufs[k], triton_bufs[k]), (
            f"Triton kernel produced different result for {k}"
        )
