"""Triton kernel skeleton for `pc_self_modify`.

Status (2026-05-14): SKELETON ONLY. Not wired into the dispatch path.
Brian's hardware is a 7800 XT via DirectML, which does not run Triton.
This file exists so the kernel structure is in place for the first
ROCm/CUDA box (Spark-class), and so the invariant tests can run as a
forcing function on bit-identity with the Python reference.

When this gets filled in:

1. The kernel body must produce results bit-identical to
   `luthi.v2.pc_ops._pc_self_modify_python` on the same inputs.
   The `tests/test_pc_ops_triton.py::test_triton_matches_python` test
   is the gate — if it doesn't pass, the kernel is wrong.
2. Wire into `luthi/v2/pc_ops.py::pc_self_modify` as a third dispatch
   tier: Triton (if available + tensor on CUDA/ROCm) -> C++ -> Python.
3. Add `sparse_gate` support directly in the kernel — the C++ path
   currently skips when a gate is active, forcing Python fallback.
   Triton can handle the broadcast multiply inline.

The skeleton below is intentionally structured to mirror
`_pc_self_modify_python` section-by-section (a-k labels), so the
implementer fills tile loops into the right slots without re-deriving
the math.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Conditional Triton import
# ---------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl
    _has_triton = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _has_triton = False


def is_triton_available() -> bool:
    """Whether the Triton python package imports cleanly.

    Note: this does NOT check whether Triton can actually compile on the
    current device — that requires running a kernel. Use
    `try_compile_pc_ops()` for an end-to-end probe.
    """
    return _has_triton


# ---------------------------------------------------------------------------
# Kernel skeleton (a-k labels match _pc_self_modify_python)
# ---------------------------------------------------------------------------

if _has_triton:

    @triton.jit
    def _pc_self_modify_triton_kernel(
        # Buffers (all modified in-place where the Python reference modifies)
        weight_ptr, prediction_ptr, set_point_ptr,
        momentum_ptr, update_ema_ptr, precision_ptr,
        error_acc_ptr, plasticity_ptr,
        # Inputs
        x_flat_ptr, output_ptr,
        # Optional sparse gate ([out_features], or null)
        sparse_gate_ptr,
        has_sparse_gate: tl.constexpr,
        # Shape
        out_features: tl.constexpr,
        in_features: tl.constexpr,
        batch_size: tl.constexpr,
        # Scalars
        pc_rate, pred_learning_rate, homeostatic_decay,
        set_point_adapt_rate, momentum_decay, update_ema_decay,
        precision_ema_decay, precision_min, precision_max,
        prediction_clamp,
        # Tile sizes
        BLOCK_OUT: tl.constexpr,
        BLOCK_IN: tl.constexpr,
    ):
        """One PC self-modification step, tiled.

        IMPLEMENTATION STATUS: skeleton. The body below is a placeholder
        — fill in the per-section tile loops referencing
        `_pc_self_modify_python` (luthi/v2/pc_ops.py) for the exact math.

        Sections to implement (lettering matches the reference):
          a. output_mean = output.mean(dim=0)
             actual_input = x_flat.mean(dim=0)
             predicted_input = output_mean @ prediction
             pred_error = actual_input - predicted_input
          b/c. weighted_error = (pred_error * precision).clamp(-1, 1)
          d. delta_w = outer(output_mean, weighted_error) * plasticity * pc_rate
             if has_sparse_gate: delta_w = delta_w * sparse_gate[:, None]
          e. adaptive_factor = (2 / (1 + |delta_w|/(update_ema+1e-8))).clamp(max=1)
          f. weight += delta_w * adaptive_factor
             momentum = momentum_decay*momentum + (1-momentum_decay)*delta_w
             update_ema = update_ema_decay*update_ema + (1-update_ema_decay)*|delta_w|
          g. weight += homeostatic_decay * (set_point - weight)
          h. set_point += set_point_adapt_rate * (weight - set_point)
          i. prediction += pred_learning_rate * outer(output_mean, clamp(pred_error, -1, 1))
             prediction.clamp_(-prediction_clamp, prediction_clamp)
          j. precision = precision_ema_decay*precision +
                         (1-precision_ema_decay) * (1 / (pred_error^2 + 1e-3))
             precision.clamp_(precision_min, precision_max)
          k. per_output_contrib = |output_mean| * mean(|pred_error|)
             error_acc = update_ema_decay*error_acc +
                         (1-update_ema_decay)*per_output_contrib

        Returns via output buffers — Triton kernels don't return values;
        the dispatcher reads salience from error_acc.mean() after launch.
        """
        # PLACEHOLDER: real kernel body goes here.
        # This pass exists so the @triton.jit decorator parses without
        # error during package import on a Triton-equipped box.
        pass

else:

    _pc_self_modify_triton_kernel = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Python entry point (the same shape as `pc_self_modify` in pc_ops.py)
# ---------------------------------------------------------------------------

def pc_self_modify_triton(
    weight: torch.Tensor,
    prediction: torch.Tensor,
    set_point: torch.Tensor,
    momentum: torch.Tensor,
    update_ema: torch.Tensor,
    precision: torch.Tensor,
    error_acc: torch.Tensor,
    plasticity: torch.Tensor,
    x_flat: torch.Tensor,
    output: torch.Tensor,
    pc_rate: float,
    pred_learning_rate: float,
    homeostatic_decay: float,
    set_point_adapt_rate: float,
    momentum_decay: float,
    update_ema_decay: float,
    precision_ema_decay: float,
    precision_min: float,
    precision_max: float,
    prediction_clamp: float,
    sparse_gate: torch.Tensor | None = None,
) -> tuple[float, torch.Tensor]:
    """Triton-backed pc_self_modify. SKELETON — raises NotImplementedError.

    Fills the same contract as `luthi.v2.pc_ops.pc_self_modify`. Routes
    via `_pc_self_modify_triton_kernel` once it is implemented. No silent
    fallback: if Triton is unavailable or the kernel is not yet wired,
    this function raises rather than dropping to the Python path. The
    dispatcher in `pc_ops.py` is responsible for choosing whether to
    call this entry point at all.
    """
    if not _has_triton:
        raise NotImplementedError(
            "Triton is not installed in this environment. "
            "Install triton for ROCm/CUDA, or call the Python path via "
            "luthi.v2.pc_ops.pc_self_modify (which dispatches to the C++ "
            "or Python reference)."
        )
    raise NotImplementedError(
        "pc_self_modify_triton kernel body is not yet filled in. See "
        "luthi/v2/pc_ops_triton.py for the implementation outline. "
        "Use luthi.v2.pc_ops.pc_self_modify for the working C++/Python path."
    )


def try_compile_pc_ops() -> bool:
    """End-to-end probe: attempts to launch the kernel on a tiny input.

    Returns True if Triton imports, the kernel compiles, and a launch
    completes without error. Used by tests and by the dispatcher's
    one-time capability detection.

    Returns False on any failure rather than raising — this is a
    capability probe, not a load-bearing call.
    """
    if not _has_triton:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        # Minimal launch shape to force compilation. Kernel body is a no-op
        # right now, so this only validates that JIT compilation works on
        # the current device — not that the math is correct.
        device = torch.device("cuda")
        out_features, in_features, batch_size = 4, 4, 2
        zeros2d = torch.zeros(out_features, in_features, device=device)
        zeros1d_out = torch.zeros(out_features, device=device)
        zeros1d_in = torch.zeros(in_features, device=device)
        x_flat = torch.zeros(batch_size, in_features, device=device)
        output = torch.zeros(batch_size, out_features, device=device)
        grid = (1,)
        _pc_self_modify_triton_kernel[grid](
            zeros2d, zeros2d, zeros2d,
            zeros2d, zeros2d, zeros1d_in,
            zeros1d_out, zeros1d_in,
            x_flat, output,
            zeros1d_out,  # sparse_gate dummy
            False,        # has_sparse_gate
            out_features, in_features, batch_size,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            BLOCK_OUT=4, BLOCK_IN=4,
        )
        torch.cuda.synchronize()
        return True
    except Exception:
        return False
