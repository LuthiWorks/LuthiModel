"""Bit-identity tests for the C++ pc_ops with sparse_gate.

The C++ extension at `luthi/csrc/pc_ops.cpp` accepts an optional
`sparse_gate: [out_features]` mask that zeros out delta_w rows for
gated-off outputs (mirroring the Python reference at
`pc_ops.py:138-139`).

These tests guard the contract: the C++ kernel must produce
bit-identical buffers to `_pc_self_modify_python` on the same inputs
with the same gate. If the C++ math drifts from Python (e.g. due to
reordered ops or a typo), these tests catch it before it reaches the
training run.

The dispatcher in `pc_ops.py` calls C++ for both dense (no gate) and
sparse (gate present) now; the dense path was already covered by the
existing pc_block / pc_layer tests, so these tests focus on the sparse
case and the dense/sparse equivalence boundaries.
"""

import pytest
import torch


# ---------------------------------------------------------------------------
# Skip the whole module when the C++ extension isn't loaded. This happens
# when Brian is on a fresh machine without a C++ compiler, or when the
# JIT build fails for any reason. The pure-Python path covers those cases
# but this test module isn't exercising it.
# ---------------------------------------------------------------------------

from luthi.v2 import pc_ops as _pc_ops_module  # noqa: E402

if not _pc_ops_module._use_cpp:
    pytest.skip(
        "C++ pc_ops extension not loaded (Python fallback only on this host).",
        allow_module_level=True,
    )

from luthi.v2.pc_ops import _pc_self_modify_python, pc_self_modify  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_buffers(out_features: int = 8, in_features: int = 16,
                  batch_size: int = 4, device: str = "cpu", seed: int = 0):
    torch.manual_seed(seed)
    return dict(
        weight=torch.randn(out_features, in_features, device=device) * 0.1,
        prediction=torch.randn(out_features, in_features, device=device) * 0.1,
        set_point=torch.randn(out_features, in_features, device=device) * 0.1,
        momentum=torch.zeros(out_features, in_features, device=device),
        update_ema=torch.ones(out_features, in_features, device=device) * 1e-4,
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


def _clone_buffers(bufs: dict) -> dict:
    return {k: v.clone() for k, v in bufs.items()}


def _modified_keys() -> list[str]:
    """Keys that pc_self_modify modifies in-place. Inputs only ('x_flat',
    'output') are excluded — they should be unchanged between calls."""
    return [
        "weight", "prediction", "set_point", "momentum",
        "update_ema", "precision", "error_acc",
    ]


def _assert_buffers_equal(py_bufs: dict, cpp_bufs: dict, *, context: str):
    for key in _modified_keys():
        assert torch.equal(py_bufs[key], cpp_bufs[key]), (
            f"{context}: C++ produced different result for {key!r}\n"
            f"  py={py_bufs[key].flatten()[:4]}\n"
            f"  cpp={cpp_bufs[key].flatten()[:4]}"
        )


# ---------------------------------------------------------------------------
# Sparse-gate tests
# ---------------------------------------------------------------------------

class TestSparseGateBitIdentity:
    def test_partial_gate_matches_python(self):
        """A mixed [0, 1] gate should produce identical buffers via either
        path. This is the core invariant: the C++ kernel's broadcast
        multiply must match the Python reference's."""
        py_bufs = _make_buffers()
        cpp_bufs = _clone_buffers(py_bufs)
        scalars = _default_scalars()

        # Alternating gate — half the outputs gated off.
        sparse_gate = torch.tensor(
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        )

        py_salience, py_pred_err = _pc_self_modify_python(
            **py_bufs, **scalars, sparse_gate=sparse_gate,
        )
        cpp_salience, cpp_pred_err = pc_self_modify(
            **cpp_bufs, **scalars, sparse_gate=sparse_gate,
        )

        assert py_salience == cpp_salience
        assert torch.equal(py_pred_err, cpp_pred_err)
        _assert_buffers_equal(py_bufs, cpp_bufs, context="partial gate")

    def test_all_ones_gate_matches_dense(self):
        """A gate of all ones must produce the same result as no gate.
        Confirms the sparse branch doesn't corrupt the dense math when
        the gate is effectively pass-through."""
        dense_bufs = _make_buffers()
        gated_bufs = _clone_buffers(dense_bufs)
        scalars = _default_scalars()

        sparse_gate = torch.ones(8)  # out_features = 8

        # No gate path (existing C++ dense behavior).
        pc_self_modify(**dense_bufs, **scalars, sparse_gate=None)
        # All-ones gate path (new sparse code, should be a no-op multiply).
        pc_self_modify(**gated_bufs, **scalars, sparse_gate=sparse_gate)

        _assert_buffers_equal(dense_bufs, gated_bufs, context="all-ones gate")

    def test_all_zeros_gate_freezes_weight(self):
        """A gate of all zeros should result in zero weight change (no
        outer-product contribution) but ALL OTHER buffers should still
        update — prediction matrix, precision EMA, error_acc, etc., all
        depend on output and pred_error, not on delta_w."""
        bufs = _make_buffers()
        weight_before = bufs["weight"].clone()
        scalars = _default_scalars()

        sparse_gate = torch.zeros(8)
        pc_self_modify(**bufs, **scalars, sparse_gate=sparse_gate)

        # Weight should still see homeostatic regulation (step g) — that's
        # NOT gated. So weight WILL change, just not via delta_w. To verify
        # the gate worked, compare against the Python reference's behavior.
        ref_bufs = _make_buffers()
        _pc_self_modify_python(**ref_bufs, **scalars, sparse_gate=sparse_gate)
        _assert_buffers_equal(ref_bufs, bufs, context="all-zeros gate")

    def test_gate_does_not_modify_input_gate_tensor(self):
        """The sparse_gate tensor itself should not be modified by either
        path. Same gate object should be reusable across calls."""
        bufs = _make_buffers()
        scalars = _default_scalars()
        sparse_gate = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        gate_before = sparse_gate.clone()

        pc_self_modify(**bufs, **scalars, sparse_gate=sparse_gate)

        assert torch.equal(sparse_gate, gate_before), (
            "sparse_gate tensor was mutated by the C++ path"
        )

    def test_dense_path_still_works_via_none(self):
        """Regression guard: the dense path (sparse_gate=None) should
        still produce identical results to the Python path. The dispatch
        change shouldn't break the no-gate case."""
        py_bufs = _make_buffers()
        cpp_bufs = _clone_buffers(py_bufs)
        scalars = _default_scalars()

        py_salience, py_pred_err = _pc_self_modify_python(
            **py_bufs, **scalars, sparse_gate=None,
        )
        cpp_salience, cpp_pred_err = pc_self_modify(
            **cpp_bufs, **scalars, sparse_gate=None,
        )

        assert py_salience == cpp_salience
        assert torch.equal(py_pred_err, cpp_pred_err)
        _assert_buffers_equal(py_bufs, cpp_bufs, context="dense (no gate)")


class TestSparseGateAcrossSteps:
    """Multiple steps with a fixed gate. The point is to verify behavior
    accumulates correctly — gated rows stay frozen across multiple
    self-modification steps, ungated rows continue learning."""

    def test_repeated_calls_match_across_paths(self):
        py_bufs = _make_buffers()
        cpp_bufs = _clone_buffers(py_bufs)
        scalars = _default_scalars()
        sparse_gate = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        for step in range(5):
            _pc_self_modify_python(
                **py_bufs, **scalars, sparse_gate=sparse_gate,
            )
            pc_self_modify(
                **cpp_bufs, **scalars, sparse_gate=sparse_gate,
            )

            # Refresh inputs each step so the test exercises distinct
            # gradients. Use the same seed offset for both to keep
            # them aligned.
            torch.manual_seed(100 + step)
            new_x = torch.randn(4, 16)
            new_out = torch.randn(4, 8)
            py_bufs["x_flat"] = new_x.clone()
            py_bufs["output"] = new_out.clone()
            cpp_bufs["x_flat"] = new_x.clone()
            cpp_bufs["output"] = new_out.clone()

        _assert_buffers_equal(py_bufs, cpp_bufs, context="5 steps with gate")
