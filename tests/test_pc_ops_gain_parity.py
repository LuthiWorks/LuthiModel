"""C++/Python parity for the inverted-U gain (spec §8 step 7).

The gain now runs on the fast path: the dispatcher routes gain-on to the C++
extension. This suite is the contract that the C++ math matches
`_pc_self_modify_python` exactly -- if the C++ gain drifts (reordered ops, a
typo, a wrong clamp bound), it must fail here before it reaches a training run.

Skipped whole when the C++ extension isn't loaded (Python-fallback hosts): the
Python path is the reference these tests check the C++ path against, so there is
nothing to compare when C++ is absent.
"""

from __future__ import annotations

import pytest
import torch

from luthi.v2 import pc_ops as m

if not m._use_cpp:
    pytest.skip(
        "C++ pc_ops extension not loaded (Python fallback only on this host).",
        allow_module_level=True,
    )


def _bufs(seed: int = 0):
    torch.manual_seed(seed)
    return dict(
        weight=torch.randn(8, 16) * 0.1,
        prediction=torch.randn(8, 16) * 0.1,
        set_point=torch.randn(8, 16) * 0.1,
        momentum=torch.zeros(8, 16),
        update_ema=torch.ones(8, 16) * 1e-4,
        precision=torch.ones(16),
        error_acc=torch.zeros(8),
        plasticity=torch.ones(16),
        x_flat=torch.randn(4, 16),
        output=torch.randn(4, 8),
    )


_SC = dict(
    pc_rate=0.01, pred_learning_rate=0.0001, homeostatic_decay=0.001,
    set_point_adapt_rate=1e-6, momentum_decay=0.9, update_ema_decay=0.99,
    precision_ema_decay=0.99, precision_min=0.1, precision_max=10.0,
    prediction_clamp=1.0,
)

_GAIN = dict(
    learning_gain_enabled=True, learning_gain_progress=0.2,
    learning_gain_rise=2.0, learning_gain_cap=3.0,
)

_STATE_KEYS = ("weight", "prediction", "set_point", "momentum",
               "update_ema", "precision", "error_acc")


def test_cpp_python_buffers_bit_identical_gain_on():
    """25 fixed-input steps with the gain amplifying: every mutated buffer is
    bit-identical between the C++ and Python paths."""
    cpp = _bufs(0)
    py = _bufs(0)
    for _ in range(25):
        m._cpp_ops.pc_self_modify(
            **cpp, **_SC, sparse_gate=None, **_GAIN,
            return_applied_change=True,
        )
        m._pc_self_modify_python(
            **py, **_SC, sparse_gate=None, **_GAIN,
            return_applied_change=True,
        )
    for k in _STATE_KEYS:
        assert torch.equal(cpp[k], py[k]), f"{k} drifted between C++ and Python"


def test_cpp_python_applied_change_matches():
    """The applied-change reduction agrees between paths at every step."""
    cpp = _bufs(1)
    py = _bufs(1)
    for _ in range(25):
        _, _, ac_c = m._cpp_ops.pc_self_modify(
            **cpp, **_SC, sparse_gate=None, **_GAIN,
            return_applied_change=True,
        )
        _, _, ac_p = m._pc_self_modify_python(
            **py, **_SC, sparse_gate=None, **_GAIN,
            return_applied_change=True,
        )
        assert ac_c.item() == pytest.approx(ac_p, abs=1e-7)


def test_cpp_pregain_histories_bit_identical_gain_on_vs_rest():
    """The measured spike-guard guarantee holds on the C++ path too: momentum
    and update_ema (adaptive_factor's inputs) are bit-identical whether the
    gain amplifies (progress<1) or rests (gain=1.0). Only the weight diverges."""
    active = _bufs(2)
    rest = _bufs(2)
    for _ in range(25):
        m._cpp_ops.pc_self_modify(
            **active, **_SC, sparse_gate=None,
            learning_gain_enabled=True, learning_gain_progress=0.2,
            learning_gain_rise=2.0, learning_gain_cap=3.0,
            return_applied_change=False,
        )
        m._cpp_ops.pc_self_modify(
            **rest, **_SC, sparse_gate=None,
            learning_gain_enabled=True, learning_gain_progress=1.0,
            learning_gain_rise=2.0, learning_gain_cap=3.0,
            return_applied_change=False,
        )
    assert torch.equal(active["momentum"], rest["momentum"])
    assert torch.equal(active["update_ema"], rest["update_ema"])
    assert not torch.allclose(active["weight"], rest["weight"])


def test_cpp_gain_off_still_bit_identical_to_python():
    """Regime f on the fast path: gain off, C++ vs Python bit-identical
    (guards that threading the new args did not perturb the legacy path)."""
    cpp = _bufs(3)
    py = _bufs(3)
    for _ in range(25):
        m._cpp_ops.pc_self_modify(**cpp, **_SC, sparse_gate=None)
        m._pc_self_modify_python(**py, **_SC, sparse_gate=None)
    for k in _STATE_KEYS:
        assert torch.equal(cpp[k], py[k]), k
