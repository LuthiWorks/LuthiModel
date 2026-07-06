"""Op-level integration tests for the inverted-U gain in pc_self_modify.

Regimes (f) legacy identity and (d) spike-guard bit-identity (gain spec §6).
The pre-gain history decision (spec §4, measured) makes (d) provable: momentum
and update_ema record delta_w regardless of the gain, so adaptive_factor's
inputs -- and thus the refinement-6 spike guard -- are bit-identical gain-on
vs off; only the applied weight diverges.

Bit-identity claims compare WITHIN the Python path (both calls gain-enabled) to
avoid C++/Python float noise; the legacy-at-rest claim uses allclose because it
may cross paths (gain-off can dispatch to C++).
"""

from __future__ import annotations

import torch

from luthi.v2.pc_ops import pc_self_modify


def _buffers(seed: int = 0):
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


def _scalars():
    return dict(
        pc_rate=0.001, pred_learning_rate=0.0001, homeostatic_decay=0.001,
        set_point_adapt_rate=1e-6, momentum_decay=0.9, update_ema_decay=0.99,
        precision_ema_decay=0.99, precision_min=0.1, precision_max=10.0,
        prediction_clamp=1.0,
    )


def _run(gain_kwargs, steps: int = 25, seed: int = 0):
    """Run N steps with fixed inputs (so delta_w is input/prediction-driven,
    not weight-driven -- the histories evolve identically across gain settings
    while only the applied weight diverges)."""
    bufs = _buffers(seed)
    sc = _scalars()
    for _ in range(steps):
        pc_self_modify(**bufs, **sc, **gain_kwargs)
    return bufs


def test_f_flag_off_is_the_default_path():
    off = _run({})
    off_explicit = _run({"learning_gain_enabled": False})
    for k in ("weight", "momentum", "update_ema"):
        assert torch.equal(off[k], off_explicit[k]), k


def test_f_gain_on_at_rest_is_numerically_legacy():
    """progress=1.0 -> fall 0 -> gain=1.0 everywhere -> legacy behavior."""
    off = _run({})
    rest = _run({"learning_gain_enabled": True, "learning_gain_progress": 1.0})
    for k in ("weight", "momentum", "update_ema"):
        assert torch.allclose(off[k], rest[k], atol=1e-6), k


def test_d_spike_guard_inputs_bit_identical_gain_on_vs_rest():
    """The load-bearing pre-gain proof: with the gain amplifying (progress<1)
    vs at rest (gain=1.0), momentum and update_ema -- adaptive_factor's inputs
    -- are BIT-IDENTICAL. The spike guard cannot be weakened by the gain.
    Only the applied weight diverges (the gain did its amplification)."""
    rest = _run({"learning_gain_enabled": True, "learning_gain_progress": 1.0})
    active = _run({"learning_gain_enabled": True, "learning_gain_progress": 0.2})
    assert torch.equal(rest["momentum"], active["momentum"]), \
        "momentum must be pre-gain identical"
    assert torch.equal(rest["update_ema"], active["update_ema"]), \
        "update_ema (spike-guard denominator) must be pre-gain identical"
    assert not torch.allclose(rest["weight"], active["weight"]), \
        "the gain must have amplified the applied weight change"


def test_gain_amplifies_the_applied_change():
    init = _buffers(seed=0)["weight"].clone()
    rest = _run({"learning_gain_enabled": True, "learning_gain_progress": 1.0})
    active = _run({"learning_gain_enabled": True, "learning_gain_progress": 0.2})
    moved_rest = (rest["weight"] - init).abs().sum()
    moved_active = (active["weight"] - init).abs().sum()
    assert moved_active > moved_rest, \
        "resolving coherent change should move the weight further than at-rest"
