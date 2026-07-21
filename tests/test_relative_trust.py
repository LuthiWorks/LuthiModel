"""Relative trust (v5 precision awakening, 2026-07-21).

The three-stage fix for the measured saturation (registry: the corrected
precision entries): numerics-only eps (legacy 1e-3 flattened a real
13-22x reliability spread to ~1.05x), freed ledger, ratio-to-median
use-time weighting with precision_min/max as RATIO bounds.

Covers:
- differentiation actually appears (the ledger spreads; nothing pins at
  the legacy cap) under tiny-error regimes;
- ratio semantics: weighting follows clamp(prec/median, 0.1, 10);
- flag-off stays legacy (eps flattening and cap pinning reproduce);
- C++/Python parity for the relative mode (skipped without the ext);
- plumbing: model kwarg reaches every layer; driver tables complete.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from luthi.v2 import pc_ops as m

_SC = dict(
    pc_rate=0.01, pred_learning_rate=0.0001, homeostatic_decay=0.001,
    set_point_adapt_rate=1e-6, momentum_decay=0.9, update_ema_decay=0.99,
    precision_ema_decay=0.9, precision_min=0.1, precision_max=10.0,
    prediction_clamp=1.0,
)


def _bufs(seed: int = 0, n_in: int = 16, n_out: int = 8):
    torch.manual_seed(seed)
    return dict(
        weight=torch.randn(n_out, n_in) * 0.1,
        prediction=torch.randn(n_out, n_in) * 0.1,
        set_point=torch.randn(n_out, n_in) * 0.1,
        momentum=torch.zeros(n_out, n_in),
        update_ema=torch.ones(n_out, n_in) * 1e-4,
        precision=torch.ones(n_in),
        error_acc=torch.zeros(n_out),
        plasticity=torch.ones(n_in),
        x_flat=torch.randn(4, n_in) * 0.01,   # tiny-error regime
        output=torch.randn(4, n_out) * 0.01,
    )


def _steps(bufs, n, **kw):
    out = None
    for _ in range(n):
        out = m._pc_self_modify_python(
            bufs["weight"], bufs["prediction"], bufs["set_point"],
            bufs["momentum"], bufs["update_ema"], bufs["precision"],
            bufs["error_acc"], bufs["plasticity"],
            bufs["x_flat"], bufs["output"], **{**_SC, **kw},
        )
    return out


class TestRelativeMode:
    def test_ledger_spreads_and_never_pins_at_legacy_cap(self):
        bufs = _bufs()
        _steps(bufs, 30, relative_trust=True)
        p = bufs["precision"]
        # Tiny errors -> huge 1/err^2 targets; the freed ledger must
        # record them (legacy would pin 100% at exactly 10.0).
        assert p.max() > 100.0
        assert not torch.any((p > 9.99) & (p < 10.01)), "values parked at legacy cap"
        # Real differentiation: the spread survives (this is the whole point).
        spread = torch.quantile(p, 0.95) / torch.quantile(p, 0.05)
        assert spread > 1.5

    def test_legacy_mode_still_flattens_and_pins(self):
        bufs = _bufs()
        _steps(bufs, 30, relative_trust=False)
        p = bufs["precision"]
        # The measured pathology reproduces under the flag-off path:
        # everything saturates at the absolute cap.
        assert torch.all(p >= 9.99), "legacy path should pin at cap in tiny-error regime"

    def test_ratio_semantics_cap_and_median(self):
        # Hand-built precision ledger; one step; the applied weight delta
        # must scale per-input by clamp(prec/median, 0.1, 10).
        bufs = _bufs()
        n_in = 16
        prec = torch.ones(n_in)
        prec[0] = 1e6   # far above median -> capped at 10x
        prec[1] = 1e-9  # far below median -> floored at 0.1x
        prec[2] = prec.median() * 3.0  # inside the band -> ~3x
        bufs["precision"] = prec.clone()
        # Neutralize the metaplasticity dampener: with a tiny update_ema,
        # adaptive_factor = 2/(1+|delta|/ema) compresses large columns
        # nonlinearly and destroys exactly the ratios under test (found
        # the hard way -- first version of this test failed on it).
        bufs["update_ema"] = torch.ones_like(bufs["update_ema"]) * 1e6
        w_before = bufs["weight"].clone()
        # Freeze the confounds: no homeostasis drift for this check.
        sc = dict(_SC, homeostatic_decay=0.0, set_point_adapt_rate=0.0)
        _, pred_error = m._pc_self_modify_python(
            bufs["weight"], bufs["prediction"], bufs["set_point"],
            bufs["momentum"], bufs["update_ema"], bufs["precision"],
            bufs["error_acc"], bufs["plasticity"],
            bufs["x_flat"], bufs["output"], **sc, relative_trust=True,
        )
        dw = (bufs["weight"] - w_before)  # [out, in]
        # Per-input applied magnitude carries trust_i * |pred_error_i|;
        # divide out each column's own error so only trust remains.
        col = dw.abs().sum(dim=0) / pred_error.abs().clamp(min=1e-12)
        ref = col[3]  # mid-band input, trust ~1x
        if ref > 0:
            assert col[0] / ref == pytest.approx(10.0, rel=0.05)
            assert col[1] / ref == pytest.approx(0.1, rel=0.05)

    def test_modes_differ(self):
        a, b = _bufs(), _bufs()
        _steps(a, 5, relative_trust=False)
        _steps(b, 5, relative_trust=True)
        assert not torch.allclose(a["precision"], b["precision"])


@pytest.mark.skipif(not m._use_cpp, reason="C++ pc_ops extension not loaded")
class TestCppParity:
    def test_relative_mode_buffers_match(self):
        keys = ("weight", "prediction", "set_point", "momentum",
                "update_ema", "precision", "error_acc")
        py, cp = _bufs(3), _bufs(3)
        for _ in range(25):
            m._pc_self_modify_python(
                py["weight"], py["prediction"], py["set_point"],
                py["momentum"], py["update_ema"], py["precision"],
                py["error_acc"], py["plasticity"],
                py["x_flat"], py["output"], **_SC, relative_trust=True,
            )
            m._cpp_ops.pc_self_modify(
                cp["weight"], cp["prediction"], cp["set_point"],
                cp["momentum"], cp["update_ema"], cp["precision"],
                cp["error_acc"], cp["plasticity"],
                cp["x_flat"], cp["output"],
                _SC["pc_rate"], _SC["pred_learning_rate"],
                _SC["homeostatic_decay"], _SC["set_point_adapt_rate"],
                _SC["momentum_decay"], _SC["update_ema_decay"],
                _SC["precision_ema_decay"], _SC["precision_min"],
                _SC["precision_max"], _SC["prediction_clamp"],
                relative_trust=True,
            )
        for k in keys:
            assert torch.allclose(py[k], cp[k], rtol=1e-5, atol=1e-7), k


class TestPlumbing:
    def test_model_kwarg_reaches_layers(self):
        from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM
        torch.manual_seed(0)
        model = MultimodalPredictiveCodingLM(
            vocab_size=64, d_model=32, n_blocks=2, n_heads=2,
            ffn_expansion=1, max_seq_len=16,
            max_audio_tokens=16, max_vision_tokens=16,
            backward_pass_enabled=False, relative_trust=True,
        )
        layers = [mm for mm in model.modules() if hasattr(mm, "relative_trust")]
        assert layers and all(mm.relative_trust for mm in layers)

    def test_driver_tables_complete(self):
        from scripts.jepa_pilot_driver import (
            ARM_CONFIGS, ARM_COSINE, ARM_FILELIST, ARM_SIGREG, ARM_TAPER,
            STAGES,
        )
        arm = "living_v5_4x_d4"
        assert STAGES[10] == [(arm, 512)]
        assert ARM_CONFIGS[arm]["relative_trust"] is True
        assert ARM_CONFIGS[arm]["n_blocks"] == 4
        assert ARM_TAPER[arm] is True
        assert arm in ARM_FILELIST
        assert ARM_SIGREG[arm] == pytest.approx(0.2)
        assert ARM_COSINE[arm] is True
