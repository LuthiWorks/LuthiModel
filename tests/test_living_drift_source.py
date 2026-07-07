"""Living-drift eye source config + band re-warm (Fable step-8 ruling,
2026-07-06).

The eye's source is an EXPLICIT M9Config knob, decoupled from the gain flag,
because "momentum" and "applied_change" are different QUANTITIES (doubly-smoothed
signed-EMA magnitude vs applied-change EMA) -- not scalings. So:
 - the source is read from config, never from attribute presence;
 - changing it re-warms the living band (fresh DriftBand), so two units never
   share one median/MAD history;
 - the applied source reads a per-layer EMA at momentum's decay (commensurate),
   not an instantaneous magnitude; None until fed (no momentum fallback).
"""

from __future__ import annotations

import torch

from luthi.v2 import PredictiveCodingLayer
from luthi.v2.m9.runner import M9Config, _living_drift_reading
from luthi.v2.m9.staleness import StalenessManager


def _layer(gain: bool, seed: int = 0) -> PredictiveCodingLayer:
    torch.manual_seed(seed)
    return PredictiveCodingLayer(
        in_features=16, out_features=8, pc_rate=0.01,
        num_episodes=8, context_dim=8, learning_gain_enabled=gain,
    )


def test_config_default_source_is_momentum():
    assert M9Config().living_drift_source == "momentum"


def test_reading_momentum_source_ignores_gain_flag():
    on = _layer(gain=True, seed=1)
    torch.manual_seed(21)
    x = torch.randn(4, 16)
    for _ in range(15):
        on(x)
    # Even with the gain on, source="momentum" reads mean |momentum|.
    assert _living_drift_reading(on, "momentum") == on.momentum.abs().mean().item()


def test_reading_applied_source_none_until_fed():
    off = _layer(gain=False, seed=2)
    on = _layer(gain=True, seed=2)
    torch.manual_seed(22)
    x = torch.randn(4, 16)
    for _ in range(15):
        off(x)
        on(x)
    # Gain off: EMA unfed -> None (never a momentum fallback -> no unit mixing).
    assert _living_drift_reading(off, "applied_change") is None
    # Gain on: reads the fair-parallel EMA.
    assert _living_drift_reading(on, "applied_change") == on._applied_ema.value


def test_applied_ema_uses_momentum_decay():
    """The fair-parallel EMA is smoothed at momentum's decay so the two eye
    sources are commensurate."""
    layer = PredictiveCodingLayer(
        16, 8, pc_rate=0.01, num_episodes=8, context_dim=8,
        learning_gain_enabled=True, momentum_decay=0.95,
    )
    assert layer._applied_ema.decay == 0.95


def test_set_source_rewarms_band_only_on_change():
    sm = StalenessManager()
    assert sm.living_drift_source == "momentum"
    # Warm the momentum-unit band.
    for v in [1.0, 2.0, 3.0]:
        sm.observe_living_drift(v)
    band_before = sm.living_band
    # Idempotent: same source -> same band object, history kept.
    sm.set_living_drift_source("momentum")
    assert sm.living_band is band_before
    assert len(sm.living_band.values) == 3
    # Change -> fresh band (re-warm), old-unit history discarded.
    sm.set_living_drift_source("applied_change")
    assert sm.living_band is not band_before
    assert len(sm.living_band.values) == 0
    assert sm.living_drift_source == "applied_change"


def test_set_source_rejects_unknown():
    sm = StalenessManager()
    try:
        sm.set_living_drift_source("bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_resolution_decay_defaults_are_generous():
    """Fable ruling: default horizons ~100/~1000 forwards, not ~10/~100."""
    layer = PredictiveCodingLayer(16, 8, num_episodes=8, context_dim=8)
    assert layer._err_short.decay == 0.99
    assert layer._err_long.decay == 0.999
