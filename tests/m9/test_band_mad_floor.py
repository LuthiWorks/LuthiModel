"""A quiet signal must not turn a spike detector into a hair trigger.

Regression pin for the 2026-08-14 audit finding. Both the staleness
spike bands and kills.TrendingBand floored MAD with a bare absolute
`max(mad, 1e-8)`. When a signal has a real scale but no spread -- 0.5
repeated, then 0.500001 -- the band collapses onto its own median and a
0.0002% move becomes a breach. The safety property was inverted: the
more stable the entity's drift, the more sensitive its detector, and a
spurious staleness spike forces failover, drops cached Q, and starts a
recovery countdown.

This is the shape the 2026-07-27 arc named -- "a dial against its stop is
a hair trigger" -- which produced the v4 trust events that turned out to
be epsilon artifacts rather than data.

The floor is RELATIVE. It deliberately does not suppress a zero-baseline
band that jumps: 0 -> 10 is a real breach and must still fire.
"""

from __future__ import annotations

import random

from luthi.v2.m9.kills import KillState, TrendingBand
from luthi.v2.m9.staleness import StalenessManager


# ----------------- staleness spike bands -----------------

def test_quiet_signal_does_not_spike_on_a_hair():
    m = StalenessManager()
    for _ in range(8):
        m.observe_drift(0.5)
    m.observe_drift(0.500001)          # +0.0002%
    assert not m.spike(), "a 0.0002% move must not force failover"


def test_real_spike_still_fires_on_a_noisy_signal():
    m = StalenessManager()
    random.seed(0)
    for _ in range(16):
        m.observe_drift(0.5 + random.uniform(-0.05, 0.05))
    m.observe_drift(0.52)
    assert not m.spike(), "in-band variation is not a spike"
    m.observe_drift(5.0)
    assert m.spike(), "a 10x jump must still spike"


def test_zero_baseline_still_spikes_and_is_counted_as_scaleless():
    """A band pinned at exactly 0 that starts moving IS a plasticity event.

    It must still fire -- but the band had no scale to judge magnitude
    with, and that blindness has to be visible rather than silent.
    """
    m = StalenessManager()
    for _ in range(8):
        m.observe_drift(0.0)
    assert not m.spike()
    assert m.snapshot()["degenerate_band_skips"].get("drift", 0) > 0
    m.observe_drift(1e-7)
    assert m.spike(), "movement off a frozen baseline is a real event"


# ----------------- kills.TrendingBand -----------------

def test_trending_band_ignores_a_hair_off_a_quiet_baseline():
    band = TrendingBand(window=16, direction="max", k=3.0,
                        sustained_cycles=3, min_warmup=4)
    for _ in range(10):
        band.observe(0.5)
    for _ in range(3):
        s = band.observe(0.500001)
    assert s == KillState.HEALTHY, "a kill must not fire on 0.0002%"


def test_trending_band_still_fires_on_a_real_breach():
    """The property tests/m9/test_kills.py already pins, restated here so
    the floor change cannot silently weaken it."""
    band = TrendingBand(window=16, direction="max", k=3.0,
                        sustained_cycles=3, min_warmup=4)
    for _ in range(10):
        band.observe(0.0)
    assert band.observe(10.0) == KillState.FLAGGED
    assert band.observe(10.0) == KillState.FLAGGED
    assert band.observe(10.0) == KillState.FIRED


def test_trending_band_fires_on_a_real_breach_off_a_nonzero_baseline():
    band = TrendingBand(window=16, direction="max", k=3.0,
                        sustained_cycles=3, min_warmup=4)
    for _ in range(10):
        band.observe(0.5)
    states = [band.observe(50.0) for _ in range(3)]
    assert states[-1] == KillState.FIRED
