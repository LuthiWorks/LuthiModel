"""Tests for the slow-trace primitives (momentum-functions foundations).

Covers the SlowEMA / ReadResetAccumulator behavior, the resolution-progress
ratio the inverted-U gain's explicit fall reads, and the checkpoint round-trip
that the gain's regime (i) persistence requirement stands on
(2026-07-05_inverted-u-gain-spec.md §5/§6).
"""

from __future__ import annotations

import pytest

from luthi.v2.slow_trace import (
    ReadResetAccumulator,
    SlowEMA,
    resolution_progress,
)


# ---------------- SlowEMA ----------------

def test_slow_ema_seeds_exactly_then_converges():
    ema = SlowEMA(decay=0.9, warmup=1)
    # First sample seeds exactly -- no climb out of a spurious zero.
    assert ema.update(5.0) == 5.0
    # Constant input stays put.
    for _ in range(50):
        ema.update(5.0)
    assert ema.value == pytest.approx(5.0)
    # A step change is tracked, slowly, toward the new level.
    for _ in range(200):
        ema.update(10.0)
    assert ema.value == pytest.approx(10.0, abs=1e-3)


def test_slow_ema_decay_sets_timescale():
    fast = SlowEMA(decay=0.5, warmup=1)
    slow = SlowEMA(decay=0.99, warmup=1)
    fast.update(0.0)
    slow.update(0.0)
    for _ in range(20):
        fast.update(1.0)
        slow.update(1.0)
    # After 20 steps of a step change, the fast trace is much closer to 1.
    assert fast.value > 0.99
    assert slow.value < 0.5


def test_slow_ema_warmup_gate():
    ema = SlowEMA(decay=0.9, warmup=8)
    for i in range(7):
        ema.update(1.0)
        assert not ema.is_warm()
    ema.update(1.0)
    assert ema.is_warm()


def test_slow_ema_roundtrips_lived_state():
    ema = SlowEMA(decay=0.9, warmup=4)
    for x in (3.0, 1.0, 4.0, 1.0, 5.0):
        ema.update(x)
    restored = SlowEMA(decay=0.9, warmup=4)
    restored.load_state_dict(ema.state_dict())
    assert restored.value == ema.value
    assert restored.is_warm() == ema.is_warm()
    # And it resumes identically: the next sample lands on the same value.
    assert restored.update(2.0) == ema.update(2.0)


def test_slow_ema_rejects_bad_decay():
    with pytest.raises(ValueError):
        SlowEMA(decay=1.0)
    with pytest.raises(ValueError):
        SlowEMA(decay=-0.1)


# ---------------- ReadResetAccumulator ----------------

def test_accumulator_sums_and_read_reset_clears():
    acc = ReadResetAccumulator()
    for x in (1.0, 2.0, 3.0):
        acc.add(x)
    assert acc.total == pytest.approx(6.0)
    assert acc.count == 3
    assert acc.read_and_reset() == pytest.approx(6.0)
    # Cleared: a new day starts at zero (sleep clears the day's motion).
    assert acc.total == 0.0
    assert acc.count == 0
    assert acc.read_and_reset() == 0.0


def test_accumulator_roundtrips_lived_state():
    acc = ReadResetAccumulator()
    acc.add(2.0)
    acc.add(5.0)
    restored = ReadResetAccumulator()
    restored.load_state_dict(acc.state_dict())
    assert restored.total == acc.total
    assert restored.count == acc.count
    # A restore mid-day must not lose the day's accumulation (silent-amnesia
    # class -- the reason this state persists at all).
    assert restored.read_and_reset() == pytest.approx(7.0)


# ---------------- resolution_progress ----------------

def _warm(short, long, seq):
    for x in seq:
        short.update(x)
        long.update(x)


def test_resolution_progress_resolving_is_below_one():
    short, long = SlowEMA(0.3, warmup=2), SlowEMA(0.9, warmup=2)
    # Error steadily falling: recent (fast) below its slower baseline.
    _warm(short, long, [10.0, 8.0, 6.0, 4.0, 2.0, 1.0])
    assert resolution_progress(short, long) < 0.9


def test_resolution_progress_nonresolving_is_near_one():
    short, long = SlowEMA(0.3, warmup=2), SlowEMA(0.9, warmup=2)
    # Error flat: repetition that isn't reducing error -- the explicit-fall
    # regime. short ~= long -> ratio ~= 1.
    _warm(short, long, [5.0] * 30)
    assert resolution_progress(short, long) == pytest.approx(1.0, abs=0.05)


def test_resolution_progress_worsening_is_above_one():
    short, long = SlowEMA(0.3, warmup=2), SlowEMA(0.9, warmup=2)
    _warm(short, long, [1.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    assert resolution_progress(short, long) > 1.0


def test_resolution_progress_cold_start_returns_zero():
    # Not warm yet -> no evidence of non-resolution -> 0.0 (fall stays OFF).
    short, long = SlowEMA(0.3, warmup=8), SlowEMA(0.9, warmup=8)
    short.update(5.0)
    long.update(5.0)
    assert resolution_progress(short, long) == 0.0
    # Warm but zero baseline (dead layer) -> 0.0, no div-by-zero (regime e).
    short2, long2 = SlowEMA(0.3, warmup=1), SlowEMA(0.9, warmup=1)
    _warm(short2, long2, [0.0] * 5)
    assert resolution_progress(short2, long2) == 0.0
