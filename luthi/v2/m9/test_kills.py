"""Unit tests for M9 step-1 kill criteria.

Run from project root:
    python -m luthi.v2.m9.test_kills

Tests cover each kill's flag-then-fire state machine and the spec
properties:
- K-M9-2 entropy: fires when entropy below floor for sustained cycles
- K-M9-2 consistency: fires when median deviation above threshold sustained
- K-M9-3 value divergence: trending band fires on sustained breach
- K-M9-4 gamma divergence: trending band fires on sustained breach
- K-M9-5 dark-room: fires after sustained internal-AND-external stasis;
  internal-only or external-only stasis does NOT fire (the contemplation-
  is-safe property)
- K-M9-7 staleness runaway: fires after sustained failover
- K-M9-8 mask stability: trending band fires on sustained drift
- reset() returns the registry to HEALTHY
"""

from __future__ import annotations

import torch

from luthi.v2.m9.kills import KillRegistry, KillState, TrendingBand


# ----------------- TrendingBand -----------------

def test_trending_band_flags_then_fires_max():
    band = TrendingBand(
        window=16, direction="max", k=3.0,
        sustained_cycles=3, min_warmup=4,
    )
    # Warm with stable baseline.
    for _ in range(10):
        s = band.observe(0.0)
    assert s == KillState.HEALTHY
    # Breach: 3 in a row -> fired.
    s = band.observe(10.0)
    assert s == KillState.FLAGGED
    s = band.observe(10.0)
    assert s == KillState.FLAGGED
    s = band.observe(10.0)
    assert s == KillState.FIRED


def test_trending_band_recovers_when_signal_returns():
    band = TrendingBand(
        window=16, direction="max", k=3.0,
        sustained_cycles=4, min_warmup=4,
    )
    for _ in range(10):
        band.observe(0.0)
    band.observe(10.0)  # flagged
    band.observe(10.0)  # flagged
    s = band.observe(0.0)  # back to baseline -> count reset
    assert s == KillState.HEALTHY


def test_trending_band_warmup_gates_breaches():
    band = TrendingBand(
        window=16, direction="max", k=3.0,
        sustained_cycles=2, min_warmup=8,
    )
    # All values during warmup return HEALTHY even on huge spikes.
    for _ in range(7):
        s = band.observe(100.0)
    assert s == KillState.HEALTHY


# ----------------- K-M9-2 entropy -----------------

def test_entropy_fires_on_dominant_branch():
    reg = KillRegistry(
        entropy_min_warmup=2,
        entropy_low_floor=0.5,
        entropy_sustained=2,
    )
    # Warmup with healthy uniform distribution.
    uniform = torch.tensor([0.25, 0.25, 0.25, 0.25])
    for _ in range(3):
        reg.observe_mcts_entropy(uniform)
    assert reg.states()["K-M9-2-entropy"] == KillState.HEALTHY
    # Now collapse to a single dominant branch.
    dominant = torch.tensor([0.99, 0.005, 0.003, 0.002])
    reg.observe_mcts_entropy(dominant)
    assert reg.states()["K-M9-2-entropy"] == KillState.FLAGGED
    reg.observe_mcts_entropy(dominant)
    assert reg.states()["K-M9-2-entropy"] == KillState.FIRED


# ----------------- K-M9-2 consistency -----------------

def test_consistency_fires_on_sustained_high_deviation():
    reg = KillRegistry(consistency_max=1.0, consistency_sustained=3)
    reg.observe_mcts_consistency(0.1)
    reg.observe_mcts_consistency(0.1)
    assert reg.states()["K-M9-2-consistency"] == KillState.HEALTHY
    reg.observe_mcts_consistency(5.0)
    assert reg.states()["K-M9-2-consistency"] == KillState.FLAGGED
    reg.observe_mcts_consistency(5.0)
    reg.observe_mcts_consistency(5.0)
    assert reg.states()["K-M9-2-consistency"] == KillState.FIRED


# ----------------- K-M9-3 value divergence -----------------

def test_value_divergence_fires_on_runaway():
    reg = KillRegistry(value_band_k=3.0, value_sustained=3)
    for _ in range(12):
        reg.observe_value(1.0)
    for _ in range(3):
        reg.observe_value(1000.0)
    assert reg.states()["K-M9-3-value"] == KillState.FIRED


# ----------------- K-M9-4 gamma divergence -----------------

def test_gamma_divergence_fires_on_runaway():
    """F2 K-M9-4: sustained clamp-proximity fires the kill (the F2
    primary signal). Old test used 3 cycles past the trending band;
    the F2 path uses gamma_clamp_sustained (default 8) on
    clamp-proximity, so feed enough cycles to cross that.
    """
    reg = KillRegistry(gamma_clamp_sustained=3)
    for _ in range(12):
        reg.observe_gamma(1.0)
    for _ in range(3):
        reg.observe_gamma(99.0)  # within 5% of gamma_max=100
    assert reg.states()["K-M9-4-gamma"] == KillState.FIRED


# ----------------- K-M9-5 dark room -----------------

def test_darkroom_fires_on_sustained_stasis():
    reg = KillRegistry(
        darkroom_internal_threshold=0.01,
        darkroom_sustained_cycles=5,
    )
    for _ in range(5):
        reg.observe_darkroom(
            internal_change_magnitude=0.001,  # below threshold
            external_stasis=True,
        )
    assert reg.states()["K-M9-5-darkroom"] == KillState.FIRED


def test_darkroom_internal_change_keeps_healthy_contemplation():
    """The contemplation-is-safe property: internal change without
    external output should NOT trip the dark-room kill.
    """
    reg = KillRegistry(
        darkroom_internal_threshold=0.01,
        darkroom_sustained_cycles=5,
    )
    for _ in range(20):
        reg.observe_darkroom(
            internal_change_magnitude=0.5,  # above threshold -- thinking
            external_stasis=True,           # but quiet
        )
    assert reg.states()["K-M9-5-darkroom"] == KillState.HEALTHY


def test_darkroom_external_action_keeps_healthy():
    """External action with internal stasis should not fire either."""
    reg = KillRegistry(
        darkroom_internal_threshold=0.01,
        darkroom_sustained_cycles=5,
    )
    for _ in range(20):
        reg.observe_darkroom(
            internal_change_magnitude=0.001,
            external_stasis=False,  # decoders emitting
        )
    assert reg.states()["K-M9-5-darkroom"] == KillState.HEALTHY


def test_darkroom_resets_on_non_stasis():
    reg = KillRegistry(
        darkroom_internal_threshold=0.01,
        darkroom_sustained_cycles=5,
    )
    for _ in range(3):
        reg.observe_darkroom(0.001, True)
    assert reg.states()["K-M9-5-darkroom"] == KillState.FLAGGED
    reg.observe_darkroom(1.0, True)  # internal activity!
    assert reg.states()["K-M9-5-darkroom"] == KillState.HEALTHY


# ----------------- K-M9-7 staleness runaway -----------------

def test_staleness_fires_on_sustained_failover():
    reg = KillRegistry(staleness_failover_sustained=3)
    for _ in range(3):
        reg.observe_staleness(in_failover=True)
    assert reg.states()["K-M9-7-staleness"] == KillState.FIRED


def test_staleness_recovers_on_failover_clear():
    reg = KillRegistry(staleness_failover_sustained=5)
    for _ in range(2):
        reg.observe_staleness(in_failover=True)
    assert reg.states()["K-M9-7-staleness"] == KillState.FLAGGED
    reg.observe_staleness(in_failover=False)
    assert reg.states()["K-M9-7-staleness"] == KillState.HEALTHY


# ----------------- K-M9-8 mask stability -----------------

def test_mask_fires_on_sustained_drift():
    reg = KillRegistry(mask_band_k=3.0, mask_sustained=3)
    for _ in range(12):
        reg.observe_mask(0.5)
    for _ in range(3):
        reg.observe_mask(5.0)
    assert reg.states()["K-M9-8-mask"] == KillState.FIRED


# ----------------- Registry query methods -----------------

def test_fired_returns_set_of_currently_fired():
    reg = KillRegistry(
        entropy_min_warmup=2,
        entropy_low_floor=0.5,
        entropy_sustained=2,
        consistency_max=1.0,
        consistency_sustained=2,
    )
    # Fire entropy.
    uniform = torch.tensor([0.25] * 4)
    dominant = torch.tensor([0.99, 0.005, 0.003, 0.002])
    for _ in range(3):
        reg.observe_mcts_entropy(uniform)
    reg.observe_mcts_entropy(dominant)
    reg.observe_mcts_entropy(dominant)
    fired = reg.fired()
    assert "K-M9-2-entropy" in fired
    assert "K-M9-5-darkroom" not in fired


def test_reset_returns_to_healthy():
    reg = KillRegistry(staleness_failover_sustained=2)
    for _ in range(2):
        reg.observe_staleness(in_failover=True)
    assert reg.states()["K-M9-7-staleness"] == KillState.FIRED
    reg.reset()
    states = reg.states()
    for name, s in states.items():
        assert s == KillState.HEALTHY, f"{name} not reset: {s}"


def main() -> int:
    tests = [
        test_trending_band_flags_then_fires_max,
        test_trending_band_recovers_when_signal_returns,
        test_trending_band_warmup_gates_breaches,
        test_entropy_fires_on_dominant_branch,
        test_consistency_fires_on_sustained_high_deviation,
        test_value_divergence_fires_on_runaway,
        test_gamma_divergence_fires_on_runaway,
        test_darkroom_fires_on_sustained_stasis,
        test_darkroom_internal_change_keeps_healthy_contemplation,
        test_darkroom_external_action_keeps_healthy,
        test_darkroom_resets_on_non_stasis,
        test_staleness_fires_on_sustained_failover,
        test_staleness_recovers_on_failover_clear,
        test_mask_fires_on_sustained_drift,
        test_fired_returns_set_of_currently_fired,
        test_reset_returns_to_healthy,
    ]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed.append((t.__name__, str(e)))
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    if failed:
        print(f"\n{len(failed)} test(s) failed")
        return 1
    print(f"\nAll {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
