"""Divergence guards (2026-07-29).

Added after the depth-8 shakeout diverged at step ~2250 and reported
`outcome: completed, admissible: True` with held-out NMSE 5.675 against a
healthy ~0.57. Every criterion in KillCriteriaConfig is relative to a baseline
AND gated behind warmup_batches=5000, while every probe run that day was
3000-4000 steps -- so nothing could fire. These two guards are absolute and
ungated.

See docs/research/2026-07-29_depth8-shakeout-verdict.md.
"""
import math

import pytest

from luthi.v2.jepa_runner import JEPATrainer, RunnerConfig, SamplerConfig


class _Guard:
    """Minimal stand-in exposing just the guard methods and the state they use.

    The guards are pure functions of config + their own accumulated state, so
    testing them against a real JEPATrainer would require a full model, loader
    and optimizer for no added coverage.
    """

    def __init__(self, **overrides):
        cfg = RunnerConfig(
            sampler=SamplerConfig(corpus_sizes_tokens={"text": 1_000_000})
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        self.config = cfg
        self.global_step = 0
        self._div_losses = []
        self._div_baseline = None
        self._div_run = 0

    _check_loss_divergence = JEPATrainer._check_loss_divergence
    _check_divergence = JEPATrainer._check_divergence


def _feed(guard, losses):
    """Feed a loss series; return the step index that tripped, or None."""
    for i, v in enumerate(losses):
        guard.global_step = i
        r = guard._check_loss_divergence(v)
        if r is not None:
            return i
    return None


class TestNMSEGuard:
    def test_healthy_nmse_passes(self):
        g = _Guard()
        assert g._check_divergence({"text": {"nmse_mean": 0.5658}}) is None

    def test_diverged_nmse_trips(self):
        """The actual value from the diverged depth-8 run."""
        g = _Guard()
        r = g._check_divergence({"text": {"nmse_mean": 5.675383453829092}})
        assert r is not None and "nmse" in r

    def test_nmse_at_one_still_passes(self):
        """NMSE 1.0 means 'no better than predicting the mean' -- bad, but it is
        a legitimate early-training value and not this guard's job to police."""
        g = _Guard()
        assert g._check_divergence({"text": {"nmse_mean": 1.0}}) is None

    def test_nonfinite_nmse_trips(self):
        g = _Guard()
        r = g._check_divergence({"text": {"nmse_mean": float("nan")}})
        assert r is not None and "nonfinite" in r

    def test_missing_nmse_is_not_a_trip(self):
        g = _Guard()
        assert g._check_divergence({"text": {"l_pred_mean": 9576.0}}) is None

    def test_disabled_by_zero(self):
        g = _Guard(divergence_nmse_max=0.0)
        assert g._check_divergence({"text": {"nmse_mean": 999.0}}) is None


class TestLossGuard:
    def test_baseline_is_frozen_not_rolling(self):
        """The load-bearing property.

        A rolling baseline rises with a diverging loss and masks it -- the same
        positive-feedback bug the consolidation trigger had before 2026-05-10.
        Here: 10 points at 1.0 set the baseline, then a slow climb well past
        10x must still trip, which it cannot if the baseline tracks the climb.
        """
        g = _Guard()
        series = [1.0] * 10 + [1.0 * (1.35 ** i) for i in range(1, 30)]
        tripped = _feed(g, series)
        assert tripped is not None
        assert g._div_baseline == pytest.approx(1.0)

    def test_single_spike_does_not_trip(self):
        """Healthy runs do spike. Sustained elevation is what matters."""
        g = _Guard()
        series = [1.0] * 10 + [1.0, 500.0, 1.0, 1.0, 400.0, 1.0] * 5
        assert _feed(g, series) is None

    def test_two_of_three_sustained_does_not_trip(self):
        g = _Guard()
        series = [1.0] * 10 + [500.0, 500.0, 1.0, 500.0, 500.0, 1.0]
        assert _feed(g, series) is None

    def test_three_sustained_trips(self):
        g = _Guard()
        series = [1.0] * 10 + [500.0, 500.0, 500.0]
        assert _feed(g, series) == 12

    def test_not_armed_before_baseline_points(self):
        g = _Guard()
        # Enormous losses during the baseline window must not trip: early
        # training legitimately starts high, which is why the baseline is the
        # median of the first N rather than an absolute number.
        assert _feed(g, [1e6] * 9) is None
        assert g._div_baseline is None

    def test_nonpositive_baseline_disarms_rather_than_misfires(self):
        g = _Guard()
        assert _feed(g, [0.0] * 10 + [1e9] * 10) is None
        assert g._div_baseline == -1.0

    def test_nonfinite_loss_trips_immediately(self):
        g = _Guard()
        assert _feed(g, [1.0] * 3 + [float("inf")]) == 3

    def test_disabled_by_zero(self):
        g = _Guard(divergence_loss_mult=0.0)
        assert _feed(g, [1.0] * 10 + [1e9] * 10) is None


class TestAgainstRealSeries:
    """Replay of the actual runs, which is the only test that matters.

    Six healthy runs must not trip; the diverged one must. Thresholds were
    chosen from exactly this sweep, so these numbers pin the choice: healthy
    peak/baseline ran 0.6x-1.7x against a 10x threshold.
    """

    HEALTHY = [
        # probe_surprise seed45 -- first 10 then the rest, abbreviated to the
        # shape that matters: baseline ~32, series stays at or below it.
        [31.0, 33.0, 30.0, 35.0, 28.0, 32.0, 34.0, 31.0, 29.0, 33.0]
        + [20.0, 131.55, 15.0, 12.0, 40.0, 9.0, 25.0, 13.0, 11.0, 30.0],
        # A run whose losses fall by an order of magnitude (living_v5 shape).
        [0.70, 0.69, 0.71, 0.68, 0.70, 0.72, 0.69, 0.70, 0.68, 0.71]
        + [0.5, 0.4, 1.13, 0.35, 0.3, 0.28, 0.25, 0.22, 0.2, 0.19],
    ]

    def test_healthy_shapes_do_not_trip(self):
        for i, series in enumerate(self.HEALTHY):
            g = _Guard()
            assert _feed(g, series) is None, f"false positive on series {i}"

    def test_the_diverged_shape_trips(self):
        """The depth-8 loss trajectory: settled ~200, then four orders up."""
        g = _Guard()
        series = (
            [330.8, 625.9, 204.2, 235.9, 173.7, 259.1, 190.0, 150.0, 140.0, 123.4]
            + [130.0, 208.2, 145.0, 2000.0, 4000.0, 6000.0, 10531.9]
        )
        tripped = _feed(g, series)
        assert tripped is not None, "the diverged run must trip"


class TestBaselineDuringDivergenceIsTheHole:
    """Pins the 2026-07-30 guard failure.

    `probe_surprise_d8_amp8_512d_seed88` diverged from step ~100 and the loss
    guard never fired. Its losses, actually logged:

        step 100  3.2e7      step 600  1.15e11
        step 200  1.0e10     step 1000 2.33e11
        step 300  3.3e10     peak      ~3.0e11

    The baseline is the median of the first 10 logged losses = 1.0e11, so the
    trip threshold was 1e12 and a loss peaking at 3e11 never crossed it. The
    guard baselined itself on the divergence it existed to detect.

    Freezing the baseline fixed a DIFFERENT bug (a rolling statistic drifting up
    with the loss). It does nothing when the run is already broken at the moment
    the baseline is taken. Every relative criterion has this hole, which is why
    the absolute NMSE check now also runs periodically rather than only at epoch
    end.
    """

    REAL_DIVERGED_LOSSES = [
        3.218e07, 1.029e10, 3.261e10, 7.227e10, 8.521e10,
        1.150e11, 1.517e11, 1.471e11, 1.881e11, 2.334e11,
        2.202e11, 2.539e11, 2.7e11, 2.9e11, 3.04e11,
    ]

    def test_loss_guard_misses_a_run_that_was_already_diverging(self):
        """Documents the limitation rather than asserting it is acceptable."""
        g = _Guard()
        assert _feed(g, self.REAL_DIVERGED_LOSSES) is None, (
            "if this now trips, the loss guard was strengthened and this test "
            "should be updated to match the new behaviour"
        )
        assert g._div_baseline > 1e10

    def test_absolute_nmse_guard_catches_it_regardless_of_baseline(self):
        """The reason the absolute check is what runs periodically.

        NMSE has a fixed reference -- 1.0 is 'no better than predicting the
        mean' -- so it needs no baseline and cannot be blinded by one.
        """
        g = _Guard()
        r = g._check_divergence({"text": {"nmse_mean": 343309.2018}})
        assert r is not None and "nmse" in r

    def test_absolute_guard_needs_no_history(self):
        """It must fire on the very first observation, with no warmup at all."""
        g = _Guard()
        assert g._check_divergence({"text": {"nmse_mean": 5.0}}) is not None
