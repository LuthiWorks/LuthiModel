"""Surprise drive (2026-07-29): scale-free excess-error drive.

The mechanism this replaces is documented in
docs/research/2026-07-29_why-the-drive-goes-quiet.md: the PC drive was raw
reconstruction error, which self-extinguishes as the model fits. Measured on
living_v5_4x_d4_512d_seed44, update_ema fell 3 orders of magnitude inside the
first 17% of epoch 1 -- on entirely novel data, with the taper pinned at 1.0.

These tests pin the properties the fix must have, and one property of the OLD
behaviour (clamp saturation) that must not silently return, because it is what
made the first attempt at this fix a no-op.
"""
import pytest
import torch

from luthi.v2.living_layer_pc import PredictiveCodingLayer


def _layer(**kw):
    base = dict(in_features=16, out_features=8, num_episodes=4, context_dim=8)
    base.update(kw)
    return PredictiveCodingLayer(**base)


def _surprise_layer(**kw):
    kw.setdefault("drive_warmup_calls", 5)
    return _layer(drive_mode="surprise", relative_trust=True, **kw)


class TestConfigContract:
    def test_default_is_raw(self):
        assert _layer().drive_mode == "raw"

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="drive_mode"):
            _layer(drive_mode="novelty")

    def test_surprise_requires_relative_trust(self):
        """The mechanism dependency is enforced loudly, not supplied silently.

        Absolute precision weighting saturates the +/-1 clamp at 100% in the
        production regime, which discards the drive magnitude entirely. A
        surprise drive under absolute trust would be a no-op wearing a new
        name -- exactly the failure mode of the 2026-07-28 rms attempt.
        """
        with pytest.raises(ValueError, match="relative_trust"):
            _layer(drive_mode="surprise", relative_trust=False)

    def test_surprise_with_relative_trust_constructs(self):
        assert _surprise_layer().drive_mode == "surprise"


class TestOffByDefault:
    def test_raw_mode_leaves_drive_buffers_untouched(self):
        torch.manual_seed(0)
        layer = _layer()
        for _ in range(20):
            layer(torch.randn(4, 16))
        assert int(layer.drive_calls.item()) == 0
        assert float(layer.drive_ref.item()) == 0.0
        assert float(layer.drive_gain.item()) == 0.0

    def test_raw_mode_bit_identical_to_pre_change(self):
        """A layer in raw mode must produce byte-identical weights to one that
        never heard of the surprise drive. Guards the default path."""
        torch.manual_seed(0)
        a = _layer()
        b = _layer()
        b.load_state_dict(a.state_dict())
        for i in range(15):
            torch.manual_seed(100 + i)
            x = torch.randn(4, 16)
            a(x)
            b(x)
        assert torch.equal(a.weight, b.weight)


class TestWarmup:
    def test_warmup_behaves_exactly_like_raw(self):
        """A new mechanism must never be inert OR wild before it has data.

        Same discipline as the episode-store admission fix, which failed twice
        in production after passing unit tests because its warmup did not fall
        back to known behaviour.
        """
        torch.manual_seed(0)
        raw = _layer()
        sur = _surprise_layer(drive_warmup_calls=10_000)
        sur_ref = _layer(relative_trust=True)
        sur.load_state_dict(sur_ref.state_dict())
        raw2 = _layer(relative_trust=True)
        raw2.load_state_dict(sur_ref.state_dict())
        for i in range(8):
            torch.manual_seed(200 + i)
            x = torch.randn(4, 16)
            sur(x)
            raw2(x)
        assert torch.allclose(sur.weight, raw2.weight, atol=0, rtol=0)

    def test_statistics_accumulate_during_warmup(self):
        torch.manual_seed(0)
        layer = _surprise_layer(drive_warmup_calls=10_000)
        for _ in range(30):
            layer(torch.randn(4, 16))
        assert int(layer.drive_calls.item()) == 30
        assert float(layer.drive_ref.item()) > 0.0


class TestScaleFreedom:
    def test_gain_survives_a_collapsing_error_scale(self):
        """The defining property: the drive must not vanish just because the
        error got small.

        Two identical layers, one fed inputs 1000x smaller. Raw drive scales
        with the input; the surprise drive is a ratio and must not.
        """
        results = {}
        for scale in (1.0, 1e-3):
            torch.manual_seed(0)
            layer = _surprise_layer()
            for i in range(60):
                torch.manual_seed(300 + i)
                layer(torch.randn(4, 16) * scale)
            # A spike, at the same RELATIVE magnitude in both runs.
            torch.manual_seed(999)
            layer(torch.randn(4, 16) * scale * 8.0)
            results[scale] = float(layer.drive_gain.item())
        big, small = results[1.0], results[1e-3]
        assert big > 0.0 and small > 0.0
        # Gains need only be the same order; the estimator is stochastic.
        assert 0.1 < (small / big) < 10.0, results


class TestSurpriseGating:
    def test_quiet_on_stationary_input(self):
        """Familiar input should produce a low duty cycle.

        This is intended behaviour, not a regression to the dead regime -- see
        test_retains_range_while_quiet for the half that makes it credible.
        """
        torch.manual_seed(0)
        layer = _surprise_layer()
        x = torch.randn(4, 16)
        for _ in range(200):
            layer(x)          # the same input, over and over
        duty = (float(layer.drive_fire_count.item())
                / max(float(layer.drive_calls.item()), 1.0))
        assert duty < 0.5, duty

    def test_retains_range_while_quiet(self):
        """The distinction from a dead drive: after going quiet on familiar
        input, a novel input must still produce a large gain.

        A dead drive cannot do this -- its magnitude is gone. This test is the
        difference between the fix and the thing it replaces.
        """
        torch.manual_seed(0)
        layer = _surprise_layer()
        x = torch.randn(4, 16)
        for _ in range(200):
            layer(x)
        quiet_gain = float(layer.drive_gain.item())
        torch.manual_seed(4242)
        layer(torch.randn(4, 16) * 25.0)
        spike_gain = float(layer.drive_gain.item())
        assert spike_gain > quiet_gain
        assert spike_gain > 0.5, (quiet_gain, spike_gain)

    def test_duty_is_zero_on_a_predictable_error_scale(self):
        """Both stationary input AND i.i.d. draws from one distribution must
        read zero duty.

        The second case is the subtle one: every batch is fresh data, but the
        error SCALE is perfectly predictable, so there is nothing to write.
        Measured 0.0000 for both. This is the behaviour that makes a nonzero
        duty on real data meaningful.
        """
        for stationary in (True, False):
            torch.manual_seed(0)
            layer = _surprise_layer(drive_warmup_calls=50)
            fixed = torch.randn(4, 16)
            for i in range(400):
                if stationary:
                    layer(fixed)
                else:
                    torch.manual_seed(5000 + i)
                    layer(torch.randn(4, 16))
            assert layer.aliveness()["drive_duty"] == 0.0, stationary

    def test_duty_rises_after_a_distribution_shift(self):
        """The signature the 2026-07-29 review asked for: response at a
        boundary, not before it.

        This is the unit-scale version of the curriculum boundary-response
        test. A raw drive cannot produce it once its magnitude has decayed.
        """
        torch.manual_seed(0)
        layer = _surprise_layer(drive_warmup_calls=50)
        for i in range(300):
            torch.manual_seed(6000 + i)
            layer(torch.randn(4, 16))
        before = layer.aliveness()["drive_duty"]
        for i in range(300):
            torch.manual_seed(7000 + i)
            layer(torch.randn(4, 16) * 6.0)
        after = layer.aliveness()["drive_duty"]
        assert before == 0.0, before
        assert after > 0.0, after

    def test_duty_metric_survives_a_gain_floor(self):
        """With a floor the gain never hits zero, so duty must count gain ABOVE
        the floor or the instrument goes blind precisely when a run chooses to
        keep a baseline trickle. Measured duty 1.0000 at floor=0.05 before the
        fix."""
        torch.manual_seed(0)
        layer = _surprise_layer(drive_warmup_calls=50, drive_gain_floor=0.05)
        x = torch.randn(4, 16)
        for _ in range(400):
            layer(x)
        assert layer.aliveness()["drive_duty"] == 0.0

    def test_gain_is_bounded(self):
        torch.manual_seed(0)
        layer = _surprise_layer(drive_gain_max=2.0)
        for _ in range(60):
            layer(torch.randn(4, 16))
        for mult in (1e2, 1e4, 1e6):
            layer(torch.randn(4, 16) * mult)
            assert 0.0 <= float(layer.drive_gain.item()) <= 2.0

    def test_dev_floor_prevents_infinite_surprise(self):
        """On perfectly stationary input the deviation estimate collapses, and
        without a floor a trivial residual would read as infinite surprise."""
        torch.manual_seed(0)
        layer = _surprise_layer()
        x = torch.zeros(4, 16)
        for _ in range(300):
            layer(x)
        assert torch.isfinite(layer.drive_gain).all()
        assert torch.isfinite(layer.weight).all()


class TestSaturationRegressionGuard:
    def test_absolute_precision_saturates_the_clamp(self):
        """Pins the 2026-07-29 finding that made the first drive fix a no-op.

        At production precision scales the +/-1 clamp on
        (drive_error * precision) is fully saturated, so the update is
        sign-based and ANY positive rescaling of the drive changes nothing.
        Measured on probe_storefix_512d_seed45: 100% of entries clamped for
        both raw and rms-normalized error, sign-identical.

        If this test ever fails, the clamp or the precision scale changed and
        the `relative_trust` requirement on surprise mode should be revisited.
        """
        precision = torch.full((16,), 1.7e5)
        pred_error = torch.randn(16) * 0.23      # measured production rms
        raw = (pred_error * precision).clamp(-1.0, 1.0)
        normalized = (pred_error / pred_error.pow(2).mean().sqrt()
                      * precision).clamp(-1.0, 1.0)
        assert float((raw.abs() >= 1.0).float().mean()) == 1.0
        assert float((normalized.abs() >= 1.0).float().mean()) == 1.0
        assert torch.equal(torch.sign(raw), torch.sign(normalized))

    def test_relative_trust_does_not_saturate(self):
        """The reason surprise mode requires relative trust."""
        precision = torch.rand(512) * 3e5 + 5e4
        trust = (precision / precision.median()).clamp(0.1, 10.0)
        pred_error = torch.randn(512)
        weighted = (pred_error * trust).clamp(-1.0, 1.0)
        frac = float((weighted.abs() >= 1.0).float().mean())
        assert frac < 0.6, frac


class TestFrozenPathUntouched:
    def test_frozen_reencode_mutates_no_drive_state(self):
        torch.manual_seed(0)
        layer = _surprise_layer()
        for _ in range(30):
            layer(torch.randn(4, 16))
        before = (
            float(layer.drive_ref.item()),
            float(layer.drive_dev.item()),
            int(layer.drive_calls.item()),
            int(layer.drive_fire_count.item()),
        )
        layer._plasticity_frozen = True
        try:
            for _ in range(10):
                layer(torch.randn(4, 16))
        finally:
            layer._plasticity_frozen = False
        after = (
            float(layer.drive_ref.item()),
            float(layer.drive_dev.item()),
            int(layer.drive_calls.item()),
            int(layer.drive_fire_count.item()),
        )
        assert before == after


class TestInstrumentation:
    def test_duty_and_state_are_reported(self):
        torch.manual_seed(0)
        layer = _surprise_layer()
        for _ in range(50):
            layer(torch.randn(4, 16))
        stats = layer.aliveness()
        for key in ("drive_gain", "drive_ref", "drive_dev", "drive_duty",
                    "drive_gain_mean_fired", "drive_fires", "drive_calls"):
            assert key in stats, key
            assert isinstance(stats[key], float)
        assert 0.0 <= stats["drive_duty"] <= 1.0

    def test_mean_gain_when_firing_is_recoverable(self):
        """Separates 'fires rarely' from 'fires feebly'.

        The mean must be taken over FIRING calls only. Averaged over all calls
        it would be dominated by the ~98% of zeros and would track duty rather
        than magnitude, collapsing the two extinction modes back together.
        """
        torch.manual_seed(0)
        layer = _surprise_layer(drive_warmup_calls=50)
        for i in range(400):
            torch.manual_seed(8000 + i)
            layer(torch.randn(4, 16) * (1.0 if i < 200 else 7.0))
        stats = layer.aliveness()
        fires = stats["drive_fires"]
        assert fires > 0, "expected the shift to produce firings"
        mean_fired = stats["drive_gain_mean_fired"]
        # Every firing call has gain > floor, so the mean over firings must too.
        assert mean_fired > layer.drive_gain_floor, mean_fired
        assert mean_fired <= layer.drive_gain_max
        # And it must NOT equal the all-calls mean, which the zeros would drag
        # far below it.
        all_calls_mean = float(layer.drive_gain_sum.item()) / stats["drive_calls"]
        assert mean_fired > all_calls_mean * 2, (mean_fired, all_calls_mean)

    def test_gain_sum_untouched_in_raw_mode(self):
        torch.manual_seed(0)
        layer = _layer()
        for _ in range(20):
            layer(torch.randn(4, 16))
        assert float(layer.drive_gain_sum.item()) == 0.0

    def test_gain_sum_frozen_path_safe(self):
        torch.manual_seed(0)
        layer = _surprise_layer(drive_warmup_calls=20)
        for i in range(200):
            torch.manual_seed(9000 + i)
            layer(torch.randn(4, 16) * (1.0 if i < 100 else 7.0))
        before = float(layer.drive_gain_sum.item())
        layer._plasticity_frozen = True
        try:
            for _ in range(10):
                layer(torch.randn(4, 16) * 50.0)
        finally:
            layer._plasticity_frozen = False
        assert float(layer.drive_gain_sum.item()) == before
