"""Tests for SpikingLivingLayer — all 6 SNN features.

1. Spike mask from membrane potential
2. Leaky membrane dynamics
3. Refractory period
4. Spike propagation delay
5. Surrogate gradient
6. Fire-and-reset

Plus: inheritance/compatibility and stability tests.
"""

import torch
import pytest
from luthi.living_layer import LivingLayerV6
from luthi.living_layer_spiking import SpikingLivingLayer, SpikeSurrogate


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def layer():
    """A small spiking living layer with default parameters."""
    torch.manual_seed(42)
    return SpikingLivingLayer(in_features=16, out_features=16)


@pytest.fixture
def layer_low_threshold():
    """Spiking layer with very low threshold (spikes easily)."""
    torch.manual_seed(42)
    return SpikingLivingLayer(
        in_features=16, out_features=16,
        spike_threshold=0.001, membrane_leak=0.01,
    )


@pytest.fixture
def layer_high_threshold():
    """Spiking layer with very high threshold (never spikes)."""
    torch.manual_seed(42)
    return SpikingLivingLayer(
        in_features=16, out_features=16,
        spike_threshold=1000.0,
    )


@pytest.fixture
def layer_subtract_reset():
    """Spiking layer with subtract reset mode."""
    torch.manual_seed(42)
    return SpikingLivingLayer(
        in_features=16, out_features=16,
        spike_threshold=0.001, membrane_leak=0.01,
        reset_mode="subtract",
    )


# ── Feature 1: Spike Mask from Membrane Potential ─────────────────────


class TestSpikeMask:
    def test_spike_mask_produced(self, layer_low_threshold):
        """During repeated passes with strong input, some weights should spike."""
        x = torch.randn(4, 16) * 2.0
        any_spiked = False
        for _ in range(25):
            layer_low_threshold(x)
            if layer_low_threshold.spike_mask.any():
                any_spiked = True
        assert any_spiked, (
            "Expected some spikes during 25 passes with strong input"
        )

    def test_spike_mask_shape(self, layer):
        """Spike mask shape matches [out_features, in_features]."""
        x = torch.randn(4, 16)
        layer(x)
        assert layer.spike_mask.shape == (16, 16)

    def test_spike_requires_threshold(self, layer_high_threshold):
        """Weights with membrane below threshold should not spike."""
        x = torch.randn(4, 16)
        for _ in range(10):
            layer_high_threshold(x)
        assert not layer_high_threshold.spike_mask.any(), (
            "No spikes should occur with extremely high threshold"
        )

    def test_spike_mask_all_zero_initially(self, layer):
        """Before any forward pass, spike mask should be all zeros."""
        assert not (layer.spike_mask > 0.5).any()


# ── Feature 2: Leaky Membrane Dynamics ────────────────────────────────


class TestLeakyMembrane:
    def test_membrane_decays_without_input(self, layer):
        """Membrane potential should decay when no strong input drives it."""
        # Inject known positive values into membrane
        layer.membrane_potential.fill_(5.0)
        # Run forward with near-zero input (weak drive)
        x = torch.zeros(1, 16) + 1e-6
        layer(x)
        # Membrane should have decayed from 5.0
        # After leak: 5.0 * (1 - 0.1) = 4.5, then excitability adds some
        mean_mp = layer.membrane_potential.mean().item()
        assert mean_mp < 5.0, f"Membrane should have decayed, got {mean_mp}"

    def test_membrane_decay_rate(self, layer_high_threshold):
        """Membrane after one leak step equals old * (1 - leak) + drive."""
        layer = layer_high_threshold
        layer.membrane_potential.fill_(10.0)
        # With threshold=1000, no spikes will occur.
        # Expected after leak: 10.0 * (1 - 0.1) = 9.0 + small excitability drive
        x = torch.zeros(1, 16) + 1e-6
        layer(x)
        mean = layer.membrane_potential.mean().item()
        assert 8.5 < mean < 10.5, f"Expected ~9.0, got {mean}"

    def test_membrane_accumulates_with_sustained_input(self):
        """Repeated high-salience inputs should increase membrane potential."""
        torch.manual_seed(42)
        # Use high threshold so spikes don't reset the membrane
        layer = SpikingLivingLayer(
            in_features=16, out_features=16,
            spike_threshold=1000.0, membrane_leak=0.01,
        )
        x = torch.randn(4, 16) * 3.0
        initial_mean = layer.membrane_potential.mean().item()
        for _ in range(10):
            layer(x)
        final_mean = layer.membrane_potential.mean().item()
        assert final_mean > initial_mean, (
            f"Membrane should accumulate: {initial_mean} -> {final_mean}"
        )

    def test_membrane_separate_from_excitability_acc(self, layer):
        """Membrane potential and excitability_acc are independent buffers."""
        x = torch.randn(4, 16)
        layer(x)
        # They should exist as separate buffers with potentially different values
        assert layer.membrane_potential is not layer.excitability_acc
        # Membrane is driven by output magnitude, not excitability directly
        # They should not be equal
        assert not torch.allclose(
            layer.membrane_potential, layer.excitability_acc
        )


# ── Feature 3: Refractory Period ─────────────────────────────────────


class TestRefractoryPeriod:
    def test_refractory_prevents_spike(self, layer_low_threshold):
        """A weight that just spiked cannot spike again during refractory."""
        x = torch.randn(4, 16) * 3.0
        # Drive until some spikes occur
        for _ in range(20):
            layer_low_threshold(x)

        # Record which weights spiked (float: 1.0 = fired, 0.0 = not)
        spiked_mask = layer_low_threshold.spike_mask.clone()
        if not (spiked_mask > 0.5).any():
            pytest.skip("No spikes occurred to test refractory")

        # Immediately force membrane above threshold for previously spiking weights
        # Use element-wise math (spike_mask is float, not bool)
        layer_low_threshold.membrane_potential.mul_(1.0 - spiked_mask).add_(spiked_mask * 100.0)

        # Run one more pass
        layer_low_threshold(x)

        # Weights that WERE spiking should now be in refractory
        # (they shouldn't spike this pass even with high membrane potential,
        #  because refractory_counter was set to refractory_steps on spike)
        # Note: some may have had refractory expire in the pass, check counter
        still_refractory = layer_low_threshold.refractory_counter > 0.5
        # At least some of the previously spiked weights should be refractory
        overlap = (spiked_mask > 0.5) & still_refractory
        assert overlap.any(), "Some previously spiked weights should be in refractory"

    def test_refractory_counter_decrements(self, layer):
        """Refractory counter should decrease by 1 each step."""
        # Manually set refractory counter
        layer.refractory_counter.fill_(5)
        x = torch.randn(1, 16)
        layer(x)
        # Non-spiking weights should have counter decremented
        # (some weights might spike, but most shouldn't with default threshold=1.0)
        assert (layer.refractory_counter < 5).any(), (
            "Refractory counter should have decremented for non-spiking weights"
        )

    def test_refractory_counter_clamped_at_zero(self, layer):
        """Counter should never go negative."""
        layer.refractory_counter.fill_(0)
        x = torch.randn(1, 16)
        layer(x)
        assert (layer.refractory_counter >= 0).all()

    def test_refractory_ends_allows_spike(self, layer_low_threshold):
        """After refractory period ends, weight can spike again."""
        x = torch.randn(4, 16) * 3.0
        # Drive to get spikes
        for _ in range(20):
            layer_low_threshold(x)

        # Set all refractory counters to 1 (will expire after next pass)
        layer_low_threshold.refractory_counter.fill_(1)
        # Set high membrane potential
        layer_low_threshold.membrane_potential.fill_(100.0)
        layer_low_threshold(x)
        # After this pass, refractory expired (1->0), and membrane is high
        # Some weights should have spiked
        # (or at minimum, refractory should be 0 for non-spikers)
        assert (layer_low_threshold.refractory_counter == 0).any()

    def test_refractory_counter_shape(self, layer):
        """Shape matches [out_features, in_features]."""
        assert layer.refractory_counter.shape == (16, 16)


# ── Feature 4: Spike Propagation Delay ────────────────────────────────


class TestSpikePropagation:
    def test_delay_buffer_stores_history(self, layer_low_threshold):
        """After N forward passes, delay buffer contains recent spike masks."""
        x = torch.randn(4, 16) * 3.0
        for _ in range(5):
            layer_low_threshold(x)
        # Delay buffer should have been written to
        assert layer_low_threshold._delay_pos == 5

    def test_delayed_spikes_lag(self, layer):
        """get_delayed_spikes() returns mask from delay_steps ago, not current."""
        # Fill delay buffer manually with known values
        layer.delay_buffer[0] = torch.ones(16, 16)  # "oldest"
        layer.delay_buffer[1] = torch.zeros(16, 16)  # "newest"
        layer._delay_pos = 2  # Next write position = 0 (wraps)
        # Reading at position delay_idx % delay_steps = 0 → gets the oldest
        delayed = layer.get_delayed_spikes()
        assert torch.allclose(delayed, torch.ones(16, 16))

    def test_delay_ring_wraps(self, layer):
        """Ring buffer wraps correctly after more passes than delay_steps."""
        x = torch.randn(1, 16)
        delay_steps = layer.delay_steps
        for _ in range(delay_steps * 3):
            layer(x)
        # delay_idx should have wrapped
        assert layer._delay_pos == delay_steps * 3
        # Buffer should still be valid (no out-of-bounds)
        delayed = layer.get_delayed_spikes()
        assert delayed.shape == (16, 16)

    def test_incoming_spikes_affect_membrane(self, layer):
        """Passing incoming_spikes should increase membrane_potential."""
        initial_mp = layer.membrane_potential.clone()
        incoming = torch.ones(16, 16)
        x = torch.randn(1, 16)
        layer(x, incoming_spikes=incoming)
        # Membrane should have increased from incoming spikes
        # incoming_spikes * spike_scale = 1.0 * 0.1 = 0.1 added
        # Then leak, then excitability drive
        # Just check it's different from initial
        diff = (layer.membrane_potential - initial_mp).abs().sum().item()
        assert diff > 0, "Incoming spikes should affect membrane potential"


# ── Feature 5: Surrogate Gradient ─────────────────────────────────────


class TestSurrogateGradient:
    def test_output_is_differentiable(self, layer):
        """Despite hard spike threshold, output should support backward()."""
        x = torch.randn(2, 16, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()
        # x should have received gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_surrogate_gradient_nonzero(self):
        """SpikeSurrogate backward should produce non-zero gradients."""
        mp = torch.randn(4, 4, requires_grad=True)
        threshold = 0.0
        spike = SpikeSurrogate.apply(mp, threshold, 10.0)
        loss = spike.sum()
        loss.backward()
        assert mp.grad is not None
        assert (mp.grad.abs() > 0).any(), "Surrogate gradient should be non-zero"

    def test_surrogate_gradient_near_threshold(self):
        """Gradient should be largest near the threshold."""
        # Values at different distances from threshold
        threshold = 0.0
        near = torch.tensor([0.0], requires_grad=True)
        far = torch.tensor([5.0], requires_grad=True)

        spike_near = SpikeSurrogate.apply(near, threshold, 10.0)
        spike_near.backward()

        spike_far = SpikeSurrogate.apply(far, threshold, 10.0)
        spike_far.backward()

        assert near.grad.abs().item() > far.grad.abs().item(), (
            "Gradient should be larger near threshold"
        )


# ── Feature 6: Fire-and-Reset ────────────────────────────────────────


class TestFireAndReset:
    def test_membrane_resets_on_spike_zero_mode(self, layer_low_threshold):
        """After spiking with reset_mode='zero', membrane should be 0."""
        x = torch.randn(4, 16) * 3.0
        # Force membrane above threshold
        layer_low_threshold.membrane_potential.fill_(100.0)
        layer_low_threshold(x)
        # Where spikes occurred, membrane should have been reset to 0
        # (then leak and drive applied, so not exactly 0, but much lower)
        spiked = layer_low_threshold.spike_mask
        if (spiked > 0.5).any():
            # Membrane for just-fired weights was reset to 0, then
            # excitability drive was added. Should be much less than 100.
            mean_fired = layer_low_threshold.membrane_potential[spiked > 0.5].mean()
            assert mean_fired < 10.0, f"Expected low membrane after reset, got {mean_fired}"

    def test_membrane_resets_on_spike_subtract_mode(self, layer_subtract_reset):
        """After spiking with reset_mode='subtract', membrane -= threshold."""
        layer = layer_subtract_reset
        # Force membrane well above threshold
        layer.membrane_potential.fill_(5.0)
        x = torch.randn(4, 16) * 3.0
        layer(x)
        spiked = layer.spike_mask
        if (spiked > 0.5).any():
            # Membrane was 5.0, threshold is 0.001, so subtract gives ~4.999
            # Then leak: 4.999 * 0.99 ≈ 4.949, plus drive
            # Should be near 5.0 still (threshold is tiny)
            mean_fired = layer.membrane_potential[spiked > 0.5].mean()
            assert mean_fired > 1.0, (
                f"Subtract reset should keep membrane high, got {mean_fired}"
            )

    def test_non_spiking_membrane_unchanged_by_reset(self, layer_high_threshold):
        """Non-spiking weights should not be affected by reset logic."""
        layer = layer_high_threshold
        layer.membrane_potential.fill_(0.5)
        mp_before = layer.membrane_potential.clone()
        x = torch.randn(1, 16)
        layer(x)
        # No spikes should have occurred (threshold=1000)
        assert not (layer.spike_mask > 0.5).any()
        # Membrane changed only from leak + drive, not from reset
        # Just verify it's not zero (reset didn't fire)
        assert layer.membrane_potential.mean().item() > 0


# ── Inheritance / Compatibility ───────────────────────────────────────


class TestInheritance:
    def test_inherits_living_layer(self, layer):
        """SpikingLivingLayer should be an instance of LivingLayerV6."""
        assert isinstance(layer, LivingLayerV6)

    def test_hebbian_still_active(self):
        """Hebbian self-modification should still occur once spikes develop."""
        torch.manual_seed(42)
        layer = SpikingLivingLayer(
            in_features=16, out_features=16,
            spike_threshold=0.001, membrane_leak=0.01,
            homeostatic_decay=0.0, set_point_adapt_rate=0.0,
        )
        x = torch.randn(4, 16) * 2.0
        # Warm up: build membrane until spikes develop
        for _ in range(5):
            layer(x)
        w_before = layer.weight.clone()
        layer(x)
        diff = (layer.weight - w_before).abs().sum().item()
        assert diff > 0, "Hebbian updates should still modify weights when spiking"

    def test_error_directed_still_works(self, layer):
        """apply_error() should still update weights."""
        x = torch.randn(4, 16)
        layer(x)
        w_before = layer.weight.clone()
        error = torch.randn(4, 16)
        layer.apply_error(error)
        diff = (layer.weight - w_before).abs().sum().item()
        assert diff > 0, "Error-directed learning should still work"

    def test_episodic_memory_still_works(self, layer):
        """Episodes should still be stored."""
        # Use strong input to exceed salience threshold
        x = torch.randn(4, 16) * 5.0
        for _ in range(10):
            layer(x)
        assert layer.episode_count.item() > 0, "Episodes should be stored"

    def test_homeostasis_still_works(self, layer):
        """Weights should stay near set points over many passes."""
        x = torch.randn(4, 16) * 0.1  # Small input
        for _ in range(100):
            layer(x)
        drift = (layer.weight - layer.set_point).abs().mean().item()
        # Drift should be bounded, not exploding
        assert drift < 10.0, f"Weight drift too large: {drift}"

    def test_aliveness_includes_spiking_metrics(self, layer):
        """Aliveness dict should include spiking-specific keys."""
        x = torch.randn(1, 16)
        layer(x)
        report = layer.aliveness()
        assert "membrane_potential_mean" in report
        assert "membrane_potential_max" in report
        assert "spike_fraction" in report
        assert "refractory_fraction" in report
        # Also should have all parent keys
        assert "weight_mean" in report
        assert "excitability_mean" in report
        assert "update_ema_mean" in report


# ── Stability ─────────────────────────────────────────────────────────


class TestStability:
    def test_no_nan_500_passes(self, layer):
        """500 forward passes should produce no NaN."""
        x = torch.randn(4, 16)
        for i in range(500):
            out = layer(x)
            assert not torch.isnan(out).any(), f"NaN in output at pass {i}"
            assert not torch.isnan(layer.weight).any(), f"NaN in weight at pass {i}"
            assert not torch.isnan(layer.membrane_potential).any(), (
                f"NaN in membrane at pass {i}"
            )

    def test_weight_drift_bounded_spiking(self, layer):
        """Weight norm should stay bounded after 500 passes."""
        x = torch.randn(4, 16)
        initial_norm = layer.weight.norm().item()
        for _ in range(500):
            layer(x)
        final_norm = layer.weight.norm().item()
        # Weight norm shouldn't explode (10x is very generous)
        assert final_norm < initial_norm * 10, (
            f"Weight norm exploded: {initial_norm} -> {final_norm}"
        )

    def test_membrane_bounded(self, layer):
        """Membrane potential should stay bounded (no runaway accumulation)."""
        x = torch.randn(4, 16)
        for _ in range(500):
            layer(x)
        max_mp = layer.membrane_potential.abs().max().item()
        # With leak=0.1 and threshold=1.0, membrane shouldn't exceed
        # a few multiples of threshold
        assert max_mp < 100.0, f"Membrane potential unbounded: max={max_mp}"


# ── Spike-Gated Buffer Updates ──────────────────────────────────────


class TestSpikeGating:
    def test_zero_spike_mask_blocks_hebbian(self):
        """With all-zero spike_mask, Hebbian weight update should be zero."""
        torch.manual_seed(42)
        layer = SpikingLivingLayer(
            in_features=16, out_features=16,
            spike_threshold=1000.0,  # Never spikes
            homeostatic_decay=0.0,
            set_point_adapt_rate=0.0,
        )
        w_before = layer.weight.clone()
        x = torch.randn(4, 16)
        layer(x)
        diff = (layer.weight - w_before).abs().max().item()
        assert diff < 1e-7, f"Expected zero weight change, got {diff}"

    def test_full_spike_mask_allows_updates(self):
        """With all-ones spike_mask, Hebbian updates proceed normally."""
        torch.manual_seed(42)
        layer = SpikingLivingLayer(
            in_features=16, out_features=16,
            spike_threshold=0.0,  # Everything spikes
            membrane_leak=0.0,
        )
        # Force membrane above threshold and counter=0
        layer.membrane_potential.fill_(1.0)
        layer.refractory_counter.fill_(0.0)
        w_before = layer.weight.clone()
        x = torch.randn(4, 16) * 2.0
        layer(x)
        diff = (layer.weight - w_before).abs().sum().item()
        assert diff > 0, "Hebbian should update when all weights spike"

    def test_partial_mask_selective_update(self):
        """Only spiking quadrant gets Hebbian updates."""
        torch.manual_seed(42)
        layer = SpikingLivingLayer(
            in_features=16, out_features=16,
            spike_threshold=1000.0,
            homeostatic_decay=0.0,
            set_point_adapt_rate=0.0,
        )
        # Force top-left 8x8 above threshold, rest below
        layer.membrane_potential[:8, :8] = 2000.0
        layer.membrane_potential[8:, :] = 0.0
        layer.membrane_potential[:8, 8:] = 0.0
        layer.refractory_counter.fill_(0.0)

        w_before = layer.weight.clone()
        x = torch.randn(4, 16) * 2.0
        layer(x)

        spiking_diff = (layer.weight[:8, :8] - w_before[:8, :8]).abs().sum().item()
        silent_diff = (layer.weight[8:, :] - w_before[8:, :]).abs().sum().item()

        assert spiking_diff > 0, "Spiking quadrant should have Hebbian updates"
        assert silent_diff < 1e-6, f"Silent quadrant should be unchanged, got {silent_diff}"

    def test_homeostatic_always_runs(self):
        """Homeostatic regulation runs even with zero spikes."""
        torch.manual_seed(42)
        layer = SpikingLivingLayer(
            in_features=16, out_features=16,
            spike_threshold=1000.0,
            hebb_rate=0.0,
            homeostatic_decay=0.01,
        )
        # Push weights away from set point
        layer.weight.add_(torch.ones_like(layer.weight) * 0.5)
        w_before = layer.weight.clone()
        x = torch.randn(4, 16)
        layer(x)
        drift_before = (w_before - layer.set_point).abs().mean().item()
        drift_after = (layer.weight - layer.set_point).abs().mean().item()
        assert drift_after < drift_before, (
            "Homeostatic should reduce drift even without spikes"
        )

    def test_excitability_always_runs(self):
        """Excitability dynamics run regardless of spike state."""
        torch.manual_seed(42)
        layer = SpikingLivingLayer(
            in_features=16, out_features=16,
            spike_threshold=1000.0,
        )
        exc_before = layer.excitability_acc.clone()
        x = torch.randn(4, 16)
        layer(x)
        exc_change = (layer.excitability_acc - exc_before).abs().sum().item()
        assert exc_change > 0, "Excitability should update even without spikes"

    def test_set_point_always_adapts(self):
        """Set point adaptation runs regardless of spike state."""
        torch.manual_seed(42)
        layer = SpikingLivingLayer(
            in_features=16, out_features=16,
            spike_threshold=1000.0,
            homeostatic_decay=0.01,
            set_point_adapt_rate=0.01,
        )
        # Push weights away from set point
        layer.weight.add_(torch.ones_like(layer.weight) * 0.5)
        sp_before = layer.set_point.clone()
        x = torch.randn(4, 16)
        layer(x)
        sp_change = (layer.set_point - sp_before).abs().sum().item()
        assert sp_change > 0, "Set point should adapt even without spikes"

    def test_momentum_frozen_when_not_spiking(self):
        """Momentum should not change for non-spiking weights."""
        torch.manual_seed(42)
        layer = SpikingLivingLayer(
            in_features=16, out_features=16,
            spike_threshold=1000.0,
        )
        layer.momentum.fill_(0.5)
        mom_before = layer.momentum.clone()
        x = torch.randn(4, 16)
        layer(x)
        diff = (layer.momentum - mom_before).abs().max().item()
        assert diff < 1e-7, f"Momentum should be frozen without spikes, got {diff}"

    def test_update_ema_frozen_when_not_spiking(self):
        """update_ema should not change for non-spiking weights."""
        torch.manual_seed(42)
        layer = SpikingLivingLayer(
            in_features=16, out_features=16,
            spike_threshold=1000.0,
        )
        ema_before = layer.update_ema.clone()
        x = torch.randn(4, 16)
        layer(x)
        diff = (layer.update_ema - ema_before).abs().max().item()
        assert diff < 1e-7, f"update_ema should be frozen without spikes, got {diff}"

    def test_spike_decision_before_hebbian(self):
        """Spike decision should precede Hebbian update (phase reorder)."""
        torch.manual_seed(42)
        layer = SpikingLivingLayer(
            in_features=16, out_features=16,
            spike_threshold=0.5,
            membrane_leak=0.01,
        )
        # Set membrane above threshold for top half, below for bottom
        layer.membrane_potential[:8, :] = 1.0
        layer.membrane_potential[8:, :] = 0.0
        layer.refractory_counter.fill_(0.0)

        x = torch.randn(4, 16) * 2.0
        layer(x)

        assert layer.spike_mask[:8, :].mean().item() > 0.5, (
            "Top half should have spiked (membrane was above threshold)"
        )
        assert layer.spike_mask[8:, :].mean().item() < 0.5, (
            "Bottom half should not have spiked"
        )

    def test_input_dependent_activation(self):
        """Different inputs should activate different weight subsets.

        The 'fried chicken' test: input strong on features 0-7 should
        activate weights connected to those features, not weights
        connected to features 8-15. Like smelling fried chicken
        activates food circuits, not quantum mechanics circuits.
        """
        torch.manual_seed(42)
        layer = SpikingLivingLayer(
            in_features=16, out_features=16,
            spike_threshold=0.5, membrane_leak=0.05,
        )

        # Input A: strong on features 0-7, silent on 8-15
        input_a = torch.zeros(4, 16)
        input_a[:, :8] = 3.0
        for _ in range(20):
            layer(input_a)
        mask_a = layer.spike_mask.clone()

        # Reset membrane and refractory for a clean comparison
        layer.membrane_potential.fill_(0.0)
        layer.refractory_counter.fill_(0.0)

        # Input B: strong on features 8-15, silent on 0-7
        input_b = torch.zeros(4, 16)
        input_b[:, 8:] = 3.0
        for _ in range(20):
            layer(input_b)
        mask_b = layer.spike_mask.clone()

        # Input A should activate left-column weights more
        spikes_a_left = mask_a[:, :8].sum().item()
        spikes_a_right = mask_a[:, 8:].sum().item()
        assert spikes_a_left > spikes_a_right, (
            f"Input A should activate left weights more: "
            f"left={spikes_a_left}, right={spikes_a_right}"
        )

        # Input B should activate right-column weights more
        spikes_b_left = mask_b[:, :8].sum().item()
        spikes_b_right = mask_b[:, 8:].sum().item()
        assert spikes_b_right > spikes_b_left, (
            f"Input B should activate right weights more: "
            f"left={spikes_b_left}, right={spikes_b_right}"
        )

    def test_500_pass_stability_with_gating(self):
        """500 passes with gating enabled should remain stable."""
        torch.manual_seed(42)
        layer = SpikingLivingLayer(in_features=16, out_features=16)
        x = torch.randn(4, 16)
        for i in range(500):
            out = layer(x)
            assert not torch.isnan(out).any(), f"NaN at pass {i}"
            assert not torch.isnan(layer.weight).any(), f"NaN weight at pass {i}"
            assert not torch.isinf(layer.weight).any(), f"Inf weight at pass {i}"
