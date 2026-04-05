"""Spiking Living Layer: LivingLayerV6 with full spiking neural dynamics.

Extends the living weight architecture with biologically-inspired spiking:
- Leaky integrate-and-fire membrane dynamics
- Refractory periods after firing
- Spike propagation between layers via delay buffers
- Surrogate gradients for backpropagation through spike decisions
- Fire-and-reset with configurable reset modes

The spiking layer inherits ALL existing living weight behavior (Hebbian
self-modification, metaplasticity, homeostatic regulation, episodic memory)
and layers spike dynamics on top. The membrane potential integrates
excitability signals; when it crosses threshold, the neuron fires,
modulating its output and communicating with downstream layers.

All operations use float32 element-wise tensor math for DirectML
compatibility — no boolean indexing, no .item() calls, no long tensors
in the forward path.

This is a prototype file — the original LivingLayerV6 is unchanged.
"""

import torch
import torch.nn as nn

from luthi.living_layer import LivingLayerV6


class SpikeSurrogate(torch.autograd.Function):
    """Hard spike in forward pass, smooth sigmoid gradient in backward.

    Enables gradient flow through the binary spike decision for upstream
    trainable parameters (attention layers). The forward pass uses a hard
    threshold comparison; the backward pass approximates the gradient with
    the derivative of a steep sigmoid centered at the threshold.

    Currently the membrane potential is a buffer (not a parameter), so
    gradients through this function are not optimized. This is ready for
    when spike_threshold or membrane dynamics become learnable.
    """

    @staticmethod
    def forward(ctx, membrane_potential, threshold, sharpness=10.0):
        ctx.save_for_backward(membrane_potential)
        ctx.threshold = threshold
        ctx.sharpness = sharpness
        return (membrane_potential >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        (membrane_potential,) = ctx.saved_tensors
        shifted = ctx.sharpness * (membrane_potential - ctx.threshold)
        sigmoid_val = torch.sigmoid(shifted)
        surrogate_grad = ctx.sharpness * sigmoid_val * (1 - sigmoid_val)
        return grad_output * surrogate_grad, None, None


class SpikingLivingLayer(LivingLayerV6):
    """A spiking extension of LivingLayerV6.

    All existing living weight dynamics are preserved unchanged. On top of
    those, this layer adds:

    1. Leaky membrane potential that integrates excitability signals
    2. Threshold-based spiking with fire-and-reset
    3. Refractory periods that prevent immediate re-firing
    4. Delay buffer for spike propagation to downstream layers
    5. Output modulation based on spike state

    The biological analogy:
    - excitability_acc (parent) = receptor density / channel expression (slow)
    - membrane_potential (new) = actual membrane voltage (fast)
    - refractory_counter = absolute refractory period
    - delay_buffer = axonal conduction delay

    All spiking buffers use float32 and all operations are element-wise
    tensor math for DirectML compatibility.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        # Spiking parameters
        spike_threshold: float = 1.0,
        membrane_leak: float = 0.1,
        refractory_steps: int = 3,
        delay_steps: int = 2,
        spike_scale: float = 0.1,
        reset_mode: str = "zero",
        spike_baseline: float = 0.3,
        surrogate_sharpness: float = 10.0,
        # All parent parameters forwarded via kwargs
        **kwargs,
    ):
        super().__init__(in_features, out_features, **kwargs)

        # Spiking hyperparameters
        self.spike_threshold = spike_threshold
        self.membrane_leak = membrane_leak
        self.refractory_steps = float(refractory_steps)
        self.delay_steps = max(1, delay_steps)
        self.spike_scale = spike_scale
        assert reset_mode in ("zero", "subtract"), (
            f"reset_mode must be 'zero' or 'subtract', got '{reset_mode}'"
        )
        self.reset_mode = reset_mode
        self.spike_baseline = spike_baseline
        self.surrogate_sharpness = surrogate_sharpness

        # --- Spiking buffers (all float32 for DirectML compatibility) ---

        # Membrane potential: leaky integrator
        self.register_buffer(
            "membrane_potential",
            torch.zeros(out_features, in_features),
        )

        # Refractory counter: float32 countdown (0.0 = ready to fire)
        self.register_buffer(
            "refractory_counter",
            torch.zeros(out_features, in_features),
        )

        # Current spike mask: float32 (1.0 = fired, 0.0 = silent)
        self.register_buffer(
            "spike_mask",
            torch.zeros(out_features, in_features),
        )

        # Delay ring buffer: stores recent spike masks for propagation
        self.register_buffer(
            "delay_buffer",
            torch.zeros(self.delay_steps, out_features, in_features),
        )

        # Ring buffer write position (plain Python int, not a tensor).
        # Rebuilds in a few forward passes after checkpoint load.
        self._delay_pos: int = 0

    def _get_update_gate(self) -> torch.Tensor:
        """Return spike_mask as the activity-dependent gate.

        Hebbian weight updates, momentum, and update_ema only apply to
        weights that fired in the current step. Quiescent weights retain
        their last state until they fire again.
        """
        return self.spike_mask

    def get_delayed_spikes(self) -> torch.Tensor:
        """Return the spike mask from delay_steps ago for propagation."""
        return self.delay_buffer[self._delay_pos % self.delay_steps]

    def forward(
        self,
        x: torch.Tensor,
        incoming_spikes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with spiking dynamics and activity-dependent gating.

        Eight phases — spike decision BEFORE parent forward so the gate
        is ready when Hebbian updates run:
          0. Receive incoming spikes → membrane
          1. Membrane leak
          2. Spike decision (from accumulated membrane, before learning)
          3. Fire-and-reset + refractory management
          4. Parent forward (Hebbian/homeostatic/episodic — gated by spike_mask)
          5. Drive membrane from input-weight alignment (for NEXT step)
          6. Delay buffer update
          7. Output modulation

        All spiking operations use element-wise tensor math only —
        no boolean indexing, no .item(), no dtype conversions in the
        critical path. DirectML-safe.

        Args:
            x: Input tensor [batch, in_features] or [batch, seq_len, in_features].
            incoming_spikes: Optional [out_features, in_features] float tensor
                from the previous block's delay buffer.

        Returns:
            Spike-modulated output tensor matching input leading dimensions.
        """
        with torch.no_grad():
            # Phase 0: Receive incoming spikes from previous block
            if incoming_spikes is not None:
                self.membrane_potential.add_(
                    incoming_spikes * self.spike_scale
                )

            # Phase 1: Membrane leak
            self.membrane_potential.mul_(1.0 - self.membrane_leak)

            # Phase 2: Spike decision (all element-wise float ops)
            # Decided BEFORE parent forward so _get_update_gate() returns
            # the current spike_mask when Hebbian code queries it.
            above = (self.membrane_potential >= self.spike_threshold).float()
            ready = (self.refractory_counter <= 0.0).float()
            self.spike_mask = above * ready  # 1.0 where fired, 0.0 where not

            # Phase 3: Fire-and-reset + refractory management
            if self.reset_mode == "zero":
                self.membrane_potential.mul_(1.0 - self.spike_mask)
            else:  # "subtract"
                self.membrane_potential.sub_(
                    self.spike_mask * self.spike_threshold
                )

            # Refractory: where fired → set to refractory_steps
            #             where not fired → decrement (clamp at 0)
            decremented = (self.refractory_counter - 1.0).clamp(min=0.0)
            self.refractory_counter = torch.where(
                self.spike_mask > 0.5,
                torch.full_like(self.refractory_counter, self.refractory_steps),
                decremented,
            )

        # Phase 4: Parent forward (Hebbian updates now gated by spike_mask)
        output = super().forward(x)

        with torch.no_grad():
            # Phase 5: Drive membrane from input-weight alignment (for NEXT step)
            # Per-weight: how relevant is this weight to the current input?
            # drive[i,j] = |weight[i,j]| * |input[j]| — weights aligned with
            # strong input features get more membrane drive and are more likely
            # to spike. This is input-dependent sparse activation: the input
            # selects which weights are relevant, like olfactory input lighting
            # up food-associated circuits but not motor circuits.
            # Normalized by mean so dynamics are scale-invariant.
            if x.dim() == 3:
                input_mag = x.detach().abs().mean(dim=(0, 1))  # [in]
            else:
                input_mag = x.detach().abs().mean(dim=0)  # [in]
            drive_signal = self.weight.abs() * input_mag.unsqueeze(0)  # [out, in]
            drive_signal = drive_signal / (drive_signal.mean() + 1e-8)
            self.membrane_potential.add_(drive_signal * self.spike_scale)

            # Phase 6: Delay buffer update
            write_pos = self._delay_pos % self.delay_steps
            self.delay_buffer[write_pos] = self.spike_mask
            self._delay_pos += 1

        # Phase 7: Output modulation
        spike_ratio = self.spike_mask.mean(dim=1)  # [out]
        modulation = self.spike_baseline + (1.0 - self.spike_baseline) * spike_ratio

        if output.dim() == 3:
            output = output * modulation.view(1, 1, -1)
        else:
            output = output * modulation.unsqueeze(0)

        return output

    def aliveness(self) -> dict[str, float]:
        """Return diagnostics including spiking-specific metrics."""
        base = super().aliveness()
        base["membrane_potential_mean"] = self.membrane_potential.mean().item()
        base["membrane_potential_max"] = self.membrane_potential.max().item()
        base["spike_fraction"] = self.spike_mask.mean().item()
        base["refractory_fraction"] = (
            (self.refractory_counter > 0).float().mean().item()
        )
        return base
