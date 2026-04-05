"""Spiking hybrid block: scalar attention + spiking living FFN + episode store.

Same architecture as HybridBlock but with SpikingLivingLayer replacing
LivingLayerV6. The forward() signature changes to accept and return spike
state for inter-block propagation.

This is a prototype file — the original HybridBlock is unchanged.
"""

import torch
import torch.nn as nn

from luthi.attention import ScalarAttention
from luthi.episode_store import EpisodeStore
from luthi.living_layer_spiking import SpikingLivingLayer


class SpikingHybridBlock(nn.Module):
    """One block of the spiking hybrid architecture.

    Identical to HybridBlock except:
    - Uses SpikingLivingLayer instead of LivingLayerV6
    - forward() accepts incoming_spikes and returns (output, delayed_spikes)
    - Aliveness report includes spiking metrics
    """

    def __init__(
        self,
        d_model: int,
        # Living layer parameters
        hebb_rate: float = 0.001,
        error_rate: float = 0.001,
        homeostatic_decay: float = 0.001,
        set_point_adapt_rate: float = 1e-6,
        eval_hebb_fraction: float = 0.33,
        # Episode store parameters
        num_episodes: int = 64,
        episode_blend: float = 0.3,
        # Spiking parameters
        spike_threshold: float = 1.0,
        membrane_leak: float = 0.1,
        refractory_steps: int = 3,
        delay_steps: int = 2,
        spike_scale: float = 0.1,
        reset_mode: str = "zero",
        spike_baseline: float = 0.3,
    ):
        super().__init__()
        self.d_model = d_model

        # Pre-norm layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Attention (trainable via backprop)
        self.attention = ScalarAttention(d_model)

        # Spiking living FFN (self-modifying + spike dynamics)
        self.living_ffn = SpikingLivingLayer(
            d_model,
            d_model,
            hebb_rate=hebb_rate,
            error_rate=error_rate,
            homeostatic_decay=homeostatic_decay,
            set_point_adapt_rate=set_point_adapt_rate,
            eval_hebb_fraction=eval_hebb_fraction,
            spike_threshold=spike_threshold,
            membrane_leak=membrane_leak,
            refractory_steps=refractory_steps,
            delay_steps=delay_steps,
            spike_scale=spike_scale,
            reset_mode=reset_mode,
            spike_baseline=spike_baseline,
        )

        # Episode store (hippocampus)
        self.episode_store = EpisodeStore(
            d_model,
            num_episodes=num_episodes,
            blend_factor=episode_blend,
        )

        # Store FFN output for error-directed learning
        self._ffn_output: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        incoming_spikes: torch.Tensor | None = None,
        causal: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the spiking hybrid block.

        Args:
            x: [batch, seq_len, d_model] input tensor.
            incoming_spikes: Optional spike tensor from previous block's
                delay buffer. Shape [d_model, d_model].
            causal: Whether to use causal masking in attention.

        Returns:
            Tuple of:
              - [batch, seq_len, d_model] output tensor
              - [d_model, d_model] delayed spikes for next block
        """
        block_input = x

        # 1. Pre-norm attention with residual
        x = x + self.attention(self.norm1(x), causal=causal)

        # 2. Pre-norm spiking living FFN with residual
        ffn_in = self.norm2(x)
        ffn_out = self.living_ffn(ffn_in, incoming_spikes=incoming_spikes)
        x = x + ffn_out

        # Store for error-directed learning (retain grad for backward hook)
        self._ffn_output = ffn_out
        if ffn_out.requires_grad:
            ffn_out.retain_grad()

        # 3. Episode store: recall, blend, and store
        x = self.episode_store(block_input, x)

        # 4. Get delayed spikes for next block
        delayed_spikes = self.living_ffn.get_delayed_spikes()

        return x, delayed_spikes

    def apply_living_error(self) -> None:
        """Apply error-directed learning using the gradient at the FFN output.

        Call this AFTER loss.backward() to update the living FFN weights
        using the error signal that flowed back through the residual path.
        """
        if self._ffn_output is None:
            return
        if self._ffn_output.grad is None:
            return

        self.living_ffn.apply_error(self._ffn_output.grad)
        self._ffn_output = None

    def apply_living_error_direct(self, error: torch.Tensor) -> None:
        """Apply error-directed learning with an explicit error signal."""
        self.living_ffn.apply_error(error)

    def aliveness(self) -> dict[str, float]:
        """Return combined diagnostics from all living components."""
        living = self.living_ffn.aliveness()
        episodes = self.episode_store.stats()
        living["output_episodes_stored"] = episodes["episodes_stored"]
        living["output_episodes_mean_salience"] = episodes["mean_salience"]
        return living
