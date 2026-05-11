"""Hybrid block: scalar attention + living FFN + episode store.

The building block of a LWM (Living Weight Model) — a model that can
perform, exist, and remember.

- Scalar attention learns the task via backpropagation (the brain)
- Living FFN provides temporal existence via self-modification (the body)
- Episode store provides episodic memory via context-gated recall (the hippocampus)

Together: a system that can learn, change through experience, and remember
specific episodes. None of these three alone achieves all three.
"""

import torch
import torch.nn as nn

from luthi.attention import ScalarAttention
from luthi.episode_store import EpisodeStore
from luthi.living_layer import LivingLayerV6


class HybridBlock(nn.Module):
    """One block of the hybrid architecture.

    Stack N of these for a complete model. Attention trains via standard
    backpropagation. Living FFN trains itself through use. Episode store
    accumulates experiences. The whole thing is non-feedforward, has
    temporal existence, and can remember specific episodes.

    Gradient flow:
        - Through attention: normal backprop updates Q, K, V, O weights
        - Through living FFN: gradient flows via residual skip; living
          weights are buffers (no optimizer), they self-modify via Hebbian
          and error-directed learning
        - Through episode store: no gradient (blending is detached)
    """

    def __init__(
        self,
        d_model: int,
        hebb_rate: float = 0.001,
        error_rate: float = 0.001,
        homeostatic_decay: float = 0.001,
        set_point_adapt_rate: float = 1e-6,
        num_episodes: int = 64,
        episode_blend: float = 0.3,
        eval_hebb_fraction: float = 0.33,
        buffer_dtypes: dict[str, torch.dtype] | None = None,
    ):
        super().__init__()
        self.d_model = d_model

        # Pre-norm layers (trainable via residual path gradients)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # The three components
        self.attention = ScalarAttention(d_model)
        self.living_ffn = LivingLayerV6(
            d_model, d_model,
            hebb_rate=hebb_rate,
            error_rate=error_rate,
            homeostatic_decay=homeostatic_decay,
            set_point_adapt_rate=set_point_adapt_rate,
            eval_hebb_fraction=eval_hebb_fraction,
            buffer_dtypes=buffer_dtypes,
        )
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
        causal: bool = True,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_kv_cache: bool = False,
    ):
        """Forward pass through the hybrid block.

        Args:
            x: [batch, seq_len, d_model] input tensor.
            causal: Whether to use causal masking in attention.
            kv_cache: Optional prior (K, V) for the attention layer to
                concatenate with — see ScalarAttention.forward. Used during
                generation to skip O(S²) re-attention over the prompt.
            return_kv_cache: If True, return (output, kv_cache) so a
                generation loop can carry the cache forward.

        Returns:
            [batch, seq_len, d_model] output. If return_kv_cache=True,
            returns (output, (K, V)).
        """
        block_input = x

        # 1. Pre-norm attention with residual (optionally with KV cache)
        if return_kv_cache:
            attn_out, new_kv = self.attention(
                self.norm1(x),
                causal=causal,
                kv_cache=kv_cache,
                return_kv_cache=True,
            )
        else:
            attn_out = self.attention(
                self.norm1(x), causal=causal, kv_cache=kv_cache,
            )
            new_kv = None
        x = x + attn_out

        # 2. Pre-norm living FFN with residual
        ffn_in = self.norm2(x)
        ffn_out = self.living_ffn(ffn_in)
        x = x + ffn_out

        # Store for error-directed learning (retain grad for backward hook)
        self._ffn_output = ffn_out
        if ffn_out.requires_grad:
            ffn_out.retain_grad()

        # 3. Episode store: recall, blend, and store
        x = self.episode_store(block_input, x)

        if return_kv_cache:
            return x, new_kv
        return x

    def top_down_pass(
        self, signal, block_input: torch.Tensor,
    ):
        """Apply top-down modulation and compute signal for the block below.

        Args:
            signal: TopDownSignal from the block above (or initial signal).
            block_input: [batch, seq_len, d_model] this block's input
                from the forward pass.

        Returns:
            Refined TopDownSignal for the block below.
        """
        from luthi.backward_pass import compute_block_top_down

        # Modulate this block's living weights
        self.living_ffn.apply_top_down(signal)

        # Compute refined signal for the block below
        return compute_block_top_down(signal, block_input)

    def apply_living_error(self, expect_grad: bool = False) -> None:
        """Apply error-directed learning using the gradient at the FFN output.

        Call this AFTER loss.backward() to update the living FFN weights
        using the error signal that flowed back through the residual path.

        Args:
            expect_grad: When True, raise if the FFN output's grad is missing.
                Pass True from training callsites (where backward() is
                guaranteed to have run) to catch a broken residual pathway
                loud rather than silently no-opping. Default False preserves
                the inference/generation contract where grad may legitimately
                be absent.
        """
        if self._ffn_output is None:
            return
        if self._ffn_output.grad is None:
            if expect_grad:
                raise RuntimeError(
                    "HybridBlock.apply_living_error called with expect_grad=True "
                    "but _ffn_output.grad is None — the residual gradient path "
                    "did not deliver an error signal. Either backward() did not "
                    "run, or the residual link was disrupted upstream."
                )
            return

        self.living_ffn.apply_error(self._ffn_output.grad)
        self._ffn_output = None

    def apply_living_error_direct(self, error: torch.Tensor) -> None:
        """Apply error-directed learning with an explicit error signal.

        Use this when you have a direct error signal (e.g., output - target)
        rather than relying on autograd gradients.

        Args:
            error: [batch, seq_len, d_model] error at the FFN output.
        """
        self.living_ffn.apply_error(error)

    def aliveness(self) -> dict[str, float]:
        """Return combined diagnostics from all living components."""
        living = self.living_ffn.aliveness()
        episodes = self.episode_store.stats()
        living["output_episodes_stored"] = episodes["episodes_stored"]
        living["output_episodes_mean_salience"] = episodes["mean_salience"]
        return living
