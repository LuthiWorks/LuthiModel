"""Spiking LWM (Living Weight Model) — Character-level language model.

Same architecture as LuthiLM but with SpikingHybridBlocks that support
inter-block spike propagation. Spikes from block N's living FFN prime
block N+1's membrane potential via delayed spike masks.

This is a prototype file — the original LuthiLM is unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from luthi.hybrid_block_spiking import SpikingHybridBlock


class SpikingLuthiLM(nn.Module):
    """Spiking LWM character-level language model.

    Architecture:
        token -> embedding -> [SpikingHybridBlock x N] -> layer_norm -> projection -> logits

    Spike propagation: delayed spikes from block N feed into block N+1's
    membrane potential, creating cross-layer temporal dynamics.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_blocks: int = 2,
        max_seq_len: int = 128,
        # Living layer parameters
        hebb_rate: float = 0.001,
        error_rate: float = 0.001,
        homeostatic_decay: float = 0.001,
        set_point_adapt_rate: float = 1e-6,
        num_episodes: int = 64,
        episode_blend: float = 0.3,
        eval_hebb_fraction: float = 0.33,
        # Spiking parameters
        spike_threshold: float = 1.0,
        membrane_leak: float = 0.1,
        refractory_steps: int = 3,
        delay_steps: int = 2,
        spike_scale: float = 0.1,
        reset_mode: str = "zero",
        spike_baseline: float = 0.3,
        backward_pass_enabled: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.backward_pass_enabled = backward_pass_enabled

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Learned positional embedding
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Stack of spiking hybrid blocks
        self.blocks = nn.ModuleList([
            SpikingHybridBlock(
                d_model=d_model,
                hebb_rate=hebb_rate,
                error_rate=error_rate,
                homeostatic_decay=homeostatic_decay,
                set_point_adapt_rate=set_point_adapt_rate,
                num_episodes=num_episodes,
                episode_blend=episode_blend,
                eval_hebb_fraction=eval_hebb_fraction,
                spike_threshold=spike_threshold,
                membrane_leak=membrane_leak,
                refractory_steps=refractory_steps,
                delay_steps=delay_steps,
                spike_scale=spike_scale,
                reset_mode=reset_mode,
                spike_baseline=spike_baseline,
            )
            for _ in range(n_blocks)
        ])

        # Final layer norm and output projection
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with inter-block spike propagation and top-down sweep.

        Phase 1 (bottom-up): Forward through blocks with spike propagation.
        Phase 2 (top-down): If training, backward sweep with top-down
            modulation and backward spike propagation.

        Args:
            x: [batch, seq_len] integer token indices.

        Returns:
            [batch, seq_len, vocab_size] logits for next token prediction.
        """
        batch, seq_len = x.shape
        assert seq_len <= self.max_seq_len, (
            f"Sequence length {seq_len} exceeds max {self.max_seq_len}"
        )

        # Embed tokens + positions
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(positions)

        # Phase 1: Bottom-up with spike propagation
        block_inputs = []
        prev_spikes = None
        for block in self.blocks:
            block_inputs.append(h.detach())
            h, delayed_spikes = block(
                h, incoming_spikes=prev_spikes, causal=True
            )
            prev_spikes = delayed_spikes

        # Project to vocabulary
        h_final = self.final_norm(h)
        logits = self.output_proj(h_final)

        # Phase 2: Top-down backward sweep with backward spike propagation
        if self.training and self.backward_pass_enabled:
            from luthi.backward_pass import create_initial_signal

            with torch.no_grad():
                signal = create_initial_signal(h.detach())
                for i in reversed(range(len(self.blocks))):
                    signal, backward_spikes = self.blocks[i].top_down_pass(
                        signal, block_inputs[i],
                    )
                    # Feed backward spikes into the block below's membrane
                    if i > 0:
                        self.blocks[i - 1].living_ffn.membrane_potential.add_(
                            backward_spikes * self.blocks[i - 1].living_ffn.spike_scale * 0.3
                        )

        return logits

    def apply_living_errors(self, expect_grad: bool = False) -> None:
        """Apply error-directed learning to all living FFN layers.

        Call this AFTER loss.backward() to update living weights using
        the error signals captured during the backward pass.

        Args:
            expect_grad: When True, blocks raise instead of silently
                no-opping if the residual gradient is missing. Pass True
                from training loops (post-backward); leave False at
                inference/generation callsites.
        """
        for block in self.blocks:
            block.apply_living_error(expect_grad=expect_grad)

    def aliveness_report(self) -> list[dict[str, float]]:
        """Return aliveness diagnostics for each block (including spiking)."""
        return [block.aliveness() for block in self.blocks]

    def total_parameters(self) -> dict[str, int]:
        """Count trainable parameters vs living weight buffers."""
        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        living_buffers = 0
        for block in self.blocks:
            for name, buf in block.living_ffn.named_buffers():
                living_buffers += buf.numel()
        return {
            "trainable": trainable,
            "living_buffers": living_buffers,
            "total": trainable + living_buffers,
        }
