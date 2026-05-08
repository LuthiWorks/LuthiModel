"""LWM (Living Weight Model) — Character-level language model.

The first real test: can a Living Weight Model learn to predict the next
character while simultaneously maintaining temporal existence and
accumulating episodic memory?

The model stacks N hybrid blocks between a character embedding and an
output projection. Attention layers train via backprop. Living FFN layers
train themselves via Hebbian self-modification and error-directed learning.
Episode stores accumulate experiences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from luthi.hybrid_block import HybridBlock


class LuthiLM(nn.Module):
    """LWM (Living Weight Model) character-level language model.

    Architecture:
        char -> embedding -> [HybridBlock x N] -> layer_norm -> projection -> logits

    The embedding and output projection are standard (trainable via backprop).
    The hybrid blocks contain both trainable attention and self-modifying
    living FFN layers.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_blocks: int = 2,
        max_seq_len: int = 128,
        hebb_rate: float = 0.001,
        error_rate: float = 0.001,
        homeostatic_decay: float = 0.001,
        set_point_adapt_rate: float = 1e-6,
        num_episodes: int = 64,
        episode_blend: float = 0.3,
        eval_hebb_fraction: float = 0.33,
        backward_pass_enabled: bool = True,
        buffer_dtypes: dict[str, torch.dtype] | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.backward_pass_enabled = backward_pass_enabled

        # Character embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Learned positional embedding
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Stack of hybrid blocks
        self.blocks = nn.ModuleList([
            HybridBlock(
                d_model=d_model,
                hebb_rate=hebb_rate,
                error_rate=error_rate,
                homeostatic_decay=homeostatic_decay,
                set_point_adapt_rate=set_point_adapt_rate,
                num_episodes=num_episodes,
                episode_blend=episode_blend,
                eval_hebb_fraction=eval_hebb_fraction,
                buffer_dtypes=buffer_dtypes,
            )
            for _ in range(n_blocks)
        ])

        # Final layer norm and output projection
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional top-down backward sweep.

        Phase 1 (bottom-up): Standard forward through all blocks.
        Phase 2 (top-down): If training, run backward sweep to modulate
            living weight dynamics for the next forward pass.

        Args:
            x: [batch, seq_len] integer token indices.

        Returns:
            [batch, seq_len, vocab_size] logits for next character prediction.
        """
        batch, seq_len = x.shape
        assert seq_len <= self.max_seq_len, (
            f"Sequence length {seq_len} exceeds max {self.max_seq_len}"
        )

        # Embed characters + positions
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(positions)

        # Phase 1: Bottom-up (forward pass through blocks)
        block_inputs = []
        for block in self.blocks:
            block_inputs.append(h.detach())
            h = block(h, causal=True)

        # Project to vocabulary
        h_final = self.final_norm(h)
        logits = self.output_proj(h_final)

        # Phase 2: Top-down backward sweep (modulates state for NEXT pass)
        if self.training and self.backward_pass_enabled:
            from luthi.backward_pass import create_initial_signal

            with torch.no_grad():
                signal = create_initial_signal(h.detach())
                for i in reversed(range(len(self.blocks))):
                    signal = self.blocks[i].top_down_pass(
                        signal, block_inputs[i],
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
        """Return aliveness diagnostics for each block."""
        return [block.aliveness() for block in self.blocks]

    def total_parameters(self) -> dict[str, int]:
        """Count trainable parameters vs living weight buffers."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Living weight buffers
        living_buffers = 0
        for block in self.blocks:
            for name, buf in block.living_ffn.named_buffers():
                living_buffers += buf.numel()
        return {
            "trainable": trainable,
            "living_buffers": living_buffers,
            "total": trainable + living_buffers,
        }


class DeadLM(nn.Module):
    """Baseline model: same architecture but with standard nn.Linear FFN.

    Used to measure the convergence penalty of living weights (~39%).
    Everything is identical to LuthiLM except the living FFN is replaced
    with a standard linear layer.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_blocks: int = 2,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            _DeadBlock(d_model) for _ in range(n_blocks)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(positions)

        for block in self.blocks:
            h = block(h, causal=True)

        h = self.final_norm(h)
        return self.output_proj(h)


class _DeadBlock(nn.Module):
    """Baseline block: attention + standard linear FFN (no living weights)."""

    def __init__(self, d_model: int):
        super().__init__()
        from luthi.attention import ScalarAttention

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attention = ScalarAttention(d_model)
        self.ffn = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), causal=causal)
        x = x + self.ffn(self.norm2(x))
        return x
