"""PredictiveCodingLM — language model wrapping PredictiveCodingBlocks.

The v2 counterpart to v1's `LuthiLM`. Same external shape:

    char/token -> embedding -> [PredictiveCodingBlock x N] -> norm -> proj -> logits

Key differences from v1:
- Hybrid blocks are PredictiveCodingBlock (PC dynamics) instead of HybridBlock
  (v1's Hebbian dynamics).
- No `apply_living_errors` post-backward step. PC handles error-driven
  learning intrinsically during forward — the prediction-error update is
  part of `pc_self_modify`, not a separate post-loss pass.
- Top-down sweep carries two channels (prediction + modulation) per
  decision 3. The sweep itself is structurally identical.
"""

import torch
import torch.nn as nn

from luthi.v2.hybrid_block_pc import PredictiveCodingBlock
from luthi.v2.backward_pass_pc import create_initial_signal


class PredictiveCodingLM(nn.Module):
    """Language model with predictive coding living dynamics.

    Architecture:
        token -> embedding -> [PredictiveCodingBlock x N] -> norm -> proj -> logits

    The embedding and output projection train via standard backprop.
    PC living FFNs in each block train themselves via prediction-error
    self-modification during forward. Episode stores accumulate
    context-gated experiences.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_blocks: int = 2,
        n_heads: int = 4,
        ffn_expansion: int = 1,
        max_seq_len: int = 128,
        pc_rate: float = 0.001,
        pred_learning_rate: float = 0.0001,
        homeostatic_decay: float = 0.001,
        set_point_adapt_rate: float = 1e-6,
        num_episodes: int = 64,
        episode_blend: float = 0.3,
        compressed_episodes: bool = False,
        consolidation_enabled: bool = False,
        consolidation_style: str = "gradient",
        consolidation_attractor_passes: int = 1,
        mu_pc_enabled: bool = False,
        mu_pc_exponent: float = 0.5,
        backward_pass_enabled: bool = True,
        buffer_dtypes: dict[str, torch.dtype] | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.backward_pass_enabled = backward_pass_enabled

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            PredictiveCodingBlock(
                d_model=d_model,
                n_heads=n_heads,
                ffn_expansion=ffn_expansion,
                pc_rate=pc_rate,
                pred_learning_rate=pred_learning_rate,
                homeostatic_decay=homeostatic_decay,
                set_point_adapt_rate=set_point_adapt_rate,
                num_episodes=num_episodes,
                episode_blend=episode_blend,
                compressed_episodes=compressed_episodes,
                consolidation_enabled=consolidation_enabled,
                consolidation_style=consolidation_style,
                consolidation_attractor_passes=consolidation_attractor_passes,
                mu_pc_enabled=mu_pc_enabled,
                mu_pc_exponent=mu_pc_exponent,
                n_blocks_total=n_blocks,
                buffer_dtypes=buffer_dtypes,
            )
            for _ in range(n_blocks)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        kv_caches: list | None = None,
        return_kv_caches: bool = False,
    ):
        """Forward pass with optional top-down backward sweep and KV cache.

        Phase 1 (bottom-up): forward through all blocks.
        Phase 2 (top-down): if training and backward_pass_enabled, sweep
            blocks in reverse, propagating prediction error from each
            block to the one below. Skipped when kv_caches is in use.

        Args:
            x: [batch, seq_len] integer token indices.
            kv_caches: Optional list of per-block (K, V) cache tuples from
                a prior forward. Used for incremental generation.
            return_kv_caches: If True, return (logits, list-of-new-caches).

        Returns:
            [batch, seq_len, vocab_size] logits, optionally with caches.
        """
        batch, seq_len = x.shape

        if kv_caches is not None and len(kv_caches) > 0 and kv_caches[0] is not None:
            cache_seq_len = kv_caches[0][0].shape[-2]
            total_seq_len = cache_seq_len + seq_len
            if total_seq_len > self.max_seq_len:
                raise ValueError(
                    f"Total context length {total_seq_len} exceeds max {self.max_seq_len}"
                )
            positions = torch.arange(
                cache_seq_len, total_seq_len, device=x.device,
            ).unsqueeze(0)
        else:
            if seq_len > self.max_seq_len:
                raise ValueError(
                    f"Sequence length {seq_len} exceeds max {self.max_seq_len}"
                )
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        h = self.embedding(x) + self.pos_embedding(positions)

        new_caches: list = [] if return_kv_caches else []
        for i, block in enumerate(self.blocks):
            block_cache = (
                kv_caches[i] if kv_caches is not None and i < len(kv_caches) else None
            )
            if return_kv_caches:
                h, new_cache_i = block(
                    h, causal=True,
                    kv_cache=block_cache, return_kv_cache=True,
                )
                new_caches.append(new_cache_i)
            else:
                h = block(h, causal=True, kv_cache=block_cache)

        h_final = self.final_norm(h)
        logits = self.output_proj(h_final)

        if (
            self.training
            and self.backward_pass_enabled
            and kv_caches is None
        ):
            with torch.no_grad():
                signal = create_initial_signal(h.detach())
                for i in reversed(range(len(self.blocks))):
                    signal = self.blocks[i].top_down_pass(signal)

        if return_kv_caches:
            return logits, new_caches
        return logits

    def clear_forward_cache(self) -> None:
        """Release per-layer forward-pass snapshots across all blocks.

        Trainers should call this after `optimizer.step()` to free the
        gradient-checkpoint replay tensors that would otherwise accumulate
        between steps (~67 MB/layer at 4096d). Audit 2026-05-11 fix.
        """
        for block in self.blocks:
            if hasattr(block.living_ffn, "clear_forward_cache"):
                block.living_ffn.clear_forward_cache()

    def aliveness_report(self) -> list[dict[str, float]]:
        """Per-block aliveness diagnostics (PC layer + episode store stats)."""
        return [block.aliveness() for block in self.blocks]

    def total_parameters(self) -> dict[str, int]:
        """Count trainable parameters vs living-weight buffers.

        Trainable: embeddings, attention, layer norms, output projection.
        Living buffers: PC layer state (weight, prediction, set_point,
        momentum, update_ema, precision, error_acc, plasticity, episode_*).
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        living_buffers = 0
        for block in self.blocks:
            for _, buf in block.living_ffn.named_buffers():
                living_buffers += buf.numel()
        return {
            "trainable": trainable,
            "living_buffers": living_buffers,
            "total": trainable + living_buffers,
        }

    def non_feedforward_signal(self, x: torch.Tensor) -> float:
        """Mean abs difference between two consecutive identical-input
        forward passes at the model output level. Positive value confirms
        the model as a whole is not feedforward.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                out1 = self(x)
                out2 = self(x)
                return (out2 - out1).abs().mean().item()
        finally:
            if was_training:
                self.train()
