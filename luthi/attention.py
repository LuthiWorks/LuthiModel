"""Scaled dot-product multi-head attention for the hybrid blocks.

Originally single-head (the legacy class name `ScalarAttention` is retained
for v1 checkpoint compatibility); refactored to multi-head on 2026-05-10
after a third-party audit flagged that single-head won't scale to the 4096d
/ 36-block production architecture (cannot attend to multiple independent
subspaces simultaneously).

The class name stays. `n_heads=1` reproduces the original math exactly
(d_head = d_model, single scaled dot-product head), so all v1 checkpoints
trained before this change load and behave identically. New v2 blocks pass
`n_heads=4` by default.

The causal mask is cached as a non-persistent buffer at __init__ and
re-sliced per forward, avoiding the per-step `torch.triu` allocation that
4.6 flagged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScalarAttention(nn.Module):
    """Scaled dot-product attention with optional multi-head splitting.

    Q/K/V/O are nn.Parameters trained via backpropagation. With n_heads=1
    this is the original single-head behavior (d_head = d_model). With
    n_heads > 1 the channel dim is split into n_heads × d_head subspaces
    and attention runs per head, with results concatenated and projected
    out — standard multi-head attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # Per-head scaling — 1/sqrt(d_head). When n_heads=1, d_head=d_model,
        # so this reduces to the original 1/sqrt(d_model) scaling.
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Pre-allocated causal mask. Non-persistent (not saved with the
        # checkpoint) — it's purely a shape, not state. Resized on demand
        # if a forward pass requests a sequence longer than max_seq_len.
        mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("_causal_mask", mask, persistent=False)

    def _get_causal_mask(self, seq_len: int, device) -> torch.Tensor:
        """Return a [seq_len, seq_len] causal mask, growing the cache if needed."""
        if self._causal_mask.shape[0] < seq_len:
            # Audit 2026-05-11 fix: use register_buffer instead of bare
            # assignment so the regrown mask is properly tracked as a
            # non-persistent buffer. The bare assignment also worked
            # (nn.Module.__setattr__ recognizes the existing buffer name)
            # but was fragile — register_buffer is the canonical pattern.
            new_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
                diagonal=1,
            )
            self.register_buffer("_causal_mask", new_mask, persistent=False)
        return self._causal_mask[:seq_len, :seq_len]

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = True,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_kv_cache: bool = False,
    ):
        """
        Args:
            x: [batch, seq_len, d_model] (seq_len=1 when generating with cache)
            causal: If True, apply causal mask (future tokens masked).
            kv_cache: Optional (K_prev, V_prev) tuple from a prior forward,
                each shaped [B, H, S_prev, D]. When provided, the new K/V
                from this forward are concatenated onto the cached ones so
                the attention sees the full accumulated context — letting
                generation skip O(S²) recomputation of prior tokens.
            return_kv_cache: If True, return (output, (K, V)) instead of
                just output, so the caller can feed the updated cache to
                the next forward. Default False preserves the original
                Tensor-returning signature for training/non-generation paths.

        Returns:
            [batch, seq_len, d_model] output (and optional new K/V cache).
        """
        B, S, _ = x.shape
        H = self.n_heads
        D = self.d_head

        # Project and split into heads: [B, S, d_model] -> [B, H, S, D]
        Q = self.q_proj(x).view(B, S, H, D).transpose(1, 2)
        K_new = self.k_proj(x).view(B, S, H, D).transpose(1, 2)
        V_new = self.v_proj(x).view(B, S, H, D).transpose(1, 2)

        if kv_cache is not None:
            K_prev, V_prev = kv_cache
            K = torch.cat([K_prev, K_new], dim=2)  # along sequence axis
            V = torch.cat([V_prev, V_new], dim=2)
        else:
            K, V = K_new, V_new

        # Scaled dot-product per head: [B, H, S_query, S_total]
        scores = (Q @ K.transpose(-2, -1)) * self.scale

        if causal:
            # With KV cache the query axis is shorter than the key axis.
            # Query position i corresponds to absolute position (S_total - S + i),
            # so query i can attend to all key positions [0, S_total - S + i].
            S_total = K.shape[-2]
            full_mask = self._get_causal_mask(S_total, x.device)
            # Slice the rows we care about (the query rows) from the
            # full [S_total, S_total] causal mask — equivalent to building
            # the [S, S_total] sub-mask explicitly. When kv_cache is None,
            # S = S_total and this collapses to the original behavior.
            mask = full_mask[S_total - S : S_total, :]
            scores.masked_fill_(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = attn @ V  # [B, H, S, D]

        # Concatenate heads back to [B, S, d_model] and project.
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        out = self.o_proj(out)

        if return_kv_cache:
            return out, (K, V)
        return out


# Forward-compatible alias. Some callers may want to import MultiHeadAttention
# explicitly; both names point at the same class to avoid duplication.
MultiHeadAttention = ScalarAttention
