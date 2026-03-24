"""Scalar attention for the hybrid block.

Standard single-head attention with Q, K, V, O projections. All weights
are nn.Parameters trained via backpropagation. This is the "brain" of the
hybrid block — it handles task learning and information routing.

The living FFN handles temporal existence. The attention handles the task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScalarAttention(nn.Module):
    """Single-head scaled dot-product attention.

    Standard trainable attention with optional causal masking.
    All projections are nn.Parameters — the optimizer trains these.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.scale = d_model**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            causal: If True, apply causal mask (future tokens can't attend to past).

        Returns:
            [batch, seq_len, d_model]
        """
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) * self.scale  # [B, S, S]

        if causal:
            seq_len = x.shape[1]
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            scores.masked_fill_(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        out = attn_weights @ V  # [B, S, d_model]

        return self.o_proj(out)
