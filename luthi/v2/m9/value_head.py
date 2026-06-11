"""V(s) value head -- bootstraps the MCTS tree; biases the habit net.

Architecture: small 2-layer MLP from state vector [B, D] to scalar
value [B]. Deliberately NOT a living-weight layer: V is the
deliberative artifact that must be *stable enough* that cached node
values can be compared across cycles. Living V would drift with
plasticity and compound the Seam-C cross-cycle tree staleness the
plan §4 staleness machinery is built to handle.

Trained downstream (training story TBD as the loop integrates):
classic Bellman-target supervision against MCTS-improved values, or
a Fountas-style policy/value distillation off the MCTS visit
distribution and Q-values. The head itself is generic.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """2-layer MLP: [B, d_model] -> [B] scalar value."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """s: [B, d_model] -> [B] scalar value."""
        return self.net(s).squeeze(-1)
