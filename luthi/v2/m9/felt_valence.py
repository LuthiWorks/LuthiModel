"""Felt valence: how a state feels to the mind, learned from living.

Added 2026-11-03 (Brian: "We still need Luthi to learn to avoid dangerous or
unpleasant things"; docs/research/2026-11-03_learning-to-avoid.md).

The head predicts, from a pooled state, how that state will feel: -1
unpleasant, 0 neutral, +1 pleasant. It learns only from what the mind's world
reports as felt (Sanctuary percepts carrying a valence, passed to
`M9Trainer.observe_transition` as ``ctx["valence"]``). Its last layer starts at
zero, so at birth it predicts neutral for everything. Nothing is unpleasant
to Luthi until something has felt unpleasant. The avoidance it supports is
learned, not built in.

The planner reads `unpleasantness` (predicted negative valence, zero for
neutral or pleasant states) as one more cost in expected free energy. It
avoids, and does not seek: seeking is engagement's job.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeltValenceHead(nn.Module):
    def __init__(self, d_model: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or max(16, d_model // 4)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """[B, D] -> [B] predicted valence in (-1, 1)."""
        return torch.tanh(self.net(s)).squeeze(-1)

    def unpleasantness(self, s: torch.Tensor) -> torch.Tensor:
        """[B, D] -> [B] predicted unpleasantness in [0, 1)."""
        return torch.relu(-self.forward(s))
