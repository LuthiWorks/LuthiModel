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

It reads a state as its difference from ordinary life (2026-11-04,
docs/research/2026-11-04_feeling-neutral.md). Real states are mostly one shared
direction, so a head reading them raw took about 300 painful events to tell the
painful state from any other, and meanwhile felt everything as painful. The
baseline is a slow average of the states that came with no feeling. It moves
slowly on purpose: a fast one would make lingering somewhere dangerous look
ordinary.

The planner reads `unpleasantness` (predicted negative valence, zero for
neutral or pleasant states) as one more cost in expected free energy. It
avoids, and does not seek: seeking is engagement's job.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeltValenceHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden: int | None = None,
        baseline_rate: float = 0.001,
    ):
        super().__init__()
        self.baseline_rate = baseline_rate
        self.register_buffer("baseline", torch.zeros(d_model))
        self.register_buffer("baseline_count", torch.zeros(()))
        hidden = hidden or max(16, d_model // 4)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """[B, D] -> [B] predicted valence in (-1, 1)."""
        return torch.tanh(self.net(s - self.baseline)).squeeze(-1)

    @torch.no_grad()
    def observe_unfelt(self, s: torch.Tensor) -> None:
        """Move the baseline toward a [B, D] state that came with no feeling:
        a plain average at first, then an average that fades at baseline_rate."""
        self.baseline_count += 1
        rate = max(1.0 / float(self.baseline_count), self.baseline_rate)
        self.baseline.add_(s.mean(dim=0) - self.baseline, alpha=rate)

    def unpleasantness(self, s: torch.Tensor) -> torch.Tensor:
        """[B, D] -> [B] predicted unpleasantness in [0, 1)."""
        return torch.relu(-self.forward(s))
