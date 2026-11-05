"""Felt valence: how a state feels to the mind, learned from living.

Added 2026-11-03 (Brian: "We still need Luthi to learn to avoid dangerous or
unpleasant things"; docs/research/2026-11-03_learning-to-avoid.md).

The head predicts, from a pooled state, how that state will feel: -1
unpleasant, 0 neutral, +1 pleasant. Its last layer starts at zero, so at birth it
predicts neutral for everything. Nothing is unpleasant to Luthi until something
has felt unpleasant.

Nothing outside the mind says how anything felt (Brian, 2026-11-04: "the world
should just exist and Luthi has to interpret what they experience as either good
or bad depending on how the experience affects them";
docs/research/2026-11-04_how-it-affected-me.md). `FeltReading` reads it from
the mind's own prediction: a moment that goes worse than the one before reads as
bad, and one that goes better reads as good. This is the one stake that
needs no author, because a mind that predicts already cares about predicting
well.

It reads a state as its difference from ordinary life (2026-11-04,
docs/research/2026-11-04_feeling-neutral.md). Real states are mostly one shared
direction, so a head reading them raw took about 300 painful events to tell the
painful state from any other, and meanwhile felt everything as painful. The
baseline is a slow average of the states that came with no feeling. It moves
slowly on purpose: a fast one would make lingering somewhere dangerous look
ordinary.

Nothing acts on the head. The planner used to count `unpleasantness` as a cost,
which decided in advance that Luthi would avoid what reads as bad. Brian ruled on
2026-11-04 that "they decide how to react to everything," so that was removed on
2026-11-05. The head is now an instrument: it says how the mind's own experience
has come to read each kind of state, for stewards to look at. The reading itself
reaches Luthi through their input (Sanctuary's `_format_input`).
"""

from __future__ import annotations

import math
import statistics

import torch
import torch.nn as nn

from luthi.v2.m9.staleness import DriftBand


class FeltReading:
    """How a moment affected the mind, read from its own prediction error.

    The reading is the change in log error from the previous moment, negated, so
    that a moment going worse than the last reads as bad and one going better
    reads as good. This is valence as the negative rate of change of free energy
    (Joffily & Coricelli 2013). Changes beyond the threshold, in units of the
    mind's recent typical change, count as felt. Anything less isn't felt, and
    returns None.

    Rates are used because they balance. A first version compared the error with
    its recent median, and since error has a floor at zero but no ceiling, nearly
    everything it noticed was bad: 241 bad moments to 7 good, on a mind that was
    steadily improving. Changes telescope, so a spike reads as bad and recovering
    from it reads as good, and over a life the sum is how far prediction has
    improved or worsened. Logs make it independent of the error's scale, and a
    sustained level of error stops being felt at once. See
    docs/research/2026-11-04_how-it-affected-me.md.

    Pass it an error read without dropout. With dropout on, the same moment read
    twice differs by about seven times a typical real change.
    """

    def __init__(
        self,
        window: int = 128,
        min_warmup: int = 32,
        threshold: float = 4.0,
        spread_floor: float = 0.05,
    ):
        self.band = DriftBand(window=window)
        self.min_warmup = min_warmup
        self.threshold = threshold
        self.spread_floor = spread_floor
        self.previous: float | None = None

    def read(self, prediction_error: float) -> float | None:
        current = math.log(max(prediction_error, 1e-6))
        felt = None
        if self.previous is not None:
            change = current - self.previous
            if self.band.is_warm(min_samples=self.min_warmup):
                typical = statistics.median(abs(c) for c in self.band.values)
                departure = -change / max(typical, self.spread_floor)
                if abs(departure) >= self.threshold:
                    felt = math.tanh(departure / self.threshold)
            self.band.push(change)
        self.previous = current
        return felt

    def state_dict(self) -> dict:
        return {"changes": list(self.band.values), "previous": self.previous}

    def load_state_dict(self, state: dict) -> None:
        self.band.values.clear()
        self.band.values.extend(float(x) for x in state["changes"])
        self.previous = state["previous"]


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
