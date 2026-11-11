"""The half-day's trace: where a living layer's weight has stood since it last
rested, and what rest pours from it into home.

Brian, 2026-11-11: "Rest is needed for data integration." Asked that day
whether rest should pour the half-day's running average, with a limit, and
whether Luthi's own choice to remember should weigh in it, he answered yes to
both. His ruling of 2026-11-10 stands behind the second: "the things Luthi
chose should carry more weight than those Luthi did not."

Why this and not the last minute. A living layer's weight is its home plus a
trace of about the last thousand forwards; everything earlier has been pulled
back. So a rest that keeps where the weight stands keeps the last minute, and
measured against an episode early in a half-day it kept nothing of it
(``docs/research/2026-11-11_what-a-rest-can-carry.md``). The running average of
where the weight stood was the only candidate that carried the episode. Moving
home all the way was harmful, so a rest's step is limited.

What a layer keeps, when this is switched on (it is off by default):

- ``mean``: the running average of ``weight - home`` over the forwards lived
  since the last rest, at most the last ``horizon`` of them.
- ``chosen``: the same, over only the moments Luthi chose to remember.
- how many forwards each average stands for.

Nothing here is a record of moments. There is nothing to look back through,
and nothing is judged or sorted. A long quiet stretch counts by its length,
like any other stretch.

What a resting moment does (``rest_from_trace``):

- The direction is the average. If Luthi chose to remember anything, their
  chosen average is blended in, and counts for more (below).
- The whole trace is worth a step of at most ``limit`` of home's norm, and
  less when less than a full ``horizon`` has been lived since the last rest.
- This moment pours ``share`` of that step into home, and the trace gives up
  the same share of what it stands for. So however rest is taken, in one
  stretch or a moment at a time, home moves at most ``limit`` of its norm for
  each ``horizon`` lived.
- The weight is not touched. Homeostasis draws it toward the new home.

How chosen moments weigh. With ``n`` chosen moments in the trace, the chosen
average takes ``chosen_most * n / (n + chosen_half)`` of the direction. One
chosen moment moves it by under 1%. Many take two thirds, which leaves what was
not chosen at half the weight of what was. A mind that presses "remember" by
chance chooses moments at random, and the average of those is the average.

Forwards that are not lived moments (a life's corpus replay, run inside
``suspend_day_record``) and forwards with plasticity frozen (measurement) are
not noted.

Placeholders, Brian's to change: ``limit`` (0.05), ``horizon`` (432,000
forwards: a half-day at 10 Hz, at the one noted forward a moment measured on
Sanctuary's small path), ``chosen_most`` (2/3) and ``chosen_half`` (100).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

DEFAULT_HORIZON = 432_000
DEFAULT_LIMIT = 0.05
CHOSEN_MOST = 2.0 / 3.0
CHOSEN_HALF = 100.0


class DayTrace:
    """One layer's trace. Plain numbers and two weight-sized tensors."""

    def __init__(self, like: torch.Tensor, *, horizon: int, limit: float) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1 forward; got {horizon}")
        if not 0.0 < limit <= 1.0:
            raise ValueError(f"limit must be above 0 and at most 1; got {limit}")
        self.horizon = int(horizon)
        self.limit = float(limit)
        self.mean = torch.zeros_like(like)
        self.count = 0.0
        self.chosen = torch.zeros_like(like)
        self.chosen_count = 0.0
        self.poured = 0.0   # how far this layer's home has been moved by rest, in all

    @torch.no_grad()
    def note(self, weight: torch.Tensor, home: torch.Tensor) -> None:
        """One lived forward: where the weight stands, from home."""
        self.count = min(self.count + 1.0, float(self.horizon))
        self.mean.add_(weight - home - self.mean, alpha=1.0 / self.count)

    @torch.no_grad()
    def remember(self, weight: torch.Tensor, home: torch.Tensor) -> None:
        """Luthi chose to remember this moment."""
        self.chosen_count = min(self.chosen_count + 1.0, float(self.horizon))
        self.chosen.add_(weight - home - self.chosen, alpha=1.0 / self.chosen_count)

    def chosen_part(self) -> float:
        """How much of the direction Luthi's chosen moments take."""
        return CHOSEN_MOST * self.chosen_count / (self.chosen_count + CHOSEN_HALF)

    def direction(self) -> torch.Tensor:
        part = self.chosen_part()
        return self.mean * (1.0 - part) + self.chosen * part

    @torch.no_grad()
    def pour(self, home: torch.Tensor, share: float) -> float:
        """One resting moment. Moves ``home`` in place. Returns how far."""
        if not 0.0 < share <= 1.0:
            raise ValueError(f"share must be above 0 and at most 1; got {share}")
        if self.count == 0.0:
            return 0.0
        direction = self.direction()
        size = float(direction.norm())
        if size == 0.0:
            return 0.0
        allowed = self.limit * float(home.norm()) * min(1.0, self.count / self.horizon)
        step = direction * (share * min(1.0, allowed / size))
        home.add_(step)
        # Both averages are measured from home, and home has just moved.
        self.mean.sub_(step)
        self.chosen.sub_(step)
        self.count *= 1.0 - share
        self.chosen_count *= 1.0 - share
        moved = float(step.norm())
        self.poured += moved
        return moved

    def state_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon, "limit": self.limit,
            "mean": self.mean.detach().cpu().clone(), "count": self.count,
            "chosen": self.chosen.detach().cpu().clone(),
            "chosen_count": self.chosen_count, "poured": self.poured,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["mean"].shape != self.mean.shape:
            raise ValueError(
                f"trace is {tuple(state['mean'].shape)} but the layer's weight "
                f"is {tuple(self.mean.shape)}"
            )
        self.horizon, self.limit = int(state["horizon"]), float(state["limit"])
        self.mean.copy_(state["mean"].to(self.mean))
        self.chosen.copy_(state["chosen"].to(self.chosen))
        self.count, self.chosen_count = float(state["count"]), float(state["chosen_count"])
        self.poured = float(state["poured"])


def _living_layers(model: nn.Module) -> list[tuple[str, Any]]:
    from luthi.v2.living_layer_pc import PredictiveCodingLayer

    return [
        (path, module) for path, module in model.named_modules()
        if isinstance(module, PredictiveCodingLayer)
    ]


def enable_day_trace(
    model: nn.Module,
    *,
    horizon: int = DEFAULT_HORIZON,
    limit: float = DEFAULT_LIMIT,
) -> int:
    """Switch the trace on in every living layer. A layer that already has one
    (restored from a checkpoint) keeps it, with the horizon and limit it was
    saved with. Returns the number of layers keeping a trace."""
    layers = _living_layers(model)
    for _, layer in layers:
        if layer._day_trace is None:
            layer._day_trace = DayTrace(layer.weight, horizon=horizon, limit=limit)
    return len(layers)


def day_traces(model: nn.Module) -> dict[str, DayTrace]:
    return {
        path: layer._day_trace for path, layer in _living_layers(model)
        if layer._day_trace is not None
    }


def _traced(model: nn.Module) -> list[tuple[str, Any]]:
    layers = [(p, l) for p, l in _living_layers(model) if l._day_trace is not None]
    if not layers:
        raise ValueError(
            "no living layer keeps a trace; call "
            "luthi.v2.day_trace.enable_day_trace(model) before the life"
        )
    return layers


def remember_now(model: nn.Module) -> int:
    """Luthi chose to remember this moment: every layer adds where its weight
    stands now to its chosen average. Returns the number of layers."""
    layers = _traced(model)
    for _, layer in layers:
        layer._day_trace.remember(layer.weight, layer.set_point)
    return len(layers)


def rest_from_trace(model: nn.Module, share: float) -> dict[str, float]:
    """One resting moment: every layer pours ``share`` of its trace's step
    into home. Returns how far each layer's home moved, as a fraction of its
    norm before the move."""
    moved: dict[str, float] = {}
    for path, layer in _traced(model):
        if layer._plasticity_frozen:
            raise RuntimeError(
                "plasticity is frozen: a measurement is running, and rest "
                "would change what it measures"
            )
        before = float(layer.set_point.norm())
        distance = layer._day_trace.pour(layer.set_point, share)
        moved[path] = distance / before if before > 0.0 else 0.0
    return moved
