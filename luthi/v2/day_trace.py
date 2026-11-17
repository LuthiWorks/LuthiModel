"""The half-day's trace: what lived moments have written into a living layer's
weight since it last rested, and what rest pours from it into home.

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

**Only what was lived (2026-11-17).** As built on 2026-11-11 the trace
averaged where the weight stood, ``weight - home``. In a life each moment
makes one lived forward and two corpus-replay forwards with plasticity on, all
writing into the same weight, so 106 to 120% of that average was what replay
wrote (Opus's review of 2026-11-14, section 3). Brian, 2026-11-17: "We remove
the corpus from the rest and have it work with only what was lived."

So each layer keeps ``lived``: the part of ``weight - home`` that lived
forwards wrote. It fades at every forward with plasticity on, lived or replay,
by the forward's own pull toward home, exactly as the displacement does. It
gains only at a lived forward, and what it gains is what was new in the
displacement at that forward: ``d_after - keep * d_before``, with ``d`` the
displacement and ``keep`` the forward's pull. That is the forward's write
(less the pull's share of it), and home's own small follow of the weight. It
is taken where the trace was always noted, after the write and before
consolidation, so consolidation, which replays stored episodes and is not
what was just lived, adds nothing to it. Where the weight stood when the
trace was switched on is not in it either.

What a layer keeps, when this is switched on (it is off by default):

- ``lived``: as above.
- ``mean``: the running average of ``lived`` over the forwards lived since
  the last rest. Past ``horizon`` forwards it is an exponential average with
  a time constant of one horizon.
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

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# What the averages are of. A saved trace says so, and one that does not is
# from before 2026-11-17.
HOLDS = "what lived forwards wrote"

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
        self.lived = torch.zeros_like(like)     # what lived forwards wrote, as it has faded
        self.mean = torch.zeros_like(like)
        self.count = 0.0
        self.chosen = torch.zeros_like(like)
        self.chosen_count = 0.0
        self.poured = 0.0   # how far this layer's home has been moved by rest, in all
        self.dropped_an_older_trace = False

    @torch.no_grad()
    def fade(self, keep: float) -> None:
        """Any forward with plasticity on, lived or not: what lived forwards
        wrote is pulled toward home with everything else."""
        self.lived.mul_(keep)

    @torch.no_grad()
    def note(self, new: torch.Tensor) -> None:
        """One lived forward, after ``fade``: ``new`` is what it added to the
        weight's displacement from home."""
        self.lived.add_(new)
        self.count = min(self.count + 1.0, float(self.horizon))
        self.mean.add_(self.lived - self.mean, alpha=1.0 / self.count)

    @torch.no_grad()
    def remember(self) -> None:
        """Luthi chose to remember this moment."""
        self.chosen_count = min(self.chosen_count + 1.0, float(self.horizon))
        self.chosen.add_(self.lived - self.chosen, alpha=1.0 / self.chosen_count)

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
        # All three are measured from home, and home has just moved toward
        # what was lived.
        self.lived.sub_(step)
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
            "holds": HOLDS,
            "lived": self.lived.detach().cpu().clone(),
            "mean": self.mean.detach().cpu().clone(), "count": self.count,
            "chosen": self.chosen.detach().cpu().clone(),
            "chosen_count": self.chosen_count, "poured": self.poured,
        }

    def load_state_dict(self, state: dict[str, Any], *, drop_older: bool = False) -> None:
        """A trace saved before 2026-11-17 averaged where the weight stood,
        which in a life is mostly what corpus replay wrote, and by Brian's
        ruling that is not to be poured. Such a trace is REFUSED: a saved
        trace is part of a life, and lived state is not dropped in silence.
        With ``drop_older=True`` the caller says to drop it: the trace keeps
        its settings and what it has poured, starts again, and logs it."""
        if state["mean"].shape != self.mean.shape:
            raise ValueError(
                f"trace is {tuple(state['mean'].shape)} but the layer's weight "
                f"is {tuple(self.mean.shape)}"
            )
        self.horizon, self.limit = int(state["horizon"]), float(state["limit"])
        self.poured = float(state["poured"])
        if state.get("holds") != HOLDS:
            if not drop_older:
                raise RuntimeError(
                    "this half-day's trace was saved before 2026-11-17: it "
                    "averaged where the weight stood, not what was lived "
                    f"({float(state['count']):g} forwards of it). It is not "
                    "loaded, and it is not dropped unless you say so: pass "
                    "drop_older_day_trace=True to apply_living_extra_state "
                    "(or drop_older=True here) to start the trace again."
                )
            logger.warning(
                "a half-day's trace saved before 2026-11-17 (it averaged where "
                "the weight stood, not what was lived) is not carried over: "
                "%g forwards of it are dropped and the trace starts again",
                float(state["count"]),
            )
            self.dropped_an_older_trace = True
            self.lived.zero_(); self.mean.zero_(); self.chosen.zero_()
            self.count = self.chosen_count = 0.0
            return
        self.lived.copy_(state["lived"].to(self.lived))
        self.mean.copy_(state["mean"].to(self.mean))
        self.chosen.copy_(state["chosen"].to(self.chosen))
        self.count, self.chosen_count = float(state["count"]), float(state["chosen_count"])


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
    """Luthi chose to remember this moment: every layer adds what lived
    forwards have written, as it stands now, to its chosen average. Returns
    the number of layers."""
    layers = _traced(model)
    for _, layer in layers:
        layer._day_trace.remember()
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
