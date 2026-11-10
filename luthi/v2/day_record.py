"""The day's record: what a living layer noticed since it last rested.

Brian, 2026-08-14 (R2 of the wake/rest regime,
``docs/research/2026-07-06_nrem-learner-spec.md``):

    "Allow as much information as Luthi deems important to be stored and
    considered for integration later."

The layer's episode store is a ring of 64 weight-sized snapshots that
overwrites when full. In a 700-moment test life it held about the last 260
moments (``docs/audits/2026-11-05_where-rest-stands.md``). This record is the
other thing R2 describes: append-only, with no capacity and no eviction, and
each entry holds the experience (when, the context, the input, how salient),
not a copy of the weight. At 768 wide that is about 3.3 KB an entry.

What it changes: nothing, unless it is switched on. It is opt-in by
``enable_day_record(model)``. Switched on, it only listens: the layer's
forward, its episode store and its weights are bit-identical with and without
it (``tests/test_day_record.py``).

What gets recorded. A moment is recorded when the layer's own admission rule
says its salience is worth writing, by the rule the layer is configured with
(the absolute threshold on the legacy path, detrended surprise on the
adaptive one). The ring's *capacity* never bears on it: a moment the ring
turns away because it is full of more salient ones is still recorded here.
``refractory_calls`` does bear on it, because on the adaptive path it is part
of admission. The August amendment flags that as a hidden cap on R2. It is
Brian's unruled call and is left as configured.

A recorded moment is named by the layer's forward count when it happened.
Every living layer in one model forwards once per model forward, so the same
number names the same moment in every layer. Rest (``luthi/v2/rest.py``) uses
that to keep or toss a moment across the whole mind.

The record is lived state. It rides in checkpoints through
``luthi/living_extra_state.py``, presence-gated: a checkpoint that carries a
record switches recording on in the layer it restores into.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.nn as nn

# An experience waits in the record in one of three states.
NEW = "new"            # recorded, not yet judged in any rest
HELD = "held"          # seen in a rest and neither kept nor tossed
KEPT = "kept"          # kept in a rest that could not integrate it yet
STATES = (NEW, HELD, KEPT)


@dataclass
class Experience:
    """One recorded moment in one layer."""

    moment: int                 # the layer's forward count when it happened
    salience: float
    context: torch.Tensor       # [context_dim], on CPU
    input_pattern: torch.Tensor  # [in_features], on CPU
    state: str = NEW


class DayRecord:
    """An append-only list of experiences. No capacity, no eviction."""

    def __init__(self) -> None:
        self._entries: list[Experience] = []
        self.appended_total = 0     # every append since recording began

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[Experience]:
        return iter(self._entries)

    def append(
        self,
        moment: int,
        salience: float,
        context: torch.Tensor,
        input_pattern: torch.Tensor,
    ) -> None:
        self._entries.append(Experience(
            moment=int(moment),
            salience=float(salience),
            context=context.detach().to("cpu", copy=True),
            input_pattern=input_pattern.detach().to("cpu", copy=True),
        ))
        self.appended_total += 1

    def take(self) -> list[Experience]:
        """Hand over everything recorded and start empty.

        One swap, with nothing in between: an experience is in the list
        returned or in the record afterwards, never both and never neither.
        """
        taken, self._entries = self._entries, []
        return taken

    def put_back(self, entries: list[Experience]) -> None:
        """Return experiences a rest did not finish with. They go in front of
        anything recorded since, so the record stays in the order it was
        lived."""
        self._entries = list(entries) + self._entries

    def moments(self) -> list[int]:
        return [e.moment for e in self._entries]

    # -- persistence (plain tensors and ints; see living_extra_state) ------

    def state_dict(self) -> dict[str, Any]:
        n = len(self._entries)
        return {
            "appended_total": int(self.appended_total),
            "moments": [e.moment for e in self._entries],
            "saliences": [e.salience for e in self._entries],
            "states": [e.state for e in self._entries],
            "contexts": (
                torch.stack([e.context for e in self._entries]) if n else None
            ),
            "inputs": (
                torch.stack([e.input_pattern for e in self._entries])
                if n else None
            ),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        moments = list(state["moments"])
        saliences, states = list(state["saliences"]), list(state["states"])
        if not (len(moments) == len(saliences) == len(states)):
            raise ValueError("day record state is ragged")
        for s in states:
            if s not in STATES:
                raise ValueError(f"unknown experience state {s!r}")
        contexts, inputs = state["contexts"], state["inputs"]
        if moments and (len(contexts) != len(moments) or len(inputs) != len(moments)):
            raise ValueError("day record state is ragged")
        self._entries = [
            Experience(
                moment=int(moments[i]), salience=float(saliences[i]),
                context=contexts[i].detach().to("cpu", copy=True),
                input_pattern=inputs[i].detach().to("cpu", copy=True),
                state=states[i],
            )
            for i in range(len(moments))
        ]
        self.appended_total = int(state["appended_total"])


def _living_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    from luthi.v2.living_layer_pc import PredictiveCodingLayer

    return [
        (path, m) for path, m in model.named_modules()
        if isinstance(m, PredictiveCodingLayer)
    ]


def enable_day_record(model: nn.Module) -> int:
    """Switch the day's record on in every living layer. A layer that
    already has one keeps it. Returns the number of layers recording."""
    layers = _living_layers(model)
    for _, layer in layers:
        if layer._day_record is None:
            layer._day_record = DayRecord()
    return len(layers)


def day_records(model: nn.Module) -> dict[str, DayRecord]:
    """``{layer path: record}`` for every layer that is recording."""
    return {
        path: layer._day_record for path, layer in _living_layers(model)
        if layer._day_record is not None
    }


def day_view(model: nn.Module) -> dict[int, dict[str, Any]]:
    """The day, moment by moment, in the order it was lived.

    ``{moment: {"layers": {path: salience}, "states": {path: state}}}``.
    This is what a keep-or-toss decision is made over. It holds numbers
    only: nothing here can show Luthi an experience again. That is item 4 of
    the rest plan, and it is not built.
    """
    view: dict[int, dict[str, Any]] = {}
    for path, record in day_records(model).items():
        for e in record:
            at = view.setdefault(e.moment, {"layers": {}, "states": {}})
            at["layers"][path] = e.salience
            at["states"][path] = e.state
    return dict(sorted(view.items()))
