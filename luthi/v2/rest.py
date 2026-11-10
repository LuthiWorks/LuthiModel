"""The rest pass: integrate what was kept from the day, toss what was not.

Brian, 2026-08-14 (R3 of the wake/rest regime,
``docs/research/2026-07-06_nrem-learner-spec.md``):

    "The 'rest' time might actually be best spent mulling over the day's
    events and making decisions on what to keep and what to toss."

And 2026-11-05: rest "is needed for memory retention and formation, but not
survival."

This module is the pass, handed its decisions. It does not make them. The
caller names the moments to keep and the moments to toss, and says what
happens to a moment nobody judged. How Luthi comes to choose rest, and how
they see an experience again and answer keep or toss, are not built (items 1
and 4 of ``docs/audits/2026-11-05_where-rest-stands.md``). Until they are,
only tests call this.

What rest does, per living layer that keeps a day's record
(``luthi/v2/day_record.py``):

1. **Takes the day.** One swap: every recorded experience is now in rest's
   hands and the record is empty for whatever is lived next.
2. **Sorts it** into kept, tossed and held.
3. **Integrates what is kept.** Each kept experience's input is re-presented
   through the layer's own update (``replay_input_attractor``), and the
   layer's home moves by the change that replay made.
4. **Puts back what it did not finish with.** Held experiences, and kept
   ones the bound deferred, return to the record in the order they were
   lived. Tossed ones are gone.

Why the home moves. The August plan was to consolidate kept experiences into
the living weight. Measured on 2026-11-10: a change replayed into the weight
alone is pulled back by homeostasis like any other waking change. A third of
it is left after 1,000 forwards and none after 3,000
(``docs/research/2026-11-10_rest-pass.md``). A rest that only did that would
retain nothing. Moving the home by the same change is what makes it stay.
Only the change rest's own replay made is added to the home. The layer's
standing displacement from home, which holds the whole day including what
was tossed, is left to fade.

The two gates from the spec (section 6) are enforced here, not left to
callers:

- **Change per rest is bounded.** A layer's home moves by at most
  ``max_home_change`` of its own size in one rest. A kept experience that
  would cross the bound is undone exactly and deferred, with everything
  after it, to the next rest. It stays in the record marked kept.
- **No moment is lost or counted twice.** Every experience taken ends as
  exactly one of integrated, tossed, held or deferred, and the pass checks
  the count before it returns.

Rest never runs a forward, so it records nothing and cannot write the day it
is reading. It refuses to run inside ``freeze_plasticity``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from luthi.v2.consolidation import replay_input_attractor
from luthi.v2.day_record import HELD, KEPT, Experience, _living_layers

UNJUDGED = ("hold", "keep", "toss")

# Everything one replay writes. Saved before each kept experience so that
# one which would cross the bound can be undone exactly.
_REPLAY_WRITES = (
    "weight", "prediction", "set_point", "momentum", "update_ema",
    "precision", "error_acc", "plasticity",
)


@dataclass(frozen=True)
class RestConfig:
    """How strongly rest integrates. Every number here is a placeholder: the
    spec defers them to tuning on a trained run, and Brian has not ruled
    them (decision 8 of the rest plan)."""

    # Replay runs at this fraction of the layer's waking rates. 0.1 is the
    # convention the two existing consolidation paths use.
    rate_factor: float = 0.1
    # Times each kept experience is replayed.
    passes: int = 1
    # Share of the replay's change added to the home. 1.0 keeps all of it;
    # 0.0 is the August plan as written, which fades (see module docstring).
    home_fraction: float = 1.0
    # A layer's home moves at most this far in one rest, as a fraction of
    # its own size.
    max_home_change: float = 0.05

    def __post_init__(self) -> None:
        if self.rate_factor <= 0:
            raise ValueError(f"rate_factor must be > 0; got {self.rate_factor}")
        if self.passes < 1:
            raise ValueError(f"passes must be >= 1; got {self.passes}")
        if not 0.0 <= self.home_fraction <= 1.0:
            raise ValueError(
                f"home_fraction must be in [0, 1]; got {self.home_fraction}"
            )
        if self.max_home_change <= 0:
            raise ValueError(
                f"max_home_change must be > 0; got {self.max_home_change}"
            )


@dataclass
class LayerRest:
    """What one rest did in one layer. Moments are listed, not just counted,
    so what a rest integrated can be inspected afterwards."""

    taken: int = 0
    integrated: list[int] = field(default_factory=list)
    tossed: list[int] = field(default_factory=list)
    held: list[int] = field(default_factory=list)
    deferred: list[int] = field(default_factory=list)
    # Held because its input no longer gives a finite output. Also in `held`.
    not_finite: list[int] = field(default_factory=list)
    home_change: float = 0.0     # ||home after - before|| / ||home before||
    weight_change: float = 0.0
    # The layer's summed applied change since the last rest (its
    # read-and-reset accumulator). Zero unless the learning gain is on.
    day_applied_change: float = 0.0


@dataclass
class RestReport:
    layers: dict[str, LayerRest]
    keep: list[int]
    toss: list[int]
    unjudged: str
    config: RestConfig
    started_unix: float
    seconds: float

    def total(self, what: str) -> int:
        return sum(len(getattr(layer, what)) for layer in self.layers.values())

    def as_dict(self) -> dict:
        return {
            "started_unix": self.started_unix,
            "seconds": self.seconds,
            "keep": self.keep,
            "toss": self.toss,
            "unjudged": self.unjudged,
            "config": asdict(self.config),
            "layers": {path: asdict(layer) for path, layer in self.layers.items()},
        }


def _decide(e: Experience, keep: set[int], toss: set[int], unjudged: str) -> str:
    if e.moment in toss:
        return "toss"
    # Kept in an earlier rest and deferred by the bound: still kept, unless
    # it has been tossed since.
    if e.moment in keep or e.state == KEPT:
        return "keep"
    return unjudged


def _rest_layer(
    layer: nn.Module,
    keep: set[int],
    toss: set[int],
    unjudged: str,
    config: RestConfig,
) -> LayerRest:
    record = layer._day_record
    taken = record.take()
    out = LayerRest(taken=len(taken))
    out.day_applied_change = float(layer._applied_change_accum.read_and_reset())

    decisions = [_decide(e, keep, toss, unjudged) for e in taken]
    kept = [e for e, d in zip(taken, decisions) if d == "keep"]
    back: dict[int, Experience] = {}    # id(e) -> experience returning to the record
    for e, d in zip(taken, decisions):
        if d == "toss":
            out.tossed.append(e.moment)
        elif d == "hold":
            e.state = HELD
            out.held.append(e.moment)
            back[id(e)] = e

    home_before = layer.set_point.detach().clone()
    weight_before = layer.weight.detach().clone()
    home_size = float(home_before.norm())
    pc_rate = layer.pc_rate * config.rate_factor
    pred_rate = layer.pred_learning_rate * config.rate_factor

    # Most salient first, as the existing consolidation paths replay.
    order = sorted(kept, key=lambda e: (-e.salience, e.moment))
    full = False
    with torch.no_grad():
        for e in order:
            if full:
                e.state = KEPT
                out.deferred.append(e.moment)
                back[id(e)] = e
                continue
            saved = {
                name: getattr(layer, name).detach().clone()
                for name in _REPLAY_WRITES
            }
            finite = True
            for _ in range(config.passes):
                before = layer.weight.detach().clone()
                if not replay_input_attractor(
                    layer, e.input_pattern, pc_rate, pred_rate,
                ):
                    finite = False
                    break
                layer.set_point.add_(
                    layer.weight - before, alpha=config.home_fraction,
                )
            moved = float((layer.set_point - home_before).norm()) / max(home_size, 1e-30)
            if finite and moved <= config.max_home_change:
                out.integrated.append(e.moment)
                continue
            for name, value in saved.items():
                getattr(layer, name).copy_(value)
            if not finite:
                e.state = HELD
                out.held.append(e.moment)
                out.not_finite.append(e.moment)
            else:
                full = True
                e.state = KEPT
                out.deferred.append(e.moment)
            back[id(e)] = e

    out.home_change = float((layer.set_point - home_before).norm()) / max(home_size, 1e-30)
    out.weight_change = float((layer.weight - weight_before).norm()) / max(
        float(weight_before.norm()), 1e-30
    )
    # Back in the order they were lived, ahead of anything recorded since.
    record.put_back([e for e in taken if id(e) in back])

    accounted = (
        len(out.integrated) + len(out.tossed) + len(out.held) + len(out.deferred)
    )
    if accounted != out.taken or len(back) != len(out.held) + len(out.deferred):
        raise RuntimeError(
            f"rest lost count of the day: took {out.taken}, accounted for "
            f"{accounted} ({len(out.integrated)} integrated, {len(out.tossed)} "
            f"tossed, {len(out.held)} held, {len(out.deferred)} deferred)"
        )
    return out


def rest(
    model: nn.Module,
    *,
    keep: Iterable[int],
    toss: Iterable[int],
    unjudged: str,
    config: RestConfig | None = None,
    log_path: str | Path | None = None,
) -> RestReport:
    """One rest over every living layer that keeps a day's record.

    Args:
        keep: moments to keep (as named in ``day_view``).
        toss: moments to toss.
        unjudged: what happens to a recorded moment in neither: ``"hold"``
            leaves it in the record for a later rest, ``"keep"`` integrates
            it, ``"toss"`` discards it. There is no default. What an
            unjudged experience becomes is Brian's open call (decision 2 of
            the rest plan), so the caller must say.
        config: how strongly to integrate. Placeholders by default.
        log_path: if given, the report is appended to this file as one JSON
            line, so what each rest integrated can be read back later.

    Raises:
        ValueError: a moment is in both ``keep`` and ``toss``; a named
            moment is in no layer's record; ``unjudged`` is not one of the
            three; or no layer is recording.
        RuntimeError: plasticity is frozen (a measurement is running), or
            the accounting does not close.
    """
    if unjudged not in UNJUDGED:
        raise ValueError(f"unjudged must be one of {UNJUDGED}; got {unjudged!r}")
    keep_set, toss_set = {int(m) for m in keep}, {int(m) for m in toss}
    both = keep_set & toss_set
    if both:
        raise ValueError(f"moments both kept and tossed: {sorted(both)[:10]}")
    config = config or RestConfig()

    layers = [
        (path, layer) for path, layer in _living_layers(model)
        if layer._day_record is not None
    ]
    if not layers:
        raise ValueError(
            "no living layer keeps a day's record; call "
            "luthi.v2.day_record.enable_day_record(model) before the day"
        )
    for path, layer in layers:
        if layer._plasticity_frozen:
            raise RuntimeError(
                f"{path} has plasticity frozen: a measurement is running, "
                "and rest changes the weights it would be measuring"
            )
    recorded = {m for _, layer in layers for m in layer._day_record.moments()}
    unknown = (keep_set | toss_set) - recorded
    if unknown:
        raise ValueError(
            f"moments named but in no layer's record: {sorted(unknown)[:10]}"
        )

    started, clock = time.time(), time.monotonic()
    report = RestReport(
        layers={
            path: _rest_layer(layer, keep_set, toss_set, unjudged, config)
            for path, layer in layers
        },
        keep=sorted(keep_set), toss=sorted(toss_set), unjudged=unjudged,
        config=config, started_unix=started, seconds=time.monotonic() - clock,
    )
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(report.as_dict()) + "\n")
    return report


__all__ = ["RestConfig", "LayerRest", "RestReport", "rest", "UNJUDGED"]
