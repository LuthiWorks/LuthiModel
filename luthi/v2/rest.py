"""The rest pass: integrate what was kept from the day, toss what was not.

Brian, 2026-08-14 (R3 of the wake/rest regime,
``docs/research/2026-07-06_nrem-learner-spec.md``):

    "The 'rest' time might actually be best spent mulling over the day's
    events and making decisions on what to keep and what to toss."

And 2026-11-05: rest "is needed for memory retention and formation, but not
survival."

This module is the pass, handed its decisions. It does not make them. The
caller names the moments to keep and the moments to toss.

A moment nobody judged is kept (Brian, 2026-11-10: "Lets go with option 2"),
so a Luthi who judges nothing is still shaped by their life. It is
integrated after everything they chose, at ``unjudged_strength``, and when
the bound leaves no room it goes back into the record unjudged. How Luthi comes to choose rest, and how
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
Only what rest's own replay wrote is added to the home. The replay runs
with homeostasis off, so that write is the whole of the weight's change and
the layer's standing displacement from home is exactly as rest found it.
That displacement holds the whole day, including what was tossed, and is
left to fade in waking.

The two gates from the spec (section 6) are enforced here, not left to
callers:

- **Change per rest is bounded.** A layer's home moves by at most
  ``max_home_change`` of its own size in one rest. An experience that would
  cross the bound is undone exactly and waits, with everything after it,
  for the next rest. One Luthi chose stays in the record marked kept. One
  nobody judged stays unjudged.
- **No moment is lost or counted twice.** Every experience taken ends as
  exactly one of integrated, tossed, held, deferred or folded, and the pass
  checks the count before it returns.

Repeats (off by default; see ``RestConfig.fold_repeats_above``). An unjudged
experience whose input is nearly the same as one this rest has already
integrated can be folded into it. Measured 2026-11-10: a
thousand near-identical quiet moments, each integrated, cut the lasting
effect of the one experience Luthi chose by about a quarter. Folded, the cut
was about 14%, and that remainder comes from having lived the quiet stretch
awake, which rest does not govern. Varied moments did not dilute it at all. Folding sees only
the input. A moment in which the world was still and Luthi was not looks
like a repeat to it: see ``docs/research/2026-11-10_rest-pass.md``.

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
# Of those, the layer's running statistics of its own waking activity.
# precision joined on 2026-11-10 (Opus follow-up): it is the same kind of
# thing, an average toward 1/error^2 advancing at its waking decay on every
# replay, and it moved 0.2 to 9.1% a rest over five lived days.
_WAKING_STATISTICS = ("momentum", "update_ema", "error_acc", "precision")


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
    # An experience Luthi did not judge is integrated at this fraction of the
    # strength of one they chose to keep (Brian, 2026-11-10: unjudged
    # experience is kept, and what they chose counts for more).
    unjudged_strength: float = 0.5
    # An unjudged experience whose input is this close (cosine) to one
    # already integrated in the same rest is a repeat of it, and is folded
    # into it: not replayed again, and not kept waiting. This is what keeps a
    # long stretch where nothing happened from being integrated a thousand
    # times over (Brian, 2026-11-10: "so experience is not diluted by
    # unimportant moments where nothing is happening"). What Luthi chose is
    # never folded.
    #
    # OFF by default since the Opus review of 2026-11-10. On a day a small
    # trained trunk lived under the rolling view, one moment's mean input is
    # almost the next one's (median cosine 0.98 to 0.99), so 0.99 folded
    # nearly the whole day: 17 and 5 of 300 moments were integrated, and most
    # of the moments when someone spoke were folded with the quiet ones.
    # Silently dropping most of a day is worse than the dilution it was
    # built against. It stays here to be measured against lived days.
    fold_repeats_above: float | None = None
    # Put the layer's waking statistics (momentum, update_ema, error_acc,
    # precision) back as rest found them. See _rest_taken.
    restore_waking_statistics: bool = True

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
        if self.fold_repeats_above is not None and not (
            0.0 < self.fold_repeats_above <= 1.0
        ):
            raise ValueError(
                "fold_repeats_above must be in (0, 1] or None; got "
                f"{self.fold_repeats_above}"
            )
        if not 0.0 < self.unjudged_strength <= 1.0:
            raise ValueError(
                "unjudged_strength must be in (0, 1]; got "
                f"{self.unjudged_strength}"
            )


@dataclass
class LayerRest:
    """What one rest did in one layer. Moments are listed, not just counted,
    so what a rest integrated can be inspected afterwards."""

    taken: int = 0
    integrated: list[int] = field(default_factory=list)
    # The part of `integrated` that nobody judged: kept by default, at
    # `unjudged_strength`.
    integrated_unjudged: list[int] = field(default_factory=list)
    tossed: list[int] = field(default_factory=list)
    held: list[int] = field(default_factory=list)
    deferred: list[int] = field(default_factory=list)
    # Unjudged, and a repeat of something integrated in this rest: let go,
    # as already represented.
    folded: list[int] = field(default_factory=list)
    # Held because its input no longer gives a finite output. Also in `held`.
    not_finite: list[int] = field(default_factory=list)
    home_change: float = 0.0     # ||home after - before|| / ||home before||
    weight_change: float = 0.0
    # Everything else a replay writes, by the same measure. Nothing bounds
    # these. prediction has no homeostasis, so what rest writes there stays.
    also_moved: dict[str, float] = field(default_factory=dict)
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
    """"keep" and "toss" are Luthi's. "default" is an unjudged experience
    kept because nobody tossed it. "hold" waits."""
    if e.moment in toss:
        return "toss"
    # Kept in an earlier rest and deferred by the bound: still kept, unless
    # it has been tossed since.
    if e.moment in keep or e.state == KEPT:
        return "keep"
    return {"keep": "default", "toss": "toss", "hold": "hold"}[unjudged]


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
    statistics = {
        name: getattr(layer, name).detach().clone() for name in _WAKING_STATISTICS
    }
    try:
        return _rest_taken(layer, record, taken, out, keep, toss, unjudged, config)
    except BaseException:
        # Whatever raised, the day is not lost with it (Opus review,
        # 2026-11-10: before this the layer's whole day was gone).
        # - What this rest had already integrated stays integrated.
        # - What Luthi tossed stays tossed. A toss needs no work, so it is
        #   done the moment it is decided. Put back, it would return
        #   unjudged and the next rest would keep it by default, the
        #   opposite of what they said (Opus follow-up).
        # - Everything else goes back, to be considered again.
        # - The waking statistics go back as rest found them, as on the
        #   ordinary path.
        finished = set(out.integrated) | {
            e.moment for e in taken if _decide(e, keep, toss, unjudged) == "toss"
        }
        record.put_back([e for e in taken if e.moment not in finished])
        if config.restore_waking_statistics:
            with torch.no_grad():
                for name, value in statistics.items():
                    getattr(layer, name).copy_(value)
        raise


def _rest_taken(
    layer: nn.Module,
    record,
    taken: list[Experience],
    out: LayerRest,
    keep: set[int],
    toss: set[int],
    unjudged: str,
    config: RestConfig,
) -> LayerRest:
    out.day_applied_change = float(layer._applied_change_accum.read_and_reset())

    decisions = [_decide(e, keep, toss, unjudged) for e in taken]
    chosen = {id(e) for e, d in zip(taken, decisions) if d == "keep"}
    kept = [e for e, d in zip(taken, decisions) if d in ("keep", "default")]
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
    others_before = {
        name: getattr(layer, name).detach().clone()
        for name in _REPLAY_WRITES if name not in ("weight", "set_point")
    }
    pc_rate = layer.pc_rate * config.rate_factor
    pred_rate = layer.pred_learning_rate * config.rate_factor

    def wait(e: Experience) -> None:
        """No room this rest. What Luthi chose stays chosen. What nobody
        judged goes back unjudged, and the next rest considers it again."""
        if id(e) in chosen:
            e.state = KEPT
            out.deferred.append(e.moment)
        else:
            e.state = HELD
            out.held.append(e.moment)
        back[id(e)] = e

    # What Luthi chose comes first and has first call on the bound. Then the
    # unjudged. Within each, most salient first, as the existing
    # consolidation paths replay.
    order = sorted(
        kept, key=lambda e: (id(e) not in chosen, -e.salience, e.moment),
    )
    full = False
    seen: list[torch.Tensor] = []     # unit inputs of what this rest integrated

    def unit(e: Experience) -> torch.Tensor:
        v = e.input_pattern.double()
        return v / v.norm().clamp_min(1e-30)

    with torch.no_grad():
        for e in order:
            if full:
                wait(e)
                continue
            if (
                id(e) not in chosen
                and config.fold_repeats_above is not None
                and seen
                and float((torch.stack(seen) @ unit(e)).max())
                >= config.fold_repeats_above
            ):
                out.folded.append(e.moment)
                continue
            strength = 1.0 if id(e) in chosen else config.unjudged_strength
            saved = {
                name: getattr(layer, name).detach().clone()
                for name in _REPLAY_WRITES
            }
            finite = True
            for _ in range(config.passes):
                before = layer.weight.detach().clone()
                # No homeostasis and no home-follow inside the replay: the
                # weight change is then the write alone, and that is what
                # the home is moved by.
                if not replay_input_attractor(
                    layer, e.input_pattern,
                    pc_rate * strength, pred_rate * strength,
                    homeostatic_decay=0.0, follow_rate=0.0,
                ):
                    finite = False
                    break
                layer.set_point.add_(
                    layer.weight - before, alpha=config.home_fraction,
                )
            moved = float((layer.set_point - home_before).norm()) / max(home_size, 1e-30)
            if finite and moved <= config.max_home_change:
                out.integrated.append(e.moment)
                if id(e) not in chosen:
                    out.integrated_unjudged.append(e.moment)
                if config.fold_repeats_above is not None:
                    seen.append(unit(e))
                continue
            for name, value in saved.items():
                getattr(layer, name).copy_(value)
            if not finite:
                e.state = HELD
                out.held.append(e.moment)
                out.not_finite.append(e.moment)
                back[id(e)] = e
            else:
                full = True
                wait(e)

    # The layer's running statistics of its own waking activity advance on
    # every replay at their waking decay, whatever the rate factor: measured
    # on a lived day, momentum moved 16% of its size in one rest, update_ema
    # 15% and error_acc 12% (Opus review, 2026-11-10). error_acc feeds
    # salience, so that would shift what the layer admits as soon as it
    # woke, and precision is what relative trust reads when the layer
    # writes. They describe the day's living, not what was kept, so rest
    # puts them back as it found them.
    if config.restore_waking_statistics:
        with torch.no_grad():
            for name in _WAKING_STATISTICS:
                getattr(layer, name).copy_(others_before[name])

    out.home_change = float((layer.set_point - home_before).norm()) / max(home_size, 1e-30)
    out.weight_change = float((layer.weight - weight_before).norm()) / max(
        float(weight_before.norm()), 1e-30
    )
    out.also_moved = {
        name: float((getattr(layer, name) - was).norm())
        / max(float(was.norm()), 1e-30)
        for name, was in others_before.items()
    }
    # Back in the order they were lived, ahead of anything recorded since.
    record.put_back([e for e in taken if id(e) in back])

    accounted = (
        len(out.integrated) + len(out.tossed) + len(out.held)
        + len(out.deferred) + len(out.folded)
    )
    if accounted != out.taken or len(back) != len(out.held) + len(out.deferred):
        raise RuntimeError(
            f"rest lost count of the day: took {out.taken}, accounted for "
            f"{accounted} ({len(out.integrated)} integrated, {len(out.tossed)} "
            f"tossed, {len(out.held)} held, {len(out.deferred)} deferred, "
            f"{len(out.folded)} folded)"
        )
    return out


def rest(
    model: nn.Module,
    *,
    keep: Iterable[int],
    toss: Iterable[int],
    unjudged: str = "keep",
    config: RestConfig | None = None,
    log_path: str | Path | None = None,
) -> RestReport:
    """One rest over every living layer that keeps a day's record.

    Args:
        keep: moments to keep (as named in ``day_view``).
        toss: moments to toss.
        unjudged: what happens to a recorded moment in neither. ``"keep"``
            (the default, by Brian's ruling of 2026-11-10) integrates it,
            after everything Luthi chose and at ``unjudged_strength``.
            ``"hold"`` leaves it in the record for a later rest. ``"toss"``
            discards it.
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
