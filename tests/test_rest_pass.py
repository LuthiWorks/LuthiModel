"""The rest pass (R3 of the 2026-08-14 wake/rest regime; built 2026-11-10).

Rest is handed its decisions here. Nothing in this file is Luthi choosing.

Pins:
- What is kept moves the home. What is tossed leaves the layer bit-identical.
- **Retention (the capacity check).** A kept experience is still in the
  weights thousands of forwards later. With ``home_fraction=0`` (the August
  plan as written) it is not.
- **Gate 1, bounded change.** One rest moves a home by at most
  ``max_home_change``. What would cross it is undone exactly and deferred.
- **Gate 2, the count closes.** Every experience taken is integrated,
  tossed, held or deferred, once. Held and deferred ones are back in the
  record in the order lived; tossed and integrated ones are not.
- A deferred experience is still kept at the next rest without being named
  again, and a later toss overrides it.
- ``unjudged`` has no default and does what it says.
- Rest runs no forward: it records nothing and advances no counter.
- It refuses bad decisions, a model with no record, and a measurement.
- The report lists the moments, and appends to a log when given one.
"""

from __future__ import annotations

import json
import random

import pytest
import torch

from luthi.v2.day_record import HELD, KEPT, DayRecord, day_records, enable_day_record
from luthi.v2.living_layer_pc import PredictiveCodingLayer
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM
from luthi.v2.plasticity import freeze_plasticity
from luthi.v2.rest import RestConfig, rest

D = 16
WRITES = (
    "weight", "prediction", "set_point", "momentum", "update_ema",
    "precision", "error_acc", "plasticity",
)
LOOSE = RestConfig(max_home_change=1e9)   # the bound out of the way


def _layer(seed: int = 0, lived: int = 12) -> PredictiveCodingLayer:
    torch.manual_seed(seed)
    layer = PredictiveCodingLayer(D, D, salience_threshold=0.0)
    layer._day_record = DayRecord()
    g = torch.Generator().manual_seed(seed + 1)
    for _ in range(lived):
        layer(torch.randn(2, 3, D, generator=g))
    return layer


def _writes(layer) -> dict:
    return {n: getattr(layer, n).detach().clone() for n in WRITES}


def _same(a: dict, b: dict) -> bool:
    return all(torch.equal(a[k], b[k]) for k in a)


def test_keeping_moves_the_home_and_tossing_changes_nothing():
    layer = _layer()
    before = _writes(layer)
    report = rest(layer, keep=[], toss=range(1, 13), unjudged="toss", config=LOOSE)
    assert _same(before, _writes(layer)), "tossing leaves the layer as it was"
    assert report.layers[""].tossed == list(range(1, 13))
    assert len(layer._day_record) == 0

    layer = _layer()
    home = layer.set_point.detach().clone()
    report = rest(layer, keep=[3, 7], toss=[], unjudged="toss", config=LOOSE)
    assert not torch.equal(home, layer.set_point)
    assert sorted(report.layers[""].integrated) == [3, 7]
    assert report.layers[""].home_change > 0


def _share_left(home_fraction: float, forwards: int = 3000) -> float:
    """Keep one experience, live on, and ask how much of the change that
    rest made is still in the weight, against a twin that tossed it."""
    kept, tossed = _layer(), _layer()
    w0 = kept.weight.detach().clone()
    cfg = RestConfig(home_fraction=home_fraction, max_home_change=1e9)
    rest(kept, keep=[5], toss=[], unjudged="toss", config=cfg)
    rest(tossed, keep=[], toss=[], unjudged="toss", config=cfg)
    change = kept.weight.detach() - w0
    g1, g2 = (torch.Generator().manual_seed(9) for _ in range(2))
    for _ in range(forwards):
        kept(torch.randn(2, 3, D, generator=g1))
        tossed(torch.randn(2, 3, D, generator=g2))
    left = kept.weight.detach() - tossed.weight.detach()
    return float((left * change).sum() / (change * change).sum())


def test_what_is_kept_is_still_there_thousands_of_forwards_later():
    assert _share_left(home_fraction=1.0) > 0.8


def test_without_the_home_what_is_kept_fades_like_any_waking_change():
    """The August plan as written: replay into the weight and nothing else.
    Characterization of why rest moves the home."""
    assert abs(_share_left(home_fraction=0.0)) < 0.2


def test_one_rest_moves_a_home_no_further_than_the_bound():
    free = _layer()
    rest(free, keep=range(1, 13), toss=[], unjudged="toss", config=LOOSE)
    unbounded = float(
        (free.set_point - _layer().set_point).norm() / _layer().set_point.norm()
    )

    layer = _layer()
    bound = 0.6 * unbounded
    report = rest(
        layer, keep=range(1, 13), toss=[], unjudged="toss",
        config=RestConfig(max_home_change=bound),
    ).layers[""]
    assert 0 < report.home_change <= bound
    assert report.integrated and report.deferred
    assert len(report.integrated) + len(report.deferred) == 12
    # Deferred ones are in the record, marked kept, in the order lived.
    waiting = list(layer._day_record)
    assert [e.moment for e in waiting] == sorted(report.deferred)
    assert all(e.state == KEPT for e in waiting)


def test_what_the_bound_defers_is_undone_exactly():
    layer = _layer()
    free = _layer()
    rest(free, keep=range(1, 13), toss=[], unjudged="toss", config=LOOSE)
    bound = 0.6 * float(
        (free.set_point - _layer().set_point).norm() / _layer().set_point.norm()
    )
    cfg = RestConfig(max_home_change=bound)
    report = rest(layer, keep=range(1, 13), toss=[], unjudged="toss", config=cfg)
    integrated = report.layers[""].integrated
    assert integrated and report.layers[""].deferred, "the bound fell mid-rest"

    twin = _layer()   # keeps only what the bounded rest managed to integrate
    rest(twin, keep=integrated, toss=[], unjudged="toss", config=LOOSE)
    assert _same(_writes(layer), _writes(twin))


def test_a_deferred_experience_is_kept_next_time_without_being_named():
    layer = _layer()
    cfg = RestConfig(max_home_change=1e-7)       # room for nothing
    first = rest(layer, keep=[2, 9], toss=[], unjudged="hold", config=cfg).layers[""]
    assert sorted(first.deferred) == [2, 9] and not first.integrated
    second = rest(layer, keep=[], toss=[], unjudged="hold", config=LOOSE).layers[""]
    assert sorted(second.integrated) == [2, 9]
    assert len(second.held) == 10


def test_a_later_toss_overrides_an_earlier_keep():
    layer = _layer()
    rest(layer, keep=[2], toss=[], unjudged="toss",
         config=RestConfig(max_home_change=1e-7))
    before = _writes(layer)
    report = rest(layer, keep=[], toss=[2], unjudged="hold", config=LOOSE).layers[""]
    assert report.tossed == [2] and not report.integrated
    assert _same(before, _writes(layer))


@pytest.mark.parametrize("seed", range(5))
def test_the_count_closes_whatever_is_decided(seed):
    rng = random.Random(seed)
    layer = _layer(lived=40)
    moments = list(range(1, 41))
    keep = set(rng.sample(moments, 12))
    toss = set(rng.sample([m for m in moments if m not in keep], 12))
    report = rest(
        layer, keep=keep, toss=toss, unjudged="hold",
        config=RestConfig(max_home_change=rng.choice([1e-7, 2e-3, 1e9])),
    ).layers[""]
    groups = [report.integrated, report.tossed, report.held, report.deferred]
    assert report.taken == 40
    assert sorted(m for g in groups for m in g) == moments, "each moment once"
    assert set(report.tossed) == toss
    assert set(report.integrated) | set(report.deferred) == keep
    back = layer._day_record.moments()
    assert back == sorted(report.held + report.deferred)
    states = {e.moment: e.state for e in layer._day_record}
    assert all(states[m] == HELD for m in report.held)
    assert all(states[m] == KEPT for m in report.deferred)


def test_unjudged_must_be_said_and_does_what_it_says():
    with pytest.raises(TypeError):
        rest(_layer(), keep=[], toss=[])   # no default
    with pytest.raises(ValueError, match="unjudged"):
        rest(_layer(), keep=[], toss=[], unjudged="decide for them")

    held = rest(_layer(), keep=[1], toss=[2], unjudged="hold", config=LOOSE).layers[""]
    assert len(held.held) == 10 and held.integrated == [1] and held.tossed == [2]
    kept = rest(_layer(), keep=[1], toss=[2], unjudged="keep", config=LOOSE).layers[""]
    assert len(kept.integrated) == 11 and kept.tossed == [2]
    tossed = rest(_layer(), keep=[1], toss=[2], unjudged="toss", config=LOOSE).layers[""]
    assert len(tossed.tossed) == 11 and tossed.integrated == [1]


def test_rest_runs_no_forward():
    layer = _layer()
    steps, appended = layer._sparse_step_count, layer._day_record.appended_total
    ring = (int(layer.episode_writes.item()), layer.episode_inputs.detach().clone())
    rest(layer, keep=[1, 2, 3], toss=[], unjudged="hold", config=LOOSE)
    assert layer._sparse_step_count == steps
    assert layer._day_record.appended_total == appended
    assert int(layer.episode_writes.item()) == ring[0]
    assert torch.equal(layer.episode_inputs, ring[1])


def test_what_is_lived_after_a_rest_starts_a_new_day_behind_what_was_held():
    layer = _layer()
    rest(layer, keep=[1], toss=[2], unjudged="hold", config=LOOSE)
    layer(torch.randn(2, 3, D))
    assert layer._day_record.moments() == list(range(3, 13)) + [13]


def test_rest_refuses_what_it_should():
    with pytest.raises(ValueError, match="both kept and tossed"):
        rest(_layer(), keep=[1, 2], toss=[2], unjudged="hold")
    with pytest.raises(ValueError, match="in no layer's record"):
        rest(_layer(), keep=[999], toss=[], unjudged="hold")
    plain = PredictiveCodingLayer(D, D)
    with pytest.raises(ValueError, match="enable_day_record"):
        rest(plain, keep=[], toss=[], unjudged="hold")
    layer = _layer()
    before = _writes(layer)
    with freeze_plasticity(layer):
        with pytest.raises(RuntimeError, match="measurement"):
            rest(layer, keep=[1], toss=[], unjudged="hold")
    assert _same(before, _writes(layer)) and len(layer._day_record) == 12


def test_an_experience_that_no_longer_resolves_is_held_and_reported():
    layer = _layer()
    list(layer._day_record)[4].input_pattern.fill_(float("nan"))
    before = _writes(layer)
    report = rest(layer, keep=[5], toss=[], unjudged="toss", config=LOOSE).layers[""]
    assert report.not_finite == [5] and report.held == [5]
    assert _same(before, _writes(layer))
    assert layer._day_record.moments() == [5]


def test_the_config_refuses_values_that_would_not_be_rest():
    for bad in (
        dict(rate_factor=0.0), dict(passes=0), dict(home_fraction=1.5),
        dict(max_home_change=0.0),
    ):
        with pytest.raises(ValueError):
            RestConfig(**bad)


def test_the_report_lists_the_moments_and_is_logged(tmp_path):
    layer = _layer()
    log = tmp_path / "life" / "rests.jsonl"
    for keep in ([1, 2], [5]):
        rest(layer, keep=keep, toss=[3], unjudged="hold", config=LOOSE, log_path=log)
        layer._day_record.append(3, 0.1, torch.zeros(2), torch.zeros(D))
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(lines) == 2
    assert sorted(lines[0]["layers"][""]["integrated"]) == [1, 2]
    assert lines[1]["layers"][""]["integrated"] == [5]
    assert lines[0]["unjudged"] == "hold"
    assert lines[0]["config"]["home_fraction"] == 1.0


def test_a_moment_is_kept_across_the_whole_mind():
    torch.manual_seed(0)
    model = MultimodalPredictiveCodingLM(
        vocab_size=50, d_model=D, n_blocks=2, n_heads=2, ffn_expansion=1,
        max_seq_len=8, max_audio_tokens=8, max_vision_tokens=8,
        backward_pass_enabled=False,
    )
    for module in model.modules():
        if isinstance(module, PredictiveCodingLayer):
            module.salience_threshold = 0.0
    enable_day_record(model)
    g = torch.Generator().manual_seed(3)
    for _ in range(6):
        model.encode(text_tokens=torch.randint(0, 50, (1, 8), generator=g))
    report = rest(model, keep=[2, 4], toss=[1], unjudged="hold", config=LOOSE)
    assert set(report.layers) == set(day_records(model))
    for layer in report.layers.values():
        assert sorted(layer.integrated) == [2, 4]
        assert layer.tossed == [1]
        assert layer.held == [3, 5, 6]
    assert report.total("integrated") == 2 * len(report.layers)
