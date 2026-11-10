"""The day's record (R2 of the 2026-08-14 wake/rest regime; built 2026-11-10).

Pins:
- It is off unless switched on, and switched on it only listens: a layer's
  weights, buffers and episode ring are bit-identical with and without it.
- It has no capacity. Moments the full ring turns away are still recorded.
- It records by the layer's own admission rule, so ``refractory_calls``
  bears on it on the adaptive path (Brian's unruled call, left as it is).
- A moment has the same name in every layer of a model.
- A measurement (frozen plasticity) records nothing.
- ``take`` and ``put_back`` never lose or duplicate an experience and keep
  the order lived.
- It survives a checkpoint, and restoring a recorded day into a layer that
  was not recording switches recording on.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from luthi.living_extra_state import (
    apply_living_extra_state,
    collect_living_extra_state,
)
from luthi.v2.day_record import (
    HELD,
    DayRecord,
    day_records,
    day_view,
    enable_day_record,
)
from luthi.v2.living_layer_pc import PredictiveCodingLayer
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM
from luthi.v2.plasticity import freeze_plasticity

D = 16


def _layer(seed: int = 0, **kw) -> PredictiveCodingLayer:
    torch.manual_seed(seed)
    kw.setdefault("salience_threshold", 0.0)   # every moment is admitted
    kw.setdefault("num_episodes", 4)
    return PredictiveCodingLayer(D, D, **kw)


def _live(layer, n: int, seed: int = 1) -> None:
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        layer(torch.randn(2, 3, D, generator=g))


def _model(seed: int = 0) -> MultimodalPredictiveCodingLM:
    torch.manual_seed(seed)
    model = MultimodalPredictiveCodingLM(
        vocab_size=50, d_model=D, n_blocks=2, n_heads=2, ffn_expansion=1,
        max_seq_len=8, max_audio_tokens=8, max_vision_tokens=8,
        backward_pass_enabled=False,
    )
    for module in model.modules():
        if isinstance(module, PredictiveCodingLayer):
            module.salience_threshold = 0.0   # every moment is admitted
    return model


def _state(module) -> dict:
    return {n: b.detach().clone() for n, b in module.named_buffers()} | {
        n: p.detach().clone() for n, p in module.named_parameters()
    }


def test_a_layer_keeps_no_record_until_it_is_switched_on():
    layer = _layer()
    _live(layer, 5)
    assert layer._day_record is None
    assert "day_record" not in collect_living_extra_state(layer).get("", {})


def test_recording_only_listens():
    plain, recording = _layer(), _layer()
    recording._day_record = DayRecord()
    _live(plain, 40)
    _live(recording, 40)
    a, b = _state(plain), _state(recording)
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert len(recording._day_record) > 0


def test_the_record_has_no_capacity():
    layer = _layer(num_episodes=4)
    layer._day_record = DayRecord()
    _live(layer, 60)
    assert int(layer.episode_count.item()) == 4, "the ring holds four"
    assert len(layer._day_record) == 60, "the record took every one"
    assert layer._day_record.moments() == list(range(1, 61))


def test_a_moment_the_full_ring_turns_away_is_still_recorded():
    layer = _layer(num_episodes=2, salience_threshold=0.1)
    layer._day_record = DayRecord()
    context, pattern = torch.zeros(layer.context_dim), torch.zeros(D)
    for salience in (5.0, 5.0):
        layer._sparse_step_count += 1
        layer._store_episode(context, salience, pattern)
    writes = int(layer.episode_writes.item())
    layer._sparse_step_count += 1
    layer._store_episode(context, 1.0, pattern)     # less salient than both
    assert int(layer.episode_writes.item()) == writes, "the ring refused it"
    assert layer._day_record.moments() == [1, 2, 3]
    layer._sparse_step_count += 1
    layer._store_episode(context, 0.01, pattern)    # below the threshold
    assert layer._day_record.moments() == [1, 2, 3], "not salient: not recorded"


def test_only_admitted_moments_are_recorded():
    layer = _layer(salience_threshold=1e9)   # nothing is salient enough
    layer._day_record = DayRecord()
    _live(layer, 20)
    assert len(layer._day_record) == 0


def test_the_refractory_floor_bears_on_the_record_on_the_adaptive_path():
    def recorded(refractory: int) -> list[int]:
        layer = _layer(
            adaptive_episodes=True, refractory_calls=refractory,
            surprise_k=0.0, salience_window_size=16,
        )
        layer._day_record = DayRecord()
        g = torch.Generator().manual_seed(4)
        context, pattern = torch.zeros(layer.context_dim), torch.zeros(D)
        for _ in range(400):
            layer._sparse_step_count += 1
            layer._store_episode(
                context, float(torch.rand(1, generator=g)), pattern,
            )
        # Past the warmup window, where the adaptive rule is in force.
        return [m for m in layer._day_record.moments() if m > 17]

    spaced, free = recorded(50), recorded(0)
    assert len(free) > len(spaced) > 0
    gaps = [b - a for a, b in zip(spaced, spaced[1:])]
    assert min(gaps) >= 50


def test_a_moment_has_one_name_in_every_layer():
    model = _model()
    assert enable_day_record(model) == len(day_records(model)) >= 2
    g = torch.Generator().manual_seed(3)
    for _ in range(6):
        model.encode(text_tokens=torch.randint(0, 50, (1, 8), generator=g))
    per_layer = [r.moments() for r in day_records(model).values()]
    assert all(m == per_layer[0] for m in per_layer)
    assert per_layer[0] == list(range(1, 7))
    view = day_view(model)
    assert list(view) == per_layer[0]
    assert set(view[1]["layers"]) == set(day_records(model))


def test_enabling_twice_keeps_the_day():
    model = _model()
    enable_day_record(model)
    model.encode(text_tokens=torch.randint(0, 50, (1, 8)))
    before = {p: len(r) for p, r in day_records(model).items()}
    enable_day_record(model)
    assert {p: len(r) for p, r in day_records(model).items()} == before


def test_a_measurement_records_nothing():
    layer = _layer()
    layer._day_record = DayRecord()
    _live(layer, 3)
    with freeze_plasticity(layer):
        _live(layer, 10)
    assert len(layer._day_record) == 3


def test_take_and_put_back_keep_every_experience_once_and_in_order():
    record = DayRecord()
    for m in range(1, 6):
        record.append(m, 0.1 * m, torch.zeros(2), torch.zeros(3))
    taken = record.take()
    assert len(record) == 0 and [e.moment for e in taken] == [1, 2, 3, 4, 5]
    record.append(6, 0.6, torch.zeros(2), torch.zeros(3))   # lived meanwhile
    record.put_back([taken[1], taken[3]])
    assert record.moments() == [2, 4, 6]
    assert record.appended_total == 6


def test_the_record_holds_its_own_copy_of_what_it_was_given():
    record = DayRecord()
    context, pattern = torch.ones(2), torch.ones(3)
    record.append(1, 0.5, context, pattern)
    context.zero_()
    pattern.zero_()
    [e] = list(record)
    assert e.context.sum() == 2 and e.input_pattern.sum() == 3


def test_the_day_survives_a_checkpoint_and_switches_recording_on():
    lived = _layer()
    lived._day_record = DayRecord()
    _live(lived, 12)
    list(lived._day_record)[2].state = HELD
    saved = collect_living_extra_state(lived)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "extra.pt"
        torch.save(saved, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)

    woken = _layer(seed=5)
    assert woken._day_record is None
    apply_living_extra_state(woken, loaded)
    a, b = list(lived._day_record), list(woken._day_record)
    assert [e.moment for e in a] == [e.moment for e in b]
    assert [e.state for e in a] == [e.state for e in b]
    assert all(torch.equal(x.context, y.context) for x, y in zip(a, b))
    assert all(torch.equal(x.input_pattern, y.input_pattern) for x, y in zip(a, b))
    assert [x.salience for x in a] == pytest.approx([y.salience for y in b])
    assert woken._day_record.appended_total == 12


def test_an_empty_day_survives_a_checkpoint_as_an_empty_day():
    lived = _layer()
    lived._day_record = DayRecord()
    woken = _layer(seed=5)
    apply_living_extra_state(woken, collect_living_extra_state(lived))
    assert woken._day_record is not None and len(woken._day_record) == 0


def test_a_checkpoint_without_a_day_leaves_a_layer_not_recording():
    lived = _layer()
    _live(lived, 4)
    woken = _layer(seed=5)
    apply_living_extra_state(woken, collect_living_extra_state(lived))
    assert woken._day_record is None


def test_a_ragged_record_is_refused():
    record = DayRecord()
    record.append(1, 0.5, torch.zeros(2), torch.zeros(3))
    state = record.state_dict()
    state["saliences"] = []
    with pytest.raises(ValueError, match="ragged"):
        DayRecord().load_state_dict(state)


def test_the_day_survives_a_trainer_checkpoint_and_resume(tmp_path):
    """The path a life is saved by: JEPATrainer._checkpoint and resume."""
    from tests.test_m9_external_actor import _build_trainer

    def trainer(name, seed):
        t = _build_trainer(tmp_path / name, seed=seed)
        for module in t.loss_module.online_encoder.modules():
            if isinstance(module, PredictiveCodingLayer):
                module.salience_threshold = 0.0
        return t

    one = trainer("a", 7)
    enable_day_record(one.loss_module.online_encoder)
    for _ in range(4):
        one.train_step("text", one.data_loader.next_batch("text"))
    lived = {
        p: (r.moments(), [e.input_pattern.clone() for e in r])
        for p, r in day_records(one.loss_module.online_encoder).items()
    }
    assert lived and all(m for m, _ in lived.values())
    one._checkpoint("test")
    path = sorted((one.run_dir / "checkpoints").glob("ckpt_*.pt"))[-1]

    two = trainer("b", 11)   # not recording until the checkpoint says so
    assert not day_records(two.loss_module.online_encoder)
    two.resume(path)
    woken = day_records(two.loss_module.online_encoder)
    assert set(woken) == set(lived)
    for p, (moments, patterns) in lived.items():
        assert woken[p].moments() == moments
        assert all(
            torch.equal(a, e.input_pattern) for a, e in zip(patterns, woken[p])
        )
    one.action_log.close()
    two.action_log.close()
