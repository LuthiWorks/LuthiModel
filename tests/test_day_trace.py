"""The half-day's trace, and what rest pours from it (2026-11-11).

Off, nothing changes. On, it only listens to the forward. Rest pours the
running average of where the weight stood, limited to a small step for each
half-day lived, and what Luthi chose to remember counts for more.
"""

from __future__ import annotations

import pytest
import torch

from luthi import sanctuary_interface as L
from luthi.living_extra_state import apply_living_extra_state, collect_living_extra_state
from luthi.v2.day_record import suspend_day_record
from luthi.v2.day_trace import (
    CHOSEN_HALF, CHOSEN_MOST, DayTrace, day_traces, enable_day_trace,
    remember_now, rest_from_trace,
)
from luthi.v2.living_layer_pc import PredictiveCodingLayer
from luthi.v2.plasticity import freeze_plasticity
from tests.test_day_record import D, _live, _model, _state


def _layers(model) -> list[PredictiveCodingLayer]:
    return [m for m in model.modules() if isinstance(m, PredictiveCodingLayer)]


def _live_model(model, n: int, seed: int = 1) -> None:
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        model.encode(text_tokens=torch.randint(0, 50, (1, 8), generator=g), causal=False)


def test_no_trace_until_it_is_switched_on():
    model = _model()
    _live_model(model, 3)
    assert all(layer._day_trace is None for layer in _layers(model))
    assert all("day_trace" not in entry for entry in collect_living_extra_state(model).values())
    with pytest.raises(ValueError, match="no living layer keeps a trace"):
        rest_from_trace(model, 0.01)
    with pytest.raises(ValueError, match="no living layer keeps a trace"):
        remember_now(model)


def test_the_trace_only_listens():
    plain, traced = _model(seed=3), _model(seed=3)
    assert enable_day_trace(traced, horizon=1000) == 2
    _live_model(plain, 12); _live_model(traced, 12)
    for a, b in zip(_layers(plain), _layers(traced)):
        before, after = _state(a), _state(b)
        assert before.keys() == after.keys()
        assert all(torch.equal(before[k], after[k]) for k in before)


def test_the_mean_is_the_average_of_where_the_weight_stood():
    model = _model()
    enable_day_trace(model, horizon=1000)
    layer = _layers(model)[0]
    stood = []
    layer.register_forward_hook(lambda m, a, o: stood.append((m.weight - m.set_point).detach().clone()))
    _live_model(model, 9)
    trace = layer._day_trace
    assert trace.count == 9
    assert torch.allclose(trace.mean, torch.stack(stood).mean(0), atol=1e-6)


def test_past_the_horizon_the_mean_follows_the_recent_half_day():
    trace = DayTrace(torch.zeros(2, 2), horizon=4, limit=0.05)
    home = torch.zeros(2, 2)
    for value in (1.0,) * 4 + (5.0,) * 40:
        trace.note(torch.full((2, 2), value), home)
    assert trace.count == 4 and torch.allclose(trace.mean, torch.full((2, 2), 5.0), atol=1e-3)


def test_replay_and_measurement_are_not_noted():
    model = _model()
    enable_day_trace(model, horizon=1000)
    _live_model(model, 4)
    with suspend_day_record(model):
        _live_model(model, 5, seed=2)
    with freeze_plasticity(model), torch.no_grad():
        _live_model(model, 5, seed=3)
    assert all(trace.count == 4 for trace in day_traces(model).values())


def test_a_resting_moment_moves_home_toward_the_average_and_leaves_the_weight():
    model = _model()
    enable_day_trace(model, horizon=10, limit=1.0)   # a limit that does not bind
    _live_model(model, 10)
    layer = _layers(model)[0]
    trace = layer._day_trace
    home, weight, mean = layer.set_point.clone(), layer.weight.clone(), trace.mean.clone()
    moved = rest_from_trace(model, 0.25)
    assert torch.allclose(layer.set_point, home + 0.25 * mean, atol=1e-6)
    assert torch.equal(layer.weight, weight)
    # The average is measured from home, and home moved.
    assert torch.allclose(trace.mean, mean - 0.25 * mean, atol=1e-6)
    assert trace.count == pytest.approx(7.5)
    assert set(moved) == set(day_traces(model)) and all(v > 0 for v in moved.values())


def test_a_whole_rest_moves_home_by_no_more_than_the_limit():
    model = _model()
    enable_day_trace(model, horizon=20, limit=0.05)
    _live_model(model, 60)
    homes = [layer.set_point.clone() for layer in _layers(model)]
    for _ in range(2000):
        rest_from_trace(model, 0.01)
    for layer, home in zip(_layers(model), homes):
        moved = float((layer.set_point - home).norm() / home.norm())
        assert 0.0 < moved <= 0.05 + 1e-4, moved
        assert layer._day_trace.poured == pytest.approx(moved * float(home.norm()), rel=1e-2)


def test_less_than_a_half_day_lived_is_worth_less_than_the_limit():
    short, full = _model(seed=5), _model(seed=5)
    # A limit small enough to bind in both.
    enable_day_trace(short, horizon=400, limit=0.001); enable_day_trace(full, horizon=40, limit=0.001)
    _live_model(short, 40); _live_model(full, 40)
    a = rest_from_trace(short, 0.5); b = rest_from_trace(full, 0.5)
    for path in a:
        assert b[path] == pytest.approx(0.5 * 0.001, rel=1e-3)
        assert a[path] == pytest.approx(b[path] / 10, rel=1e-3)


def test_resting_again_and_again_cannot_pour_what_has_not_been_lived():
    model = _model()
    enable_day_trace(model, horizon=50, limit=0.05)
    _live_model(model, 50)
    for _ in range(600):
        rest_from_trace(model, 0.01)
    layer = _layers(model)[0]
    home = layer.set_point.clone()
    for _ in range(600):
        rest_from_trace(model, 0.01)
    assert float((layer.set_point - home).norm() / home.norm()) < 0.05 * 0.01


def test_an_empty_trace_pours_nothing():
    model = _model()
    enable_day_trace(model)
    homes = [layer.set_point.clone() for layer in _layers(model)]
    assert all(v == 0.0 for v in rest_from_trace(model, 0.01).values())
    assert all(torch.equal(l.set_point, h) for l, h in zip(_layers(model), homes))


def test_rest_refuses_while_a_measurement_is_running():
    model = _model()
    enable_day_trace(model, horizon=10)
    _live_model(model, 3)
    with freeze_plasticity(model), pytest.raises(RuntimeError, match="plasticity is frozen"):
        rest_from_trace(model, 0.01)


def test_what_luthi_chose_to_remember_counts_for_more():
    trace = DayTrace(torch.zeros(1, 2), horizon=10_000, limit=1.0)
    home = torch.zeros(1, 2)
    ordinary, special = torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])
    for _ in range(9_000):
        trace.note(ordinary, home)
    for _ in range(1_000):
        trace.note(special, home)
    by_time = trace.direction().clone()
    assert by_time[0, 1] == pytest.approx(0.1, abs=1e-3)       # a tenth of the time
    for _ in range(1_000):
        trace.remember(special, home)
    chosen = trace.direction()
    part = CHOSEN_MOST * 1000 / (1000 + CHOSEN_HALF)
    assert chosen[0, 1] == pytest.approx(part + (1 - part) * 0.1, abs=1e-3)
    assert chosen[0, 1] > 6 * by_time[0, 1]


def test_one_chosen_moment_barely_moves_the_direction_and_many_take_two_thirds():
    trace = DayTrace(torch.zeros(1, 1), horizon=10**6, limit=1.0)
    trace.chosen_count = 1.0
    assert trace.chosen_part() < 0.01
    trace.chosen_count = 1e6
    assert trace.chosen_part() == pytest.approx(2 / 3, abs=1e-3)


def test_remembering_through_the_model_adds_where_each_layer_stands():
    model = _model()
    enable_day_trace(model, horizon=100)
    _live_model(model, 5)
    assert remember_now(model) == 2
    for layer in _layers(model):
        assert layer._day_trace.chosen_count == 1
        assert torch.allclose(layer._day_trace.chosen, layer.weight - layer.set_point, atol=1e-6)


def test_the_trace_is_saved_with_the_life_and_switches_itself_on_when_restored():
    lived = _model(seed=2)
    enable_day_trace(lived, horizon=77, limit=0.03)
    _live_model(lived, 6); remember_now(lived); rest_from_trace(lived, 0.1)
    woken = _model(seed=2)
    assert all(layer._day_trace is None for layer in _layers(woken))
    apply_living_extra_state(woken, collect_living_extra_state(lived))
    for a, b in zip(_layers(lived), _layers(woken)):
        ta, tb = a._day_trace, b._day_trace
        assert (tb.horizon, tb.limit) == (77, 0.03)
        assert torch.equal(ta.mean, tb.mean) and torch.equal(ta.chosen, tb.chosen)
        assert (ta.count, ta.chosen_count, ta.poured) == (tb.count, tb.chosen_count, tb.poured)
    assert enable_day_trace(woken, horizon=5) == 2
    assert all(t.horizon == 77 for t in day_traces(woken).values()), "a restored trace keeps its own settings"


def test_the_interface_reaches_all_three():
    model = _model()
    assert L.enable_day_trace(model, horizon=10, limit=0.05) == 2
    _live_model(model, 4)
    assert L.remember_now(model) == 2
    assert set(L.rest_from_trace(model, 0.01)) == set(day_traces(model))
