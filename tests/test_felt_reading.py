"""How an experience affected the mind, read from its own prediction (2026-11-04).

Brian: "the world should just exist and Luthi has to interpret what they
experience as either good or bad depending on how the experience affects them."

Pins:
- Nothing outside the mind can say how something felt.
- A moment going worse than the last reads as negative, better as positive, and
  an ordinary one isn't felt at all.
- It doesn't depend on the error's scale, and a steady signal can't turn tiny
  changes into feelings.
- A sustained level of error stops being felt, and recovering reads as good.
- It is balanced: a mind is not pushed toward feeling bad.
- The error it reads is taken without dropout.
- What is read reaches the head and the account, and survives a checkpoint.
- Without felt valence, nothing is read.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import pytest
import torch

from luthi.sanctuary_interface import felt_now, felt_summary
from luthi.v2.m9.felt_valence import FeltReading

from tests.test_m9_external_actor import D, _build_trainer


@pytest.fixture
def tmp():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _settled(level: float = 1.0, noise: float = 0.01, n: int = 64) -> FeltReading:
    rng = random.Random(0)
    reading = FeltReading()
    for _ in range(n):
        reading.read(level * (1 + rng.gauss(0, noise)))
    return reading


def test_nothing_is_felt_before_it_knows_what_change_is_typical():
    reading = FeltReading(min_warmup=32)
    rng = random.Random(0)
    assert all(reading.read(10.0 * rng.random() + 0.1) is None for _ in range(33))


def test_going_worse_is_bad_and_going_better_is_good():
    assert _settled().read(3.0) < -0.5
    assert _settled().read(0.3) > 0.5


def test_an_ordinary_moment_is_not_felt():
    assert _settled().read(1.01) is None


def test_it_does_not_depend_on_the_scale_of_the_error():
    small, large = _settled(1.0), _settled(1000.0)
    assert small.read(3.0) == pytest.approx(large.read(3000.0), abs=1e-6)


def test_a_steady_signal_does_not_turn_tiny_changes_into_feelings():
    reading = FeltReading()
    for _ in range(64):
        reading.read(1.0)
    assert reading.read(1.05) is None


def test_a_sustained_level_stops_being_felt():
    reading = _settled()
    assert reading.read(3.0) is not None
    assert reading.read(3.0) is None


def test_recovering_from_a_bad_moment_reads_as_good():
    reading = _settled()
    assert reading.read(3.0) < 0
    assert reading.read(1.0) > 0


def test_a_mind_is_not_pushed_toward_feeling_bad():
    """Error has a floor and no ceiling, so comparing it with its median read 241
    bad moments to 7 good. A skewed sequence like that must come out balanced."""
    rng = random.Random(0)
    reading = FeltReading()
    felt = [reading.read(rng.lognormvariate(0.0, 0.8)) for _ in range(4000)]
    got = [f for f in felt if f is not None]
    bad, good = sum(f < 0 for f in got), sum(f > 0 for f in got)
    assert got and 0.8 < bad / good < 1.25
    assert abs(sum(got)) / len(felt) < 0.02


def test_steady_improvement_reads_as_good_on_balance():
    rng = random.Random(0)
    reading = FeltReading()
    felt = [
        reading.read(5.0 * (0.995 ** i) * rng.lognormvariate(0.0, 0.3))
        for i in range(2000)
    ]
    assert sum(f for f in felt if f is not None) > 0


def _observe(trainer, l_pred: float, s_next: torch.Tensor, **ctx) -> dict:
    trainer._run_lived_update = lambda *a, **k: {"lived_l_pred_clean": l_pred}
    trainer._run_corpus_replay = lambda: 0
    trainer._check_retention_gate = lambda: {}
    return trainer.observe_transition(
        torch.zeros(1, D), torch.randn(1, D), s_next, {"context_obs": {}, **ctx},
    )


def _state(seed: int) -> torch.Tensor:
    return torch.randn(1, D, generator=torch.Generator().manual_seed(seed))


def test_a_bad_moment_teaches_the_head_and_is_counted(tmp):
    trainer = _build_trainer(tmp, felt_valence=True)
    for i in range(40):
        _observe(trainer, 1.0 + 0.01 * (i % 3), _state(i))
    assert felt_now(trainer) is None
    assert not trainer.felt_valence.net[-1].weight.any(), "nothing felt, nothing learned"

    hurt = _state(100)
    out = _observe(trainer, 5.0, hurt)
    assert out["valence_felt"] < -0.5
    assert felt_now(trainer) == out["valence_felt"]
    with torch.no_grad():
        assert trainer.felt_valence(hurt).item() < 0
    summary = felt_summary(trainer)
    assert summary["read"] == 41 and summary["negative"] == 1 and summary["positive"] == 0
    trainer.action_log.close()


def test_the_world_cannot_say_how_something_felt(tmp):
    trainer = _build_trainer(tmp, felt_valence=True)
    for i in range(32):
        _observe(trainer, 1.0, _state(i), valence=-1.0)
    assert felt_summary(trainer)["negative"] == 0
    assert not trainer.felt_valence.net[-1].weight.any()
    trainer.action_log.close()


def test_the_error_it_reads_has_no_dropout_noise(tmp):
    trainer = _build_trainer(tmp, felt_valence=True)
    calls = []
    original = trainer.loss_module.compute_lived_loss

    def spy(*args, **kwargs):
        calls.append(trainer.loss_module.predictor.training)
        return {"l_pred": torch.tensor(1.0)}

    trainer.loss_module.compute_lived_loss = spy
    trainer._prediction_error_without_dropout({}, torch.zeros(1, D), torch.zeros(1, D), "text")
    assert calls == [False], "read with dropout off"
    assert trainer.loss_module.predictor.training, "the predictor goes back to training mode"
    trainer.loss_module.compute_lived_loss = original
    trainer.action_log.close()


def test_a_moment_without_a_lived_update_is_not_read(tmp):
    trainer = _build_trainer(tmp, felt_valence=True)
    trainer.observe_transition(torch.zeros(1, D), torch.randn(1, D), _state(0), {})
    assert felt_summary(trainer)["read"] == 0
    assert felt_now(trainer) is None
    trainer.action_log.close()


def test_without_felt_valence_nothing_is_read(tmp):
    trainer = _build_trainer(tmp)
    _observe(trainer, 1.0, _state(0))
    assert felt_summary(trainer)["read"] == 0
    assert "m9_felt_reading" not in trainer._extra_checkpoint_state()
    trainer.action_log.close()


def test_the_reading_survives_a_checkpoint(tmp):
    trainer = _build_trainer(tmp / "a", felt_valence=True)
    for i in range(40):
        _observe(trainer, 1.0 + 0.01 * (i % 3), _state(i))
    _observe(trainer, 5.0, _state(100))
    state = trainer._extra_checkpoint_state()
    expected = trainer.felt_reading.state_dict()
    counts = felt_summary(trainer)
    trainer.action_log.close()

    restored = _build_trainer(tmp / "b", felt_valence=True)
    restored.load_m9_state(state)
    assert restored.felt_reading.state_dict() == expected
    assert felt_summary(restored) == counts
    restored.action_log.close()
