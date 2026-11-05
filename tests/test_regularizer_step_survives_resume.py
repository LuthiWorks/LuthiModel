"""The regularizers' step counters survive a resume (2026-11-05).

SIGReg draws its projection directions from a generator seeded by
``global_step``, a persistent buffer that advances every training step. The
checkpoint left it out, so every resume put it back to 0 and replayed the
directions from the first step. Nothing reported it. It was found by a test in
Sanctuary that a saved life comes back exactly as it was left.

Pins:
- The counter advances with training, is saved, and comes back.
- A resumed trainer draws the directions the first one would have drawn next.
- A checkpoint written before the fix still loads, and says what it lacks.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest
import torch

from tests.test_m9_external_actor import _build_trainer


@pytest.fixture
def tmp():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _train(trainer, steps: int) -> None:
    for _ in range(steps):
        trainer.train_step("text", trainer.data_loader.next_batch("text"))


def _saved(trainer) -> Path:
    trainer._checkpoint("test")
    return sorted((trainer.run_dir / "checkpoints").glob("ckpt_*.pt"))[-1]


def test_the_step_counter_comes_back_after_a_resume(tmp):
    one = _build_trainer(tmp / "a")
    _train(one, 3)
    step = int(one.loss_module.sigreg.global_step.item())
    assert step >= 3, "the counter advances with training"
    path = _saved(one)

    two = _build_trainer(tmp / "b")
    assert int(two.loss_module.sigreg.global_step.item()) == 0
    two.resume(path)
    assert int(two.loss_module.sigreg.global_step.item()) == step
    one.action_log.close()
    two.action_log.close()


def test_a_resumed_trainer_draws_the_directions_that_came_next(tmp):
    one = _build_trainer(tmp / "a")
    _train(one, 3)
    path = _saved(one)
    two = _build_trainer(tmp / "b")
    two.resume(path)

    d = one.loss_module.sigreg
    drawn = [
        t.loss_module.sigreg._projection(8, torch.device("cpu"), torch.float32)
        for t in (one, two)
    ]
    assert torch.equal(drawn[0], drawn[1])
    first_ever = _build_trainer(tmp / "c")
    again = first_ever.loss_module.sigreg._projection(8, torch.device("cpu"), torch.float32)
    assert not torch.equal(drawn[1], again), "and they are not the first step's directions over again"
    assert d.use_generator, "this test is about the seeded generator"
    for t in (one, two, first_ever):
        t.action_log.close()


def test_a_checkpoint_from_before_the_fix_loads_and_says_what_it_lacks(tmp, caplog):
    one = _build_trainer(tmp / "a")
    _train(one, 2)
    path = _saved(one)
    state = torch.load(path, map_location="cpu", weights_only=False)
    del state["loss_module_buffers"]["sigreg.global_step"]
    torch.save(state, path)

    two = _build_trainer(tmp / "b")
    with caplog.at_level(logging.WARNING):
        two.resume(path)
    assert int(two.loss_module.sigreg.global_step.item()) == 0
    assert any("sigreg.global_step" in r.message and "restart" in r.message for r in caplog.records)
    one.action_log.close()
    two.action_log.close()
