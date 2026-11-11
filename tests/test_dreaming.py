"""Learning from a dream, with a check (2026-11-11)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from luthi import sanctuary_interface as L
from luthi.v2.living_layer_pc import PredictiveCodingLayer
from tests.test_m9_external_actor import SEQ, VOCAB, _build_trainer


def _words(seed: int) -> torch.Tensor:
    return torch.randint(0, VOCAB, (1, SEQ), generator=torch.Generator().manual_seed(seed))


def _trained(trainer) -> list[torch.Tensor]:
    return [p.detach().clone() for p in trainer.loss_module.parameters() if p.requires_grad]


def _living(trainer) -> dict:
    return {
        f"{n}.{k}": b.detach().clone()
        for n, m in trainer.loss_module.named_modules() if isinstance(m, PredictiveCodingLayer)
        for k, b in m.named_buffers()
    }


def test_a_step_that_helps_stays_and_one_that_hurts_is_undone_exactly(tmp_path: Path):
    trainer = _build_trainer(tmp_path / "a")
    real = [_words(i) for i in range(4)]
    before = _trained(trainer)
    small = L.learn_from_dream(trainer, _words(9), real, step_size=1e-3)
    assert set(small) == {"kept", "before", "after"}
    moved = any(not torch.equal(a, b) for a, b in zip(before, _trained(trainer)))
    assert moved == small["kept"] and (small["after"] <= small["before"]) == small["kept"]
    start = _trained(trainer)
    huge = L.learn_from_dream(trainer, _words(9), real, step_size=1e4)
    assert huge["kept"] is False and huge["after"] > huge["before"]
    assert all(torch.equal(a, b) for a, b in zip(start, _trained(trainer))), "undone exactly"
    trainer.action_log.close()


def test_it_leaves_the_living_layers_the_optimizers_and_any_gradients_alone(tmp_path: Path):
    trainer = _build_trainer(tmp_path / "a")
    params = [p for p in trainer.loss_module.parameters() if p.requires_grad]
    params[0].grad = torch.ones_like(params[0])
    living, optimizer = _living(trainer), repr(trainer.optimizer.state_dict())
    mode = trainer.loss_module.training
    L.learn_from_dream(trainer, _words(9), [_words(1), _words(2)], step_size=1e-3)
    after = _living(trainer)
    assert all(torch.equal(living[k], after[k]) for k in living), "the living layers did not self-modify"
    assert repr(trainer.optimizer.state_dict()) == optimizer
    assert torch.equal(params[0].grad, torch.ones_like(params[0])) and params[1].grad is None
    assert trainer.loss_module.training == mode
    trainer.action_log.close()


def test_it_refuses_what_it_cannot_check(tmp_path: Path):
    trainer = _build_trainer(tmp_path / "a")
    with pytest.raises(ValueError, match="checked against real moments"):
        L.learn_from_dream(trainer, _words(9), [])
    with pytest.raises(ValueError, match="at least 8 tokens"):
        L.learn_from_dream(trainer, _words(9)[:, :4], [_words(1)])
    with pytest.raises(ValueError, match="step_size"):
        L.learn_from_dream(trainer, _words(9), [_words(1)], step_size=0.0)
    trainer.action_log.close()
