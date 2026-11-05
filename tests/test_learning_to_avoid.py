"""Learning to avoid what is unpleasant (2026-11-03; Brian: "We still need
Luthi to learn to avoid dangerous or unpleasant things";
docs/research/2026-11-03_learning-to-avoid.md).

Pins:
- Felt valence is off by default: nothing is built and every G is unchanged.
- When on, the head starts neutral: nothing is unpleasant until felt so.
- It learns from the world's reported valence, faster for strongly felt
  moments, and learns the felt state rather than everything at once.
- The planner does not count it: how to react is Luthi's call.
- It survives a checkpoint round trip.
- `keep_recent` moves each living home toward its weight by the given
  fraction.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from luthi.sanctuary_interface import encode_state, keep_recent
from luthi.v2.m9.felt_valence import FeltValenceHead
from luthi.v2.m9.runner import M9Config

from tests.test_m9_external_actor import D, SEQ, VOCAB, _build_trainer


def _trainer(run_dir: Path, **m9):
    return _build_trainer(run_dir, **m9)


@pytest.fixture
def tmp():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _states(trainer, seed: int = 0):
    torch.manual_seed(seed)
    enc = trainer.loss_module.online_encoder
    calm = encode_state(enc, text_tokens=torch.randint(0, VOCAB // 2, (1, SEQ)))
    hurt = encode_state(enc, text_tokens=torch.randint(VOCAB // 2, VOCAB, (1, SEQ)))
    return calm, hurt


def test_off_by_default_nothing_is_built(tmp):
    trainer = _trainer(tmp)
    assert trainer.felt_valence is None and trainer.valence_optimizer is None
    assert "m9_felt_valence_state_dict" not in trainer._extra_checkpoint_state()
    trainer.action_log.close()


def test_the_head_starts_neutral():
    head = FeltValenceHead(D)
    s = torch.randn(5, D)
    assert torch.equal(head(s), torch.zeros(5))
    assert torch.equal(head.unpleasantness(s), torch.zeros(5))


def _feel(trainer, s_t, s_next, valence, n=1):
    a_t = torch.randn(1, D)
    for _ in range(n):
        out = trainer._learn_felt_valence(s_next, valence)
    return out


def test_it_learns_what_felt_unpleasant_and_not_everything(tmp):
    trainer = _trainer(tmp, felt_valence=True)
    calm, hurt = _states(trainer)
    before = trainer.felt_valence.unpleasantness(hurt).item()
    out = _feel(trainer, calm, hurt, valence=-1.0, n=40)
    assert out["valence_felt"] == -1.0 and "valence_loss" in out
    with torch.no_grad():
        hurt_after = trainer.felt_valence.unpleasantness(hurt).item()
        calm_after = trainer.felt_valence.unpleasantness(calm).item()
    assert hurt_after > before + 0.3, "the felt state is learned as unpleasant"
    assert calm_after < hurt_after, "a different state is not tarred with it"
    trainer.action_log.close()


def test_strongly_felt_moments_teach_faster(tmp):
    deltas = {}
    for gain in (0.0, 4.0):
        trainer = _trainer(tmp / str(gain), felt_valence=True, valence_salience_gain=gain)
        calm, hurt = _states(trainer)
        _feel(trainer, calm, hurt, valence=-1.0, n=5)
        with torch.no_grad():
            deltas[gain] = trainer.felt_valence.unpleasantness(hurt).item()
        trainer.action_log.close()
    assert deltas[4.0] > deltas[0.0]


def test_what_was_learned_survives_a_checkpoint(tmp):
    trainer = _trainer(tmp / "a", felt_valence=True)
    calm, hurt = _states(trainer)
    _feel(trainer, calm, hurt, valence=-1.0, n=20)
    state = trainer._extra_checkpoint_state()
    learned = trainer.felt_valence(hurt).item()
    trainer.action_log.close()

    restored = _trainer(tmp / "b", felt_valence=True)
    restored.load_m9_state(state)
    assert restored.felt_valence(hurt).item() == pytest.approx(learned)
    restored.action_log.close()


def test_keep_recent_moves_each_home_by_the_fraction(tmp):
    trainer = _trainer(tmp)
    model = trainer.loss_module.online_encoder
    layers = [b.living_ffn for b in model.blocks]
    with torch.no_grad():
        for layer in layers:
            layer.weight.add_(0.1)
    gaps = [(l.weight - l.set_point).clone() for l in layers]
    homes = [l.set_point.clone() for l in layers]
    assert keep_recent(model, 0.25) == len(layers)
    for layer, gap, home in zip(layers, gaps, homes):
        assert torch.allclose(layer.set_point, home + 0.25 * gap)
    with pytest.raises(ValueError):
        keep_recent(model, 1.5)
    trainer.action_log.close()
