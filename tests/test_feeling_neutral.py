"""Teaching the felt-valence head what neutral is (2026-11-04).

The head trained only on felt moments, so it had no examples to separate them
from. Measured through Sanctuary's runner: after 100 painful events, ordinary
states that were never felt read -0.69, against -0.91 for the painful one. Brian
ruled that an unfelt moment should count as neutral.

Unfelt states are remembered, and taught as neutral alongside each felt moment.
The head reads a state relative to a slow average of unfelt states.
Pins:
- A mind that has felt nothing has learned nothing: the head stays exactly zero.
- After pain, the painful state reads unpleasant and unfelt states stay close to
  neutral, so the planner has a reason to prefer one over the other.
- A moment with a feeling never joins the neutral examples.
- With no room for examples, the old spread is what you get (characterization).
- Without felt valence, none of this exists.
- The examples survive a checkpoint.
- The metrics say how far the feeling has spread.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch

from tests.test_m9_external_actor import D, _build_trainer


@pytest.fixture
def tmp():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _unit_states(n: int, seed: int) -> torch.Tensor:
    """States shaped like real ones. Measured through Sanctuary's runner at this
    size, pooled states have norm ~7 and sit ~2.4 apart: mostly a shared
    direction plus a smaller difference. Fully random states are so far apart
    that the feeling barely spreads even without this fix, which would make these
    tests pass for the wrong reason."""
    shared = torch.randn(D, generator=torch.Generator().manual_seed(1234))
    shared = 7.0 * shared / shared.norm()
    generator = torch.Generator().manual_seed(seed)
    return shared + 0.3 * torch.randn(n, D, generator=generator)


def _live(trainer, s_next: torch.Tensor, valence=None) -> dict:
    ctx = {} if valence is None else {"valence": valence}
    s_t = torch.zeros(1, D)
    return trainer.observe_transition(s_t, torch.randn(1, D), s_next.reshape(1, D), ctx)


def _live_unfelt(trainer, states: torch.Tensor) -> None:
    for state in states:
        _live(trainer, state)


def test_a_mind_that_has_felt_nothing_has_learned_nothing(tmp):
    trainer = _build_trainer(tmp, felt_valence=True)
    _live_unfelt(trainer, _unit_states(40, seed=0))
    last = trainer.felt_valence.net[-1]
    assert not last.weight.any() and not last.bias.any()
    trainer.action_log.close()


def test_pain_stays_with_what_hurt(tmp):
    trainer = _build_trainer(tmp, felt_valence=True)
    _live_unfelt(trainer, _unit_states(16, seed=0))
    hurt = _unit_states(1, seed=1)
    for _ in range(100):
        _live(trainer, hurt, valence=-0.9)
    never_seen = _unit_states(16, seed=2)
    with torch.no_grad():
        assert trainer.felt_valence(hurt).item() < -0.6
        assert trainer.felt_valence(never_seen).mean().item() > -0.35
    trainer.action_log.close()


def test_the_planner_can_tell_what_hurt_from_what_did_not(tmp):
    trainer = _build_trainer(tmp, felt_valence=True)
    _live_unfelt(trainer, _unit_states(16, seed=0))
    hurt = _unit_states(1, seed=1)
    for _ in range(100):
        _live(trainer, hurt, valence=-0.9)
    fine = _unit_states(1, seed=2)
    s_t, a_t = torch.zeros(1, D), torch.randn(1, D)
    cost_hurt = trainer.efe.compute_g_realized(s_t=s_t, a_t=a_t, s_next=hurt)["c_avoid"].item()
    cost_fine = trainer.efe.compute_g_realized(s_t=s_t, a_t=a_t, s_next=fine)["c_avoid"].item()
    assert cost_hurt > cost_fine + 0.3
    trainer.action_log.close()


def test_a_felt_moment_is_not_a_neutral_example(tmp):
    trainer = _build_trainer(tmp, felt_valence=True)
    _live(trainer, _unit_states(1, seed=0), valence=-0.9)
    assert len(trainer._unfelt_states) == 0
    _live(trainer, _unit_states(1, seed=1))
    assert len(trainer._unfelt_states) == 1
    trainer.action_log.close()


def test_neutral_examples_keep_the_feeling_from_spreading(tmp):
    """Their effect shows over many events (-0.32 without, -0.09 with, at 1,000)."""
    spread = {}
    for examples in (0, 16):
        trainer = _build_trainer(tmp / str(examples), felt_valence=True, valence_neutral_examples=examples)
        _live_unfelt(trainer, _unit_states(16, seed=0))
        hurt = _unit_states(1, seed=1)
        for _ in range(400):
            _live(trainer, hurt, valence=-0.9)
        with torch.no_grad():
            spread[examples] = trainer.felt_valence(_unit_states(32, seed=2)).mean().item()
        trainer.action_log.close()
    assert spread[0] < spread[16] - 0.05
    assert spread[16] > -0.25


def test_the_head_reads_states_against_ordinary_life(tmp):
    trainer = _build_trainer(tmp, felt_valence=True)
    states = _unit_states(16, seed=0)
    _live_unfelt(trainer, states)
    assert torch.allclose(trainer.felt_valence.baseline, states.mean(dim=0), atol=1e-4)
    trainer.action_log.close()


def test_lingering_somewhere_does_not_make_it_ordinary_quickly(tmp):
    """The baseline fades slowly, so a stretch in one place barely moves it."""
    trainer = _build_trainer(tmp, felt_valence=True)
    head = trainer.felt_valence
    head.baseline_count.fill_(10_000)
    before = head.baseline.clone()
    elsewhere = _unit_states(1, seed=3) + 5.0
    for _ in range(50):
        head.observe_unfelt(elsewhere)
    moved = (head.baseline - before).norm() / (elsewhere - before).norm()
    assert moved < 0.06, "50 moments moved the baseline about 5% of the way, no more"
    trainer.action_log.close()


def test_the_baseline_survives_a_checkpoint(tmp):
    trainer = _build_trainer(tmp / "a", felt_valence=True)
    _live_unfelt(trainer, _unit_states(4, seed=0))
    state = trainer._extra_checkpoint_state()
    baseline = trainer.felt_valence.baseline.clone()
    trainer.action_log.close()

    restored = _build_trainer(tmp / "b", felt_valence=True)
    restored.load_m9_state(state)
    assert torch.equal(restored.felt_valence.baseline, baseline)
    assert restored.felt_valence.baseline_count.item() == 4
    restored.action_log.close()


def test_without_felt_valence_nothing_is_remembered(tmp):
    trainer = _build_trainer(tmp)
    _live_unfelt(trainer, _unit_states(4, seed=0))
    assert len(trainer._unfelt_states) == 0
    assert "m9_valence_unfelt_states" not in trainer._extra_checkpoint_state()
    trainer.action_log.close()


def test_the_examples_survive_a_checkpoint(tmp):
    trainer = _build_trainer(tmp / "a", felt_valence=True)
    states = _unit_states(4, seed=0)
    _live_unfelt(trainer, states)
    state = trainer._extra_checkpoint_state()
    trainer.action_log.close()

    restored = _build_trainer(tmp / "b", felt_valence=True)
    restored.load_m9_state(state)
    assert torch.equal(torch.stack(list(restored._unfelt_states)), states)
    assert restored._unfelt_states.maxlen == restored.m9_config.valence_neutral_examples
    restored.action_log.close()


def test_a_checkpoint_from_before_this_change_still_loads(tmp):
    trainer = _build_trainer(tmp / "a", felt_valence=True)
    state = trainer._extra_checkpoint_state()
    del state["m9_valence_unfelt_states"]
    trainer.action_log.close()

    restored = _build_trainer(tmp / "b", felt_valence=True)
    restored.load_m9_state(state)
    assert len(restored._unfelt_states) == 0
    restored.action_log.close()


def test_the_metrics_say_how_far_it_has_spread(tmp):
    trainer = _build_trainer(tmp, felt_valence=True)
    _live_unfelt(trainer, _unit_states(4, seed=0))
    out = _live(trainer, _unit_states(1, seed=1), valence=-0.9)
    assert out["valence_neutral_examples"] == 4
    assert "valence_unfelt_predicted" in out
    trainer.action_log.close()


def test_a_first_felt_moment_with_no_examples_still_teaches(tmp):
    trainer = _build_trainer(tmp, felt_valence=True)
    hurt = _unit_states(1, seed=0)
    out = _live(trainer, hurt, valence=-1.0)
    assert out["valence_neutral_examples"] == 0
    assert math.isnan(out["valence_unfelt_predicted"])
    with torch.no_grad():
        assert trainer.felt_valence(hurt).item() < 0
    trainer.action_log.close()
