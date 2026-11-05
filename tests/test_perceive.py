"""`perceive`: one forward, no generation (2026-11-05, for Sanctuary's small path).

Pins:
- It gives the same state as `encode_state`, the kind the action layer trains on.
- It returns everything a moment needs: state, latents, and the inputs.
- The inputs it returns are its own copy.
- Perception is alive: one call changes the living weights, as one encode does.
- The learner can use what it returns.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from luthi.sanctuary_interface import (
    decode_action,
    encode_state,
    observe_transition,
    perceive,
    select_action,
)

from tests.test_m9_external_actor import D, SEQ, VOCAB, _build_trainer


@pytest.fixture
def tmp():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _tokens(seed: int = 0) -> torch.Tensor:
    return torch.randint(0, VOCAB, (1, SEQ), generator=torch.Generator().manual_seed(seed))


def test_it_gives_the_state_the_action_layer_was_trained_on(tmp):
    one, two = _build_trainer(tmp / "a"), _build_trainer(tmp / "b")
    tokens = _tokens()
    state = perceive(one.loss_module.online_encoder, text_tokens=tokens)
    expected = encode_state(two.loss_module.online_encoder, text_tokens=tokens)
    assert torch.equal(state.s_t, expected)
    assert state.s_t.shape == (1, D) and state.ctx_latents.shape == (1, SEQ, D)
    assert not state.s_t.requires_grad and not state.ctx_latents.requires_grad
    one.action_log.close()
    two.action_log.close()


def test_the_inputs_it_returns_are_its_own_copy(tmp):
    trainer = _build_trainer(tmp)
    tokens = _tokens()
    state = perceive(trainer.loss_module.online_encoder, text_tokens=tokens)
    assert torch.equal(state.context_obs["text_tokens"], tokens)
    tokens.zero_()
    assert state.context_obs["text_tokens"].any()
    trainer.action_log.close()


def test_perceiving_changes_the_living_weights_once(tmp):
    one, two = _build_trainer(tmp / "a"), _build_trainer(tmp / "b")
    tokens = _tokens()
    layer = one.loss_module.online_encoder.blocks[0].living_ffn
    before = layer.weight.clone()
    perceive(one.loss_module.online_encoder, text_tokens=tokens)
    encode_state(two.loss_module.online_encoder, text_tokens=tokens)
    assert not torch.equal(layer.weight, before), "the read is a write"
    assert torch.equal(layer.weight, two.loss_module.online_encoder.blocks[0].living_ffn.weight)
    one.action_log.close()
    two.action_log.close()


def test_a_whole_moment_runs_on_what_it_returns(tmp):
    trainer = _build_trainer(tmp)
    model = trainer.loss_module.online_encoder
    first = perceive(model, text_tokens=_tokens(0))
    selection = select_action(trainer, first.s_t, context_latents=first.ctx_latents)
    decode_action(trainer, selection.action, s_t=first.s_t, context_latents=first.ctx_latents)
    second = perceive(model, text_tokens=_tokens(1))
    out = observe_transition(
        trainer, first.s_t, selection.action, second.s_t,
        plan_snapshot=selection.plan_snapshot, context_obs=first.context_obs,
    )
    assert "lived_l_pred" in out, "the learner learned from the lived moment"
    trainer.action_log.close()


def test_a_model_without_encode_is_refused():
    with pytest.raises(AttributeError):
        perceive(torch.nn.Linear(2, 2), text_tokens=_tokens())
