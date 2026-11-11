"""Sight and sound through ``perceive`` (2026-11-11): an open channel into the
model's own encoders, in the same forward as the text."""

from pathlib import Path

import pytest
import torch

from luthi import sanctuary_interface as L
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM
from tests.test_m9_external_actor import SEQ, VOCAB, _build_trainer

# Small enough for the test model's 16-place encoders: 4 places of image and 3
# of sound.
IMAGE = (1, 3, 32, 32)
SOUND = (1, 8000)


def _inputs(seed: int = 0, text: int = 8):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randint(0, VOCAB, (1, text), generator=generator),
        torch.randn(IMAGE, generator=generator),
        torch.randn(SOUND, generator=generator),
    )


def test_text_alone_is_as_it_was(tmp_path: Path):
    a = _build_trainer(tmp_path / "a").loss_module.online_encoder
    b = _build_trainer(tmp_path / "b").loss_module.online_encoder
    tokens, _, _ = _inputs()
    state = L.perceive(a, text_tokens=tokens)
    expected = b.encode(text_tokens=tokens, causal=False)["latents"]
    assert torch.equal(state.ctx_latents, expected)
    assert set(state.context_obs) == {"text_tokens"}


def test_sight_and_sound_join_the_same_view(tmp_path: Path):
    model = _build_trainer(tmp_path / "a").loss_module.online_encoder
    tokens, image, sound = _inputs()
    state = L.perceive(model, text_tokens=tokens, image=image, audio_waveform=sound)
    assert state.ctx_latents.shape[1] == 4 + 3 + tokens.shape[1]
    assert set(state.context_obs) == {"text_tokens", "image", "audio_waveform"}
    assert torch.equal(state.context_obs["image"], image)
    assert state.context_obs["image"] is not image


def test_what_is_seen_changes_what_is_perceived(tmp_path: Path):
    tokens, image, sound = _inputs()
    states = []
    for other in (image, image + 1.0):
        model = _build_trainer(tmp_path / f"m{len(states)}").loss_module.online_encoder
        states.append(L.perceive(model, text_tokens=tokens, image=other, audio_waveform=sound).s_t)
    assert not torch.allclose(states[0], states[1])


def test_a_whole_moment_with_sight_and_sound_is_lived_and_learned_from(tmp_path: Path):
    trainer = _build_trainer(tmp_path / "a")
    model = trainer.loss_module.online_encoder
    eye_before = model.vision_encoder.patch_embed.weight.detach().clone()
    ear_before = model.audio_encoder.patch_embed.weight.detach().clone()
    previous, learned = None, {}
    for moment in range(4):
        tokens, image, sound = _inputs(seed=moment)
        state = L.perceive(model, text_tokens=tokens, image=image, audio_waveform=sound)
        if previous is not None:
            learned = L.observe_transition(
                trainer, previous[0], previous[1], state.s_t,
                plan_snapshot=previous[2], context_obs=previous[3],
            )
        selection = L.select_action(trainer, state.s_t, context_latents=state.ctx_latents)
        L.decode_action(trainer, selection.action, s_t=state.s_t, context_latents=state.ctx_latents)
        previous = (state.s_t, selection.action, selection.plan_snapshot, state.context_obs)
    assert "lived_l_pred" in learned
    # The eye and the ear learn from what is lived: nothing else trains them.
    assert not torch.equal(model.vision_encoder.patch_embed.weight, eye_before)
    assert not torch.equal(model.audio_encoder.patch_embed.weight, ear_before)
    trainer.action_log.close()


def test_the_contract_says_what_the_model_takes(tmp_path: Path):
    trainer = _build_trainer(tmp_path / "a")
    model = MultimodalPredictiveCodingLM(
        vocab_size=VOCAB, d_model=32, n_blocks=1, n_heads=2, ffn_expansion=1,
        max_seq_len=SEQ, max_audio_tokens=SEQ, max_vision_tokens=SEQ,
        vision_image_size=32, backward_pass_enabled=False,
    )
    contract = L.sense_contract(model, trainer)
    assert contract == {
        "image_size": 32, "image_tokens": 4, "max_vision_tokens": SEQ,
        "sample_rate": 16000,
        "samples_per_audio_token": 2560, "max_audio_tokens": SEQ,
        "view_limit": SEQ + 8 - 1,
    }
    trainer.action_log.close()


def test_the_contract_shows_a_model_that_cannot_place_its_own_image(tmp_path: Path):
    trainer = _build_trainer(tmp_path / "a")   # 224 image, 16 places
    contract = L.sense_contract(trainer.loss_module.online_encoder, trainer)
    assert contract["image_tokens"] == 196 > contract["max_vision_tokens"] == SEQ
    trainer.action_log.close()
