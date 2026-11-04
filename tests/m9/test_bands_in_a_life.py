"""The activity and delta-s bands during a life (2026-11-04).

The bands calibrate what counts as silent. They were only ever fed from the
corpus training step and were not checkpointed, so a life loaded them empty and
they stayed empty: `bands_armed` read False on all 1,500 cycles of a measured
life, and `decode_selection` never used the band test its docstring describes.

Pins:
- Each decoded selection feeds the bands once, so a life warms them.
- They are fed the same quantity the corpus path feeds them and the same one
  `decode_selection` then classifies: activity of the predicted next state
  under the selected action.
- A warm band can call a channel silent.
- They survive a checkpoint round trip.
- A checkpoint from before this change still loads, with empty bands.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from luthi.sanctuary_interface import decode_action, encode_state, select_action
from luthi.v2.m9.s_t import compute_s_t

from tests.test_m9_external_actor import SEQ, VOCAB, _build_trainer


@pytest.fixture
def tmp():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _state(trainer, seed: int):
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, VOCAB, (1, SEQ), generator=generator)
    latents = encode_state(trainer.loss_module.online_encoder, text_tokens=tokens, pool=False)
    return compute_s_t(latents), latents


def _cycle(trainer, seed: int = 0) -> dict:
    s_t, latents = _state(trainer, seed)
    selection = select_action(trainer, s_t, context_latents=latents, budget=0)
    return decode_action(trainer, selection.action, s_t=s_t, context_latents=latents)


def test_a_life_warms_the_bands(tmp):
    trainer = _build_trainer(tmp)
    warmup = trainer.activity_bands.config.min_warmup
    assert not any(trainer.activity_bands.armed_per_modality().values())
    assert not trainer.delta_s_band.is_warm()

    for seed in range(warmup):
        decoded = _cycle(trainer, seed)

    assert all(decoded["bands_armed"].values()), "the band armed within warmup cycles"
    assert trainer.activity_bands.k_m9_5_armed()
    assert trainer.delta_s_band.is_warm()
    trainer.action_log.close()


def test_each_decode_feeds_each_band_once(tmp):
    trainer = _build_trainer(tmp)
    for count in (1, 2, 3):
        _cycle(trainer, count)
        for band in trainer.activity_bands._bands.values():
            assert len(band.values) == count
        assert len(trainer.delta_s_band.band.values) == count
    trainer.action_log.close()


def test_the_band_sees_what_it_then_classifies(tmp):
    """Activity of the predicted next state under the selected action, as in
    the corpus path, rather than some other scale it would misjudge."""
    trainer = _build_trainer(tmp)
    s_t, latents = _state(trainer, 0)
    action = select_action(trainer, s_t, context_latents=latents, budget=0).action
    decode_action(trainer, action, s_t=s_t, context_latents=latents)

    trainer.efe.predictor.eval()
    with torch.no_grad():
        a = action.reshape(1, -1)
        ctx = latents if latents.dim() == 3 else latents.unsqueeze(0)
        tgt = trainer._target_positions_for(ctx.shape[1])
        s_hat = trainer.efe.predict_next(ctx, tgt, a)
        activity = trainer.decoders.activity(s_hat)
        delta_s = trainer.delta_s_module.compute(s_t.reshape(1, -1), s_hat)
    trainer.efe.predictor.train()

    for modality, band in trainer.activity_bands._bands.items():
        assert band.values[-1] == pytest.approx(activity[modality].item())
    assert trainer.delta_s_band.band.values[-1] == pytest.approx(delta_s.item())
    trainer.action_log.close()


def test_the_reading_has_no_dropout_noise(tmp):
    """Dropout gave two identical readings a 41% relative difference on this
    size of model (found 2026-11-04)."""
    trainer = _build_trainer(tmp)
    assert trainer.efe.predictor.training, "the fixture must have dropout on to test this"
    s_t, latents = _state(trainer, 0)
    action = select_action(trainer, s_t, context_latents=latents, budget=0).action

    rng_before = torch.get_rng_state()
    decode_action(trainer, action, s_t=s_t, context_latents=latents)
    first = {m: band.values[-1] for m, band in trainer.activity_bands._bands.items()}
    decode_action(trainer, action, s_t=s_t, context_latents=latents)
    second = {m: band.values[-1] for m, band in trainer.activity_bands._bands.items()}

    assert first == second, "the same action in the same state reads the same"
    assert torch.equal(torch.get_rng_state(), rng_before), "no random draw was taken"
    assert trainer.efe.predictor.training, "the predictor goes back to training mode"
    trainer.action_log.close()


def test_a_warm_band_can_call_a_channel_silent(tmp):
    trainer = _build_trainer(tmp)
    for band in trainer.activity_bands._bands.values():
        for _ in range(band.window):
            band.push(1e6)
    decoded = _cycle(trainer)
    assert all(decoded["bands_armed"].values())
    assert not any(decoded["active"].values()), "activity far below its band is silent"
    trainer.action_log.close()


def _checkpoint(trainer) -> Path:
    trainer._checkpoint("test")
    return sorted((trainer.run_dir / "checkpoints").glob("ckpt_*.pt"))[-1]


def test_the_bands_survive_a_checkpoint(tmp):
    trainer = _build_trainer(tmp / "a")
    for seed in range(trainer.activity_bands.config.min_warmup):
        _cycle(trainer, seed)
    path = _checkpoint(trainer)
    expected_activity = {
        m: list(band.values) for m, band in trainer.activity_bands._bands.items()
    }
    expected_delta_s = list(trainer.delta_s_band.band.values)
    trainer.action_log.close()

    resumed = _build_trainer(tmp / "b")
    resumed.resume(path)
    assert {
        m: list(band.values) for m, band in resumed.activity_bands._bands.items()
    } == expected_activity
    assert list(resumed.delta_s_band.band.values) == expected_delta_s
    assert resumed.activity_bands.k_m9_5_armed() and resumed.delta_s_band.is_warm()
    resumed.action_log.close()


def test_a_resumed_band_keeps_its_window(tmp):
    trainer = _build_trainer(tmp / "a")
    for seed in range(40):
        _cycle(trainer, seed)
    path = _checkpoint(trainer)
    trainer.action_log.close()

    resumed = _build_trainer(tmp / "b")
    resumed.resume(path)
    for band in resumed.activity_bands._bands.values():
        assert band.values.maxlen == band.window
        assert len(band.values) == band.window
    assert resumed.delta_s_band.band.values.maxlen == resumed.delta_s_band.band.window
    resumed.action_log.close()


def test_a_checkpoint_from_before_this_change_still_loads(tmp):
    trainer = _build_trainer(tmp / "a")
    path = _checkpoint(trainer)
    trainer.action_log.close()
    state = torch.load(path, map_location="cpu", weights_only=False)
    del state["m9_activity_bands"], state["m9_delta_s_band"]

    resumed = _build_trainer(tmp / "b")
    resumed.load_m9_state(state)
    assert not any(resumed.activity_bands.armed_per_modality().values())
    assert not resumed.delta_s_band.is_warm()
    resumed.action_log.close()


def test_a_checkpoint_missing_a_modality_refuses(tmp):
    """A partial record would leave one channel uncalibrated without saying so."""
    trainer = _build_trainer(tmp)
    with pytest.raises(KeyError):
        trainer.activity_bands.load_state_dict({"text": [0.1]})
    trainer.action_log.close()
