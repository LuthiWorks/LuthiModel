"""The episode-store fix must be reachable from the model, and OFF by default.

The 2026-07-27 defect fix is an opt-in arm setting, like relative_trust before
it. Two properties matter and both have bitten this project before: a flag that
cannot be reached from the model level is a flag that never runs (the
learning-gain machinery sat unreachable from 2026-07-05 until run 3), and a
default that changes silently rewrites what completed families meant.
"""

import torch

from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM


def _model(**kw):
    return MultimodalPredictiveCodingLM(
        vocab_size=32,
        d_model=16,
        n_heads=2,
        n_blocks=2,
        max_seq_len=8,
        **kw,
    )


def _living_layers(model):
    return [m for name, m in model.named_modules()
            if name.endswith("living_ffn") and hasattr(m, "adaptive_episodes")]


def test_default_is_legacy_behaviour():
    layers = _living_layers(_model())
    assert layers, "no living layers found ??? plumbing test cannot verify anything"
    for layer in layers:
        assert layer.adaptive_episodes is False
        assert layer.adaptive_recall is False


def test_optin_reaches_every_living_layer():
    layers = _living_layers(_model(adaptive_episodes=True, adaptive_recall=True))
    assert layers
    for layer in layers:
        assert layer.adaptive_episodes is True
        assert layer.adaptive_recall is True
        # the anti-fossil buffers must exist and checkpoint with the model
        assert hasattr(layer, "episode_steps")
        assert hasattr(layer, "salience_window")
        assert layer.episode_steps.numel() == layer.num_episodes


def test_new_buffers_round_trip_through_state_dict():
    a = _model(adaptive_episodes=True)
    layer = _living_layers(a)[0]
    layer.episode_steps[0] = 1234
    layer.salience_window[0] = 0.5
    layer.episode_step_counter.fill_(99)
    b = _model(adaptive_episodes=True)
    b.load_state_dict(a.state_dict())
    lb = _living_layers(b)[0]
    assert int(lb.episode_steps[0].item()) == 1234
    assert float(lb.salience_window[0].item()) == 0.5
    assert int(lb.episode_step_counter.item()) == 99

