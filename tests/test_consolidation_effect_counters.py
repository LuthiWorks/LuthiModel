"""consolidation_fires counts TRIGGERS. These counters count EFFECT.

Regression pin for the 2026-08-14 audit finding. Both replay pathways
return the number of episodes they actually replayed and return 0
immediately on an empty store; both return values were discarded while
`_consolidation_fire_count` incremented unconditionally. On the 768x8
family (seed 97, 54,000 steps) blocks 0-4 each reported ~1,000
`consolidation_fires` having replayed zero episodes for the entire run,
because their episode stores were empty throughout.

The invariant these tests protect: it must be possible to tell, from the
metrics alone, whether consolidation DID anything. `noop_fires == fires`
means it did not.
"""

import torch

from luthi.v2.consolidation import consolidate_layer, consolidate_layer_attractor
from luthi.v2.living_layer_pc import PredictiveCodingLayer

D = 8


def _layer(**kw):
    base = dict(
        in_features=D, out_features=D, num_episodes=4, context_dim=4,
        consolidation_enabled=True,
        consolidation_window=8,
        consolidation_trigger_window=2,
        consolidation_threshold_factor=1e9,  # anything is sub-threshold -> fires
    )
    base.update(kw)
    return PredictiveCodingLayer(**base)


def _drive(layer, steps=60, scale=1.0):
    torch.manual_seed(0)
    x = torch.randn(4, D) * scale
    for _ in range(steps):
        layer(x)


def test_replay_pathways_return_zero_on_empty_store():
    """The contract the fire-site relies on. Both pathways, explicitly."""
    layer = _layer(salience_threshold=1e9)
    assert int(layer.episode_count.item()) == 0
    assert consolidate_layer(layer) == 0
    assert consolidate_layer_attractor(layer) == 0


def test_fires_without_episodes_are_counted_as_noops():
    """The seed-97 blocks 0-4 case: triggers fire, nothing is replayed."""
    layer = _layer(salience_threshold=1e9)  # nothing is ever salient enough
    _drive(layer)

    a = layer.aliveness()
    assert a["consolidation_fires"] > 0, "test did not trigger consolidation"
    assert int(layer.episode_count.item()) == 0
    # The finding: fires look healthy, effect is zero.
    assert a["consolidation_replayed_total"] == 0.0
    assert a["consolidation_noop_fires"] == a["consolidation_fires"]


def test_fires_with_episodes_replay_something():
    """The control: a store with episodes must show non-zero effect."""
    layer = _layer(salience_threshold=0.0)  # store aggressively
    _drive(layer)

    a = layer.aliveness()
    assert a["consolidation_fires"] > 0, "test did not trigger consolidation"
    assert int(layer.episode_count.item()) > 0, "test stored no episodes"
    assert a["consolidation_replayed_total"] > 0.0
    assert a["consolidation_noop_fires"] < a["consolidation_fires"]


def test_effect_counters_survive_resume():
    """They persist via living_extra_state, like the trigger count."""
    from luthi.living_extra_state import (
        apply_living_extra_state,
        collect_living_extra_state,
    )

    class _Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = _layer(salience_threshold=0.0)

    src = _Holder()
    _drive(src.layer)
    assert src.layer.aliveness()["consolidation_replayed_total"] > 0

    state = collect_living_extra_state(src)
    dst = _Holder()
    apply_living_extra_state(dst, state, source="test roundtrip")

    for k in ("consolidation_replayed_total", "consolidation_noop_fires",
              "consolidation_fires"):
        assert dst.layer.aliveness()[k] == src.layer.aliveness()[k], k


def test_pre_change_checkpoints_restore_without_the_new_keys():
    """Old living_extra_state has no effect counters; restore must not fail."""
    from luthi.living_extra_state import apply_living_extra_state

    class _Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = _layer(salience_threshold=0.0)

    dst = _Holder()
    apply_living_extra_state(
        dst, {"layer": {"consolidation_fire_count": 7}}, source="old ckpt",
    )
    assert dst.layer.aliveness()["consolidation_fires"] == 7.0
    assert dst.layer.aliveness()["consolidation_replayed_total"] == 0.0
