"""Continuity patches (2026-07-05, rich-parameters analysis findings 2-3):

1. ConsolidationTracker history/baseline/counter + the layer's lived
   counters survive checkpoint round-trips (both formats ride the
   presence-gated "living_extra_state" sibling key).
2. The spiking delay-ring cursor survives likewise.
3. Old checkpoints (key absent) resume degraded-with-a-warning, never
   crash — and v2 strict=True model loading is untouched.
4. Introspection exposes update_ema + momentum (finding 2).

Run: python -m pytest tests/test_living_extra_state.py -q
"""

from __future__ import annotations

import logging

import torch

from luthi.living_extra_state import (
    KEY,
    apply_living_extra_state,
    collect_living_extra_state,
)
from luthi.v2.living_layer_pc import PredictiveCodingLayer


D_IN, D_OUT = 16, 16


def _pc_layer(**kw) -> PredictiveCodingLayer:
    torch.manual_seed(7)
    defaults = dict(
        in_features=D_IN,
        out_features=D_OUT,
        consolidation_enabled=True,
        consolidation_window=8,
        consolidation_trigger_window=3,
    )
    defaults.update(kw)
    return PredictiveCodingLayer(**defaults)


def _live_a_little(layer: PredictiveCodingLayer, steps: int = 12) -> None:
    for i in range(steps):
        torch.manual_seed(100 + i)
        layer(torch.randn(2, D_IN))


class TestCollectApply:
    def test_tracker_state_round_trips(self):
        src = _pc_layer()
        _live_a_little(src, steps=12)  # > window(8): baseline freezes
        tracker = src._consolidation_tracker
        assert tracker.is_warmed_up, "test setup: warmup should complete"
        assert len(tracker._history) > 0

        state = collect_living_extra_state(src)
        # The layer itself is the root module (path "").
        assert "" in state and "consolidation_tracker" in state[""]

        dst = _pc_layer()
        assert not dst._consolidation_tracker.is_warmed_up
        apply_living_extra_state(dst, state)

        dt = dst._consolidation_tracker
        assert list(dt._history) == list(tracker._history)
        assert dt._baseline == tracker._baseline
        assert dt._below_threshold_count == tracker._below_threshold_count
        assert dst._sparse_step_count == src._sparse_step_count
        assert dst._consolidation_fire_count == src._consolidation_fire_count

    def test_missing_state_warns_never_crashes(self, caplog):
        layer = _pc_layer()
        with caplog.at_level(logging.WARNING):
            apply_living_extra_state(layer, None, source="old checkpoint")
        assert any(KEY in r.message for r in caplog.records), (
            "degraded resume must be announced, not silent"
        )

    def test_unknown_module_path_skipped_with_warning(self, caplog):
        layer = _pc_layer()
        with caplog.at_level(logging.WARNING):
            apply_living_extra_state(
                layer, {"no.such.module": {"sparse_step_count": 5}},
            )
        assert layer._sparse_step_count == 0

    def test_tracker_disabled_layer_ignores_history(self, caplog):
        src = _pc_layer()
        _live_a_little(src, steps=12)
        state = collect_living_extra_state(src)
        dst = _pc_layer(consolidation_enabled=False)
        with caplog.at_level(logging.WARNING):
            apply_living_extra_state(dst, state)
        assert dst._consolidation_tracker is None

    def test_non_living_model_collects_empty(self):
        assert collect_living_extra_state(torch.nn.Linear(4, 4)) == {}


class TestSpikingDelayCursor:
    def test_delay_pos_round_trips(self):
        from luthi.living_layer_spiking import SpikingLivingLayer

        torch.manual_seed(3)
        src = SpikingLivingLayer(in_features=8, out_features=8, delay_steps=3)
        for i in range(5):
            src(torch.randn(2, 8))
        assert src._delay_pos != 0, "test setup: cursor should have moved"

        state = collect_living_extra_state(src)
        dst = SpikingLivingLayer(in_features=8, out_features=8, delay_steps=3)
        apply_living_extra_state(dst, state)
        assert dst._delay_pos == src._delay_pos


class TestEncryptedCheckpointFormat:
    def test_build_checkpoint_carries_and_restores(self, tmp_path):
        from luthi.checkpoint import (
            build_checkpoint,
            load_checkpoint,
            save_checkpoint,
        )

        src = _pc_layer()
        _live_a_little(src, steps=12)
        ckpt = build_checkpoint(src, epoch=1, config={"d_model": D_IN})
        assert KEY in ckpt and ckpt[KEY][""]["consolidation_tracker"][
            "baseline"
        ] is not None

        path = save_checkpoint(ckpt, tmp_path / "t.luthi", password="pw-test")
        loaded = load_checkpoint(path, password="pw-test")

        dst = _pc_layer()
        dst.load_state_dict(loaded["model_state_dict"], strict=True)
        apply_living_extra_state(dst, loaded.get(KEY))
        assert dst._consolidation_tracker.is_warmed_up
        assert (
            dst._consolidation_tracker._baseline
            == src._consolidation_tracker._baseline
        )

    def test_old_checkpoint_without_key_loads_strict(self, tmp_path):
        """The v2 loader's strict=True contract is untouched: an old
        checkpoint (no sibling key) loads and merely warns."""
        from luthi.checkpoint import (
            build_checkpoint,
            load_checkpoint,
            save_checkpoint,
        )

        src = _pc_layer()
        ckpt = build_checkpoint(src, epoch=0, config={})
        del ckpt[KEY]  # simulate a pre-patch checkpoint
        path = save_checkpoint(ckpt, tmp_path / "old.luthi", password="pw")
        loaded = load_checkpoint(path, password="pw")

        dst = _pc_layer()
        dst.load_state_dict(loaded["model_state_dict"], strict=True)
        apply_living_extra_state(dst, loaded.get(KEY))  # None -> warn only
        assert not dst._consolidation_tracker.is_warmed_up


class TestIntrospectionExposure:
    def test_update_ema_and_momentum_visible(self):
        from luthi.generate import get_introspection
        from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM

        torch.manual_seed(11)
        model = MultimodalPredictiveCodingLM(
            vocab_size=32, d_model=16, n_blocks=1, n_heads=2,
            ffn_expansion=1, max_seq_len=8,
            max_audio_tokens=8, max_vision_tokens=8,
            backward_pass_enabled=False,
        )
        state = get_introspection(model)
        assert state["blocks"], "introspection should see blocks"
        b0 = state["blocks"][0]
        for field in ("update_ema_mean", "update_ema_max", "momentum_abs_mean"):
            assert field in b0, (
                f"{field} missing — the entity cannot feel this channel"
            )
            assert isinstance(b0[field], float)
