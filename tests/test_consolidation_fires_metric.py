"""Consolidation-fire count surfaced as a metric (Brian, 2026-07-18).

Requested after a consolidation event was identified from its forensic
signature alone (pred_frob step-jump + err_acc/update_ema flash with all
external channels silent, live 4x run, step ~28000). The counter already
existed on the layer and already persisted through checkpoints via
living_extra_state; this pins the new surfacing path:

  layer.aliveness()["consolidation_fires"]
    -> runner _substrate_health_metrics (SUM across blocks -- event
       counts, not levels)
    -> per-block substrate_blocks entries (heatmap row)
"""

from __future__ import annotations

import torch

from luthi.v2.jepa_runner import _substrate_health_metrics
from luthi.v2.living_layer_pc import PredictiveCodingLayer


D = 16


def test_aliveness_reports_fires_and_counts_up():
    layer = PredictiveCodingLayer(
        D, D, num_episodes=4, context_dim=8,
        salience_threshold=0.0,
        consolidation_enabled=True,
        consolidation_window=8,
        consolidation_trigger_window=4,
    )
    assert layer.aliveness()["consolidation_fires"] == 0.0
    layer._consolidation_fire_count = 3  # simulate three fires
    assert layer.aliveness()["consolidation_fires"] == 3.0


def test_runner_aggregates_fires_by_sum():
    aliveness = [
        {"prediction_norm": 1.0, "error_acc_mean": 0.01,
         "consolidation_fires": 2.0},
        {"prediction_norm": 2.0, "error_acc_mean": 0.02,
         "consolidation_fires": 5.0},
    ]
    m = _substrate_health_metrics(aliveness)
    assert m["consolidation_fires"] == 7.0, "event counts aggregate by SUM"
    assert m["pred_frob"] == 1.5, "levels still aggregate by mean"


def test_absent_fires_degrade_to_nan_not_crash():
    """Dead-arm blocks report {'dead_ffn': 1.0}; the aggregate must
    degrade honestly (nan) rather than fake a zero or crash."""
    m = _substrate_health_metrics([{"dead_ffn": 1.0}])
    import math
    assert math.isnan(m["consolidation_fires"])


def test_fires_survive_checkpoint_roundtrip():
    from luthi.living_extra_state import (
        apply_living_extra_state,
        collect_living_extra_state,
    )

    class _Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = PredictiveCodingLayer(
                D, D, num_episodes=4, context_dim=8,
                consolidation_enabled=True,
            )

    src = _Holder()
    src.layer._consolidation_fire_count = 11
    state = collect_living_extra_state(src)

    dst = _Holder()
    apply_living_extra_state(dst, state, source="test roundtrip")
    assert dst.layer._consolidation_fire_count == 11
    assert dst.layer.aliveness()["consolidation_fires"] == 11.0
