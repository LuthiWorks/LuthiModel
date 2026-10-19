"""The LC-768 living-channel program family is a DECOMPOSITION: each arm
must differ from its comparison arm in exactly the variable the comparison
is about, and every arm must share the same loss / data / trainer settings
(2026-10-19; registry entry E3, docs/research/2026-10-19_the-living-channel-program.md).

A matched control that is not matched is worse than none -- it produces a
confident attribution to the wrong variable. These tests pin the matching.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

drv = pytest.importorskip("scripts.jepa_pilot_driver")

ARMS = ("lc768_dead", "lc768_living", "lc768_storeoff", "lc768_cov", "lc768_dead_x4")


def _diff(a: str, b: str) -> set[str]:
    ca, cb = drv.ARM_CONFIGS[a], drv.ARM_CONFIGS[b]
    keys = set(ca) | set(cb)
    return {k for k in keys if ca.get(k, "<absent>") != cb.get(k, "<absent>")}


def test_living_arm_is_the_ruled_768_configuration():
    assert drv.ARM_CONFIGS["lc768_living"] == drv.ARM_CONFIGS["probe_768_visreg"]


@pytest.mark.parametrize("a,b,expected", [
    ("lc768_storeoff", "lc768_living", {"block_episode_store"}),
    ("lc768_cov", "lc768_storeoff", {"write_rule"}),
    ("lc768_dead", "lc768_living", {"dead_ffn", "consolidation_enabled"}),
    ("lc768_dead_x4", "lc768_dead", {"ffn_expansion"}),
])
def test_each_comparison_differs_in_exactly_its_variable(a, b, expected):
    assert _diff(a, b) == expected


def test_all_arms_share_loss_data_and_trainer_settings():
    tables = {
        "ARM_VISREG": drv.ARM_VISREG, "ARM_VISREG_NUM_PROJ": drv.ARM_VISREG_NUM_PROJ,
        "ARM_SIGREG": drv.ARM_SIGREG, "ARM_SIGREG_PROJ": drv.ARM_SIGREG_PROJ,
        "ARM_FILELIST": drv.ARM_FILELIST, "ARM_COSINE": drv.ARM_COSINE,
        "ARM_TAPER": drv.ARM_TAPER, "ARM_DEEP_CADENCE": drv.ARM_DEEP_CADENCE,
        "ARM_GUARD_MIN_STEP": drv.ARM_GUARD_MIN_STEP,
        "ARM_LR_WARMUP": drv.ARM_LR_WARMUP, "ARM_GRAD_CLIP": drv.ARM_GRAD_CLIP,
    }
    for name, table in tables.items():
        vals = {table.get(arm, "<absent>") for arm in ARMS}
        assert len(vals) == 1, f"{name} differs across LC arms: {vals}"
    assert drv.ARM_GRAD_CLIP["lc768_dead"] == 5.0e5


def test_stages_are_768():
    for stage, arm in zip(range(60, 65), ("lc768_dead", "lc768_living",
                                          "lc768_storeoff", "lc768_cov",
                                          "lc768_dead_x4")):
        assert drv.STAGES[stage] == [(arm, 768)]
