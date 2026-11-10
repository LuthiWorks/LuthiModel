"""scripts/run_no_training_experiments.py: which runs it picks and what it
plans (2026-11-10). The instruments themselves need trained runs; the
selection and the plan are pure and are pinned here.

Pins:
- It finds a run by its ``pilot_result.json`` and needs a checkpoint.
- By default it selects the registered runs only: the 768x8 family, the 512
  v5 seeds, and ``dead_4x`` as E1's controls.
- E2 runs on living runs only. E1 runs on living and dead, both document
  sets. E4 runs on 768-wide living runs, four ways.
- Every job writes its own file. E4's four variants do not share one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_no_training_experiments import (  # noqa: E402
    find_runs,
    is_dead,
    plan,
    select,
)


def _run(root: Path, arm: str, d: int, seed: int, ckpt: bool = True) -> None:
    run = root / f"{arm}_{d}d_seed{seed}"
    (run / "checkpoints").mkdir(parents=True)
    (run / "pilot_result.json").write_text(json.dumps(
        {"arm": arm, "d_model": d, "seed": seed, "outcome": "completed"}
    ))
    if ckpt:
        (run / "checkpoints" / "ckpt_00000000.pt").write_bytes(b"")


def _world(tmp_path: Path) -> list[dict]:
    for seed in (46, 95, 97):
        _run(tmp_path, "probe_768_visreg", 768, seed)
    _run(tmp_path, "living_v5_4x_d4", 512, 44)
    _run(tmp_path, "dead_4x", 512, 44)
    _run(tmp_path, "probe_d8_wsig", 512, 46)             # not registered
    _run(tmp_path, "probe_768_visreg", 768, 11, ckpt=False)   # no checkpoint
    return find_runs([tmp_path])


def test_it_finds_runs_and_selects_the_registered_ones(tmp_path):
    runs = _world(tmp_path)
    assert len(runs) == 7
    chosen = select(runs, everything=False, match="")
    names = [r["name"] for r in chosen]
    assert len(chosen) == 5
    assert all("probe_768_visreg" in n for n in names[:3]), "the 768 family first"
    assert "probe_d8_wsig_512d_seed46" not in names
    assert "probe_768_visreg_768d_seed11" not in names, "no checkpoint, not run"
    assert len(select(runs, everything=True, match="")) == 6
    assert [r["name"] for r in select(runs, False, "seed95")] == [
        "probe_768_visreg_768d_seed95"
    ]


def test_dead_is_read_from_the_arm_name():
    assert is_dead("dead_4x") and is_dead("lc768_dead") and is_dead("lc768_dead_x4")
    assert not is_dead("probe_768_visreg") and not is_dead("living_v5_4x_d4")


def test_the_plan_is_e2_then_e1_then_e4_on_the_right_runs(tmp_path):
    jobs = plan(select(_world(tmp_path), False, ""), {"e2", "e1", "e4"})
    by = {}
    for j in jobs:
        by.setdefault(j["experiment"], []).append(j)
    assert [j["experiment"] for j in jobs] == sorted(
        (j["experiment"] for j in jobs), key=["E2", "E1", "E4"].index
    )
    assert len(by["E2"]) == 4 and not any("dead" in j["run"] for j in by["E2"])
    assert len(by["E1"]) == 10, "five runs, two document sets each"
    assert sum("dead_4x" in j["run"] for j in by["E1"]) == 2
    assert len(by["E4"]) == 12 and all("768d" in j["run"] for j in by["E4"])
    assert plan(select(_world_again(tmp_path), False, ""), {"e2"})[0]["experiment"] == "E2"


def _world_again(tmp_path: Path) -> list[dict]:
    return find_runs([tmp_path])


def test_every_job_writes_its_own_file(tmp_path):
    jobs = plan(select(_world(tmp_path), False, ""), {"e2", "e1", "e4"})
    outs = [str(j["out"]) for j in jobs]
    assert len(outs) == len(set(outs))
    one_run = [j["out"].name for j in jobs
               if j["experiment"] == "E4" and "seed46" in j["run"]]
    assert sorted(one_run) == [
        "per_block_probe_causal.json", "per_block_probe_offset+1.json",
        "per_block_probe_offset-1.json", "per_block_probe_offset0.json",
    ]
    for j in jobs:
        assert "--out" in j["cmd"] and str(j["out"]) in j["cmd"]
