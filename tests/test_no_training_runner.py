"""scripts/run_no_training_experiments.py: which runs it picks and what it
plans (2026-11-10). The instruments themselves need trained runs; the
selection and the plan are pure and are pinned here.

Since 2026-11-13 it also pins how a job is waited on: by its result, not by
its exit, because on Brian's card a job wrote a whole result and then spun
for an hour.

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
    drop_copies,
    find_runs,
    reruns_beside_originals,
    is_dead,
    plan,
    result_of,
    run_job,
    select,
    set_aside,
    stale,
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
    assert len(by["E1"]) == 11, (
        "five runs, two document sets each, and the July living run once "
        "more at the rate its taper ended on"
    )
    floor = [j for j in by["E1"] if j["out"].name == "adaptation_eval_taper_floor.json"]
    assert [j["run"] for j in floor] == ["living_v5_4x_d4_512d_seed44"]
    assert floor[0]["cmd"][-2:] == ["--rate-scale", "0.2"]
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


def test_a_copy_of_a_run_is_measured_once_and_said_so(tmp_path):
    """Opus review, 2026-11-10: with an archive beside the original, the
    second result overwrote the first and the run counted twice."""
    _run(tmp_path / "runs", "probe_768_visreg", 768, 46)
    _run(tmp_path / "archive" / "2026-08-20_visreg", "probe_768_visreg", 768, 46)
    _run(tmp_path / "runs", "probe_768_visreg", 768, 95)
    chosen = select(find_runs([tmp_path / "runs", tmp_path / "archive"]), False, "")
    kept, dropped = drop_copies(chosen)
    assert len(chosen) == 3 and len(kept) == 2 and len(dropped) == 1
    assert "archive" in str(dropped[0]["dir"])
    assert "runs" in str(dropped[0]["same_as"])
    outs = [str(j["out"]) for j in plan(kept, {"e2", "e1", "e4"})]
    assert len(outs) == len(set(outs))


def test_a_rerun_beside_its_original_is_named(tmp_path):
    _run(tmp_path, "living_v5_4x_d4", 512, 44)
    _run(tmp_path, "living_v5_4x_d4_rerun", 512, 44)
    _run(tmp_path, "living_v5_4x_d4_rerun", 512, 45)
    chosen = select(find_runs([tmp_path]), False, "")
    assert reruns_beside_originals(chosen) == [
        ("living_v5_4x_d4_512d_seed44", "living_v5_4x_d4_rerun_512d_seed44")
    ]


def test_a_smoke_run_is_not_taken_for_a_run_of_the_arm_it_smoked(tmp_path):
    """Found on Brian's archive, 2026-11-13: a smoke run records the arm it
    smoked, and was selected beside the registered family."""
    _run(tmp_path / "real", "probe_768_visreg", 768, 46)
    _run(tmp_path / "2026-08-11_smoke_768_b16", "probe_768_visreg", 768, 7)
    runs = find_runs([tmp_path])
    assert sorted(r["smoke"] for r in runs) == [False, True]
    assert [r["seed"] for r in select(runs, everything=False, match="")] == [46]
    assert len(select(runs, everything=True, match="")) == 2, "--all still takes it"


def test_two_roots_give_the_living_runs_and_their_controls_together(tmp_path):
    """Brian's runs sit in two places: the living family in one, the dead
    controls in the other, and the 768 family in both."""
    here, archive = tmp_path / "repo_runs", tmp_path / "archive"
    for seed in (46, 95, 97):
        _run(here, "probe_768_visreg", 768, seed)
        _run(archive, "probe_768_visreg", 768, seed)
    for seed in (42, 43, 44):
        _run(here, "living_v5_4x_d4", 512, seed)
        _run(archive, "dead_4x", 512, seed)
    kept, dropped = drop_copies(select(find_runs([here, archive]), everything=False, match=""))
    assert len(dropped) == 3 and all(r["arm"] == "probe_768_visreg" for r in dropped)
    arms = sorted({r["arm"] for r in kept})
    assert arms == ["dead_4x", "living_v5_4x_d4", "probe_768_visreg"]
    assert all(here in r["dir"].parents for r in kept if r["arm"] == "probe_768_visreg"), "the first root's copy is kept"



def test_a_run_that_recorded_its_taper_is_read_at_the_floor_too(tmp_path):
    """And a dead run never is: it has no living rate."""
    made = (
        ("probe_768_visreg", 97, True, "completed"),
        # Stopped at a sixth of its run. The taper begins at half, so this
        # run never left the full rate, and the floor is not its rate.
        ("probe_768_visreg", 46, True, "killed"),
        ("lc768_living", 97, False, "completed"),
        ("lc768_dead", 97, True, "completed"),
    )
    for arm, seed, taper, outcome in made:
        run = tmp_path / f"{arm}_768d_seed{seed}"
        (run / "checkpoints").mkdir(parents=True)
        (run / "checkpoints" / "ckpt_00000000.pt").write_bytes(b"")
        (run / "pilot_result.json").write_text(json.dumps({
            "arm": arm, "d_model": 768, "seed": seed, "outcome": outcome,
            "config": {"model_kwargs": {"d_model": 768}, "taper": taper},
        }))
    jobs = plan(find_runs([tmp_path]), {"e1"})
    floor = [j["run"] for j in jobs if j["out"].name == "adaptation_eval_taper_floor.json"]
    assert floor == ["probe_768_visreg_768d_seed97"]


def test_the_floor_is_the_trainers():
    from luthi.v2.jepa_runner import TaperConfig
    from luthi.v2.run_rebuild import TAPER_FLOOR

    assert TAPER_FLOOR == TaperConfig().floor


# -- waiting on a job by its result (2026-11-13) --------------------------

_WRITES = "import json, sys; open(sys.argv[1], 'w').write(json.dumps({'x': 1, 'reconstruction': {}}))"


def _job(tmp_path: Path, code: str, **kw) -> tuple[dict, Path]:
    out = tmp_path / "result.json"
    outcome = run_job(
        [sys.executable, "-c", code, str(out)], out, tmp_path / "result.log",
        poll=0.05, **{"exit_grace": 30.0, "timeout": 30.0, **kw},
    )
    return outcome, out


def test_a_job_that_writes_and_exits_is_done(tmp_path):
    outcome, out = _job(tmp_path, _WRITES)
    assert outcome["ok"] and outcome["killed"] is None and outcome["code"] == 0
    assert result_of(out) == {"x": 1, "reconstruction": {}}


def test_a_job_that_writes_and_then_spins_is_killed_and_counts(tmp_path):
    """What happened on Brian's card. The result is whole, so the work is
    done; the process is not waited on for an hour."""
    outcome, out = _job(tmp_path, _WRITES + "\nwhile True: pass", exit_grace=0.5)
    assert outcome["ok"] and outcome["killed"] == "did not exit after writing its result"
    assert outcome["seconds"] < 20
    assert result_of(out) is not None


def test_a_job_that_spins_with_no_result_fails_at_the_limit(tmp_path):
    outcome, out = _job(tmp_path, "while True: pass", timeout=0.5)
    assert not outcome["ok"] and outcome["killed"] == "wrote no result in time"
    assert not out.exists()


def test_a_job_that_exits_badly_or_writes_nothing_fails(tmp_path):
    outcome, _ = _job(tmp_path, "raise SystemExit(3)")
    assert not outcome["ok"] and outcome["code"] == 3
    outcome, _ = _job(tmp_path, "pass")
    assert not outcome["ok"] and outcome["code"] == 0


def test_a_half_written_result_is_not_a_result(tmp_path):
    """The poll can land while the job is writing. That is not yet done."""
    out = tmp_path / "result.json"
    out.write_text('{"x": 1, "reconstr', encoding="utf-8")
    assert result_of(out) is None
    assert result_of(tmp_path / "absent.json") is None


def test_a_result_that_does_not_say_what_was_reconstructed_is_not_kept_as_done(tmp_path):
    out = tmp_path / "result.json"
    out.write_text(json.dumps({"summary": {}}), encoding="utf-8")
    assert result_of(out) is None
    out.write_text(json.dumps({"summary": {}, "reconstruction": {}}), encoding="utf-8")
    assert result_of(out) is not None


def test_an_earlier_result_is_moved_aside_and_never_overwritten(tmp_path):
    out = tmp_path / "result.json"
    assert set_aside(out) is None
    for n in (1, 2):
        out.write_text(str(n), encoding="utf-8")
        assert set_aside(out) == tmp_path / f"result.earlier{n}.json"
        assert not out.exists()
    assert [(tmp_path / f"result.earlier{n}.json").read_text() for n in (1, 2)] == ["1", "2"]


def test_each_instrument_writes_what_was_reconstructed():
    """The runner only keeps a result that says so. The three scripts it
    runs must each write it, or every run would be measured again forever."""
    for name in ("living_write_probe.py", "adaptation_eval.py", "per_block_probe.py"):
        text = (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert '"reconstruction"' in text and "reconstruction(" in text, name


def test_e4_fits_on_the_runs_device_and_an_earlier_cpu_result_is_measured_again(tmp_path):
    """Twelve jobs took 14 to 16 hours fitted on the CPU. All twelve are to
    be fitted the same way, so a result from before is not kept as done."""
    jobs = [j for j in plan(select(_world(tmp_path), False, ""), {"e4"})]
    assert jobs and all(j["cmd"][-2:] == ["--fit-device", "run"] for j in jobs)
    job = jobs[0]
    assert stale(job) == ""                                  # nothing there yet
    job["out"].write_text(json.dumps({"reconstruction": {}, "settings": {"offset": 1}}))
    assert "before the probe could use the card" in stale(job)
    job["out"].write_text(json.dumps(
        {"reconstruction": {}, "settings": {"offset": 1, "fit_device": "cuda:0"}}
    ))
    assert stale(job) == ""
    job["out"].write_text(json.dumps({"settings": {"fit_device": "cpu"}}))
    assert "does not say what was reconstructed" in stale(job)
