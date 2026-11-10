"""Run E2, E1 and E4 of the living-channel program over existing runs.

These three need no training. They read checkpoints that already exist and
never modify one (``docs/research/2026-10-19_the-living-channel-program.md``;
the order below is To-Do.md's). This script finds the runs, runs each
instrument on each, and copies the result files into the repository so they
can be committed and read.

    python scripts/run_no_training_experiments.py --runs-root E:\\runs --dry-run
    python scripts/run_no_training_experiments.py --runs-root E:\\runs

What it runs, in order:

  E2  living_write_probe          every selected living run
  E1  adaptation_eval             every selected living run, and every dead
                                  run as a control; document sets 0 and 1
  E4  per_block_probe             every selected 768-wide run, four ways:
                                  offset +1, offset -1, offset 0, causal

Which runs are selected, by the arm each run recorded:

  living   probe_768_visreg (the 768x8 family) and living_v5_* at 512
  dead     dead_4x          (E1's positive controls: exact_null must be true)

``--all`` selects every completed run under the roots instead, and
``--match`` narrows either selection to run names containing a string.

It is safe to stop and start again: a result that already exists is skipped
(``--redo`` runs it again). One failure does not stop the rest. The exit
code is 0 only if nothing failed.

Each E4 variant gets its own file name. Run by hand with no ``--out``, all
four write ``per_block_probe.json`` and each overwrites the one before.

Results are written next to each run, as the instruments always have, and
copied to ``runs_meta/living_channel_e1_e2_e4/<run name>/`` in this repo.
This script does not score the registry's frozen predictions. It gathers.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
COLLECT = REPO_ROOT / "runs_meta" / "living_channel_e1_e2_e4"


def find_runs(roots: list[Path]) -> list[dict]:
    """Every run directory under the roots: one that holds a
    ``pilot_result.json`` and at least one checkpoint."""
    seen, runs = set(), []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"runs root does not exist: {root}")
        for result_path in sorted(root.rglob("pilot_result.json")):
            run_dir = result_path.parent
            if run_dir in seen:
                continue
            seen.add(run_dir)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            runs.append({
                "dir": run_dir,
                "name": run_dir.name,
                "arm": str(result.get("arm", "")),
                "d_model": int(result.get("d_model", 0)),
                "seed": result.get("seed"),
                "outcome": str(result.get("outcome", "")),
                "has_checkpoint": any((run_dir / "checkpoints").glob("ckpt_*.pt")),
            })
    return runs


def is_dead(arm: str) -> bool:
    return "dead" in arm.split("_")


def select(runs: list[dict], everything: bool, match: str) -> list[dict]:
    out = []
    for run in runs:
        if not run["has_checkpoint"]:
            continue
        if match and match not in run["name"]:
            continue
        arm, d = run["arm"], run["d_model"]
        registered = (
            (arm == "probe_768_visreg" and d == 768)
            or (arm.startswith("living_v5") and d == 512)
            or arm == "dead_4x"
        )
        if everything or registered:
            out.append(run)
    # The 768x8 family first, then the rest, as To-Do.md orders them.
    out.sort(key=lambda r: (r["d_model"] != 768, r["arm"], str(r["seed"])))
    return out


def drop_copies(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    """One run per (arm, width, seed). A second directory holding the same
    run is a copy (an archive beside the original, say). Measured twice, its
    result would overwrite the first under the same name and it would count
    twice in E1's checkpoint count (Opus review, 2026-11-10, B5). Returns
    (kept, dropped); the first found is kept."""
    seen, kept, dropped = {}, [], []
    for run in runs:
        key = (run["arm"], run["d_model"], str(run["seed"]))
        if key in seen:
            dropped.append({**run, "same_as": seen[key]["dir"]})
        else:
            seen[key] = run
            kept.append(run)
    return kept, dropped


def reruns_beside_originals(runs: list[dict]) -> list[tuple[str, str]]:
    """Pairs where both an arm and its ``_rerun`` are selected at the same
    width and seed. Both are measured. Which one the registry means is not
    this script's call."""
    by = {(r["arm"], r["d_model"], str(r["seed"])): r["name"] for r in runs}
    return [
        (by[(arm[: -len("_rerun")], d, seed)], name)
        for (arm, d, seed), name in by.items()
        if arm.endswith("_rerun") and (arm[: -len("_rerun")], d, seed) in by
    ]


def plan(runs: list[dict], only: set[str]) -> list[dict]:
    """The jobs, in order. Each is one instrument on one run, writing one
    file."""
    jobs = []

    def job(exp, run, script, out_name, extra):
        jobs.append({
            "experiment": exp, "run": run["name"], "run_dir": run["dir"],
            "out": run["dir"] / out_name,
            "cmd": [
                sys.executable, str(SCRIPTS / script),
                "--run-dir", str(run["dir"]),
                "--out", str(run["dir"] / out_name), *extra,
            ],
        })

    living = [r for r in runs if not is_dead(r["arm"])]
    dead = [r for r in runs if is_dead(r["arm"])]
    if "e2" in only:
        for run in living:
            job("E2", run, "living_write_probe.py", "living_write_probe.json", [])
    if "e1" in only:
        for run in living + dead:
            common = ["--docs", "32", "--read", "4", "--score", "2"]
            job("E1", run, "adaptation_eval.py", "adaptation_eval.json", common)
            job("E1", run, "adaptation_eval.py", "adaptation_eval_phase1.json",
                common + ["--doc-phase", "1"])
    if "e4" in only:
        for run in [r for r in living if r["d_model"] == 768] or (
            living if not any(r["d_model"] == 768 for r in runs) else []
        ):
            job("E4", run, "per_block_probe.py", "per_block_probe_offset+1.json",
                ["--offset", "1"])
            job("E4", run, "per_block_probe.py", "per_block_probe_offset-1.json",
                ["--offset", "-1"])
            job("E4", run, "per_block_probe.py", "per_block_probe_offset0.json",
                ["--offset", "0"])
            job("E4", run, "per_block_probe.py", "per_block_probe_causal.json",
                ["--causal"])
    return jobs


def collect(job: dict, into: Path) -> None:
    target = into / job["run"]
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(job["out"], target / job["out"].name)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs-root", dest="roots", action="append", required=True,
                   help="A directory holding run directories. Repeatable.")
    p.add_argument("--only", default="e2,e1,e4",
                   help="Comma-separated subset of e2,e1,e4.")
    p.add_argument("--all", dest="everything", action="store_true",
                   help="Every completed run, not only the registered ones.")
    p.add_argument("--match", default="",
                   help="Only runs whose directory name contains this.")
    p.add_argument("--redo", action="store_true",
                   help="Run again even where a result exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="List what would run and stop.")
    p.add_argument("--collect-into", default=str(COLLECT),
                   help="Where result files are copied. '' to skip.")
    args = p.parse_args()

    only = {x.strip().lower() for x in args.only.split(",") if x.strip()}
    if not only <= {"e2", "e1", "e4"}:
        p.error("--only takes e2, e1 and e4")
    runs = find_runs([Path(r) for r in args.roots])
    chosen, copies = drop_copies(select(runs, args.everything, args.match))
    jobs = plan(chosen, only)

    print(f"[experiments] {len(runs)} run(s) found, {len(chosen)} selected:")
    for run in chosen:
        kind = "dead" if is_dead(run["arm"]) else "living"
        print(f"  {run['name']}  ({kind}, {run['d_model']}d, {run['outcome']})")
    for run in copies:
        print(f"[experiments] SKIPPED A COPY: {run['dir']} is the same run "
              f"(arm, width, seed) as {run['same_as']}. Only the first is "
              "measured. Pass one root, or --match, if the other is the one "
              "you mean.")
    for original, rerun in reruns_beside_originals(chosen):
        print(f"[experiments] NOTE: both {original} and {rerun} are selected "
              "and both will be measured. Say which the registry means when "
              "you report E1's checkpoint count.")
    if not chosen:
        print("[experiments] nothing selected. --all selects every run; "
              "--dry-run shows what is found. The driver's default output "
              "is the repo's runs\\jepa_pilot, not E:\\runs: try that as "
              "--runs-root.")
        return 2
    killed = [r["name"] for r in chosen if r["outcome"] != "completed"]
    if killed:
        print(f"[experiments] note: not 'completed': {', '.join(killed)}. "
              "They are run anyway; the registry decides whether they count.")
    print(f"[experiments] {len(jobs)} job(s)")

    into = Path(args.collect_into) if args.collect_into else None
    done, skipped, failed = [], [], []
    for i, job in enumerate(jobs, 1):
        label = f"{job['experiment']} {job['run']} -> {job['out'].name}"
        if job["out"].exists() and not args.redo:
            skipped.append(label)
            if into and not args.dry_run:
                collect(job, into)
            print(f"[{i}/{len(jobs)}] have  {label}")
            continue
        if args.dry_run:
            print(f"[{i}/{len(jobs)}] would {label}")
            continue
        print(f"[{i}/{len(jobs)}] run   {label}", flush=True)
        log = job["out"].with_suffix(".log")
        started = time.monotonic()
        with open(log, "wb") as f:
            code = subprocess.run(
                job["cmd"], cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT,
            ).returncode
        minutes = (time.monotonic() - started) / 60
        if code == 0 and job["out"].exists():
            done.append(label)
            if into:
                collect(job, into)
            print(f"          done in {minutes:.1f} min")
        else:
            failed.append(label)
            print(f"          FAILED (exit {code}); see {log}")

    if args.dry_run:
        return 0
    print(f"[experiments] ran {len(done)}, already had {len(skipped)}, "
          f"failed {len(failed)}")
    for label in failed:
        print(f"  failed: {label}")
    if into:
        print(f"[experiments] results copied to {into}")
        print("[experiments] to share them: git add runs_meta && git commit "
              "-m \"E1, E2, E4 results\" && git push")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
