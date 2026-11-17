"""Run the jobs of brief 4 (2026-11-16): M1 and M2.

Results go to ``runs_meta/living_gradient_2026_11_17/<run>/<what>.json`` with
the log beside them. A result that is there and whole is kept. Each job is
waited on by its result, with the no-training runner's ``run_job``. If a job
fails, nothing further is started and the exit code is 1.

    python redteam/terminal_fable_2026_11_17/run_gradient_jobs.py --plan m1
    python redteam/terminal_fable_2026_11_17/run_gradient_jobs.py --plan m2 --dry-run

Three jobs at a time is the ceiling on Brian's machine (memory, not the
card); the default is two.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_no_training_experiments import result_of, run_job  # noqa: E402

RUNS = REPO_ROOT / "runs" / "jepa_pilot"
INTO = REPO_ROOT / "runs_meta" / "living_gradient_2026_11_17"
HERE = Path(__file__).resolve().parent

JULY42 = "living_v5_4x_d4_512d_seed42"
P97 = "probe_768_visreg_768d_seed97"
# Amendment 5's layout: 64 windows read, 8 scored; 48 documents a set at
# 512 and 64 at 768.
LAYOUT = {JULY42: ["--docs", "48", "--read", "64", "--score", "8"],
          P97: ["--docs", "64", "--read", "64", "--score", "8"]}
PHASES = [("", []), ("_phase1", ["--doc-phase", "1"])]
EVAL = str(REPO_ROOT / "scripts" / "adaptation_eval.py")
M1 = str(HERE / "probe_helpful_direction.py")


def m1() -> list[tuple[str, str, str, list[str]]]:
    return [(run, f"helpful_direction{suffix}", M1, LAYOUT[run] + ["--windows", "4", "64"] + phase)
            for suffix, phase in PHASES for run in (JULY42, P97)]


def m2() -> list[tuple[str, str, str, list[str]]]:
    """The gradient control restricted to the living weights: three rates,
    each once, both document sets; and the whole-model control at its frozen
    recipe (1e-5) in the same layout for the same documents."""
    jobs = []
    for suffix, phase in PHASES:
        for run in (JULY42, P97):
            for rate in ("1e-5", "1e-4", "1e-3"):
                jobs.append((run, f"e1_read64_living_gradient_{rate}{suffix}", EVAL,
                             LAYOUT[run] + phase + ["--read-by-gradient", rate,
                                                    "--gradient-steps", "1",
                                                    "--gradient-read-what", "living"]))
            jobs.append((run, f"e1_read64_gradient_1e-5{suffix}", EVAL,
                         LAYOUT[run] + phase + ["--read-by-gradient", "1e-5",
                                                "--gradient-steps", "1"]))
    return jobs


PLANS = {"m1": m1, "m2": m2}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", choices=sorted(PLANS), required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--job-timeout-minutes", type=float, default=90.0)
    ap.add_argument("--exit-grace-seconds", type=float, default=30.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.workers > 3:
        raise SystemExit("three jobs at a time is the ceiling on this machine")

    todo = []
    for run, what, script, flags in PLANS[args.plan]():
        out = INTO / run / f"{what}.json"
        cmd = [sys.executable, "-u", script, "--run-dir", str(RUNS / run),
               "--out", str(out)] + flags
        job = {"label": f"{run} -> {what}.json", "out": out, "cmd": cmd}
        have = result_of(out) is not None
        print(f"{'have ' if have else 'would' if args.dry_run else 'queue'} {job['label']}"
              f"    {' '.join(flags)}", flush=True)
        if not have:
            todo.append(job)
    if args.dry_run or not todo:
        return 0

    lock = threading.Lock()
    stop = threading.Event()
    done, failed = [], []
    began = time.monotonic()

    def run_one(job: dict) -> None:
        if stop.is_set():
            return
        job["out"].parent.mkdir(parents=True, exist_ok=True)
        if job["out"].exists():
            raise RuntimeError(f"{job['out']} exists and is not a whole result")
        outcome = run_job(job["cmd"], job["out"], job["out"].with_suffix(".log"),
                          exit_grace=args.exit_grace_seconds,
                          timeout=args.job_timeout_minutes * 60, heartbeat=1e9)
        with lock:
            at = time.strftime("%H:%M:%S")
            if outcome["ok"]:
                done.append(job["label"])
                print(f"[{at}] done   {job['label']}  {outcome['seconds'] / 60:.1f} min"
                      f"{'  (did not exit; killed)' if outcome['killed'] else ''}", flush=True)
            else:
                failed.append(job["label"])
                stop.set()
                print(f"[{at}] FAILED {job['label']}  after {outcome['seconds'] / 60:.1f} min "
                      f"({outcome['killed'] or 'exit ' + str(outcome['code'])}); "
                      f"see {job['out'].with_suffix('.log')}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_one, todo))
    never = len(todo) - len(done) - len(failed)
    print(f"ran {len(done)}, failed {len(failed)}, not started {never}; "
          f"{(time.monotonic() - began) / 60:.1f} min in all", flush=True)
    return 1 if failed or never else 0


if __name__ == "__main__":
    sys.exit(main())
