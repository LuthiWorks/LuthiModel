"""Run the adaptation_eval jobs of the 2026-11-16 card brief (jobs 3 and 4).

The no-training runner (``scripts/run_no_training_experiments.py``) knows
none of these jobs, and they are exploratory, so they are listed here and
not taught to it. Each job is waited on by its result and not by its exit,
with the runner's own ``run_job``: ``adaptation_eval.py`` writes a whole
result and then does not exit on Brian's card (38 of 38 on 2026-11-13).

Results go to ``runs_meta/living_channel_dose_2026_11_16/<run>/<what>.json``
with the log beside them. A result that is already there and whole is kept.

The first job is run alone and timed. The rest run ``--workers`` at a time:
a read is one window per forward, so one job leaves most of the card idle.
Times for those are therefore times with the card shared.

If a job fails, or writes no result within ``--job-timeout-minutes``,
nothing further is started and the exit code is 1.

    python redteam/terminal_fable_2026_11_16/run_dose_jobs.py --plan ladder
    python redteam/terminal_fable_2026_11_16/run_dose_jobs.py --plan switches --dry-run
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
INTO = REPO_ROOT / "runs_meta" / "living_channel_dose_2026_11_16"

JULY42 = "living_v5_4x_d4_512d_seed42"
P97 = "probe_768_visreg_768d_seed97"
OTHER_FINISHED = [
    "living_v5_4x_d4_512d_seed43", "living_v5_4x_d4_512d_seed44",
    "living_v5_4x_d4_512d_seed45", "living_v5_4x_d4_512d_seed46",
    "living_v5_4x_d4_rerun_512d_seed44",
]

FIFTH = ["--rate-scale", "0.2"]
# Job 3's rungs: name, flags.
RUNGS = [
    ("read64", ["--docs", "32", "--read", "64", "--score", "2"]),
    ("read256", ["--docs", "30", "--read", "256", "--score", "2"]),
    ("read256x4", ["--docs", "30", "--read", "256", "--score", "2", "--repeats", "4"]),
]
REGISTERED = ["--docs", "32", "--read", "4", "--score", "2"]
# What each way of reaching the score is switched off by.
NO_WRITE = ["--rate-scale", "0"]
NO_PULL = ["--homeostatic-decay", "0"]
NO_CONSOLIDATION = ["--no-read-consolidation"]
NO_EPISODES = ["--no-read-episode-writes"]


def ladder(runs: list[str]) -> list[tuple[str, str, list[str]]]:
    jobs = []
    for name, flags in RUNGS:                 # rung by rung, so the ladder stops low
        for run in runs:
            jobs.append((run, f"e1_{name}", flags))
            jobs.append((run, f"e1_{name}_rate0.2", flags + FIFTH))
    return jobs


def switches() -> list[tuple[str, str, list[str]]]:
    jobs = []
    for run in (JULY42, P97):
        # 1. What carries the change at the registered dose (four windows).
        #    The brief asked for the pull off on July seed 42. The rest take
        #    seconds each and say which of the four ways it is.
        for what, flags in [
            ("pull_off", NO_PULL),
            ("write_off", NO_WRITE),
            ("episodes_off", NO_EPISODES),
            ("consolidation_off", NO_CONSOLIDATION),
            ("only_the_write", NO_PULL + NO_CONSOLIDATION + NO_EPISODES),
            ("only_the_pull", NO_WRITE + NO_CONSOLIDATION + NO_EPISODES),
            ("only_episodes", NO_WRITE + NO_PULL + NO_CONSOLIDATION),
            ("everything_off", NO_WRITE + NO_PULL + NO_CONSOLIDATION + NO_EPISODES),
        ]:
            jobs.append((run, f"e1_read4_{what}", REGISTERED + flags))
        # 2. A repeat read.
        repeat = ["--docs", "32", "--read", "16", "--score", "16", "--repeats", "4",
                  "--read-what", "tail"]
        jobs.append((run, "e1_repeat16x4", repeat))
        jobs.append((run, "e1_repeat16x4_rate0.2", repeat + FIFTH))
        # 3. A read by gradient, one step a window, at 64 windows.
        for rate in ("1e-5", "1e-4", "1e-3"):
            jobs.append((run, f"e1_read64_gradient_{rate}",
                         RUNGS[0][1] + ["--read-by-gradient", rate, "--gradient-steps", "1"]))
    return jobs


# -- the second brief (2026-11-16) ------------------------------------------
# At 768 the first brief's 30 documents of 2 scored windows let one document
# supply most of a mean. The held-out set there (17,657 windows) allows 64
# documents AND 8 scored windows at both rungs, so both are used.
WIDE64 = ["--docs", "64", "--read", "64", "--score", "8"]
WIDE256 = ["--docs", "64", "--read", "256", "--score", "8"]
GRADIENT = ["--read-by-gradient", "1e-5", "--gradient-steps", "1"]
PHASE1 = ["--doc-phase", "1"]
ONLY = [
    ("only_the_write", NO_PULL + NO_CONSOLIDATION + NO_EPISODES),
    ("only_the_pull", NO_WRITE + NO_CONSOLIDATION + NO_EPISODES),
    ("only_consolidation", NO_WRITE + NO_PULL + NO_EPISODES),
    ("only_episodes", NO_WRITE + NO_PULL + NO_CONSOLIDATION),
]
EVERYTHING_OFF = NO_WRITE + NO_PULL + NO_CONSOLIDATION + NO_EPISODES


def control2() -> list[tuple[str, str, list[str]]]:
    """The gradient control on the second document set. First like for like
    (the layout it passed on, shifted to the disjoint set); then, at 768, a
    pair of disjoint sets at the wider layout. The wider layout's sets are
    disjoint from each other but not from the first layout's."""
    like = RUNGS[0][1] + GRADIENT
    return [
        (JULY42, "e1_read64_gradient_1e-5_phase1", like + PHASE1),
        (P97, "e1_read64_gradient_1e-5_phase1", like + PHASE1),
        (P97, "e1_read64_64docs_score8_gradient_1e-5", WIDE64 + GRADIENT),
        (P97, "e1_read64_64docs_score8_gradient_1e-5_phase1", WIDE64 + GRADIENT + PHASE1),
    ]


def parts() -> list[tuple[str, str, list[str]]]:
    """Which way carries the drift at 64 and 256 windows: each way alone.
    July seed 42 at the first brief's layout, to sit beside its ladder.
    768 seed 97 at the wider layout, with an everything-on read at that
    layout to compare against. Everything off is read at 64 windows, where
    it must be an exact null."""
    jobs = []
    for rung, july, wide in (("read64", RUNGS[0][1], WIDE64), ("read256", RUNGS[1][1], WIDE256)):
        for what, flags in ONLY:
            jobs.append((JULY42, f"e1_{rung}_{what}", july + flags))
        if rung == "read64":
            jobs.append((JULY42, f"e1_{rung}_everything_off", july + EVERYTHING_OFF))
            jobs.append((P97, f"e1_{rung}_64docs_score8_everything_off", wide + EVERYTHING_OFF))
        jobs.append((P97, f"e1_{rung}_64docs_score8", wide))
        for what, flags in ONLY:
            jobs.append((P97, f"e1_{rung}_64docs_score8_{what}", wide + flags))
    return jobs


PLANS = {
    "control2": control2,
    "parts": parts,
    "ladder": lambda: ladder([P97, JULY42]),
    "ladder-rest": lambda: ladder(OTHER_FINISHED),
    "switches": switches,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", choices=sorted(PLANS), required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--job-timeout-minutes", type=float, default=40.0)
    ap.add_argument("--exit-grace-seconds", type=float, default=30.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = []
    for run, what, flags in PLANS[args.plan]():
        out = INTO / run / f"{what}.json"
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "adaptation_eval.py"),
               "--run-dir", str(RUNS / run), "--out", str(out)] + flags
        jobs.append({"label": f"{run} -> {what}.json", "out": out, "cmd": cmd})

    todo = []
    for job in jobs:
        have = result_of(job["out"]) is not None
        print(f"{'have ' if have else 'would' if args.dry_run else 'queue'} {job['label']}"
              f"    {' '.join(job['cmd'][6:])}", flush=True)
        if not have:
            todo.append(job)
    if args.dry_run or not todo:
        return 0

    lock = threading.Lock()
    stop = threading.Event()
    done, failed, hung = [], [], []
    began = time.monotonic()

    def run(job: dict) -> None:
        if stop.is_set():
            return
        job["out"].parent.mkdir(parents=True, exist_ok=True)
        if job["out"].exists():        # there, and not whole: never overwrite silently
            raise RuntimeError(f"{job['out']} exists and is not a whole result")
        outcome = run_job(job["cmd"], job["out"], job["out"].with_suffix(".log"),
                          exit_grace=args.exit_grace_seconds,
                          timeout=args.job_timeout_minutes * 60, heartbeat=1e9)
        minutes = outcome["seconds"] / 60
        with lock:
            at = time.strftime("%H:%M:%S")
            if outcome["ok"]:
                done.append(job["label"])
                if outcome["killed"]:
                    hung.append(job["label"])
                # The grace is waited out before a job that does not exit is
                # killed, so its own time is that much less.
                own = minutes - (args.exit_grace_seconds / 60 if outcome["killed"] else 0)
                print(f"[{at}] done   {job['label']}  {own:.1f} min"
                      f"{'  (did not exit; killed)' if outcome['killed'] else ''}", flush=True)
            else:
                failed.append(job["label"])
                stop.set()
                print(f"[{at}] FAILED {job['label']}  after {minutes:.1f} min "
                      f"({outcome['killed'] or 'exit ' + str(outcome['code'])}); "
                      f"see {job['out'].with_suffix('.log')}", flush=True)

    print(f"first job alone: {todo[0]['label']}", flush=True)
    run(todo[0])
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, todo[1:]))

    never = len(todo) - len(done) - len(failed)
    print(f"ran {len(done)}, failed {len(failed)}, not started {never}, "
          f"did not exit {len(hung)} of {len(done)}; "
          f"{(time.monotonic() - began) / 60:.1f} min in all", flush=True)
    for label in failed:
        print(f"  failed: {label}")
    return 1 if failed or never else 0


if __name__ == "__main__":
    raise SystemExit(main())
