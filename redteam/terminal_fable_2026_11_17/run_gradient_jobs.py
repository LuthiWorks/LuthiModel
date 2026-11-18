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


# -- brief 8, Part 2: the read by a fixed map of the prediction's error -----
RATES = ("1e-5", "1e-4", "1e-3")
# The repeat read of the first card brief: B reads the windows it is scored on.
REPEAT = ["--docs", "32", "--read", "16", "--score", "16", "--repeats", "4",
          "--read-what", "tail"]


def _by_map(run: str, name: str, layout: list[str], kind: str, rate: str):
    kind_name = "" if kind == "map" else f"_{kind}"
    return (run, f"e1_{name}_error_map{kind_name}_{rate}", EVAL,
            layout + ["--read-by-error-map", rate, "--error-map-kind", kind])


def error_map() -> list[tuple[str, str, str, list[str]]]:
    """In the order the brief gives: the repeat read first, then 64 windows
    on both document sets; the map itself before what stands beside it
    (the map fitted with the errors shuffled, the common part alone, and M2
    on the repeat read's documents; M2 at 64 windows is in the m2 plan)."""
    jobs = []
    for run in (JULY42, P97):
        jobs += [_by_map(run, "repeat16x4", REPEAT, "map", rate) for rate in RATES]
        for suffix, phase in PHASES:
            jobs += [_by_map(run, f"read64{suffix}", LAYOUT[run] + phase, "map", rate)
                     for rate in RATES]
    for run in (JULY42, P97):
        for kind in ("shuffled", "common"):
            for suffix, phase in PHASES:
                jobs += [_by_map(run, f"read64{suffix}", LAYOUT[run] + phase, kind, rate)
                         for rate in RATES]
            jobs += [_by_map(run, "repeat16x4", REPEAT, kind, rate) for rate in RATES]
        jobs += [(run, f"e1_repeat16x4_living_gradient_{rate}", EVAL,
                  REPEAT + ["--read-by-gradient", rate, "--gradient-steps", "1",
                            "--gradient-read-what", "living"]) for rate in RATES]
    return jobs


# -- addendum 2 to brief 8: the error read that needs nothing fitted -------
def _error_rates(run: str) -> list[tuple[str, str]]:
    """(name, rate) for the three rates of the error read. ``error_rates.json``
    (set_error_rates.py) holds the rate at which one window's step is the
    share of the weight M2's is at 1e-5. The other two are 10 and 100 times
    it. They are not matched to M2's own steps at 1e-4 and 1e-3, because at
    768 M2 ran away at those rates (its step grew to 2 and 6 percent of the
    weight a window, and nmse rose by more than 1), so there is no M2 step
    of the intended size to match there. On July the two ways agree to 16%."""
    import json
    low = json.loads((INTO / run / "error_rates.json").read_text(encoding="utf-8"))[
        "error_rate_for_m2_rate"]["1e-5"]
    return [("low", f"{low:.6g}"), ("mid", f"{10 * low:.6g}"), ("high", f"{100 * low:.6g}")]


def _by_error(run: str, name: str, layout: list[str], kind: str, rate_name: str, rate: str,
              extra: tuple[str, ...] = ()):
    tag = "_in_context" if "--error-map-in-context" in extra else ""
    return (run, f"e1_{name}_{kind}{tag}_{rate_name}", EVAL,
            layout + ["--read-by-error-map", rate, "--error-map-kind", kind, "--held-score",
                      *extra])


def error() -> list[tuple[str, str, str, list[str]]]:
    """The first arm on both runs, the repeat read first; then the sign
    turned; then once in context at the middle rate; then the bias; then
    the errors of another document. Every read reports the held score."""
    layouts = lambda run: [("repeat16x4", REPEAT)] + [
        (f"read64{suffix}", LAYOUT[run] + phase) for suffix, phase in PHASES]
    jobs = []
    for kind in ("error", "error_turned"):
        for run in (JULY42, P97):
            for name, layout in layouts(run):
                jobs += [_by_error(run, name, layout, kind, n, r) for n, r in _error_rates(run)]
    for run in (JULY42, P97):
        mid = _error_rates(run)[1]
        jobs += [_by_error(run, name, layout, "error", mid[0], mid[1], ("--error-map-in-context",))
                 for name, layout in layouts(run)]
    for kind in ("bias", "error_shuffled"):
        for run in (JULY42, P97):
            for name, layout in layouts(run):
                jobs += [_by_error(run, name, layout, kind, n, r) for n, r in _error_rates(run)]
    return jobs


def m2_held() -> list[tuple[str, str, str, list[str]]]:
    """M2 and the whole-model control at 1e-5 again, with the held score:
    both meet the registered criterion on nmse, and so does the error read
    with its sign turned, which lowers nmse by moving the targets. So which
    of the two these do has to be read."""
    jobs = []
    for run in (JULY42, P97):
        for suffix, phase in PHASES:
            jobs.append((run, f"e1_read64_living_gradient_1e-5_held{suffix}", EVAL,
                         LAYOUT[run] + phase + ["--read-by-gradient", "1e-5", "--gradient-steps", "1",
                                                "--gradient-read-what", "living", "--held-score"]))
    for run in (JULY42, P97):
        for suffix, phase in PHASES:
            jobs.append((run, f"e1_read64_gradient_1e-5_held{suffix}", EVAL,
                         LAYOUT[run] + phase + ["--read-by-gradient", "1e-5", "--gradient-steps", "1",
                                                "--held-score"]))
    return jobs


def made_pull() -> list[tuple[str, str, str, list[str]]]:
    """Amendment 6 (e'), the guard's own control: no read; on arm A's own
    tensors the targets are pulled toward the prediction by 0.3 and by 0.01,
    and the guard and the centred score are taken against the pulled targets
    and against the targets as they are. Both runs, both sets."""
    return [(run, f"e1_read64_made_pull{suffix}", EVAL,
             LAYOUT[run] + phase + ["--held-score", "--made-pull-only"])
            for run in (JULY42, P97) for suffix, phase in PHASES]


def amendment_6() -> list[tuple[str, str, str, list[str]]]:
    """The reads of reports 4 and 5 again under Amendment 6 (registered
    2026-11-18 08:49): nmse, the centred score and the information guard,
    with the four numbers and the drift's shift share as description. The
    whole-model control first, then M2, then the plain error read and the
    turned one at the low rate, on both runs and both sets.

    Two files named ``..._four`` were written by an earlier form of this plan
    before the guard was built (the whole-model control on July, both sets,
    at 08:26, before the amendment was entered). They have no guard and are
    description."""
    gradient = ["--read-by-gradient", "1e-5", "--gradient-steps", "1", "--held-score"]
    jobs = []
    for what, more in (("gradient", []), ("living_gradient", ["--gradient-read-what", "living"])):
        for run in (JULY42, P97):
            for suffix, phase in PHASES:
                jobs.append((run, f"e1_read64_{what}_1e-5_a6{suffix}", EVAL,
                             LAYOUT[run] + phase + gradient + more))
    for kind in ("error", "error_turned"):
        for run in (JULY42, P97):
            low = _error_rates(run)[0][1]
            for suffix, phase in PHASES:
                jobs.append((run, f"e1_read64{suffix}_{kind}_low_a6", EVAL,
                             LAYOUT[run] + phase + ["--read-by-error-map", low, "--error-map-kind", kind,
                                                    "--held-score"]))
    return jobs


PLANS = {"m1": m1, "m2": m2, "error_map": error_map, "error": error, "m2_held": m2_held,
         "made_pull": made_pull, "amendment_6": amendment_6}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", choices=sorted(PLANS), required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--job-timeout-minutes", type=float, default=90.0)
    ap.add_argument("--exit-grace-seconds", type=float, default=30.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="",
                    help="take only the jobs whose result name holds this text")
    ap.add_argument("--keep-going", action="store_true",
                    help="a job that fails does not stop the rest (a read that runs away "
                         "raises, and that is a result); failures are listed at the end "
                         "and the exit code is still 1. A result left by a failed job is "
                         "never there: the job writes it last.")
    args = ap.parse_args()
    if args.workers > 3:
        raise SystemExit("three jobs at a time is the ceiling on this machine")

    todo = []
    for run, what, script, flags in PLANS[args.plan]():
        out = INTO / run / f"{what}.json"
        if args.only and args.only not in f"{run}/{what}":
            continue
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
        if args.keep_going and job["out"].with_suffix(".failed").exists():
            return      # it failed before, with the same code; its log is beside it
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
                if not args.keep_going:
                    stop.set()
                job["out"].with_suffix(".failed").write_text(
                    f"{outcome['killed'] or 'exit ' + str(outcome['code'])}", encoding="utf-8")
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
