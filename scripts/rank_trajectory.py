"""Effective-rank trajectories from jepa_pilot training logs.

Answers one question the existing logs already contain the data for:

    Did effective rank RISE THEN FALL (representational forgetting -- there
    was structure and the run lost it), or NEVER RISE (failure to acquire --
    there was never structure to lose)?

Those two diagnoses call for completely different fixes, and conflating them
sends you building consolidation machinery to protect something that was
never acquired. See docs/research/2026-08-05_rank-trajectory-at-depth.md.

`effective_rank` is written by jepa_runner on the DEEP logging cadence
(``logging.deep_interval_batches``, typically 1000), so a 3000-step run
yields three observations per block. That is coarse. This script reports the
cadence and the unobserved head of the run explicitly rather than presenting
a three-point series as a trajectory -- a sparse read that looks dense is
exactly the kind of quiet instrument this project keeps getting burned by.

Usage
-----
    python scripts/rank_trajectory.py probe_surprise_512d_seed45 probe_surprise_d8_512d_seed96
    python scripts/rank_trajectory.py --all-matching probe_surprise_d8
    python scripts/rank_trajectory.py --runs-dir E:/runs/... <run>

Reads ``training_log.jsonl`` and ``run_config.json`` from each run directory.
Read-only; writes nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_RUNS_DIR = Path(__file__).resolve().parent.parent / "runs" / "jepa_pilot"

# A block is treated as having acquired structure if its rank ever clears this.
# 512d trunks at depth 4 run 100-230; collapsed depth-8 blocks sit at 1-10.
# The threshold is nowhere near either population, so it does not need tuning --
# it exists to make the verdict explicit rather than eyeballed.
ACQUIRED_RANK = 20.0

# Rise-then-fall detection. Deliberately loose: with n=3 observations these are
# hints for a human, never verdicts. Reported as such.
RISE_FACTOR = 1.25
FALL_FACTOR = 0.75


def _walk_ranks(obj, step, out, path=""):
    """Collect every effective_rank in a log record, keyed by its nesting path."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "effective_rank" and isinstance(v, (int, float)):
                out.append((step, path or "top", float(v)))
            else:
                _walk_ranks(v, step, out, f"{path}/{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk_ranks(v, step, out, f"{path}[{i}]")


def read_ranks(run: Path) -> list[tuple[int, str, float]]:
    log = run / "training_log.jsonl"
    if not log.exists():
        raise FileNotFoundError(f"No training_log.jsonl in {run}")
    rows: list[tuple[int, str, float]] = []
    with log.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial final line on a killed run
            step = rec.get("step")
            if step is None:
                continue
            _walk_ranks(rec, step, rows)
    return rows


def read_meta(run: Path) -> dict:
    meta = {"n_blocks": None, "deep_interval": None, "arm": None, "d_model": None}
    cfg = run / "run_config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text(encoding="utf-8", errors="replace"))
        meta["deep_interval"] = c.get("logging", {}).get("deep_interval_batches")
    res = run / "pilot_result.json"
    if res.exists():
        r = json.loads(res.read_text(encoding="utf-8", errors="replace"))
        meta["arm"] = r.get("arm")
        meta["d_model"] = r.get("d_model")
        meta["n_blocks"] = (r.get("config") or {}).get("n_blocks")
    return meta


def report(run: Path) -> None:
    rows = read_ranks(run)
    meta = read_meta(run)
    print(f"\n=== {run.name} ===")
    print(
        f"    arm={meta['arm']}  d_model={meta['d_model']}  "
        f"n_blocks={meta['n_blocks']}  deep_cadence={meta['deep_interval']}"
    )
    if not rows:
        print("    NO effective_rank records -- deep cadence may never have fired.")
        return

    # `deep` duplicates the final block in jepa_runner's metric bundle; drop it
    # so a block is not silently counted twice in the verdict.
    by_path: dict[str, list[tuple[int, float]]] = {}
    for step, path, val in rows:
        if path == "deep":
            continue
        by_path.setdefault(path, []).append((step, val))

    first_step = min(s for s, _, _ in rows)
    print(
        f"    UNOBSERVED HEAD: steps 0-{first_step} have no rank measurement. "
        f"A rise-and-fall inside that window is not visible here."
    )

    acquired_any = False
    for path, series in sorted(by_path.items()):
        series.sort(key=lambda t: t[0])
        vals = [v for _, v in series]
        if not vals:
            continue
        peak = max(vals)
        peak_step = series[vals.index(peak)][0]
        first, last = vals[0], vals[-1]
        acquired = peak >= ACQUIRED_RANK
        acquired_any = acquired_any or acquired
        hint = ""
        if peak > first * RISE_FACTOR and last < peak * FALL_FACTOR:
            hint = "  [rise-then-fall hint]"
        verdict = "ACQUIRED" if acquired else "never cleared floor"
        print(
            f"    {path:22s} n={len(vals):3d}  first={first:8.2f}  "
            f"peak={peak:8.2f}@{peak_step:<6}  last={last:8.2f}  {verdict}{hint}"
        )
        traj = "  ".join(f"{s}:{v:.1f}" for s, v in series)
        print(f"        {traj}")

    print(
        f"    VERDICT: {'some blocks acquired structure' if acquired_any else 'NO block ever cleared rank ' + str(ACQUIRED_RANK)}"
        f" (threshold {ACQUIRED_RANK}; n={len(next(iter(by_path.values()), []))} obs/block)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="*", help="run directory names")
    ap.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    ap.add_argument(
        "--all-matching",
        help="report every run whose directory name contains this substring",
    )
    args = ap.parse_args()

    base = Path(args.runs_dir)
    if not base.exists():
        raise SystemExit(f"runs dir not found: {base}")

    names = list(args.runs)
    if args.all_matching:
        names += sorted(
            p.name for p in base.iterdir()
            if p.is_dir() and args.all_matching in p.name
        )
    if not names:
        raise SystemExit("no runs given (positional names or --all-matching)")

    for name in dict.fromkeys(names):  # dedupe, preserve order
        run = base / name
        if not run.is_dir():
            print(f"\n=== {name} ===\n    NOT FOUND under {base}")
            continue
        report(run)


if __name__ == "__main__":
    main()
