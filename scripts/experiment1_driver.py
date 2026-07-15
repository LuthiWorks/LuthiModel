"""Experiment 1 driver — matched-capacity control sweep (protocol
docs/research/living-weights-experiments.md section 2; thresholds
pre-registered in docs/research/2026-07-15_falsification-preregistration.md,
KF1/KF2).

Runs the falsification-critical comparison: v2 (living) at the tested
256d configuration vs DeadLM (static vanilla) across a CAPACITY SWEEP,
5 seeds per condition (Brian's ruling, 2026-07-15). The clean test is
whether the living model sits ABOVE the static loss-vs-capacity curve at
its own effective-capacity point, by more than pooled seed variance.

Base configuration is the M5 256d rerun's exactly (the KF2 result this
experiment disambiguates): 256d / 2 blocks / 4 heads / ffn_expansion 1 /
seq_len 128 / batch 32 / stride 64 / 30 epochs / lr 3e-4 cosine /
Gutenberg-100 / BPE. Only --arch, --d-model (dead bracket), and --seed
vary across conditions.

Stages (sequential single-run execution -- this box is shared; ~5.5h per
v2 run, ~3.3h per dead run at 256d, scaling roughly with d_model^2):

  stage 1  v2@256 x5  + dead@256 x5   (~44h)  the matched point, 5 seeds
  stage 2  dead@192 x5 + dead@384 x5  (~37h)  the curve's shape
  stage 3  dead@512 x5                (~35h)  the upper bracket

Bracket {192, 256, 384, 512} is Fable's proposal (nominal to ~2x
effective capacity, one point below to anchor the slope) -- flagged for
Brian's ratification alongside the pre-registration doc.

Resumable: a run whose output dir already contains results.json is
skipped, so the driver can be re-launched after interruptions (game
nights, reboots) and continues where it stopped.

Usage:
  python scripts/experiment1_driver.py --stage 1 --dry-run   # print plan
  python scripts/experiment1_driver.py --stage 1             # run stage 1
  python scripts/experiment1_driver.py --aggregate           # summarize
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "runs" / "experiment1"

SEEDS = (42, 43, 44, 45, 46)  # 5 seeds per condition -- Brian, 2026-07-15

# The M5 256d rerun configuration (docs/M5_RERUN_256D_RESULTS.md).
BASE_ARGS = [
    "--data_dir", "corpus_build/gutenberg_100",
    "--n_blocks", "2",
    "--n_heads", "4",
    "--ffn_expansion", "1",
    "--seq_len", "128",
    "--batch_size", "32",
    "--stride", "64",
    "--epochs", "30",
    "--lr", "3e-4",
    "--tokenizer", "bpe",
]

# (arch, d_model) conditions per stage.
STAGES: dict[int, list[tuple[str, int]]] = {
    1: [("v2", 256), ("dead", 256)],
    2: [("dead", 192), ("dead", 384)],
    3: [("dead", 512)],
}


def _run_name(arch: str, d_model: int, seed: int) -> str:
    return f"{arch}_{d_model}d_seed{seed}"


def _run_dir(arch: str, d_model: int, seed: int) -> Path:
    return OUTPUT_ROOT / _run_name(arch, d_model, seed)


def _is_complete(arch: str, d_model: int, seed: int) -> bool:
    return (_run_dir(arch, d_model, seed) / "results.json").exists()


def _command(arch: str, d_model: int, seed: int) -> list[str]:
    return [
        sys.executable, "-m", "luthi.v2.m5_runner",
        "--arch", arch,
        "--d_model", str(d_model),
        "--seed", str(seed),
        "--output_dir", str(OUTPUT_ROOT),
        "--run_name", _run_name(arch, d_model, seed),
        *BASE_ARGS,
    ]


def _plan(stages: list[int]) -> list[tuple[str, int, int]]:
    plan = []
    for stage in stages:
        for arch, d_model in STAGES[stage]:
            for seed in SEEDS:
                plan.append((arch, d_model, seed))
    return plan


def run(stages: list[int], dry_run: bool) -> int:
    plan = _plan(stages)
    done = [(a, d, s) for a, d, s in plan if _is_complete(a, d, s)]
    todo = [(a, d, s) for a, d, s in plan if not _is_complete(a, d, s)]
    print(f"[experiment1] plan: {len(plan)} runs "
          f"({len(done)} already complete, {len(todo)} to run)")

    for arch, d_model, seed in todo:
        cmd = _command(arch, d_model, seed)
        if dry_run:
            print("  DRY-RUN:", " ".join(cmd))
            continue
        print(f"[experiment1] starting {_run_name(arch, d_model, seed)}")
        log_path = _run_dir(arch, d_model, seed).with_suffix(".launch.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as log:
            result = subprocess.run(
                cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
            )
        if result.returncode != 0:
            # Fail loud and stop the queue: a failed run mid-sweep is a
            # condition with fewer seeds than its siblings, which corrupts
            # the variance estimate the whole comparison leans on. Fix,
            # then relaunch -- the completed runs are skipped.
            print(f"[experiment1] FAILED (exit {result.returncode}): "
                  f"{_run_name(arch, d_model, seed)} -- see {log_path}")
            return result.returncode
        print(f"[experiment1] completed {_run_name(arch, d_model, seed)}")
    return 0


def aggregate() -> int:
    """Collect per-run results.json into a per-condition summary.

    Reports mean +/- population-corrected std of best_val per condition.
    The pre-registered read (KF2): the living model must sit ABOVE the
    static capacity curve at its effective-capacity point by more than
    pooled seed variance -- but the CURVE-level judgment (effective-
    capacity placement) is the analysis doc's job, not this script's.
    This prints the ingredients, it does not declare the verdict.
    """
    conditions: dict[str, list[float]] = {}
    for results_path in sorted(OUTPUT_ROOT.glob("*/results.json")):
        data = json.loads(results_path.read_text())
        name = results_path.parent.name
        condition = name.rsplit("_seed", 1)[0]
        best_val = data.get("best_val")
        if best_val is None:
            print(f"[aggregate] WARNING: {name} has no best_val; skipped")
            continue
        conditions.setdefault(condition, []).append(float(best_val))

    if not conditions:
        print("[aggregate] no completed runs found")
        return 1

    summary = {}
    print(f"{'condition':<16} {'n':>2} {'best_val mean':>14} {'std':>8}")
    for condition, vals in sorted(conditions.items()):
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else float("nan")
        summary[condition] = {
            "n_seeds": len(vals), "best_val_mean": mean,
            "best_val_std": std, "best_val_all": vals,
        }
        print(f"{condition:<16} {len(vals):>2} {mean:>14.4f} {std:>8.4f}")
        if len(vals) < len(SEEDS):
            print(f"  WARNING: {condition} has {len(vals)}/{len(SEEDS)} "
                  f"seeds -- underpowered is not null (protocol section 1)")

    out = OUTPUT_ROOT / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[aggregate] written to {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", type=str, default=None,
                        help="1, 2, 3, or 'all'")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate", action="store_true",
                        help="Summarize completed runs and exit.")
    args = parser.parse_args()

    if args.aggregate:
        return aggregate()
    if args.stage is None:
        parser.error("--stage is required unless --aggregate")
    stages = [1, 2, 3] if args.stage == "all" else [int(args.stage)]
    return run(stages, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
