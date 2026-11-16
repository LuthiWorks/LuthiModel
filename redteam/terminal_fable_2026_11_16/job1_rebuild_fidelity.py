"""Job 1 of the 2026-11-16 card brief: is a rebuilt July run the run it claims to be?

Each run's ``pilot_result.json`` recorded its held-out error when training
finished (``heldout``: ``l_pred_mean``, ``nmse_mean``, ``n_batches``), computed
by the driver as::

    heldout_latent_prediction(loss_module, loader.holdout_batches("text", 8),
                              "text", max_batches=args.heldout_batches)

This rebuilds each run with ``rebuild_run`` and computes the same thing over
the same number of batches, with plasticity frozen (``heldout_latent_prediction``
freezes it itself), and sets the two side by side.

It is measured twice per run, under two seeds of the global random stream, so
that a difference between recorded and rebuilt can be set beside how much the
number moves from the random stream alone.

Nothing in a run directory is written. Exploratory; informs a registration and
is not one.

Usage:
    python redteam/terminal_fable_2026_11_16/job1_rebuild_fidelity.py \
        --out runs_meta/living_channel_dose_2026_11_16/rebuild_fidelity.json \
        <run dir> [<run dir> ...]
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

METRICS = ("l_pred_mean", "nmse_mean")


def measure(run_dir: Path, device, seeds: tuple[int, ...]) -> dict:
    import torch

    from luthi.v2.eval_heldout import heldout_latent_prediction
    from luthi.v2.run_rebuild import rebuild_run, reconstruction

    run = rebuild_run(run_dir, device=device)
    recorded = run.result["heldout"]
    n_batches = int(recorded["n_batches"])

    def batches():
        for batch in run.text_ds.holdout_batches(8):
            yield {key: value.to(device) for key, value in batch.items()}

    rebuilt = {}
    for seed in seeds:
        torch.manual_seed(seed)
        got = heldout_latent_prediction(
            run.loss_module, batches(), "text", max_batches=n_batches,
        )
        if got["n_batches"] != n_batches:
            raise RuntimeError(
                f"{run_dir}: recorded over {n_batches} batches, but only "
                f"{got['n_batches']} held-out batches of 8 exist now"
            )
        rebuilt[f"seed{seed}"] = {m: got[m] for m in METRICS}

    first = rebuilt[f"seed{seeds[0]}"]
    row = {
        "run": run_dir.name,
        "arm": run.result["arm"],
        "seed": run.result["seed"],
        "outcome": run.result.get("outcome"),
        "checkpoint": run.checkpoint.name,
        "global_step": run.global_step,
        "n_batches": n_batches,
        "recorded": {m: recorded[m] for m in METRICS},
        "rebuilt": rebuilt,
        "relative_difference": {
            m: (first[m] - recorded[m]) / recorded[m] for m in METRICS
        },
        "relative_spread_across_seeds": {
            m: (max(r[m] for r in rebuilt.values()) - min(r[m] for r in rebuilt.values()))
            / recorded[m]
            for m in METRICS
        },
        "reconstruction": reconstruction(run.result["config"]),
    }
    del run
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    from luthi.v2 import result_json
    from scripts.jepa_pilot_driver import _device

    device = _device()
    print(f"device={device}", flush=True)
    rows = []
    for run_dir in args.run_dirs:
        started = time.monotonic()
        row = measure(Path(run_dir), device, tuple(args.seeds))
        rows.append(row)
        rec, reb, rel = row["recorded"], row["rebuilt"][f"seed{args.seeds[0]}"], row["relative_difference"]
        print(
            f"{row['run']:<36} step={row['global_step']} n={row['n_batches']}  "
            f"l_pred rec={rec['l_pred_mean']:.6f} reb={reb['l_pred_mean']:.6f} "
            f"rel={rel['l_pred_mean']:+.2e}  "
            f"nmse rec={rec['nmse_mean']:.6f} reb={reb['nmse_mean']:.6f} "
            f"rel={rel['nmse_mean']:+.2e}  "
            f"seed-spread nmse={row['relative_spread_across_seeds']['nmse_mean']:.1e}  "
            f"({time.monotonic() - started:.0f}s)",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result_json.dumps({
        "what": "recorded held-out error against the same thing computed from the rebuilt run",
        "plasticity": "frozen (heldout_latent_prediction freezes it)",
        "device": str(device),
        "holdout_batch_size": 8,
        "seeds": args.seeds,
        "runs": rows,
    }), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
