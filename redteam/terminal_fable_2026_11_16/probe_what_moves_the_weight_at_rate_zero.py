"""What changes a living weight when the write and the pull are both off?

Job 2 (``redteam/fable_2026_11_16/probe_one_forwards_write.py``) reads the
same windows at living rate 0 with ``homeostatic_decay`` also 0 and calls the
change ``no_pull``. On a small run it was exactly 0 everywhere. On July seed
42 on the card it was 0 at 1 and 4 windows and then not 0 in the last layer.

The layer has one more thing that adds to the weight: consolidation
(``luthi/v2/consolidation.py``), which fires on its own trigger and uses
``layer.pc_rate`` unscaled by ``rate_scale``. This reads one document with the
write and the pull both off and records, per living layer and per window, the
size of the weight's change in that forward and whether consolidation fired
in it. If the two coincide exactly, consolidation is the whole of ``no_pull``.

    python redteam/terminal_fable_2026_11_16/probe_what_moves_the_weight_at_rate_zero.py \
        --run-dir <run> --windows 1024 --out <file.json>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--windows", type=int, default=1024)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    from luthi.v2 import result_json
    from luthi.v2.living_layer_pc import PredictiveCodingLayer
    from luthi.v2.run_rebuild import rebuild_run, reconstruction
    from scripts.jepa_pilot_driver import _device

    device = _device()
    run = rebuild_run(args.run_dir, device=device)
    model = run.model.eval()
    layers = [m for m in model.modules() if isinstance(m, PredictiveCodingLayer)]
    windows = run.text_ds.holdout_window_tokens().to(device)[: args.windows]

    trackers = [layer._consolidation_tracker for layer in layers]
    at_load = [{
        "tracker": tracker is not None,
        "history": len(tracker._history) if tracker is not None else None,
        "baseline": tracker._baseline if tracker is not None else None,
        "below_threshold_count": tracker._below_threshold_count if tracker is not None else None,
        "fires_before": layer._consolidation_fire_count,
        "episodes": int(layer.episode_count.item()),
        "consolidation_style": getattr(layer, "consolidation_style", None),
        "pc_rate": layer.pc_rate,
        "consolidation_rate_factor": getattr(layer, "consolidation_rate_factor", None),
    } for layer, tracker in zip(layers, trackers)]

    for layer in layers:
        layer.rate_scale = 0.0
        layer.homeostatic_decay = 0.0

    weight_norm = [float(layer.weight.double().norm()) for layer in layers]
    start = [layer.weight.detach().clone() for layer in layers]
    moved_in = [[] for _ in layers]       # windows (1-based) whose forward changed the weight
    fired_in = [[] for _ in layers]       # windows whose forward fired consolidation
    step_size = [[] for _ in layers]      # that forward's change, as a share of the weight
    with torch.no_grad():
        for n, w in enumerate(windows, 1):
            before = [layer.weight.detach().clone() for layer in layers]
            fires = [layer._consolidation_fire_count for layer in layers]
            model.encode(text_tokens=w.unsqueeze(0), causal=False)
            clear = getattr(model, "clear_forward_cache", None)
            if callable(clear):
                clear()
            for i, layer in enumerate(layers):
                change = float((layer.weight - before[i]).double().norm())
                if change > 0.0:
                    moved_in[i].append(n)
                    step_size[i].append(change / weight_norm[i])
                if layer._consolidation_fire_count != fires[i]:
                    fired_in[i].append(n)

    rows = []
    for i, layer in enumerate(layers):
        total = float((layer.weight - start[i]).double().norm()) / weight_norm[i]
        rows.append({
            "layer": i,
            **at_load[i],
            "forwards_that_moved_the_weight": len(moved_in[i]),
            "forwards_that_fired_consolidation": len(fired_in[i]),
            "the_same_forwards": moved_in[i] == fired_in[i],
            "first_fire_at_window": fired_in[i][0] if fired_in[i] else None,
            "fired_at_windows": fired_in[i][:50],
            "change_per_firing_forward": (
                {"min": min(step_size[i]), "max": max(step_size[i])} if step_size[i] else None
            ),
            "total_change": total,
        })
        print(f"layer {i}: episodes={at_load[i]['episodes']} "
              f"tracker history={at_load[i]['history']} baseline={at_load[i]['baseline']} "
              f"moved in {len(moved_in[i])} forwards, consolidation fired in {len(fired_in[i])}, "
              f"same forwards={moved_in[i] == fired_in[i]}, "
              f"first fire at window {rows[-1]['first_fire_at_window']}, total change {total:.1e}",
              flush=True)

    report = {
        "run": {"arm": run.result["arm"], "seed": run.result["seed"],
                "checkpoint": run.checkpoint.name, "global_step": run.global_step},
        "reconstruction": reconstruction(run.result["config"]),
        "settings": {"windows": int(windows.shape[0]), "rate_scale": 0.0,
                     "homeostatic_decay": 0.0, "device": str(device)},
        "layers": rows,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result_json.dumps(report), encoding="utf-8")
        print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
