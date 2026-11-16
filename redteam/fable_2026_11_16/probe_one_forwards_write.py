"""What reading writes into a living weight, with the write told apart from
the pull toward home. (2026-11-16)

Not a test and not a registered instrument. A working-session measurement
(CLAUDE.md, standing rule 1) for two questions that the first read of E1,
E2 and E4 and Opus's review of it left open:

1. Opus proposes, for the revision of E3 that E2 requires, to "measure the
   write itself, the change one forward makes". Under the mean-field rule
   the write is an outer product of two vectors, so one forward's change is
   one direction whatever the model has learned. Is that so when measured?
2. E1 read four windows and could not see the write. After how many windows
   is the write's part of the change larger than everything else's?

A forward changes a living weight in two ways. The write scales with the
living rate (``rate_scale``). The pull toward home, and anything else, does
not. So from the same start, on the same windows:

    whole   = the change at rate 1
    rest    = the change at rate 0
    write   = whole - rest
    no_pull = the change at rate 0 with homeostatic_decay also 0

Reported per living layer, after 1, 4, 16, ... windows of one document:
each part's size as a share of the weight's norm; the top direction's share
of the write; and the cosine between the writes of two different documents.
Plasticity is on throughout: that is the thing measured. Every pass starts
from the checkpoint's state, restored exactly.

    python redteam/fable_2026_11_16/probe_one_forwards_write.py \\
        --run-dir <run_dir> --windows 1 4 64 256 1024 --out <file.json>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--out", default="")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    from luthi.living_extra_state import apply_living_extra_state, collect_living_extra_state
    from luthi.v2 import result_json
    from luthi.v2.living_layer_pc import PredictiveCodingLayer
    from luthi.v2.run_rebuild import rebuild_run, reconstruction
    from scripts.jepa_pilot_driver import _device
    import copy

    device = torch.device("cpu") if args.cpu else _device()
    run = rebuild_run(args.run_dir, device=device)
    model = run.model.eval()
    layers = [m for m in model.modules() if isinstance(m, PredictiveCodingLayer)]
    stops = sorted(set(args.windows))
    longest = stops[-1]
    windows = run.text_ds.holdout_window_tokens().to(device)
    if windows.shape[0] < 3 * longest:
        raise SystemExit(f"{windows.shape[0]} held-out windows cannot give two "
                         f"documents of {longest} windows a document apart")
    documents = {"a": windows[:longest], "b": windows[2 * longest:3 * longest]}

    start = copy.deepcopy(model.state_dict())
    start_extra = copy.deepcopy(collect_living_extra_state(model))
    decay = [layer.homeostatic_decay for layer in layers]

    def read(doc: torch.Tensor, rate: float, pull: bool) -> dict[int, list[torch.Tensor]]:
        model.load_state_dict(start)
        apply_living_extra_state(model, copy.deepcopy(start_extra), source="probe")
        for layer, d in zip(layers, decay):
            layer.rate_scale = rate
            layer.homeostatic_decay = d if pull else 0.0
        before = [layer.weight.detach().clone() for layer in layers]
        at: dict[int, list[torch.Tensor]] = {}
        with torch.no_grad():
            for i, w in enumerate(doc, 1):
                model.encode(text_tokens=w.unsqueeze(0), causal=False)
                clear = getattr(model, "clear_forward_cache", None)
                if callable(clear):
                    clear()
                if i in stops:
                    at[i] = [(layer.weight - b).double().cpu() for layer, b in zip(layers, before)]
        return at

    passes = {
        (name, kind): read(doc, rate, pull)
        for name, doc in documents.items()
        for kind, rate, pull in (("whole", 1.0, True), ("rest", 0.0, True), ("no_pull", 0.0, False))
    }
    for layer, d in zip(layers, decay):
        layer.rate_scale, layer.homeostatic_decay = 1.0, d
    weight_norm = [float(layer.weight.double().norm()) for layer in layers]
    home_gap = [float((layer.weight - layer.set_point).double().norm()) for layer in layers]

    def top_share(change: torch.Tensor) -> float:
        s = torch.linalg.svdvals(change)
        total = float((s ** 2).sum())
        return float(s[0] ** 2) / total if total > 0 else float("nan")

    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        denom = float(a.norm() * b.norm())
        return float((a * b).sum()) / denom if denom > 0 else float("nan")

    rows = []
    for n in stops:
        for i in range(len(layers)):
            whole = {d: passes[(d, "whole")][n][i] for d in documents}
            rest = {d: passes[(d, "rest")][n][i] for d in documents}
            write = {d: whole[d] - rest[d] for d in documents}
            rows.append({
                "windows": n, "layer": i,
                "whole": float(whole["a"].norm()) / weight_norm[i],
                "rest": float(rest["a"].norm()) / weight_norm[i],
                "write": float(write["a"].norm()) / weight_norm[i],
                "no_pull": float(passes[("a", "no_pull")][n][i].norm()) / weight_norm[i],
                "write_top_share": top_share(write["a"]),
                "write_cosine_between_documents": cosine(write["a"], write["b"]),
                "rest_cosine_between_documents": cosine(rest["a"], rest["b"]),
            })
    report = {
        "run": {"arm": run.result["arm"], "seed": run.result["seed"],
                "checkpoint": run.checkpoint.name, "global_step": run.global_step},
        "reconstruction": reconstruction(run.result["config"]),
        "settings": {"windows": stops, "device": str(device),
                     "homeostatic_decay": decay, "living_layers": len(layers)},
        "displacement_from_home": [g / w for g, w in zip(home_gap, weight_norm)],
        "rows": rows,
    }
    print(f"{run.result['arm']} seed {run.result['seed']} step {run.global_step}; "
          f"{len(layers)} living layers; sizes are shares of the weight's norm")
    print("displacement from home:", " ".join(f"{v:.0e}" for v in report["displacement_from_home"]))
    for n in stops:
        here = [r for r in rows if r["windows"] == n]
        print(f"after {n} windows")
        for key, spec in (("write", ".1e"), ("rest", ".1e"), ("no_pull", ".1e"),
                          ("write_top_share", ".3f"), ("write_cosine_between_documents", "+.2f"),
                          ("rest_cosine_between_documents", "+.2f")):
            print(f"  {key:32s}", " ".join(format(r[key], spec) for r in here))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result_json.dumps(report), encoding="utf-8")
        print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
