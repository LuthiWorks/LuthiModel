"""The weight at home, one living layer at a time.

From chat Fable's reply to card report 2 (2026-11-16), as a test of a
hypothesis: with every living weight set to its home (``set_point``) the
July runs score 1.2 to 6.0% worse. If nearly all of that is the last
layer's, the model is leaning on a standing offset there.

For each run: the frozen score by document (``--docs`` documents of
``--score`` held-out windows, nothing read), as saved; with every living
weight at home; and with each living weight alone at home. Only ``weight``
is set, in the online encoder, and it is put back and checked.

Nothing in a run directory is written. Exploratory.

    python redteam/terminal_fable_2026_11_17/probe_home_one_layer_at_a_time.py \\
        --out <file.json> <run dir> [<run dir> ...]
"""
from __future__ import annotations

import argparse
import gc
import math
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def paired(a: list[float], b: list[float]) -> dict:
    d = [y - x for x, y in zip(a, b)]
    mean = statistics.fmean(d)
    se = statistics.stdev(d) / math.sqrt(len(d))
    return {"mean": mean, "t": (mean / se) if se > 0 else None,
            "relative_to_saved": mean / statistics.fmean(a),
            "n_higher": sum(x > 0 for x in d), "n": len(d)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--docs", type=int, default=64)
    ap.add_argument("--score", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from luthi.v2 import adaptation_eval as ae
    from luthi.v2 import result_json
    from luthi.v2.living_weight_gradient import living_layers
    from luthi.v2.run_rebuild import rebuild_run, reconstruction
    from scripts.jepa_pilot_driver import _device

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting it")
    device = _device()
    rows = []
    for run_dir in args.runs:
        started = time.monotonic()
        run = rebuild_run(run_dir, device=device)
        lm = run.loss_module.eval()
        layers = living_layers(lm.online_encoder)
        windows = run.text_ds.holdout_window_tokens().to(device)
        docs = ae.holdout_documents(windows, n_read=0, n_score=args.score, n_docs=args.docs)

        def scores() -> list[float]:
            return [ae._score(lm, doc)["nmse"] for doc in docs]

        @torch.no_grad()
        def at_home(which: list[int]) -> list[float]:
            saved = [layers[i].weight.clone() for i in which]
            try:
                for i in which:
                    layers[i].weight.copy_(layers[i].set_point.to(layers[i].weight.dtype))
                got = scores()
            finally:
                for i, w in zip(which, saved):
                    layers[i].weight.copy_(w)
            if not all(torch.equal(layers[i].weight, w) for i, w in zip(which, saved)):
                raise RuntimeError(f"{run_dir}: a weight was not put back")
            return got

        as_saved = scores()
        every = paired(as_saved, at_home(list(range(len(layers)))))
        alone = [paired(as_saved, at_home([i])) for i in range(len(layers))]
        all_but = [paired(as_saved, at_home([j for j in range(len(layers)) if j != i]))
                   for i in range(len(layers))]
        if scores() != as_saved:
            raise RuntimeError(f"{run_dir}: the score as saved changed")
        with torch.no_grad():
            distance = [float((m.weight.double() - m.set_point.double()).norm()
                              / m.weight.double().norm()) for m in layers]
        row = {
            "run": Path(run_dir).name, "arm": run.result["arm"], "seed": run.result["seed"],
            "checkpoint": run.checkpoint.name, "global_step": run.global_step,
            "nmse_as_saved": statistics.fmean(as_saved),
            "distance_from_home_share_of_weight": distance,
            "home_minus_saved": {"every_layer": every, "each_layer_alone": alone,
                                 "every_layer_but_this": all_but},
            "sum_of_each_alone_over_every_layer": (
                sum(a["mean"] for a in alone) / every["mean"] if every["mean"] else None),
            "minutes": (time.monotonic() - started) / 60,
            "reconstruction": reconstruction(run.result["config"]),
        }
        rows.append(row)
        print(f"{row['run']}: every layer at home {every['relative_to_saved']:+.2%} "
              f"(worse on {every['n_higher']}/{every['n']})")
        print("  each alone      ", " ".join(f"{a['relative_to_saved']:+.3%}" for a in alone))
        print("  worse on        ", " ".join(f"{a['n_higher']:>7d}" for a in alone))
        print("  all but this one", " ".join(f"{a['relative_to_saved']:+.3%}" for a in all_but))
        print("  distance        ", " ".join(f"{x:.3%}" for x in distance), flush=True)
        del run, lm, layers, windows, docs
        gc.collect()
        torch.cuda.empty_cache()

    report = {"what": "the frozen score with living weights at home, one layer at a time",
              "documents": {"n_docs": args.docs, "n_score": args.score, "n_read": 0},
              "runs": rows,
              "reconstruction": rows[0]["reconstruction"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result_json.dumps(report), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
