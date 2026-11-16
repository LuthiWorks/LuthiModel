"""Does the living part of the weight help at all, as saved?

Read 2 of the second card brief (2026-11-16). On two runs the pull toward
home alone improved the frozen score on 24 of 32 documents. So: take the
frozen score as saved, then set every living layer's weight to its home
(``set_point``) and take the same frozen score on the same windows.

Two scores, each as saved and at home:

- the run's own held-out error, as ``pilot_result.json`` recorded it
  (``heldout_latent_prediction`` over the recorded number of batches of 8);
- a paired score by document: ``--docs`` documents of ``--score`` held-out
  windows, spread evenly (``holdout_documents`` with nothing read), scored
  by ``adaptation_eval``'s own frozen ``_score``.

"At home" is measured twice: with the weight set to home in the online
encoder's living layers only, and in every living layer the loss module
holds (the target encoder has its own). Only ``weight`` is set. Every other
buffer is left as saved, and the weights are put back and checked.

Also recorded by layer: how far the weight stands from home as saved.

Nothing in a run directory is written. Exploratory.

    python redteam/terminal_fable_2026_11_16/probe_weight_at_home.py \
        --out runs_meta/living_channel_dose_2026_11_16/weight_at_home.json \
        <run dir> [<run dir> ...]
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HELDOUT = ("l_pred_mean", "nmse_mean")
BY_DOC = ("nmse", "l_pred", "target_eff_rank")


def paired(a: list[float], b: list[float]) -> dict:
    """b minus a over documents."""
    d = [y - x for x, y in zip(a, b)]
    mean = statistics.fmean(d)
    se = statistics.stdev(d) / math.sqrt(len(d))
    return {
        "mean": mean, "se": se, "t": (mean / se) if se > 0 else None,
        "median": statistics.median(d),
        "n_lower": sum(x < 0 for x in d), "n_higher": sum(x > 0 for x in d),
        "n": len(d),
    }


def measure(run_dir: Path, device, n_docs: int, n_score: int) -> dict:
    import torch

    from luthi.v2 import adaptation_eval as ae
    from luthi.v2.eval_heldout import heldout_latent_prediction
    from luthi.v2.living_layer_pc import PredictiveCodingLayer
    from luthi.v2.run_rebuild import rebuild_run, reconstruction

    run = rebuild_run(run_dir, device=device)
    lm = run.loss_module
    online = [(n, m) for n, m in lm.online_encoder.named_modules()
              if isinstance(m, PredictiveCodingLayer)]
    everywhere = [(n, m) for n, m in lm.named_modules()
                  if isinstance(m, PredictiveCodingLayer)]
    if not online:
        raise RuntimeError(f"{run_dir}: no living layer in the online encoder")
    n_batches = int(run.result["heldout"]["n_batches"])
    windows = run.text_ds.holdout_window_tokens().to(device)
    docs = ae.holdout_documents(windows, n_read=0, n_score=n_score, n_docs=n_docs)

    def batches():
        for batch in run.text_ds.holdout_batches(8):
            yield {key: value.to(device) for key, value in batch.items()}

    def scores() -> dict:
        torch.manual_seed(0)
        held = heldout_latent_prediction(lm, batches(), "text", max_batches=n_batches)
        if held["n_batches"] != n_batches:
            raise RuntimeError(f"{run_dir}: {held['n_batches']} batches, wanted {n_batches}")
        by_doc = [ae._score(lm, doc) for doc in docs]
        return {"heldout": {m: held[m] for m in HELDOUT},
                "by_doc": {m: [s[m] for s in by_doc] for m in BY_DOC}}

    @torch.no_grad()
    def at_home(layers) -> dict:
        saved = [m.weight.clone() for _, m in layers]
        try:
            for _, m in layers:
                m.weight.copy_(m.set_point.to(m.weight.dtype))
            got = scores()
        finally:
            for (_, m), w in zip(layers, saved):
                m.weight.copy_(w)
        if not all(torch.equal(m.weight, w) for (_, m), w in zip(layers, saved)):
            raise RuntimeError(f"{run_dir}: a weight was not put back")
        return got

    as_saved = scores()
    variants = {"online_encoder": at_home(online), "every_living_layer": at_home(everywhere)}
    again = scores()
    if again != as_saved:
        raise RuntimeError(f"{run_dir}: the score as saved changed after the weights were put back")

    def against_saved(got: dict) -> dict:
        return {
            "heldout": got["heldout"],
            "heldout_relative_change": {
                m: (got["heldout"][m] - as_saved["heldout"][m]) / as_saved["heldout"][m]
                for m in HELDOUT},
            "by_doc_home_minus_saved": {
                m: paired(as_saved["by_doc"][m], got["by_doc"][m]) for m in BY_DOC},
            "by_doc": got["by_doc"],
        }

    with torch.no_grad():
        distance = [{
            "layer": name,
            "in_online_encoder": any(m is o for _, o in online),
            "weight_norm": float(m.weight.float().norm().item()),
            "home_norm": float(m.set_point.float().norm().item()),
            "distance_from_home": float((m.weight.float() - m.set_point.float()).norm().item()),
            "relative_distance": float(((m.weight.float() - m.set_point.float()).norm()
                                        / m.set_point.float().norm()).item()),
        } for name, m in everywhere]

    row = {
        "run": run_dir.name, "arm": run.result["arm"], "seed": run.result["seed"],
        "outcome": run.result.get("outcome"), "checkpoint": run.checkpoint.name,
        "global_step": run.global_step,
        "living_layers": {"online_encoder": len(online), "everywhere": len(everywhere)},
        "heldout_batches": n_batches,
        "documents": {"n_docs": n_docs, "n_score": n_score, "n_read": 0, "phase": 0,
                      "holdout_windows": int(windows.shape[0])},
        "as_saved": {"heldout": as_saved["heldout"],
                     "recorded_heldout": {m: run.result["heldout"][m] for m in HELDOUT},
                     "by_doc": as_saved["by_doc"]},
        "at_home": {k: against_saved(v) for k, v in variants.items()},
        "distance_from_home": distance,
        "reconstruction": reconstruction(run.result["config"]),
    }
    del run, lm
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--docs", type=int, default=64)
    ap.add_argument("--score", type=int, default=8)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    from luthi.v2 import result_json
    from scripts.jepa_pilot_driver import _device

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting it")
    device = _device()
    rows = []
    for run_dir in args.run_dirs:
        started = time.monotonic()
        row = measure(Path(run_dir), device, args.docs, args.score)
        rows.append(row)
        print(f"{row['run']}  ({time.monotonic() - started:.0f}s)  living layers "
              f"{row['living_layers']}", flush=True)
        for name, v in row["at_home"].items():
            d = v["by_doc_home_minus_saved"]["nmse"]
            t = "n/a" if d["t"] is None else f"{d['t']:+.1f}"
            print(f"    home in {name:<19} heldout nmse {v['heldout_relative_change']['nmse_mean']:+.2e} rel"
                  f"   by doc nmse {d['mean']:+.3e} (t {t}), lower on {d['n_lower']} of {d['n']}",
                  flush=True)
        far = [f"{x['relative_distance']:.3f}" for x in row["distance_from_home"]
               if x["in_online_encoder"]]
        print(f"    relative distance from home, online layers: {' '.join(far)}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result_json.dumps({
        "what": "the frozen score as saved against the same with living weights set to home",
        "plasticity": "frozen for every score",
        "device": str(device),
        "runs": rows,
    }), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
