"""What holds consolidation and the pull against each other.

From card report 2 (2026-11-16), suggestion 2, taken up in chat Fable's
reply: on July seed 42 at 256 windows consolidation alone costs +1.16e-03
and the pull alone +4.40e-04, and a whole read costs +4.85e-05. This reads
the weight itself: its distance from home after each kind of read.

Reads, each from the saved state restored exactly, over ``--docs`` documents
spread evenly over the held-out set, with stops at ``--windows``:

    everything_on
    only_the_write             pull 0, no consolidation, no episode written
    only_the_pull              rate 0, no consolidation, no episode written
    only_consolidation         rate 0, pull 0, no episode written
    consolidation_and_the_pull rate 0, no episode written

By layer, after each stop: the distance from home as a share of the
weight's norm; the size of the change; how much of the change points toward
home (the cosine between the change and home minus the saved weight); how
far home itself moved; and the cosine between what consolidation alone and
the pull alone did.

Nothing in a run directory is written. Exploratory.

    python redteam/terminal_fable_2026_11_17/probe_what_holds_the_weight.py \\
        --run-dir <run> --windows 64 256 --docs 6 --out <file.json>
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

READS = {
    # name: (rate on, pull on, consolidation, episode writes)
    "everything_on": (True, True, True, True),
    "only_the_write": (True, False, False, False),
    "only_the_pull": (False, True, False, False),
    "only_consolidation": (False, False, True, False),
    "consolidation_and_the_pull": (False, True, True, False),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--windows", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--docs", type=int, default=6)
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
    started = time.monotonic()
    run = rebuild_run(args.run_dir, device=device)
    model = run.model.eval()
    layers = living_layers(model)
    stops = sorted(set(args.windows))
    windows = run.text_ds.holdout_window_tokens().to(device)
    stride = windows.shape[0] // args.docs
    if stride < stops[-1]:
        raise SystemExit("the held-out set is too short for that many documents")
    documents = [windows[i * stride: i * stride + stops[-1]] for i in range(args.docs)]

    base = ae._snapshot(model)
    decay = [layer.homeostatic_decay for layer in layers]
    rate = [layer.rate_scale for layer in layers]
    w0 = [layer.weight.detach().double().clone() for layer in layers]
    h0 = [layer.set_point.detach().double().clone() for layer in layers]
    norm = [float(w.norm()) for w in w0]
    to_home = [h - w for h, w in zip(h0, w0)]

    def cos(a: torch.Tensor, b: torch.Tensor) -> float | None:
        denom = float(a.norm() * b.norm())
        return float((a * b).sum()) / denom if denom > 0 else None

    # found[read][stop][layer] -> list over documents of dicts; changes kept for two reads
    found: dict = {name: {n: [[] for _ in layers] for n in stops} for name in READS}
    between: dict = {n: [[] for _ in layers] for n in stops}
    fires: dict = {name: [] for name in READS}
    try:
        for di, doc in enumerate(documents):
            kept: dict = {}
            for name, (rate_on, pull, consolidation, episodes) in READS.items():
                ae._restore(model, base)
                for layer, d, r in zip(layers, decay, rate):
                    layer.rate_scale = r if rate_on else 0.0
                    layer.homeostatic_decay = d if pull else 0.0
                fired_before = [int(layer._consolidation_fire_count) for layer in layers]
                done = 0
                for n in stops:
                    ae._read(model, doc[done:n], repeats=1, read_train_mode=False,
                             consolidation=consolidation, episode_writes=episodes)
                    done = n
                    for li, layer in enumerate(layers):
                        w = layer.weight.detach().double()
                        h = layer.set_point.detach().double()
                        change = w - w0[li]
                        found[name][n][li].append({
                            "distance_from_home": float((w - h).norm()) / norm[li],
                            "change": float(change.norm()) / norm[li],
                            "toward_home": cos(change, to_home[li]),
                            "home_moved": float((h - h0[li]).norm()) / norm[li],
                        })
                        if name in ("only_the_pull", "only_consolidation"):
                            kept[(name, n, li)] = change.clone()
                fires[name].append([int(layer._consolidation_fire_count) - b
                                    for layer, b in zip(layers, fired_before)])
            for n in stops:
                for li in range(len(layers)):
                    between[n][li].append(cos(kept[("only_the_pull", n, li)],
                                              kept[("only_consolidation", n, li)]))
            print(f"document {di + 1} of {len(documents)} "
                  f"({(time.monotonic() - started) / 60:.1f} min)", flush=True)
    finally:
        ae._restore(model, base)
        for layer, d, r in zip(layers, decay, rate):
            layer.rate_scale, layer.homeostatic_decay = r, d

    def med(values: list) -> float | None:
        values = [v for v in values if v is not None]
        return statistics.median(values) if values else None

    rows = []
    for name in READS:
        for n in stops:
            rows.append({"read": name, "windows": n, **{
                key: [med([x[key] for x in found[name][n][li]]) for li in range(len(layers))]
                for key in ("distance_from_home", "change", "toward_home", "home_moved")}})
    report = {
        "what": "the weight's distance from home after each kind of read",
        "run": {"arm": run.result["arm"], "seed": run.result["seed"],
                "checkpoint": run.checkpoint.name, "global_step": run.global_step},
        "settings": {"windows": stops, "documents": len(documents),
                     "homeostatic_decay": decay, "living_layers": len(layers),
                     "reads": {k: {"rate": v[0], "pull": v[1], "consolidation": v[2],
                                   "episode_writes": v[3]} for k, v in READS.items()}},
        "as_saved": {"distance_from_home": [float((w - h).norm()) / n_
                                            for w, h, n_ in zip(w0, h0, norm)]},
        "medians_over_documents": rows,
        "pull_alone_against_consolidation_alone_cosine": {
            str(n): [med(between[n][li]) for li in range(len(layers))] for n in stops},
        "consolidation_firings_by_document_and_layer": fires,
        "minutes": (time.monotonic() - started) / 60,
        "reconstruction": reconstruction(run.result["config"]),
    }

    def line(values, spec):
        return " ".join("    n/a" if v is None else format(v, spec) for v in values)

    print(f"{run.result['arm']} seed {run.result['seed']}; medians over {len(documents)} documents")
    print(f"as saved: distance from home   {line(report['as_saved']['distance_from_home'], '.2e')}")
    for r in rows:
        print(f"{r['read']}, {r['windows']} windows")
        print(f"  distance from home   {line(r['distance_from_home'], '.2e')}")
        print(f"  change               {line(r['change'], '.2e')}")
        print(f"  toward home          {line(r['toward_home'], '+.2f')}")
        print(f"  home moved           {line(r['home_moved'], '.2e')}")
    for n in stops:
        print(f"pull alone against consolidation alone, {n} windows: "
              f"{line(report['pull_alone_against_consolidation_alone_cosine'][str(n)], '+.2f')}")
    print("consolidation firings, first document:",
          {k: v[0] for k, v in fires.items()})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result_json.dumps(report), encoding="utf-8")
    print(f"wrote {out} ({report['minutes']:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
