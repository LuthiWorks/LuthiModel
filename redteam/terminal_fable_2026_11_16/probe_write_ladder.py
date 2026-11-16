"""The write ladder, retaken: what the write puts into a living weight.

Brief 3 of 2026-11-16, item 4, after Opus's review of the same day. The
first ladder (``redteam/fable_2026_11_16/probe_one_forwards_write.py``)
predates the read switches. From 64 windows consolidation moved the July
runs' last layer with rate and pull both off, so its ``write = whole - rest``
carried a part that does not depend on the document; and it compared one
pair of documents a run. This one:

- holds consolidation and episode writes off for every pass, with the same
  code ``adaptation_eval`` uses (``_read``), so what moves a weight is the
  write and the pull and nothing else. That is checked, not assumed: a pass
  with rate 0 and pull 0 must leave every weight exactly where it was;
- reads ``--docs`` documents a run, spread evenly over the held-out set, and
  gives every between-document number over all pairs;
- splits each between-document cosine into its two sides. The write is
  ``outer(output_mean, weighted_error)`` ([out, in]; ``pc_ops.py``), so the
  top left singular vector is the output side and the top right singular
  vector is the error side. For two one-direction writes the cosine between
  them is the product of the two sides' cosines. The sign of a singular
  pair is arbitrary, so the output side's cosine is reported as positive
  and the sign is carried on the error side; the product is unaffected;
- rebuilds the run once.

Three passes a document, each from the checkpoint's state restored exactly:

    whole       rate 1, pull on
    rest        rate 0, pull on            (the pull alone)
    write_alone rate 1, pull off

and ``write_in_context = whole - rest``. "The write" is everything
``rate_scale`` scales, so it includes what the prediction's learning does to
later writes. Sizes are shares of the weight's norm.

Nothing in a run directory is written. Exploratory.

    python redteam/terminal_fable_2026_11_16/probe_write_ladder.py \\
        --run-dir <run> --windows 1 4 64 256 1024 --docs 6 --out <file.json>
"""
from __future__ import annotations

import argparse
import itertools
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 4, 64, 256, 1024])
    ap.add_argument("--docs", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from luthi.v2 import adaptation_eval as ae
    from luthi.v2 import result_json
    from luthi.v2.living_layer_pc import PredictiveCodingLayer
    from luthi.v2.run_rebuild import rebuild_run, reconstruction
    from scripts.jepa_pilot_driver import _device

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting it")
    if args.docs < 2:
        raise SystemExit("--docs must be at least 2: the cosines are between documents")
    device = _device()
    started = time.monotonic()
    run = rebuild_run(args.run_dir, device=device)
    model = run.model.eval()
    layers = [m for m in model.modules() if isinstance(m, PredictiveCodingLayer)]
    stops = sorted(set(args.windows))
    longest = stops[-1]
    windows = run.text_ds.holdout_window_tokens().to(device)
    stride = windows.shape[0] // args.docs
    if stride < longest:
        raise SystemExit(f"{windows.shape[0]} held-out windows cannot give {args.docs} "
                         f"separate documents of {longest} windows")
    documents = [windows[i * stride: i * stride + longest] for i in range(args.docs)]

    base = ae._snapshot(model)
    decay = [layer.homeostatic_decay for layer in layers]
    rate = [layer.rate_scale for layer in layers]
    weight_norm = [float(layer.weight.double().norm()) for layer in layers]
    home_gap = [float((layer.weight - layer.set_point).double().norm()) for layer in layers]

    def read(doc: torch.Tensor, rate_scale: float, pull: bool) -> dict[int, list[torch.Tensor]]:
        """The change in each living weight after each stop, on the card."""
        ae._restore(model, base)
        for layer, d in zip(layers, decay):
            layer.rate_scale = rate_scale
            layer.homeostatic_decay = d if pull else 0.0
        before = [layer.weight.detach().clone() for layer in layers]
        at: dict[int, list[torch.Tensor]] = {}
        done = 0
        for stop in stops:
            ae._read(model, doc[done:stop], repeats=1, read_train_mode=False,
                     consolidation=False, episode_writes=False)
            done = stop
            at[stop] = [(layer.weight - b).detach().clone() for layer, b in zip(layers, before)]
        return at

    def top(change: torch.Tensor) -> dict:
        """Size, the top direction's share, and its two sides."""
        m = change.double().cpu()
        size = float(m.norm())
        if size == 0.0:
            return {"size": 0.0, "top_share": None, "left": None, "right": None}
        u, s, vh = torch.linalg.svd(m, full_matrices=False)
        return {"size": size, "top_share": float(s[0] ** 2 / (s ** 2).sum()),
                "left": u[:, 0].clone(), "right": vh[0].clone()}

    def cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
        denom = float(a.norm() * b.norm())
        return float((a * b).sum()) / denom if denom > 0 else None

    try:
        # The check that the switches leave nothing else moving a weight.
        still = read(documents[0], 0.0, False)
        moved = max(float(c.abs().max()) for cs in still.values() for c in cs)
        if moved != 0.0:
            raise RuntimeError(
                f"with rate 0, pull 0, consolidation and episode writes off, a weight "
                f"still moved (largest change {moved:.3e}); the parts below would not "
                f"be the write and the pull")

        # per[kind][doc][stop][layer] -> top(); full[doc][stop][layer] -> the write in context
        per: dict[str, list] = {"write_in_context": [], "write_alone": [], "rest": []}
        full: list = []
        for i, doc in enumerate(documents):
            whole = read(doc, 1.0, True)
            rest = read(doc, 0.0, True)
            alone = read(doc, 1.0, False)
            in_context = {n: [w - r for w, r in zip(whole[n], rest[n])] for n in stops}
            per["write_in_context"].append({n: [top(c) for c in in_context[n]] for n in stops})
            per["write_alone"].append({n: [top(c) for c in alone[n]] for n in stops})
            per["rest"].append({n: [{"size": float(c.double().norm())} for c in rest[n]]
                                for n in stops})
            full.append({n: [c.float().cpu() for c in in_context[n]] for n in stops})
            del whole, rest, alone, in_context
            print(f"document {i + 1} of {len(documents)} read "
                  f"({(time.monotonic() - started) / 60:.1f} min)", flush=True)
    finally:
        ae._restore(model, base)
        for layer, r, d in zip(layers, rate, decay):
            layer.rate_scale, layer.homeostatic_decay = r, d

    def spread(values: list[float]) -> dict:
        values = [v for v in values if v is not None]
        if not values:
            return {"n": 0}
        return {"n": len(values), "median": statistics.median(values),
                "min": min(values), "max": max(values)}

    pairs = list(itertools.combinations(range(len(documents)), 2))
    rows = []
    for n in stops:
        for li in range(len(layers)):
            row = {"windows": n, "layer": li}
            for kind in ("write_in_context", "write_alone"):
                tops = [per[kind][d][n][li] for d in range(len(documents))]
                sides = {"left": [], "right": [], "product": []}
                for a, b in pairs:
                    if tops[a]["left"] is None or tops[b]["left"] is None:
                        continue
                    left = cosine(tops[a]["left"], tops[b]["left"])
                    right = cosine(tops[a]["right"], tops[b]["right"])
                    if left < 0:
                        left, right = -left, -right
                    sides["left"].append(left)
                    sides["right"].append(right)
                    sides["product"].append(left * right)
                row[kind] = {
                    "size": spread([t["size"] / weight_norm[li] for t in tops]),
                    "top_share": spread([t["top_share"] for t in tops]),
                    "output_side_cosine_between_documents": spread(sides["left"]),
                    "error_side_cosine_between_documents": spread(sides["right"]),
                    "product_of_the_sides": spread(sides["product"]),
                }
            row["write_in_context"]["cosine_between_documents"] = spread(
                [cosine(full[a][n][li], full[b][n][li]) for a, b in pairs])
            row["rest"] = {"size": spread([per["rest"][d][n][li]["size"] / weight_norm[li]
                                           for d in range(len(documents))])}
            rows.append(row)

    report = {
        "what": "the write into each living weight, consolidation and episode writes held off",
        "run": {"arm": run.result["arm"], "seed": run.result["seed"],
                "checkpoint": run.checkpoint.name, "global_step": run.global_step},
        "settings": {"windows": stops, "documents": len(documents), "pairs": len(pairs),
                     "document_starts": [i * stride for i in range(len(documents))],
                     "holdout_windows": int(windows.shape[0]), "device": str(device),
                     "homeostatic_decay": decay, "living_layers": len(layers),
                     "read_consolidation": False, "read_episode_writes": False},
        "nothing_else_moves_a_weight": {"largest_change_at_rate_0_pull_0": moved},
        "displacement_from_home": [g / w for g, w in zip(home_gap, weight_norm)],
        "rows": rows,
        "minutes": (time.monotonic() - started) / 60,
        "reconstruction": reconstruction(run.result["config"]),
    }
    print(f"{run.result['arm']} seed {run.result['seed']}; {len(layers)} living layers; "
          f"{len(documents)} documents, {len(pairs)} pairs; medians by layer")
    for n in stops:
        here = [r for r in rows if r["windows"] == n]
        print(f"after {n} windows")
        for label, get, spec in (
            ("write in context, size", lambda r: r["write_in_context"]["size"], ".1e"),
            ("write alone, size", lambda r: r["write_alone"]["size"], ".1e"),
            ("the pull, size", lambda r: r["rest"]["size"], ".1e"),
            ("top share", lambda r: r["write_in_context"]["top_share"], ".3f"),
            ("cosine between documents", lambda r: r["write_in_context"]["cosine_between_documents"], "+.2f"),
            ("  output side", lambda r: r["write_in_context"]["output_side_cosine_between_documents"], "+.2f"),
            ("  error side", lambda r: r["write_in_context"]["error_side_cosine_between_documents"], "+.2f"),
        ):
            print(f"  {label:26s}", " ".join(
                format(get(r)["median"], spec) if get(r).get("n") else "  n/a" for r in here))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result_json.dumps(report), encoding="utf-8")
    print(f"wrote {out} ({report['minutes']:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
