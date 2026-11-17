"""M1 of brief 4: where the helpful direction lies, and whether the write
points along it.

For each held-out document and each living layer, with the model as saved
and plasticity frozen:

    g           the gradient of the tail's score (nmse on the scored windows,
                the quantity E1 judges) with respect to the layer's weight.
    dW(own)     what a read of the document's own windows does to that weight.
    dW(other)   what a read of another document's windows does to it.

A read is taken three ways, each from the saved state restored exactly:

    everything_on        the read as E1's first read takes it
    episode_writes_off   the same with no episode written
    only_the_write       pull 0, consolidation off, no episode written

and at each of ``--windows`` lengths. A read of n windows reads the n
windows that come just before the scored ones.

Reported by layer and by document, for each way and length:

    identity        <g, dW(own) - dW(other)>. Summed over layers it should
                    approximate B minus C for the route through the living
                    weights (Opus). The measured B minus C of the same read
                    is beside it. It knows nothing of the episode route.
    its cosine      between -g and dW(own) - dW(other)
    the write       the cosine between -g and dW(own)
    one direction   the top direction's share of g, and the share of g that
                    lies along the top direction of dW(own)
    own or common   the cosine between g for different documents

Guards (Brian's ruling of 2026-11-16, 10:29): the gradient is taken by
``luthi.v2.living_weight_gradient`` with plasticity frozen, in eval mode,
and the model is checked bit-identical at the end.

Nothing in a run directory is written. Exploratory.

    python redteam/terminal_fable_2026_11_17/probe_helpful_direction.py \\
        --run-dir <run> --docs 48 --read 64 --score 8 --windows 4 64 --out <file>
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

WAYS = {
    # name: (pull on, consolidation, episode writes)
    "everything_on": (True, True, True),
    "episode_writes_off": (True, True, False),
    "only_the_write": (False, False, False),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--docs", type=int, required=True)
    ap.add_argument("--read", type=int, default=64)
    ap.add_argument("--score", type=int, default=8)
    ap.add_argument("--windows", type=int, nargs="+", default=[4, 64])
    ap.add_argument("--doc-phase", type=int, default=0)
    ap.add_argument("--pair-docs", type=int, default=12,
                    help="documents, spread evenly, whose gradients are compared in pairs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from luthi.v2 import adaptation_eval as ae
    from luthi.v2 import result_json
    from luthi.v2.living_weight_gradient import living_layers, score_and_its_gradient
    from luthi.v2.run_rebuild import rebuild_run, reconstruction
    from scripts.jepa_pilot_driver import _device

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting it")
    if max(args.windows) > args.read:
        raise SystemExit("--windows cannot exceed --read")
    device = _device()
    started = time.monotonic()
    run = rebuild_run(args.run_dir, device=device)
    loss_module = run.loss_module.eval()
    model = loss_module.online_encoder
    layers = living_layers(model)
    n_layers = len(layers)
    windows = run.text_ds.holdout_window_tokens().to(device)
    docs = ae.holdout_documents(windows, n_read=args.read, n_score=args.score,
                                n_docs=args.docs, phase=args.doc_phase)
    d = len(docs)
    shift = max(1, d // 2)

    base = ae._snapshot(model)
    whole_before = {k: v.detach().clone() for k, v in loss_module.state_dict().items()}
    decay = [layer.homeostatic_decay for layer in layers]
    weight_norm = [float(layer.weight.double().norm()) for layer in layers]
    lengths = sorted(set(args.windows))
    pair_docs = sorted({round(i * (d - 1) / (args.pair_docs - 1))
                        for i in range(args.pair_docs)}) if args.pair_docs > 1 else []
    kept_g: dict[int, list[torch.Tensor]] = {}

    def read(head: torch.Tensor, way: str) -> list[torch.Tensor]:
        """Read from the saved state; the change in each living weight."""
        pull, consolidation, episodes = WAYS[way]
        ae._restore(model, base)
        for layer, dec in zip(layers, decay):
            layer.homeostatic_decay = dec if pull else 0.0
        before = [layer.weight.detach().clone() for layer in layers]
        try:
            ae._read(model, head, repeats=1, read_train_mode=False,
                     consolidation=consolidation, episode_writes=episodes)
        finally:
            for layer, dec in zip(layers, decay):
                layer.homeostatic_decay = dec
        return [(layer.weight - b).detach().double() for layer, b in zip(layers, before)]

    def cos(a: torch.Tensor, b: torch.Tensor) -> float | None:
        denom = float(a.norm() * b.norm())
        return float((a * b).sum()) / denom if denom > 0 else None

    def top_pair(m: torch.Tensor) -> tuple[float | None, torch.Tensor | None, torch.Tensor | None]:
        """Top direction's share of ``m`` ([out, in]) and its two sides."""
        total = float((m * m).sum())
        if total == 0.0:
            return None, None, None
        vals, vecs = torch.linalg.eigh((m @ m.T).cpu())
        u = vecs[:, -1].to(m.device)
        v = m.T @ u
        v = v / v.norm()
        return float(vals[-1]) / total, u, v

    per_doc = []
    try:
        for i, doc in enumerate(docs):
            head, tail = doc[:args.read], doc[args.read:]
            other_head = docs[(i + shift) % d][:args.read]
            ae._restore(model, base)
            a = ae._score(loss_module, tail)
            score, grads = score_and_its_gradient(loss_module, tail)
            if abs(score["nmse"] - a["nmse"]) > 1e-4 * abs(a["nmse"]):
                raise RuntimeError(
                    f"document {i}: the score the gradient was taken of ({score['nmse']!r}) "
                    f"is not the harness's score ({a['nmse']!r})")
            g = [x.double() for x in grads]
            g_norm = [float(x.norm()) for x in g]
            g_top = [top_pair(x)[0] for x in g]
            if i in pair_docs:
                kept_g[i] = [x.float().cpu() for x in g]
            row = {"doc": i, "A": a["nmse"], "g_norm": g_norm, "g_top_share": g_top,
                   "reads": {}}
            for n in lengths:
                for way in WAYS:
                    own = read(head[args.read - n:], way)
                    b = ae._score(loss_module, tail)
                    other = read(other_head[args.read - n:], way)
                    c = ae._score(loss_module, tail)
                    by_layer = []
                    for li in range(n_layers):
                        diff = own[li] - other[li]
                        ip_own = float((g[li] * own[li]).sum())
                        ip_other = float((g[li] * other[li]).sum())
                        share, u, v = top_pair(own[li])
                        along = (None if u is None or g_norm[li] == 0 else
                                 float(u @ g[li] @ v) ** 2 / g_norm[li] ** 2)
                        by_layer.append({
                            "g_dot_own": ip_own, "g_dot_other": ip_other,
                            "identity": ip_own - ip_other,
                            "cos_downhill_own_minus_other": (
                                None if (c_ := cos(g[li], diff)) is None else -c_),
                            "cos_downhill_own": (
                                None if (c_ := cos(g[li], own[li])) is None else -c_),
                            "own_size": float(own[li].norm()) / weight_norm[li],
                            "own_minus_other_size": float(diff.norm()) / weight_norm[li],
                            "own_top_share": share,
                            "g_share_along_own_top_direction": along,
                        })
                    row["reads"][f"{way}_{n}"] = {
                        "B": b["nmse"], "C": c["nmse"],
                        "identity_sum": sum(x["identity"] for x in by_layer),
                        "g_dot_own_sum": sum(x["g_dot_own"] for x in by_layer),
                        "g_dot_other_sum": sum(x["g_dot_other"] for x in by_layer),
                        "by_layer": by_layer,
                    }
                    del own, other
            per_doc.append(row)
            if (i + 1) % 8 == 0 or i + 1 == d:
                print(f"document {i + 1} of {d} ({(time.monotonic() - started) / 60:.1f} min)",
                      flush=True)
    finally:
        ae._restore(model, base)
        for layer, dec in zip(layers, decay):
            layer.homeostatic_decay = dec

    # Guard 3: bit-identical afterwards, nothing left on the graph.
    after = loss_module.state_dict()
    changed = [k for k in whole_before if not torch.equal(whole_before[k], after[k])]
    left = [k for k, b in loss_module.named_buffers() if b.requires_grad]
    if changed or left:
        raise RuntimeError(f"the model is not as it was: changed {changed[:5]}, "
                           f"left requiring a gradient {left[:5]}")

    def med(values: list) -> float | None:
        values = [v for v in values if v is not None]
        return statistics.median(values) if values else None

    def pearson(x: list[float], y: list[float]) -> float | None:
        mx, my = sum(x) / len(x), sum(y) / len(y)
        sx = sum((a - mx) ** 2 for a in x) ** 0.5
        sy = sum((b - my) ** 2 for b in y) ** 0.5
        if sx == 0 or sy == 0:
            return None
        return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)

    summary: dict = {}
    for key in per_doc[0]["reads"]:
        reads = [p["reads"][key] for p in per_doc]
        bc = [r["B"] - r["C"] for r in reads]
        ba = [r["B"] - p["A"] for r, p in zip(reads, per_doc)]
        ident = [r["identity_sum"] for r in reads]
        gown = [r["g_dot_own_sum"] for r in reads]
        summary[key] = {
            "B_minus_C_measured": ae._summ(bc),
            "identity_sum": ae._summ(ident),
            "identity_against_measured_by_document": pearson(ident, bc),
            "B_minus_A_measured": ae._summ(ba),
            "g_dot_own_sum": ae._summ(gown),
            "g_dot_own_against_B_minus_A_by_document": pearson(gown, ba),
            "identity_by_layer_mean": [
                sum(r["by_layer"][li]["identity"] for r in reads) / len(reads)
                for li in range(n_layers)],
            "by_layer_median": {
                name: [med([r["by_layer"][li][name] for r in reads]) for li in range(n_layers)]
                for name in ("cos_downhill_own_minus_other", "cos_downhill_own",
                             "own_size", "own_minus_other_size", "own_top_share",
                             "g_share_along_own_top_direction")},
            "by_layer_downhill": {
                "own_minus_other": [sum((r["by_layer"][li]["cos_downhill_own_minus_other"] or 0) > 0
                                        for r in reads) for li in range(n_layers)],
                "own": [sum((r["by_layer"][li]["cos_downhill_own"] or 0) > 0
                            for r in reads) for li in range(n_layers)],
                "of": len(reads)},
        }
    pairs = list(itertools.combinations(sorted(kept_g), 2))
    between = []
    for li in range(n_layers):
        cs = [cos(kept_g[a][li].double(), kept_g[b][li].double()) for a, b in pairs]
        cs = [c for c in cs if c is not None]
        between.append({"n": len(cs), "median": med(cs), "min": min(cs) if cs else None,
                        "max": max(cs) if cs else None})
    summary["gradient"] = {
        "top_share_median": [med([p["g_top_share"][li] for p in per_doc]) for li in range(n_layers)],
        "norm_median": [med([p["g_norm"][li] for p in per_doc]) for li in range(n_layers)],
        "cosine_between_documents": between,
        "pair_documents": sorted(kept_g), "pairs": len(pairs),
    }

    report = {
        "what": "M1 of brief 4: the gradient of the tail's nmse for each living weight, "
                "beside what a read writes there",
        "run": {"arm": run.result["arm"], "seed": run.result["seed"],
                "checkpoint": run.checkpoint.name, "global_step": run.global_step},
        "documents": {"n_docs": d, "n_read": args.read, "n_score": args.score,
                      "phase": args.doc_phase, "control_shift": shift,
                      "holdout_windows": int(windows.shape[0]),
                      "a_read_of_n_windows": "the n windows just before the scored ones"},
        "settings": {"windows": lengths, "ways": {k: {"pull": v[0], "consolidation": v[1],
                                                      "episode_writes": v[2]}
                                                  for k, v in WAYS.items()},
                     "living_layers": n_layers, "device": str(device),
                     "weight_shape": [list(layer.weight.shape) for layer in layers]},
        "guards": {"bit_identical_afterwards": True, "buffers_left_requiring_grad": 0,
                   "plasticity_frozen_for_the_gradient": True, "training_mode": False},
        "summary": summary,
        "per_doc": per_doc,
        "minutes": (time.monotonic() - started) / 60,
        "reconstruction": reconstruction(run.result["config"]),
    }

    def row(values, spec):
        return " ".join("   n/a" if v is None else format(v, spec) for v in values)

    print(f"{run.result['arm']} seed {run.result['seed']}; {n_layers} living layers; "
          f"{d} documents, phase {args.doc_phase}")
    gs = summary["gradient"]
    print("gradient: top share      ", row(gs["top_share_median"], ".3f"))
    print("gradient: between docs   ", row([b["median"] for b in gs["cosine_between_documents"]], "+.2f"))
    for key, s in summary.items():
        if key == "gradient":
            continue
        m, idn = s["B_minus_C_measured"], s["identity_sum"]
        print(f"{key}")
        print(f"  B-C measured {m['mean']:+.3e} (t {m['t']:+.1f})   identity {idn['mean']:+.3e} "
              f"(t {idn['t']:+.1f})   by document r "
              f"{s['identity_against_measured_by_document']}")
        ba, go = s["B_minus_A_measured"], s["g_dot_own_sum"]
        print(f"  B-A measured {ba['mean']:+.3e}   <g, own> {go['mean']:+.3e}   by document r "
              f"{s['g_dot_own_against_B_minus_A_by_document']}")
        bl = s["by_layer_median"]
        print("  downhill, own - other  ", row(bl["cos_downhill_own_minus_other"], "+.3f"))
        print("  downhill, own          ", row(bl["cos_downhill_own"], "+.3f"))
        print("  g along the write      ", row(bl["g_share_along_own_top_direction"], ".1e"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result_json.dumps(report), encoding="utf-8")
    print(f"wrote {out} ({report['minutes']:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
