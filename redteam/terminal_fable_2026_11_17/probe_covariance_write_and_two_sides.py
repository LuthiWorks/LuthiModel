"""Brief 7 (2026-11-17), items 2 and 3, on a finished run. Nothing is applied.

Item 2, Opus's capacity test (review of 2026-11-17, Part 1). The covariance
write each document would make, from Opus's closed form
(``redteam/opus_review_2026_11_17/probe_what_the_covariance_write_is.py``):

    write = W  S  (I - W^T P)  T

with W the layer's weight as saved, P its prediction, T = diag(precision),
and S the token covariance of the layer's input. S is taken window by
window over the document's read windows (the mean removed within each
window, as the rule removes it within each forward) and averaged. The
rule's norm cap and its clamp are left out: they change a write's size and
not, while the clamp does not bind, its direction.

Reported: its cosine between documents, whole and split into its two sides
(the top output direction and the top input direction); its top direction's
share; and its cosine with the helpful direction from
``living_weight_gradient`` (minus the gradient of the tail's score), for the
document's own tail and for another document's.

Item 3, the two sides of the helpful direction. M1 found it is close to one
direction in each layer. For each document and layer, its top pair:

    the cosine between the input side and the layer's mean input
    the cosine between the output side and the layer's mean output
    the cosine between different documents' output sides, and input sides

The sign of a top pair is arbitrary. It is set so that the input side
points along the layer's mean input, and the output side carries the sign.
The mean input and output are over the tokens of the tail's two forwards
(the context, and the whole window), which are the forwards the gradient
is of.

Chat Fable's prediction for item 2, written before this was run (reply to
Opus and report 3): on July seed 42 the between-document cosine will be
above 0.9, and the cosine with the helpful direction near zero.

Guards as in M1: eval mode, plasticity frozen for every forward, the model
checked bit-identical at the end. Exploratory.

    python redteam/terminal_fable_2026_11_17/probe_covariance_write_and_two_sides.py \\
        --run-dir <run> --docs 48 --out <file.json>
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
    ap.add_argument("--docs", type=int, required=True)
    ap.add_argument("--read", type=int, default=64)
    ap.add_argument("--score", type=int, default=8)
    ap.add_argument("--doc-phase", type=int, default=0)
    ap.add_argument("--pair-docs", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from luthi.v2 import adaptation_eval as ae
    from luthi.v2 import result_json
    from luthi.v2.living_weight_gradient import living_layers, score_and_its_gradient
    from luthi.v2.plasticity import freeze_plasticity
    from luthi.v2.run_rebuild import rebuild_run, reconstruction
    from scripts.jepa_pilot_driver import _device

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting it")
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
    whole_before = {k: v.detach().clone() for k, v in loss_module.state_dict().items()}
    relative_trust = [bool(getattr(layer, "relative_trust", False)) for layer in layers]

    captured: list = []

    def hook(module, inputs, output):
        captured.append((id(module), inputs[0].detach().reshape(-1, inputs[0].shape[-1]),
                         output.detach().reshape(-1, output.shape[-1])))

    index = {id(layer): i for i, layer in enumerate(layers)}
    handles = [layer.register_forward_hook(hook) for layer in layers]

    def cos(a: torch.Tensor, b: torch.Tensor) -> float | None:
        denom = float(a.norm() * b.norm())
        return float((a * b).sum()) / denom if denom > 0 else None

    def top_pair(m: torch.Tensor) -> tuple[float, torch.Tensor, torch.Tensor]:
        vals, vecs = torch.linalg.eigh((m @ m.T).cpu())
        u = vecs[:, -1].to(m.device)
        v = m.T @ u
        return float(vals[-1] / vals.sum()), u, v / v.norm()

    eye = [torch.eye(layer.weight.shape[1], dtype=torch.float64, device=device) for layer in layers]

    def covariance_write(li: int, s: torch.Tensor) -> torch.Tensor:
        layer = layers[li]
        w = layer.weight.double()
        p = layer.prediction.double()
        trust = layer.precision.double()
        return w @ s @ (eye[li] - w.T @ p) * trust.unsqueeze(0)      # ... T: scales the columns

    try:
        # ---- pass 1: each document's covariance write, from its read windows
        writes: list[list[torch.Tensor]] = []        # [doc][layer], float32 on the CPU
        write_top: list[list[tuple]] = []            # [doc][layer] -> (share, u, v)
        with freeze_plasticity(model), torch.no_grad():
            for di, doc in enumerate(docs):
                s = [torch.zeros_like(e) for e in eye]
                for w in doc[:args.read]:
                    captured.clear()
                    model.encode(text_tokens=w.unsqueeze(0), causal=False)
                    for mod, x, _ in captured:
                        x = x.double()
                        xc = x - x.mean(dim=0, keepdim=True)
                        s[index[mod]] += xc.T @ xc / x.shape[0]
                    clear = getattr(model, "clear_forward_cache", None)
                    if callable(clear):
                        clear()
                row, tops = [], []
                for li in range(n_layers):
                    c = covariance_write(li, s[li] / args.read)
                    share, u, v = top_pair(c)
                    row.append(c.float().cpu())
                    tops.append((share, u.cpu(), v.cpu()))
                writes.append(row)
                write_top.append(tops)
        print(f"covariance writes of {d} documents ({(time.monotonic() - started) / 60:.1f} min)",
              flush=True)

        # ---- pass 2: the helpful direction of each document's tail
        per_doc = []
        g_sides: list[list[tuple]] = []              # [doc][layer] -> (u, v), signed
        for di, doc in enumerate(docs):
            tail = doc[args.read:]
            captured.clear()
            _, grads = score_and_its_gradient(loss_module, tail)
            sums: dict = {}
            for mod, x, y in captured:
                li = index[mod]
                sx, sy, n = sums.get(li, (0, 0, 0))
                sums[li] = (sx + x.double().sum(0), sy + y.double().sum(0), n + x.shape[0])
            row, sides = [], []
            for li, g in enumerate(grads):
                g = g.double()
                helpful = -g
                x_mean, y_mean = sums[li][0] / sums[li][2], sums[li][1] / sums[li][2]
                share, u, v = top_pair(helpful)
                # helpful ~ sigma * outer(u, v) with u = eigvec and v = helpful^T u / |.|,
                # so the pair's joint sign is fixed; flip both so v points along the mean input.
                if float(v @ x_mean) < 0:
                    u, v = -u, -v
                own = writes[di][li].to(device).double()
                other = writes[(di + shift) % d][li].to(device).double()
                row.append({
                    "helpful_top_share": share,
                    "input_side_against_mean_input": cos(v, x_mean),
                    "output_side_against_mean_output": cos(u, y_mean),
                    "covariance_write_against_helpful_own": cos(own, helpful),
                    "covariance_write_against_helpful_others": cos(other, helpful),
                    "covariance_write_top_share": write_top[di][li][0],
                    "covariance_write_size_share_of_weight": float(
                        own.norm() / layers[li].weight.double().norm()),
                })
                sides.append((u.cpu(), v.cpu()))
            per_doc.append({"doc": di, "by_layer": row})
            g_sides.append(sides)
            if (di + 1) % 8 == 0 or di + 1 == d:
                print(f"document {di + 1} of {d} ({(time.monotonic() - started) / 60:.1f} min)",
                      flush=True)
    finally:
        for h in handles:
            h.remove()

    after = loss_module.state_dict()
    changed = [k for k in whole_before if not torch.equal(whole_before[k], after[k])]
    left = [k for k, b in loss_module.named_buffers() if b.requires_grad]
    if changed or left:
        raise RuntimeError(f"the model is not as it was: changed {changed[:5]}, "
                           f"left requiring a gradient {left[:5]}")

    def spread(values: list) -> dict:
        values = [v for v in values if v is not None]
        if not values:
            return {"n": 0}
        return {"n": len(values), "median": statistics.median(values),
                "min": min(values), "max": max(values)}

    all_pairs = list(itertools.combinations(range(d), 2))
    some = sorted({round(i * (d - 1) / (args.pair_docs - 1)) for i in range(args.pair_docs)})
    some_pairs = list(itertools.combinations(some, 2))
    summary = []
    for li in range(n_layers):
        rows = [p["by_layer"][li] for p in per_doc]
        # the covariance write's sides: sign set by the pair's own product, so a
        # side's cosine is given as positive on the output side and signed on the input side
        out_side, in_side = [], []
        for a, b in all_pairs:
            cu = float(write_top[a][li][1] @ write_top[b][li][1])
            cv = float(write_top[a][li][2] @ write_top[b][li][2])
            if cu < 0:
                cu, cv = -cu, -cv
            out_side.append(cu)
            in_side.append(cv)
        whole = [cos(writes[a][li].double(), writes[b][li].double()) for a, b in some_pairs]
        summary.append({
            "layer": li, "relative_trust": relative_trust[li],
            "covariance_write": {
                "cosine_between_documents": spread(whole),
                "output_side_cosine_between_documents": spread(out_side),
                "input_side_cosine_between_documents": spread(in_side),
                "top_share": spread([r["covariance_write_top_share"] for r in rows]),
                "size_share_of_weight": spread([r["covariance_write_size_share_of_weight"] for r in rows]),
                "against_helpful_own": spread([r["covariance_write_against_helpful_own"] for r in rows]),
                "against_helpful_others": spread([r["covariance_write_against_helpful_others"] for r in rows]),
                "own_above_others_in": sum(
                    r["covariance_write_against_helpful_own"] > r["covariance_write_against_helpful_others"]
                    for r in rows),
                "documents": len(rows),
            },
            "helpful_direction": {
                "top_share": spread([r["helpful_top_share"] for r in rows]),
                "input_side_against_mean_input": spread([r["input_side_against_mean_input"] for r in rows]),
                "output_side_against_mean_output": spread([r["output_side_against_mean_output"] for r in rows]),
                "output_side_cosine_between_documents": spread(
                    [float(g_sides[a][li][0] @ g_sides[b][li][0]) for a, b in all_pairs]),
                "input_side_cosine_between_documents": spread(
                    [float(g_sides[a][li][1] @ g_sides[b][li][1]) for a, b in all_pairs]),
            },
        })

    report = {
        "what": "brief 7, items 2 and 3: the covariance write each document would make "
                "(closed form, nothing applied), and the two sides of the helpful direction",
        "run": {"arm": run.result["arm"], "seed": run.result["seed"],
                "checkpoint": run.checkpoint.name, "global_step": run.global_step},
        "documents": {"n_docs": d, "n_read": args.read, "n_score": args.score,
                      "phase": args.doc_phase, "control_shift": shift,
                      "pairs_for_sides": len(all_pairs), "pairs_for_whole_writes": len(some_pairs)},
        "settings": {"closed_form": "W S (I - W^T P) T; S averaged over the read windows, "
                                    "centred within each; no cap, no clamp",
                     "trust": "precision as saved",
                     "relative_trust_layers": sum(relative_trust),
                     "living_layers": n_layers, "device": str(device)},
        "guards": {"bit_identical_afterwards": True, "buffers_left_requiring_grad": 0,
                   "plasticity_frozen_for_every_forward": True},
        "summary_by_layer": summary, "per_doc": per_doc,
        "minutes": (time.monotonic() - started) / 60,
        "reconstruction": reconstruction(run.result["config"]),
    }

    def line(get, spec="+.2f"):
        return " ".join("  n/a" if get(s).get("n", 0) == 0 else format(get(s)["median"], spec)
                        for s in summary)

    print(f"{run.result['arm']} seed {run.result['seed']}; {d} documents; medians by layer")
    print("covariance write")
    print("  between documents        ", line(lambda s: s["covariance_write"]["cosine_between_documents"]))
    print("    output side            ", line(lambda s: s["covariance_write"]["output_side_cosine_between_documents"]))
    print("    input side             ", line(lambda s: s["covariance_write"]["input_side_cosine_between_documents"]))
    print("  top share                ", line(lambda s: s["covariance_write"]["top_share"], ".3f"))
    print("  against helpful, own     ", line(lambda s: s["covariance_write"]["against_helpful_own"], "+.3f"))
    print("  against helpful, others  ", line(lambda s: s["covariance_write"]["against_helpful_others"], "+.3f"))
    print("  own above others, of", d, " ", " ".join(str(s["covariance_write"]["own_above_others_in"]) for s in summary))
    print("helpful direction")
    print("  top share                ", line(lambda s: s["helpful_direction"]["top_share"], ".3f"))
    print("  input side on mean input ", line(lambda s: s["helpful_direction"]["input_side_against_mean_input"], "+.3f"))
    print("  output side on mean out  ", line(lambda s: s["helpful_direction"]["output_side_against_mean_output"], "+.3f"))
    print("  output sides between docs", line(lambda s: s["helpful_direction"]["output_side_cosine_between_documents"]))
    print("  input sides between docs ", line(lambda s: s["helpful_direction"]["input_side_cosine_between_documents"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result_json.dumps(report), encoding="utf-8")
    print(f"wrote {out} ({report['minutes']:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
