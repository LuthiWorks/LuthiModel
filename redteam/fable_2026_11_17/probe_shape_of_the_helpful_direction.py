"""What the helpful direction for a living weight is made of. (2026-11-17)

Card report 3 (M1) found, on July seed 42, that the gradient of what
follows with respect to a living weight is close to one direction, is the
document's own, and is at right angles to the living write. Brian asked
that this be fully explored. This probe asks what that one direction is,
so that a rule could be designed to point along it.

A measurement under Brian's ruling of 2026-11-16: a gradient is taken with
respect to living weights with plasticity frozen, in eval mode, and nothing
is stepped or kept. It uses ``luthi.v2.living_weight_gradient``'s guard.

A living layer computes y = W x for every token. So the gradient of a score
for W is the sum over tokens of d ⊗ x, where x is the token's input to the
layer and d is the gradient of the score for the token's output: what the
rest of the model would have had this layer put out. This probe captures x
and d for every token and asks, per layer and per document:

1. the check: does the sum of d ⊗ x equal the gradient autograd gives?
2. an offset: how much of the helpful direction is (the sum of d) ⊗ (the
   mean x)? If nearly all, what would help is a constant added to the
   layer's output, the same for every token of the document.
3. one signal for all layers: how much of it is recovered if every layer
   uses the last living layer's d in place of its own?
4. does it carry over: the cosine between the direction a learner would
   follow on the windows just read (the head, with the target held still,
   as the training loss holds it) and the helpful direction on the windows
   that follow (the tail), for the same document and for another.
5. which way of helping it is. The score falls if the model predicts
   better, and it also falls if what it perceives becomes easier to
   predict. The first is learning; the second is perception bending toward
   expectation. Holding the target still leaves only the first. Reported:
   the cosine between the two, and the first's share of the size.
6. a rule with no backward pass at all: (the prediction's error at each
   target token) ⊗ (that token's input to the layer). Every part of it is
   in hand during the forward pass. Its cosine with the helpful direction.
   (By construction this is perception bending toward expectation.)
7. the same three questions for the learning part alone, which is the part
   a living weight should follow: is it an offset; would one signal serve
   every layer; and does a rule with no backward pass point along it? That
   rule is (the mean error over the target tokens) ⊗ (the summed inputs of
   the context tokens), with its sign set so that it moves the context's
   outputs against the error. It is right only if the predictor passes a
   shift in what it is given through to what it predicts.

8. could one fixed map carry it down? The learning part is nearly an
   offset, so what a layer needs is one vector a window: the summed
   gradient for its context tokens' outputs. The model has one vector in
   hand with no backward pass: the mean error of its prediction. With
   ``--fit-feedback``, a single linear map from the second to the first is
   fitted (ridge) on the windows of half the documents and read on the
   other half: the cosine between what the map gives and what the gradient
   says, whole and with the part common to every window removed. Beside
   it: the same with no map at all (minus the error itself), and the
   common part alone.

    python redteam/fable_2026_11_17/probe_shape_of_the_helpful_direction.py \\
        --run-dir <run> --docs 12 --read 8 --score 8 --fit-feedback --out <file.json>
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--docs", type=int, default=12)
    ap.add_argument("--read", type=int, default=8)
    ap.add_argument("--score", type=int, default=8)
    ap.add_argument("--out", default="")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--fit-feedback", action="store_true")
    ap.add_argument("--ridge", type=float, default=1e-2)
    args = ap.parse_args()

    from luthi.v2 import result_json
    from luthi.v2.living_weight_gradient import living_weights_on_the_graph
    from luthi.v2.run_rebuild import rebuild_run, reconstruction
    from scripts.jepa_pilot_driver import _device

    device = torch.device("cpu") if args.cpu else _device()
    run = rebuild_run(args.run_dir, device=device)
    loss_module = run.loss_module.eval()
    model = loss_module.online_encoder
    before = {k: v.clone() for k, v in model.state_dict().items()}
    windows = run.text_ds.holdout_window_tokens().to(device)
    per = args.read + args.score
    stride = windows.shape[0] // args.docs
    if stride < per:
        raise SystemExit(f"{windows.shape[0]} held-out windows cannot give {args.docs} "
                         f"documents of {per}")

    def measure(block: torch.Tensor) -> dict:
        """The nmse of ``block`` ([n, L]) and, per living layer: autograd's
        gradient for the weight, and every call's token inputs and the
        gradient for its token outputs."""
        seq_len = block.shape[1]
        ctx_len = int(seq_len * loss_module.context_fraction)
        calls: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        with living_weights_on_the_graph(model) as layers, torch.enable_grad():
            index = {id(layer): i for i, layer in enumerate(layers)}

            def hook(module, inputs, output):
                output.retain_grad()
                calls.append((index[id(module)], inputs[0].detach(), output))

            handles = [layer.register_forward_hook(hook) for layer in layers]
            try:
                context = model.encode(text_tokens=block[:, :ctx_len], causal=False)["per_modality"]["text"]
                full = model.encode(text_tokens=block, causal=False)["per_modality"]["text"]
                target = full[:, ctx_len:, :]
                positions = torch.arange(ctx_len, seq_len, device=block.device).unsqueeze(0).expand(block.shape[0], -1)
                predicted = loss_module.predictor(context, positions, loss_module.action_token)
                l_pred = (predicted - target).pow(2).mean()
                var = (target - target.mean(dim=(0, 1), keepdim=True)).pow(2).mean()
                nmse = l_pred / var
                held = (predicted - target.detach()).pow(2).mean() / var.detach()
                grads = torch.autograd.grad(nmse, [layer.weight for layer in layers], retain_graph=True)
                learning = torch.autograd.grad(held, [layer.weight for layer in layers], retain_graph=True)
                outs = torch.autograd.grad(nmse, [c[2] for c in calls], retain_graph=True)
                outs_learning = torch.autograd.grad(held, [c[2] for c in calls], allow_unused=True)
                surprise = (predicted - target).detach()
            finally:
                for h in handles:
                    h.remove()
        clear = getattr(model, "clear_forward_cache", None)
        if callable(clear):
            clear()
        by_layer = [[] for _ in grads]
        by_layer_learning = [[] for _ in grads]
        raw = [[] for _ in grads]
        for (i, x, y), d, dl in zip(calls, outs, outs_learning):
            flat = x.reshape(-1, x.shape[-1]).double()
            by_layer[i].append((flat, d.detach().reshape(-1, d.shape[-1]).double()))
            dl = torch.zeros_like(y) if dl is None else dl
            by_layer_learning[i].append((flat, dl.detach().reshape(-1, dl.shape[-1]).double()))
            raw[i].append(x)
        # The second call of each layer is the whole window, whose later
        # tokens are the targets. Pull each target token's output toward
        # what was predicted for it: (predicted - target) ⊗ input.
        no_backward = []
        for i in range(len(grads)):
            if len(raw[i]) != 2 or raw[i][1].dim() != 3:
                raise RuntimeError(f"living layer {i} was called {len(raw[i])} times; "
                                   "this probe expects the context and then the whole window")
            x_target = raw[i][1][:, ctx_len:, :].reshape(-1, raw[i][1].shape[-1]).double()
            no_backward.append(surprise.reshape(-1, surprise.shape[-1]).double().T @ x_target)
        mean_surprise = surprise.reshape(-1, surprise.shape[-1]).double().mean(0)
        no_backward_learning = [
            -torch.outer(mean_surprise, raw[i][0].reshape(-1, raw[i][0].shape[-1]).double().sum(0))
            for i in range(len(grads))
        ]
        wanted = [by_layer_learning[i][0][1].sum(0) for i in range(len(grads))]   # context call
        return {"nmse": float(nmse.detach()), "g": [g.detach().double() for g in grads],
                "mean_surprise": mean_surprise, "wanted": wanted,
                "learning": [g.detach().double() for g in learning],
                "no_backward": no_backward, "no_backward_learning": no_backward_learning,
                "calls": by_layer, "calls_learning": by_layer_learning}

    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        denom = float(a.norm() * b.norm())
        return float((a * b).sum()) / denom if denom > 0 else float("nan")

    docs = []
    for i in range(args.docs):
        start = i * stride
        head, tail = windows[start:start + args.read], windows[start + args.read:start + per]
        docs.append({"tail": measure(tail), "head": measure(head)})

    rows = []
    n_layers = len(docs[0]["tail"]["g"])
    for i, doc in enumerate(docs):
        t, other = doc["tail"], docs[(i + args.docs // 2) % args.docs]
        for layer in range(n_layers):
            g = t["g"][layer]
            total = sum(d.T @ x for x, d in t["calls"][layer])
            offset = sum(torch.outer(d.sum(0), x.mean(0)) for x, d in t["calls"][layer])
            top = t["calls"][n_layers - 1]
            same_calls = len(top) == len(t["calls"][layer]) and all(
                a[1].shape == b[1].shape for a, b in zip(top, t["calls"][layer]))
            shared = (sum(dt.T @ x for (x, _), (_, dt) in zip(t["calls"][layer], top))
                      if same_calls else None)
            lg = t["learning"][layer]
            l_calls, l_top = t["calls_learning"][layer], t["calls_learning"][n_layers - 1]
            l_offset = sum(torch.outer(d.sum(0), x.mean(0)) for x, d in l_calls)
            l_shared = sum(dt.T @ x for (x, _), (_, dt) in zip(l_calls, l_top)) if same_calls else None
            rows.append({
                "learning_offset_cosine": cosine(lg, l_offset),
                "learning_one_signal_for_all_layers_cosine": None if l_shared is None else cosine(lg, l_shared),
                "learning_no_backward_rule_cosine_with_downhill": cosine(t["no_backward_learning"][layer], -lg),
                "learning_no_backward_rule_from_head_cosine_with_tail_downhill":
                    cosine(doc["head"]["no_backward_learning"][layer], -lg),
                "learning_head_to_tail_cosine_own": cosine(doc["head"]["learning"][layer], lg),
                "learning_head_to_tail_cosine_other": cosine(other["head"]["learning"][layer], lg),
                "doc": i, "layer": layer,
                "check_relative_error": float((total - g).norm() / g.norm()),
                "offset_cosine": cosine(g, offset),
                "one_signal_for_all_layers_cosine": None if shared is None else cosine(g, shared),
                "head_to_tail_cosine_own": cosine(doc["head"]["learning"][layer], g),
                "head_to_tail_cosine_other": cosine(other["head"]["learning"][layer], g),
                "learning_part_cosine": cosine(t["learning"][layer], g),
                "learning_part_share_of_size": float(t["learning"][layer].norm() / g.norm()),
                "no_backward_rule_cosine_with_downhill": cosine(t["no_backward"][layer], -g),
                "no_backward_rule_from_head_cosine_with_downhill": cosine(doc["head"]["no_backward"][layer], -g),
                "g_norm": float(g.norm()),
            })

    feedback = None
    if args.fit_feedback:
        half = args.docs // 2
        samples = []          # (document, mean surprise, [wanted by layer])
        for i in range(args.docs):
            start = i * stride
            for w in range(args.read):
                m = measure(windows[start + w:start + w + 1])
                samples.append((i, m["mean_surprise"], m["wanted"]))
        fit = [s for s in samples if s[0] < half]
        read = [s for s in samples if s[0] >= half]
        E_fit = torch.stack([s[1] for s in fit]); E_read = torch.stack([s[1] for s in read])
        e_mean = E_fit.mean(0)
        feedback = {"windows_fitted_on": len(fit), "windows_read": len(read), "ridge": args.ridge, "by_layer": []}
        for layer in range(n_layers):
            U_fit = torch.stack([s[2][layer] for s in fit]); U_read = torch.stack([s[2][layer] for s in read])
            u_mean = U_fit.mean(0)
            A = E_fit - e_mean
            lam = args.ridge * float((A * A).sum() / A.shape[0])
            gram = A @ A.T + lam * torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
            F = (U_fit - u_mean).T @ torch.linalg.solve(gram, A)          # [d, d], through the dual
            given = u_mean + (E_read - e_mean) @ F.T
            med = lambda xs: statistics.median(xs)
            feedback["by_layer"].append({
                "map_cosine": med([cosine(a, b) for a, b in zip(given, U_read)]),
                "map_cosine_common_part_removed": med(
                    [cosine(a - u_mean, b - u_mean) for a, b in zip(given, U_read)]),
                "no_map_cosine": med([cosine(-e, u) for e, u in zip(E_read, U_read)]),
                "common_part_alone_cosine": med([cosine(u_mean, u) for u in U_read]),
                "wanted_between_windows_of_one_document_cosine": med([
                    cosine(a[2][layer], b[2][layer]) for a in read for b in read
                    if a[0] == b[0] and a is not b]),
                "wanted_between_documents_cosine": med([
                    cosine(a[2][layer], b[2][layer]) for a in read[::args.read] for b in read[::args.read]
                    if a[0] != b[0]]),
            })

    after = model.state_dict()
    identical = all(torch.equal(before[k], after[k]) for k in before)
    left = [n for n, b in model.named_buffers() if b.requires_grad]

    def med(key: str, layer: int):
        vals = [r[key] for r in rows if r["layer"] == layer and r[key] is not None]
        return statistics.median(vals) if vals else None

    summary = {key: [med(key, layer) for layer in range(n_layers)] for key in (
        "check_relative_error", "offset_cosine", "one_signal_for_all_layers_cosine",
        "head_to_tail_cosine_own", "head_to_tail_cosine_other",
        "learning_part_cosine", "learning_part_share_of_size",
        "no_backward_rule_cosine_with_downhill",
        "no_backward_rule_from_head_cosine_with_downhill",
        "learning_offset_cosine", "learning_one_signal_for_all_layers_cosine",
        "learning_head_to_tail_cosine_own", "learning_head_to_tail_cosine_other",
        "learning_no_backward_rule_cosine_with_downhill",
        "learning_no_backward_rule_from_head_cosine_with_tail_downhill", "g_norm")}
    report = {
        "run": {"arm": run.result["arm"], "seed": run.result["seed"], "global_step": run.global_step},
        "reconstruction": reconstruction(run.result["config"]),
        "settings": {"docs": args.docs, "read": args.read, "score": args.score, "device": str(device)},
        "guards": {"bit_identical_afterwards": identical, "buffers_left_requiring_grad": left},
        "summary_median_by_layer": summary, "one_map_from_the_error": feedback, "rows": rows,
    }
    print(f"{run.result['arm']} seed {run.result['seed']} step {run.global_step}; {args.docs} documents; "
          f"model bit-identical afterwards: {identical}; buffers left on the graph: {len(left)}")
    for key, values in summary.items():
        print(f"  {key:62s}", " ".join("   -   " if v is None else f"{v:+.3e}" if "norm" in key or "error" in key else f"{v:+.3f}" for v in values))
    if feedback:
        print(f"one fixed map from the prediction's mean error to what each layer's learning wants "
              f"(fitted on {feedback['windows_fitted_on']} windows, read on {feedback['windows_read']})")
        for key in feedback["by_layer"][0]:
            print(f"  {key:62s}", " ".join(f"{layer[key]:+.3f}" for layer in feedback["by_layer"]))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result_json.dumps(report), encoding="utf-8")
        print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
