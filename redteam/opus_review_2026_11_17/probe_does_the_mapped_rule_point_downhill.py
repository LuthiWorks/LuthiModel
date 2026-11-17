"""Does the proposed write point where learning points? The whole chain, on held-out windows.

Opus 5.5 review of 2026-11-17 (Fable's "a living write that hears the
prediction error"). Fable's probe measures two links of the chain separately:

1. how well the learning gradient is an offset, (the summed output gradient)
   ⊗ (the mean input): ``learning_offset_cosine``;
2. how well one fixed map from the prediction's mean error gives the summed
   output gradient: ``map_cosine``.

The rule is both at once: it moves a living weight by minus (map · error) ⊗
(the summed context inputs). This measures that product against the learning
gradient itself (target held still, as the training loss holds it), on
windows the map was not fitted on. Beside it, the rule with a perfect map
(the true summed output gradient in place of map · error), which is the most
the offset form allows. It also asks *why* the learning gradient is or is not
an offset:

- the share of the context tokens' inputs that is their common part, and
- the share of their output gradients that is common.

If the inputs are nearly one vector, any rank-one write acts as a shift of
the output, whatever learning wants.

A measurement under Brian's ruling of 2026-11-16, through
``living_weights_on_the_graph``: eval mode, plasticity frozen, nothing stepped
or kept, the model checked bit-identical afterwards.

Run: python redteam/opus_review_2026_11_17/probe_does_the_mapped_rule_point_downhill.py --run-dir <run> [--docs 24 --read 6 --cpu]
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
    ap.add_argument("--docs", type=int, default=24)
    ap.add_argument("--read", type=int, default=6)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    from luthi.v2.living_weight_gradient import living_weights_on_the_graph
    from luthi.v2.run_rebuild import rebuild_run
    from scripts.jepa_pilot_driver import _device

    device = torch.device("cpu") if args.cpu else _device()
    run = rebuild_run(args.run_dir, device=device)
    loss_module = run.loss_module.eval()
    model = loss_module.online_encoder
    before = {k: v.clone() for k, v in model.state_dict().items()}
    windows = run.text_ds.holdout_window_tokens().to(device)
    stride = windows.shape[0] // args.docs

    def window_signals(block: torch.Tensor) -> dict:
        """For one window: the learning gradient per living layer, the
        prediction's mean error, and per layer the context tokens' inputs and
        output gradients (target held still)."""
        L = block.shape[1]
        ctx = int(L * loss_module.context_fraction)
        calls = []
        with living_weights_on_the_graph(model) as layers, torch.enable_grad():
            index = {id(layer): i for i, layer in enumerate(layers)}

            def hook(module, inputs, output):
                output.retain_grad()
                calls.append((index[id(module)], inputs[0].detach(), output))

            handles = [layer.register_forward_hook(hook) for layer in layers]
            try:
                context = model.encode(text_tokens=block[:, :ctx], causal=False)["per_modality"]["text"]
                n_context_calls = len(calls)
                full = model.encode(text_tokens=block, causal=False)["per_modality"]["text"]
                target = full[:, ctx:, :].detach()
                positions = torch.arange(ctx, L, device=block.device).unsqueeze(0)
                predicted = loss_module.predictor(context, positions, loss_module.action_token)
                var = (target - target.mean(dim=(0, 1), keepdim=True)).pow(2).mean()
                held = (predicted - target).pow(2).mean() / var
                lg = torch.autograd.grad(held, [layer.weight for layer in layers], retain_graph=True)
                d_out = torch.autograd.grad(held, [c[2] for c in calls[:n_context_calls]])
            finally:
                for h in handles:
                    h.remove()
        clear = getattr(model, "clear_forward_cache", None)
        if callable(clear):
            clear()
        per_layer = []
        for (i, x, _), d in zip(calls[:n_context_calls], d_out):
            x = x.reshape(-1, x.shape[-1]).double()
            d = d.reshape(-1, d.shape[-1]).double()
            n = x.shape[0]
            per_layer.append({
                "lg": lg[i].detach().double(),
                "x_sum": x.sum(0), "d_sum": d.sum(0),
                "x_common_share": float(n * x.mean(0).pow(2).sum() / x.pow(2).sum()),
                "d_common_share": float(n * d.mean(0).pow(2).sum() / d.pow(2).sum()),
            })
        error = (predicted - target).detach().reshape(-1, target.shape[-1]).double().mean(0)
        return {"error": error, "layers": per_layer}

    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-300))

    samples = []
    for doc in range(args.docs):
        for w in range(args.read):
            start = doc * stride + w
            samples.append((doc, window_signals(windows[start:start + 1])))
    half = args.docs // 2
    fit = [s for d, s in samples if d < half]
    read = [s for d, s in samples if d >= half]
    n_layers = len(fit[0]["layers"])
    E_fit = torch.stack([s["error"] for s in fit])
    e_mean = E_fit.mean(0)
    A = E_fit - e_mean
    lam = args.ridge * float((A * A).sum() / A.shape[0])
    solve = torch.linalg.solve(A @ A.T + lam * torch.eye(A.shape[0], dtype=A.dtype), A)
    print(f"{run.result['arm']} seed {run.result['seed']} step {run.global_step}; "
          f"map fitted on {len(fit)} windows of {half} documents, read on {len(read)} windows of the others")
    print("per living layer, medians over the read windows:")
    for layer in range(n_layers):
        U_fit = torch.stack([s["layers"][layer]["d_sum"] for s in fit])
        u_mean = U_fit.mean(0)
        F = (U_fit - u_mean).T @ solve
        rule, perfect, mapped = [], [], []
        for s in read:
            L_ = s["layers"][layer]
            given = u_mean + F @ (s["error"] - e_mean)
            mapped.append(cosine(given, L_["d_sum"]))
            rule.append(cosine(torch.outer(given, L_["x_sum"]), L_["lg"]))
            perfect.append(cosine(torch.outer(L_["d_sum"], L_["x_sum"]), L_["lg"]))
        xs = [s["layers"][layer]["x_common_share"] for s in read]
        ds = [s["layers"][layer]["d_common_share"] for s in read]
        med = statistics.median
        print(f"  layer {layer}: map against the summed output gradient {med(mapped):+.3f}; "
              f"the rule against the learning gradient {med(rule):+.3f} "
              f"(with a perfect map {med(perfect):+.3f}); common share of the inputs {med(xs):.3f}, "
              f"of the output gradients {med(ds):.3f}")
    after = model.state_dict()
    identical = all(torch.equal(before[k], after[k]) for k in before)
    left = [n for n, b in model.named_buffers() if b.requires_grad]
    print(f"model bit-identical afterwards: {identical}; buffers left on the graph: {len(left)}")
    return 0 if identical and not left else 1


if __name__ == "__main__":
    sys.exit(main())
