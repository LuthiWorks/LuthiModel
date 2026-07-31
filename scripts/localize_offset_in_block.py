"""Where inside a block is the batch-constant offset manufactured?

A block computes:

    x1 = x0 + residual_scale * attention(norm1(x0))
    x2 = x1 + residual_scale * ffn(norm2(x1))

Offset dominance is measured at x0, x1 and x2, so the question "which half of
the block creates the offset" gets a direct answer instead of another
hypothesis. Four hypotheses died in one night by theorizing before localizing.

    offset_dominance = ||mean(h)|| / mean(||h||)

MEASURED ON REAL CORPUS TEXT. An earlier version of the per-block scan used
`torch.randint` over the vocabulary and reported block 0 at 0.989; that input is
far out of distribution and inverted the ranking of four separate runs when
re-measured (see scripts/measure_input_sensitivity.py). Any number in this file
produced from random tokens would be untrustworthy for the same reason.

`x1` is recovered from `norm2`'s input, which is exactly x-after-the-attention-
residual, so no model surgery is needed -- only forward hooks.

Read-only.
"""
import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(REPO, "runs", "jepa_pilot")


def offset_dominance(x: torch.Tensor) -> float:
    x = x.reshape(-1, x.shape[-1]).float()
    return float(x.mean(dim=0).norm() / x.norm(dim=1).mean().clamp(min=1e-12))


def load_real_text(n_seq: int, seq_len: int) -> torch.Tensor:
    files = sorted(glob.glob(os.path.join(REPO, "corpus_build", "cache", "*.pt")),
                   key=os.path.getsize, reverse=True)
    toks = torch.load(files[0], map_location="cpu", weights_only=False)
    stride = max(len(toks) // (n_seq + 1), seq_len)
    return torch.stack(
        [toks[i * stride: i * stride + seq_len] for i in range(n_seq)]
    ).long()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--blocks", type=int, default=3,
                    help="How many leading blocks to report.")
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=128)
    args = ap.parse_args()

    from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM
    from scripts.jepa_pilot_driver import ARM_CONFIGS

    real = load_real_text(args.n_seq, args.seq_len)
    print(f"real text: {args.n_seq} passages x {args.seq_len} tokens\n")

    for run in args.runs:
        d = os.path.join(BASE, run, "checkpoints")
        if not os.path.isdir(d):
            print(f"{run}: no checkpoints")
            continue
        arm = run.rsplit("_", 2)[0]
        name = sorted(n for n in os.listdir(d) if n.endswith(".pt"))[-1]
        st = torch.load(os.path.join(d, name), map_location="cpu",
                        weights_only=False)["online_state_dict"]
        vocab = int(st["embedding.weight"].shape[0])
        max_seq = int(st["pos_embedding.weight"].shape[0])
        d_model = int(run.rsplit("_", 2)[1].rstrip("d"))
        model = MultimodalPredictiveCodingLM(
            d_model=d_model, vocab_size=vocab, max_seq_len=max_seq,
            **dict(ARM_CONFIGS.get(arm, {}))
        )
        model.load_state_dict(st, strict=False)
        model.eval()

        rec: dict = {}
        hooks = []
        n_show = min(args.blocks, len(model.blocks))
        for i in range(n_show):
            blk = model.blocks[i]

            def pre_norm1(mod, inp, i=i):
                rec[(i, "x0")] = inp[0].detach()

            def pre_norm2(mod, inp, i=i):
                rec[(i, "x1_post_attn")] = inp[0].detach()

            def post_block(mod, inp, out, i=i):
                o = out[0] if isinstance(out, tuple) else out
                rec[(i, "x2_post_ffn")] = o.detach()

            hooks.append(blk.norm1.register_forward_pre_hook(pre_norm1))
            hooks.append(blk.norm2.register_forward_pre_hook(pre_norm2))
            hooks.append(blk.register_forward_hook(post_block))

        with torch.no_grad():
            model.encode(text_tokens=real.clamp(max=vocab - 1))
        for h in hooks:
            h.remove()

        rs = float(getattr(model.blocks[0], "residual_scale", 1.0))
        print(f"=== {run}   (residual_scale {rs:.4f})")
        print(f"   {'block':<7} {'x0 in':>9} {'x1 post-attn':>14} "
              f"{'x2 post-ffn':>13} {'attn delta':>12} {'ffn delta':>11}")
        for i in range(n_show):
            x0 = rec.get((i, "x0"))
            x1 = rec.get((i, "x1_post_attn"))
            x2 = rec.get((i, "x2_post_ffn"))
            if x0 is None or x1 is None or x2 is None:
                print(f"   {i:<7} (hooks did not fire)")
                continue
            a, b, c = offset_dominance(x0), offset_dominance(x1), offset_dominance(x2)
            print(f"   {i:<7} {a:>9.4f} {b:>14.4f} {c:>13.4f} "
                  f"{b - a:>+12.4f} {c - b:>+11.4f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
