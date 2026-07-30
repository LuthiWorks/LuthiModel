"""Does the trunk survive production depth? Forward-pass only, no training.

Production target is 36 blocks (ML_GLOSSARY: "2 (M5), 12 (current decisive
run), 36 (production target)"). The 2026-07-30 result -- that disabling muPC
fixes the depth-8 collapse -- was established at 8 blocks and nowhere else, and
disabling muPC sets `residual_scale` to 1.0, i.e. NO residual attenuation at any
depth. Removing attenuation is precisely the choice most likely to fail as depth
grows, so "depth 8 works" is not "depth works".

This sweeps depth with muPC on and off and measures, at initialization:

  * within-batch pairwise cosine -- does the encoder distinguish its inputs?
    (the metric the muPC verdict was scored on: 0.023 healthy at depth 4,
    0.970 collapsed at depth 8)
  * per-block activation RMS -- does signal decay or explode with depth?

At init rather than after training, deliberately: it costs minutes instead of
hours per depth, and it answers whether a configuration is even viable before
committing training time. A configuration healthy at init can still fail during
training (depth 8 with muPC did exactly that), so a clean result here is
necessary and not sufficient. A DIRTY result here is decisive -- a trunk that
cannot pass a signal at initialization will not learn to.

Read-only. Builds models from config; loads no checkpoints, writes nothing.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def within_batch_cosine(latents: torch.Tensor) -> float:
    x = latents.reshape(-1, latents.shape[-1]).float()
    n = x.shape[0] // 2
    return float(
        torch.nn.functional.cosine_similarity(x[:n], x[n:2 * n]).mean()
    )


def offset_dominance(x: torch.Tensor) -> float:
    x = x.reshape(-1, x.shape[-1]).float()
    return float(x.mean(dim=0).norm() / x.norm(dim=1).mean().clamp(min=1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="4,8,12,16,24,36")
    ap.add_argument("--exponents", default="",
                    help="Comma-separated mu_pc_exponent sweep. When set, "
                         "sweeps exponents instead of the on/off pair. "
                         "0.0 is equivalent to muPC off for the residual path.")
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=32000)
    args = ap.parse_args()

    from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM

    depths = [int(d) for d in args.depths.split(",")]
    print(f"d_model={args.d_model}, batch={args.batch}, seq={args.seq}, "
          f"at initialization (no training)\n")
    header = (f"{'depth':>6} {'muPC/e':>6} {'resid_scale':>12} "
              f"{'within-cos':>11} {'offset_dom':>11} "
              f"{'rms blk0':>10} {'rms last':>10} {'rms ratio':>10}")
    print(header)
    print("-" * len(header))

    exponents = ([float(e) for e in args.exponents.split(",")]
                 if args.exponents else None)
    for depth in depths:
        settings = ([(True, e) for e in exponents] if exponents
                    else [(True, 0.25), (False, 0.25)])
        for mu, exponent in settings:
            torch.manual_seed(0)
            model = MultimodalPredictiveCodingLM(
                d_model=args.d_model,
                vocab_size=args.vocab,
                max_seq_len=args.seq,
                n_blocks=depth,
                mu_pc_enabled=mu,
                mu_pc_exponent=exponent,
            )
            model.eval()
            resid = model.blocks[0].residual_scale
            torch.manual_seed(1)
            tokens = torch.randint(0, args.vocab, (args.batch, args.seq))
            with torch.no_grad():
                out = model.encode(
                    text_tokens=tokens, collect_block_latents=True
                )
            latents = out["latents"]
            blocks = out.get("block_latents") or []
            rms0 = float(blocks[0].float().pow(2).mean().sqrt()) if blocks else float("nan")
            rmsL = float(blocks[-1].float().pow(2).mean().sqrt()) if blocks else float("nan")
            label = f"{exponent:g}" if exponents else str(mu)
            print(f"{depth:>6} {label:>6} {resid:>12.4f} "
                  f"{within_batch_cosine(latents):>11.4f} "
                  f"{offset_dominance(latents):>11.4f} "
                  f"{rms0:>10.4f} {rmsL:>10.4f} "
                  f"{rmsL / max(rms0, 1e-12):>10.3f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
