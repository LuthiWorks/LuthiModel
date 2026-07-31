"""Within-batch input sensitivity, measured on REAL TEXT.

Does the encoder distinguish its inputs? Mean pairwise cosine between the
latents of different sequences in one batch. ~0 means different inputs give
different representations. ~1 means the encoder has stopped responding to what
it is given.

MEASURED ON REAL CORPUS TEXT, and that is the entire point of this script
existing. The first version of this measurement used `torch.randint` over the
vocabulary -- uniformly random token IDs -- against models trained on Gutenberg.
That is far out of distribution and it INVERTED the ranking:

    arm                cos RANDOM    cos REAL
    depth 4 healthy       0.0231     -0.0008
    d8 muPC off           0.0111      0.0038
    d8 power 0            0.9704      0.3333
    d8 power -1           0.6384      0.5667
    d8 power -2           0.4127      0.1319
    d8 power -4           0.9528     -0.0294   <- worst on random, best on real

Four consecutive gates (2026-07-30, stages 17-20) were scored on the random-token
version, and stage 20's verdict inverted when re-measured. The real-text version
orders held-out NMSE monotonically across four arms; the random-token version
does not.

What made the broken version look trustworthy: the depth-4 control read 0.0231
on random tokens, matching expectations, so the instrument appeared validated. A
control passing does NOT validate the input distribution.

Read-only. Loads checkpoints and the corpus cache; writes nothing.
"""
import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(REPO, "runs", "jepa_pilot")


def load_real_text(n_seq: int, seq_len: int) -> torch.Tensor:
    """Distinct passages from the largest corpus cache, spread far apart."""
    files = sorted(glob.glob(os.path.join(REPO, "corpus_build", "cache", "*.pt")),
                   key=os.path.getsize, reverse=True)
    if not files:
        raise SystemExit("no corpus cache found")
    toks = torch.load(files[0], map_location="cpu", weights_only=False)
    stride = max(len(toks) // (n_seq + 1), seq_len)
    return torch.stack(
        [toks[i * stride: i * stride + seq_len] for i in range(n_seq)]
    ).long()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--also-random", action="store_true",
                    help="Also report the discredited random-token value, for "
                         "comparison against the historical gates.")
    args = ap.parse_args()

    from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM
    from scripts.jepa_pilot_driver import ARM_CONFIGS

    real = load_real_text(args.n_seq, args.seq_len)
    print(f"real text: {args.n_seq} passages x {args.seq_len} tokens\n")
    hdr = f"{'run':<40} {'cos REAL':>10} {'NMSE':>8} {'lift':>7}"
    if args.also_random:
        hdr += f" {'cos RANDOM':>11}"
    print(hdr)
    print("-" * len(hdr))

    for run in args.runs:
        d = os.path.join(BASE, run, "checkpoints")
        if not os.path.isdir(d):
            print(f"{run:<40} no checkpoints")
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

        def cos(tokens):
            with torch.no_grad():
                lat = model.encode(text_tokens=tokens)["latents"]
            a = lat.reshape(-1, lat.shape[-1]).float()
            half = min(200, a.shape[0] // 2)
            return float(torch.nn.functional.cosine_similarity(
                a[:half], a[half:2 * half]).mean())

        c_real = cos(real.clamp(max=vocab - 1))
        nmse = lift = float("nan")
        res = os.path.join(BASE, run, "pilot_result.json")
        if os.path.exists(res):
            p = json.load(open(res))
            nmse = p["heldout"]["nmse_mean"]
            fl = p["probe_shuffled_floor"]["top1"]
            lift = p["probe"]["top1"] / fl if fl else float("nan")
        line = f"{run:<40} {c_real:>10.4f} {nmse:>8.4f} {lift:>6.2f}x"
        if args.also_random:
            torch.manual_seed(1)
            line += f" {cos(torch.randint(0, vocab, (args.n_seq, args.seq_len))):>11.4f}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
