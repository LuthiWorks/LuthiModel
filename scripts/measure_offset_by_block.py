"""Where in the trunk does the batch-constant offset appear?

The 2026-07-30 verdict refuted the projection-bias hypothesis and localized the
depth-8 offset to "the trunk", which is not a location. Trunk LayerNorm gains
were then ruled out from existing logs (flat at ~0.99-1.00 at both depths). This
measures offset dominance at EVERY block output, so "the trunk" becomes a block
index.

    offset_dominance = ||mean(latents)|| / mean(||latents||)

0 = no shared direction. 1 = every sample is the same vector.

If the offset is manufactured progressively, this rises monotonically with block
index. If one block is responsible, there is a step. If it is present at the
embedding and merely preserved, block 0 already carries it -- which would point
at the input side, not the trunk.

Read-only: loads a checkpoint, runs forward passes, writes nothing.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "runs", "jepa_pilot")


def offset_dominance(x: torch.Tensor) -> float:
    """x: [N, D] -- flattened batch/sequence of latents."""
    x = x.reshape(-1, x.shape[-1]).float()
    num = x.mean(dim=0).norm()
    den = x.norm(dim=1).mean().clamp(min=1e-12)
    return float(num / den)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=128)
    args = ap.parse_args()

    from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM
    from scripts.jepa_pilot_driver import ARM_CONFIGS

    for run in args.runs:
        ck_dir = os.path.join(BASE, run, "checkpoints")
        if not os.path.isdir(ck_dir):
            print(f"{run}: no checkpoints")
            continue
        name = sorted(n for n in os.listdir(ck_dir) if n.endswith(".pt"))[-1]
        ck = torch.load(os.path.join(ck_dir, name), map_location="cpu",
                        weights_only=False)
        state = ck.get("online_state_dict") or {}

        # Recover the arm from the run name: <arm>_<d>d_seed<n>.
        arm = run.rsplit("_", 2)[0]
        cfg = dict(ARM_CONFIGS.get(arm, {}))
        d_model = int(run.rsplit("_", 2)[1].rstrip("d"))
        n_blocks = int(cfg.get("n_blocks", 4))

        # Read vocab / max_seq_len from the checkpoint rather than assuming
        # them -- the pilot uses a 32k tokenizer and seq_len 128, and guessing
        # 50257/512 silently builds a different model.
        vocab = int(state["embedding.weight"].shape[0])
        max_seq = int(state["pos_embedding.weight"].shape[0])
        model = MultimodalPredictiveCodingLM(
            d_model=d_model, vocab_size=vocab, max_seq_len=max_seq, **cfg
        )
        missing = model.load_state_dict(state, strict=False)
        model.eval()

        print(f"\n=== {run}  ({name}, step {ck.get('global_step')}), "
              f"{n_blocks} blocks")
        if getattr(missing, "missing_keys", None):
            print(f"  (state_dict: {len(missing.missing_keys)} missing, "
                  f"{len(missing.unexpected_keys)} unexpected)")

        per_block = [[] for _ in range(n_blocks)]
        embed_od = []
        torch.manual_seed(0)
        with torch.no_grad():
            for _ in range(args.batches):
                tokens = torch.randint(
                    0, vocab, (args.batch_size, min(args.seq_len, max_seq))
                )
                out = model.encode(
                    text_tokens=tokens, collect_block_latents=True
                )
                blocks = out.get("block_latents") or []
                for i, bl in enumerate(blocks):
                    if i < n_blocks:
                        per_block[i].append(offset_dominance(bl))
                emb = out.get("embedded")
                if emb is not None:
                    embed_od.append(offset_dominance(emb))

        if embed_od:
            print(f"  embedding      offset_dominance {sum(embed_od)/len(embed_od):.4f}")
        for i, vals in enumerate(per_block):
            if not vals:
                print(f"  block {i}        (no latents collected)")
                continue
            print(f"  block {i}        offset_dominance "
                  f"{sum(vals)/len(vals):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
