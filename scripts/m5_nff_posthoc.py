"""M5 post-hoc NFF audit fix.

Loads each saved M5 checkpoint, measures the non-feedforward signal on
a probe input, and writes the result back into the corresponding
results.json. Fills the gap 4.6's audit flagged: the original M5
runner didn't instrument NFF, so we couldn't distinguish "PC active
and useful" from "PC collapsed; v2 ≈ slightly worse DeadLM."

Run as::

    python -m scripts.m5_nff_posthoc

Expects LUTHI_CHECKPOINT_KEY in the environment (the same key that
encrypted the checkpoints during training).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from luthi.checkpoint import load_checkpoint
from luthi.tokenizer import BPETokenizer
from luthi.v2 import PredictiveCodingLM
from luthi.model import DeadLM


M5_DIR = Path("E:/runs/m5")  # pre-JEPA runs moved to E: 2026-07-22
TOKENIZER_PATH = "corpus_build/gutenberg_100_bpe32k.json"
PROBE_BATCH = 4
PROBE_SEQ_LEN = 64


def _build_model(arch: str, args: dict, vocab_size: int) -> torch.nn.Module:
    """Reconstruct the model from results.json args."""
    if arch == "v2":
        return PredictiveCodingLM(
            vocab_size=vocab_size,
            d_model=args["d_model"],
            n_blocks=args["n_blocks"],
            n_heads=args["n_heads"],
            ffn_expansion=args["ffn_expansion"],
            max_seq_len=args["seq_len"],
            pc_rate=args["pc_rate"],
            pred_learning_rate=args["pred_learning_rate"],
            homeostatic_decay=args["homeostatic_decay"],
            set_point_adapt_rate=args["set_point_adapt_rate"],
            compressed_episodes=args["compressed_episodes"],
            consolidation_enabled=not args["no_consolidation"],
        )
    elif arch == "dead":
        return DeadLM(
            vocab_size=vocab_size,
            d_model=args["d_model"],
            n_blocks=args["n_blocks"],
            n_heads=args["n_heads"],
            ffn_expansion=args["ffn_expansion"],
            max_seq_len=args["seq_len"],
        )
    raise ValueError(f"unknown arch {arch!r}")


def _measure_nff(model: torch.nn.Module, x: torch.Tensor) -> float:
    """Mean abs difference between two consecutive identical-input forward
    passes. Positive value = non-feedforward (PC self-mod is active);
    near-zero = effectively feedforward (PC collapsed or absent).
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
            return (out2 - out1).abs().mean().item()
    finally:
        if was_training:
            model.train()


def main():
    # Load tokenizer once to know vocab_size.
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    vocab_size = tokenizer.vocab_size

    # Probe input — fixed seed so all checkpoints see the same tokens.
    torch.manual_seed(0)
    probe = torch.randint(0, vocab_size, (PROBE_BATCH, PROBE_SEQ_LEN))

    print(f"[probe] batch={PROBE_BATCH}, seq_len={PROBE_SEQ_LEN}, vocab={vocab_size}")
    print()

    results_by_arch: dict[str, list[float]] = {"v2": [], "dead": []}

    for arch in ("v2", "dead"):
        for seed in (42, 1337, 2026):
            run_dir = M5_DIR / f"{arch}_seed{seed}"
            results_path = run_dir / "results.json"
            ckpt_path = run_dir / "model_final.luthi"

            if not results_path.exists() or not ckpt_path.exists():
                print(f"[skip] {arch}_seed{seed}: missing files")
                continue

            with results_path.open() as f:
                meta = json.load(f)
            cfg = meta["config"]

            model = _build_model(arch, cfg, vocab_size)
            # trusted=True: the checkpoint was encrypted with our own
            # LUTHI_CHECKPOINT_KEY and we just decrypted it; we own both
            # ends of the trust boundary. DirectML's save format uses
            # `_rebuild_device_tensor_from_numpy` which isn't on torch's
            # default safe-globals allowlist, so weights_only=True would
            # reject it. The encryption gate is the real security boundary.
            ckpt = load_checkpoint(ckpt_path, trusted=True)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            nff = _measure_nff(model, probe)
            results_by_arch[arch].append(nff)

            # Write back into results.json so the doc + downstream tools
            # can read it without recomputing.
            meta["posthoc_nff_at_final"] = nff
            with results_path.open("w") as f:
                json.dump(meta, f, indent=2)

            print(f"[{arch}_seed{seed}] NFF = {nff:.6e}")

    print()
    for arch, vals in results_by_arch.items():
        if vals:
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
            print(f"[{arch} mean ± std] NFF = {mean:.6e} ± {std:.6e}")

    # Interpretation
    print()
    print("Interpretation:")
    v2_vals = results_by_arch["v2"]
    dead_vals = results_by_arch["dead"]
    if v2_vals and dead_vals:
        v2_mean = sum(v2_vals) / len(v2_vals)
        dead_mean = sum(dead_vals) / len(dead_vals)
        ratio = v2_mean / max(dead_mean, 1e-12)
        if v2_mean > 1e-4 and ratio > 10:
            print(
                "  v2 NFF is well above DeadLM NFF — PC self-modification is "
                "active at end of training. M5 PASS verdict holds."
            )
        elif v2_mean < 1e-4:
            print(
                "  v2 NFF is near zero — PC self-modification may have collapsed. "
                "M5 PASS verdict needs re-evaluation."
            )
        else:
            print(
                "  v2 NFF is positive but smaller than expected. Inspect "
                "per-block NFF before final verdict."
            )


if __name__ == "__main__":
    main()
