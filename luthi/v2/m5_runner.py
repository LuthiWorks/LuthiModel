"""M5: v2 PC vs DeadLM head-to-head training harness.

Single script that trains either `PredictiveCodingLM` (v2) or `DeadLM`
(vanilla transformer + episode store baseline) at matched config — same
d_model, n_blocks, n_heads, ffn_expansion, seq_len, batch, stride,
epochs, seed, optimizer, data, LR schedule. Per the V2 plan's M5 spec
post-strategic-shift, this is the falsification-criteria comparison
that decides whether v2 (the now-primary substrate) actually beats a
vanilla baseline.

Usage::

    python -m luthi.v2.m5_runner --arch v2 --seed 42 ...
    python -m luthi.v2.m5_runner --arch dead --seed 42 ...

Falsification criteria (per docs/V2_IMPLEMENTATION_PLAN.md M5):
- v2 convergence penalty worse than DeadLM by ≥20% at matched scale
- v2 cascade fails at depths where DeadLM succeeds
- v2 attractor dynamics indistinguishable from random control (separate
  attractor-dynamics script handles this)
- M4 STOP GATE already passed (consolidation has measurable effect)
- VRAM exceeded at equivalent parameter count

If any falsify, v2 needs revisiting; v1 ablation pipeline revives.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from luthi.data import (
    CharDataset,
    CharTokenizer,
    load_corpus,
    load_corpus_as_tensor,
    load_corpus_sample,
    load_file_list,
)
from luthi.tokenizer import BPETokenizer
from luthi.v2 import PredictiveCodingLM
from luthi.model import DeadLM


def _build_model(args, vocab_size: int, device: torch.device) -> torch.nn.Module:
    """Construct either v2 PredictiveCodingLM or DeadLM at matched config.

    Notes on matched-config fairness:
    - d_model, n_blocks, n_heads, ffn_expansion, max_seq_len all transfer
      identically to both architectures.
    - v2 turns consolidation ON for M5 (the M4 STOP GATE result is part
      of what's being validated end-to-end here).
    - DeadLM doesn't have living state, so PC-specific hyperparameters
      (pc_rate, pred_learning_rate, etc.) are silently ignored.
    """
    if args.arch == "v2":
        return PredictiveCodingLM(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_blocks=args.n_blocks,
            n_heads=args.n_heads,
            ffn_expansion=args.ffn_expansion,
            max_seq_len=args.seq_len,
            pc_rate=args.pc_rate,
            pred_learning_rate=args.pred_learning_rate,
            homeostatic_decay=args.homeostatic_decay,
            set_point_adapt_rate=args.set_point_adapt_rate,
            compressed_episodes=args.compressed_episodes,
            consolidation_enabled=not args.no_consolidation,
        ).to(device)
    elif args.arch == "dead":
        return DeadLM(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_blocks=args.n_blocks,
            n_heads=args.n_heads,
            ffn_expansion=args.ffn_expansion,
            max_seq_len=args.seq_len,
        ).to(device)
    else:
        raise ValueError(f"Unknown --arch {args.arch!r}; expected 'v2' or 'dead'")


def _compute_lr(args, epoch: int) -> float:
    """Cosine LR schedule with linear warmup, same as train_pc.py."""
    if args.lr_schedule == "constant":
        return args.lr
    if epoch < args.lr_warmup_epochs:
        return args.lr * (epoch + 1) / max(1, args.lr_warmup_epochs)
    progress = (epoch - args.lr_warmup_epochs) / max(
        1, args.epochs - args.lr_warmup_epochs
    )
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, int]:
    model.train()
    total = 0.0
    n_batches = 0
    nan_count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        optimizer.step()
        if hasattr(model, "clear_forward_cache"):
            model.clear_forward_cache()
        total += loss.item()
        n_batches += 1
    return (total / max(n_batches, 1)), nan_count


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    n_batches = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        total += loss.item()
        n_batches += 1
    return total / max(n_batches, 1)


def _save_checkpoint(model, path: Path, epoch: int, config: dict) -> None:
    """Encrypted checkpoint when LUTHI_CHECKPOINT_KEY is set, plaintext
    with loud warning otherwise — same policy as train_pc._save_checkpoint_v2.
    """
    try:
        from luthi.checkpoint import build_checkpoint, save_checkpoint
        ckpt = build_checkpoint(model, epoch=epoch, config=config)
        encrypted_path = save_checkpoint(ckpt, path)
        print(f"[checkpoint] encrypted save: {encrypted_path}")
    except ValueError as e:
        plain_path = path.with_suffix(".pt")
        torch.save(model.state_dict(), plain_path)
        print(
            f"[checkpoint] WARNING: no LUTHI_CHECKPOINT_KEY set; saved "
            f"plaintext to {plain_path}. Reason: {e}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="M5: v2 PC vs DeadLM head-to-head training harness.",
    )
    parser.add_argument(
        "--arch", type=str, required=True, choices=["v2", "dead"],
        help="Architecture: 'v2' = PredictiveCodingLM, 'dead' = DeadLM baseline.",
    )
    parser.add_argument("--data_dir", type=str, default="corpus_build/gutenberg_100")
    parser.add_argument("--file-list", dest="file_list", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30,
                        help="M5 default 30 epochs (per V2 plan).")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--ffn_expansion", type=int, default=1,
                        help="FFN expansion factor. Default 1 (no expansion) "
                             "for matched comparison with v1 baseline reference.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--pc_rate", type=float, default=0.001,
                        help="v2 only; ignored for --arch dead.")
    parser.add_argument("--pred_learning_rate", type=float, default=0.0001,
                        help="v2 only; ignored for --arch dead.")
    parser.add_argument("--homeostatic_decay", type=float, default=0.001,
                        help="v2 only; ignored for --arch dead.")
    parser.add_argument("--set_point_adapt_rate", type=float, default=1e-6,
                        help="v2 only; ignored for --arch dead.")
    parser.add_argument("--compressed_episodes", action="store_true", default=False,
                        help="v2 only; ignored for --arch dead.")
    parser.add_argument(
        "--no_consolidation", action="store_true", default=False,
        help="v2 only: disable consolidation during training. By default "
             "M5 runs with consolidation ON (the M4 STOP GATE result is "
             "part of what's being validated end-to-end). Set this only "
             "for ablation comparisons.",
    )
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="runs/m5")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Defaults to <arch>_seed<seed>.")
    parser.add_argument("--tokenizer", type=str, default="bpe", choices=["char", "bpe"])
    parser.add_argument("--bpe_vocab_size", type=int, default=32000)
    parser.add_argument("--load_tokenizer", type=str, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--lr_schedule", type=str, default="cosine",
                        choices=["constant", "cosine"])
    parser.add_argument("--lr_warmup_epochs", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    run_name = args.run_name or f"{args.arch}_seed{args.seed}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] {output_dir}")
    print(f"[arch] {args.arch}")

    # --- Tokenizer ---
    if args.tokenizer == "bpe":
        if args.load_tokenizer:
            tokenizer = BPETokenizer.load(args.load_tokenizer)
            print(f"[tokenizer] loaded BPE from {args.load_tokenizer}")
        else:
            sample_text = load_corpus_sample(args.data_dir)
            tokenizer = BPETokenizer(vocab_size=args.bpe_vocab_size)
            tokenizer.train(sample_text)
            print(f"[tokenizer] trained BPE vocab={tokenizer.vocab_size}")
    else:
        sample_text = load_corpus(args.data_dir)
        tokenizer = CharTokenizer(sample_text)
        print(f"[tokenizer] char vocab={tokenizer.vocab_size}")

    # --- Data ---
    if args.file_list:
        file_paths = load_file_list(args.file_list)
        data = load_corpus_as_tensor(*file_paths, tokenizer=tokenizer)
    else:
        data = load_corpus_as_tensor(args.data_dir, tokenizer=tokenizer)
    print(f"[data] {data.numel():,} tokens")

    n = data.numel()
    val_fraction = max(0.0, min(0.5, args.val_fraction))
    split = int(n * (1.0 - val_fraction))
    train_data, val_data = data[:split], data[split:]
    print(f"[split] {1.0 - val_fraction:.0%} train / {val_fraction:.0%} val")

    train_ds = CharDataset(
        train_data, seq_len=args.seq_len, tokenizer=tokenizer, stride=args.stride,
    )
    val_ds = CharDataset(
        val_data, seq_len=args.seq_len, tokenizer=tokenizer, stride=args.stride,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"[batches] train={len(train_loader)} val={len(val_loader)}")

    # --- Model ---
    model = _build_model(args, tokenizer.vocab_size, device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_buffer = sum(b.numel() for b in model.buffers())
    print(f"[model] arch={args.arch} trainable={n_train:,} buffers={n_buffer:,}")

    # --- Optimizer (DirectMLAdamW for DirectML compatibility) ---
    from luthi.optimizer import DirectMLAdamW
    optimizer = DirectMLAdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
    )

    # --- Training loop ---
    train_losses: list[float] = []
    val_losses: list[float] = []
    total_nan_events = 0
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        lr_for_epoch = _compute_lr(args, epoch - 1)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_for_epoch

        train_loss, nans = _train_one_epoch(
            model, train_loader, optimizer, device
        )
        val_loss = _evaluate(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        total_nan_events += nans
        best_val = min(best_val, val_loss)

        print(
            f"[epoch {epoch:3d}/{args.epochs}] "
            f"train={train_loss:.4f} val={val_loss:.4f} "
            f"lr={lr_for_epoch:.2e} nans={nans}"
        )

    # --- Save final checkpoint + metrics ---
    _save_checkpoint(model, output_dir / "model_final", args.epochs, vars(args))
    metrics = {
        "arch": args.arch,
        "seed": args.seed,
        "config": vars(args),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val": best_val,
        "nan_events": total_nan_events,
        "trainable_params": n_train,
        "buffer_params": n_buffer,
    }
    (output_dir / "results.json").write_text(json.dumps(metrics, indent=2))
    print(f"[done] saved to {output_dir}")
    print(
        f"[summary] arch={args.arch} seed={args.seed} "
        f"best_val={best_val:.4f} final_train={train_losses[-1]:.4f}"
    )


if __name__ == "__main__":
    main()
