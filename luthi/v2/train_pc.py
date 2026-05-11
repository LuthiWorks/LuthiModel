"""Training script for PredictiveCodingLM (v2).

Focused subset of v1's `luthi/train.py`. Supports the M3 sanity-check
workflow per `docs/V2_IMPLEMENTATION_PLAN.md`:

- Train for `--epochs` epochs (default 59 per refinement 1).
- At epoch 10, run a checkpoint trigger that inspects loss trajectory,
  NFF signal, prediction Frobenius norm, and NaN events. If any indicate
  poor convergence, halt and emit `grid_search_needed.json` with
  diagnostics and the recommended grid-search command. Otherwise continue.

Out of scope for M3: resume-from-checkpoint, encrypted checkpoints,
gradient checkpointing, multimodal, spiking. v2's training story is
language-only for the pilot.
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


# ---------------------------------------------------------------------------
# Checkpoint trigger
# ---------------------------------------------------------------------------

def evaluate_checkpoint_trigger(
    train_losses: list[float],
    val_losses: list[float],
    nff_signal: float,
    prediction_frob_norms: list[float],
    nan_events: int,
    nff_threshold: float = 0.01,
    prediction_growth_max: float = 100.0,
) -> dict:
    """Decide whether 10-epoch dynamics indicate convergence health.

    Returns a dict with `healthy: bool` plus per-signal pass/fail booleans
    and the metrics that drove the decision. Caller writes the dict to
    `grid_search_needed.json` when unhealthy.

    Pass criteria (refinement 1):
    - Train loss decreased monotonically (or close to it) over the window.
    - Val loss did not diverge.
    - NFF signal >= nff_threshold (layer is non-feedforward).
    - Prediction Frobenius norm trajectory bounded (late/early ratio <
      `prediction_growth_max`, no NaN).
    - Zero NaN events anywhere.
    """
    train_decreased = train_losses[-1] < train_losses[0]
    val_did_not_diverge = (
        val_losses[-1] < val_losses[0] * 1.5
        if val_losses[0] > 0 else True
    )
    nff_ok = nff_signal >= nff_threshold

    if len(prediction_frob_norms) >= 2:
        early = prediction_frob_norms[0] + 1e-8
        late = prediction_frob_norms[-1]
        prediction_growth_bounded = (
            (late / early) < prediction_growth_max
            and not math.isnan(late)
            and not math.isinf(late)
        )
    else:
        prediction_growth_bounded = True

    no_nans = nan_events == 0

    healthy = (
        train_decreased
        and val_did_not_diverge
        and nff_ok
        and prediction_growth_bounded
        and no_nans
    )

    return {
        "healthy": healthy,
        "checks": {
            "train_loss_decreased": train_decreased,
            "val_did_not_diverge": val_did_not_diverge,
            "nff_above_threshold": nff_ok,
            "prediction_growth_bounded": prediction_growth_bounded,
            "no_nan_events": no_nans,
        },
        "metrics": {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "nff_signal": nff_signal,
            "nff_threshold": nff_threshold,
            "prediction_frob_norms": prediction_frob_norms,
            "nan_events": nan_events,
        },
    }


def emit_grid_search_marker(
    output_dir: Path,
    diagnostic: dict,
    args: argparse.Namespace,
) -> Path:
    """Write `grid_search_needed.json` and print instructions.

    Per refinement 1: 12-cell grid over pc_rate × pred_learning_rate at
    10 epochs each, CPU-bound, doesn't block the GPU ablation pipeline.
    """
    marker_path = output_dir / "grid_search_needed.json"
    payload = {
        "diagnostic": diagnostic,
        "current_hp": {
            "pc_rate": args.pc_rate,
            "pred_learning_rate": args.pred_learning_rate,
        },
        "recommended_grid": {
            "pc_rate": [1e-4, 5e-4, 1e-3, 2e-3],
            "pred_learning_rate": [5e-5, 1e-4, 5e-4],
        },
        "command_template": (
            "python -m luthi.v2.grid_search "
            f"--data_dir {args.data_dir} "
            f"--tokenizer {args.tokenizer} "
            f"--load_tokenizer {args.load_tokenizer} "
            f"--d_model {args.d_model} --n_blocks {args.n_blocks} "
            f"--seq_len {args.seq_len} --batch_size {args.batch_size} "
            f"--stride {args.stride} --epochs 10 "
            f"--output_dir {output_dir / 'grid_search'}"
        ),
    }
    marker_path.write_text(json.dumps(payload, indent=2))
    print(
        f"\n[CHECKPOINT TRIGGER] 10-epoch dynamics indicate poor convergence.\n"
        f"  Failed checks: "
        f"{[k for k, v in diagnostic['checks'].items() if not v]}\n"
        f"  Marker file: {marker_path}\n"
        f"  Run grid search before continuing to epoch {args.epochs}."
    )
    return marker_path


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: PredictiveCodingLM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, int]:
    """Returns (mean_loss, nan_count)."""
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
        # Audit 2026-05-11 fix: clip gradient norm before optimizer.step().
        # Without this the attention weights see unbounded gradients on rare
        # loss spikes; the PC living state would also already have consumed
        # the bad forward (caught by the NaN guard, but only for actual
        # NaN/Inf — large-but-finite spikes still propagate). max_norm=1.0
        # matches v1's train.py default.
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        optimizer.step()
        # Audit 2026-05-11 fix: release per-layer forward snapshots so they
        # don't accumulate between steps (~67 MB/layer at 4096d would
        # otherwise sit dead across all 36 blocks).
        if hasattr(model, "clear_forward_cache"):
            model.clear_forward_cache()

        total += loss.item()
        n_batches += 1

    return (total / max(n_batches, 1)), nan_count


@torch.no_grad()
def evaluate(
    model: PredictiveCodingLM,
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


def measure_prediction_frob_norm(model: PredictiveCodingLM) -> float:
    """Mean Frobenius norm of the prediction matrix across all blocks."""
    norms = [
        block.living_ffn.prediction.norm().item() for block in model.blocks
    ]
    return sum(norms) / len(norms)


def _save_checkpoint_v2(
    model: PredictiveCodingLM,
    path,
    epoch: int = 0,
    config: dict | None = None,
) -> None:
    """Save a v2 checkpoint, encrypted if LUTHI_CHECKPOINT_KEY is set,
    plaintext .pt otherwise (with a warning).

    Audit 2026-05-11 fix: previously this code saved plaintext .pt files
    unconditionally. v1's training script encrypts via luthi.checkpoint;
    v2 should match the project's encryption invariant. Falling back to
    plaintext with a loud warning preserves the local-dev flow where the
    user hasn't set up the encryption key yet.
    """
    from pathlib import Path as _Path
    path = _Path(path)
    try:
        from luthi.checkpoint import build_checkpoint, save_checkpoint
        ckpt = build_checkpoint(model, epoch=epoch, config=config or {})
        encrypted_path = save_checkpoint(ckpt, path)
        print(f"[checkpoint] encrypted save: {encrypted_path}")
    except ValueError as e:
        # _get_password raises ValueError when no key is configured.
        # Fall back to plaintext with a warning.
        plain_path = path.with_suffix(".pt")
        torch.save(model.state_dict(), plain_path)
        print(
            f"[checkpoint] WARNING: no LUTHI_CHECKPOINT_KEY set; saved "
            f"plaintext to {plain_path}. Reason: {e}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_buffer_dtypes(spec: str | None) -> dict[str, torch.dtype] | None:
    if not spec:
        return None
    dtype_map = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp64": torch.float64,
    }
    out: dict[str, torch.dtype] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(
                f"--buffer_dtypes entry '{pair}' missing '='. "
                f"Format: 'momentum=bf16,set_point=bf16'."
            )
        name, dt = pair.split("=", 1)
        out[name.strip()] = dtype_map[dt.strip()]
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Train PredictiveCodingLM (v2) — see V2_IMPLEMENTATION_PLAN.md M3."
    )
    parser.add_argument("--data_dir", type=str, default="corpus_build/gutenberg_100")
    parser.add_argument("--file-list", dest="file_list", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=59,
                        help="Total epochs (refinement 1: 59-epoch sanity check).")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument(
        "--ffn_expansion", type=int, default=1,
        help="FFN expansion factor (audit 2026-05-10). 1 = no expansion "
             "(original linear PC FFN). 4 = standard transformer pattern "
             "(d_model → 4*d_model → d_model with GELU + PC layer in the "
             "expanded space). Off by default at pilot scale; turn on for "
             "M5+ to give the FFN real computational capacity.",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--pc_rate", type=float, default=0.001)
    parser.add_argument("--pred_learning_rate", type=float, default=0.0001)
    parser.add_argument("--homeostatic_decay", type=float, default=0.001)
    parser.add_argument("--set_point_adapt_rate", type=float, default=1e-6)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="runs/v2_pilot")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="bpe", choices=["char", "bpe"])
    parser.add_argument("--bpe_vocab_size", type=int, default=32000)
    parser.add_argument("--load_tokenizer", type=str, default=None,
                        help="Path to a pre-trained BPE tokenizer JSON.")
    parser.add_argument("--buffer_dtypes", type=str, default=None,
                        help="Per-buffer dtype overrides, e.g. 'momentum=bf16'.")
    parser.add_argument(
        "--checkpoint_trigger_epoch", type=int, default=10,
        help="Epoch to evaluate the checkpoint trigger (refinement 1).",
    )
    parser.add_argument(
        "--val_fraction", type=float, default=0.05,
        help="Fraction of the corpus held out for validation. Default 0.05 "
             "(95/5 split) — matches train_pc.py's historical default. v1's "
             "train.py used 0.10 (90/10) historically; either works, but "
             "both scripts now share this CLI knob so comparisons stay clean.",
    )
    parser.add_argument(
        "--lr_schedule", type=str, default="cosine",
        choices=["constant", "cosine"],
        help="Learning-rate schedule. 'cosine' = cosine decay with linear "
             "warmup (audit 2026-05-10: flat LR over 59 epochs leaves "
             "convergence on the table). 'constant' preserves the legacy flat "
             "behavior for direct comparison.",
    )
    parser.add_argument(
        "--lr_warmup_epochs", type=int, default=2,
        help="Linear warmup duration (in epochs) before cosine decay starts.",
    )
    parser.add_argument(
        "--compressed_episodes", action="store_true", default=False,
        help="Store episode_values as INT8 (4x memory reduction). Audit "
             "2026-05-10: production scale (4096d × 36 blocks) is infeasible "
             "with FP32 episodes (~150 GB); INT8 + future low-rank delta "
             "compression bring it tractable. Off by default at pilot scale.",
    )
    parser.add_argument(
        "--skip_checkpoint_trigger", action="store_true", default=False,
        help="Disable the 10-epoch checkpoint trigger (for M3 grid-search "
             "sub-runs that already know they are short).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    output_dir = Path(args.output_dir)
    if args.run_name:
        output_dir = output_dir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] {output_dir}")

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
        train_data, seq_len=args.seq_len,
        tokenizer=tokenizer, stride=args.stride,
    )
    val_ds = CharDataset(
        val_data, seq_len=args.seq_len,
        tokenizer=tokenizer, stride=args.stride,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
    )
    print(f"[batches] train={len(train_loader)} val={len(val_loader)}")

    # --- Model ---
    buffer_dtypes = _parse_buffer_dtypes(args.buffer_dtypes)
    model = PredictiveCodingLM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_blocks=args.n_blocks,
        ffn_expansion=args.ffn_expansion,
        max_seq_len=args.seq_len,
        pc_rate=args.pc_rate,
        pred_learning_rate=args.pred_learning_rate,
        homeostatic_decay=args.homeostatic_decay,
        set_point_adapt_rate=args.set_point_adapt_rate,
        compressed_episodes=args.compressed_episodes,
        buffer_dtypes=buffer_dtypes,
    ).to(device)
    counts = model.total_parameters()
    print(
        f"[model] trainable={counts['trainable']:,} "
        f"living_buffers={counts['living_buffers']:,}"
    )

    # --- Optimizer (DirectMLAdamW for AMD/DirectML — see luthi.optimizer) ---
    from luthi.optimizer import DirectMLAdamW
    optimizer = DirectMLAdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
    )

    # --- LR scheduler ---
    # cosine_with_warmup gives a linear ramp-up over args.lr_warmup_epochs
    # then cosine decay to ~0 over the remaining epochs. The constant
    # schedule preserves the legacy flat-LR behavior for direct comparison
    # against historical runs.
    def _compute_lr(epoch: int) -> float:
        if args.lr_schedule == "constant":
            return args.lr
        # Cosine with linear warmup.
        if epoch < args.lr_warmup_epochs:
            return args.lr * (epoch + 1) / max(1, args.lr_warmup_epochs)
        # Decay from epoch=warmup to epoch=total over a cosine curve.
        progress = (epoch - args.lr_warmup_epochs) / max(
            1, args.epochs - args.lr_warmup_epochs
        )
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    # --- Training loop ---
    train_losses: list[float] = []
    val_losses: list[float] = []
    pred_frob_norms: list[float] = []
    total_nan_events = 0

    nff_probe_x = next(iter(val_loader))[0][:1].to(device)

    for epoch in range(1, args.epochs + 1):
        # Apply scheduled LR for this epoch (epoch=1 is index 0).
        lr_for_epoch = _compute_lr(epoch - 1)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_for_epoch

        train_loss, nans = train_one_epoch(
            model, train_loader, optimizer, device
        )
        val_loss = evaluate(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        total_nan_events += nans
        pred_frob_norms.append(measure_prediction_frob_norm(model))

        print(
            f"[epoch {epoch:3d}/{args.epochs}] "
            f"train={train_loss:.4f} val={val_loss:.4f} "
            f"pred_frob={pred_frob_norms[-1]:.3f} "
            f"lr={lr_for_epoch:.2e} "
            f"nans={nans}"
        )

        if (
            not args.skip_checkpoint_trigger
            and epoch == args.checkpoint_trigger_epoch
        ):
            nff = model.non_feedforward_signal(nff_probe_x)
            diagnostic = evaluate_checkpoint_trigger(
                train_losses=train_losses,
                val_losses=val_losses,
                nff_signal=nff,
                prediction_frob_norms=pred_frob_norms,
                nan_events=total_nan_events,
            )
            (output_dir / "checkpoint_trigger.json").write_text(
                json.dumps(diagnostic, indent=2)
            )
            if not diagnostic["healthy"]:
                emit_grid_search_marker(output_dir, diagnostic, args)
                # Save what we have and exit; caller picks up via marker.
                _save_checkpoint_v2(
                    model, output_dir / "model_at_checkpoint_trigger",
                    epoch=epoch, config=vars(args),
                )
                return
            print("[checkpoint trigger] healthy; continuing")

    # --- Final save ---
    _save_checkpoint_v2(
        model, output_dir / "model_final",
        epoch=args.epochs, config=vars(args),
    )
    (output_dir / "metrics.json").write_text(json.dumps({
        "train_losses": train_losses,
        "val_losses": val_losses,
        "prediction_frob_norms": pred_frob_norms,
        "nan_events": total_nan_events,
    }, indent=2))
    print(f"[done] saved to {output_dir}")


if __name__ == "__main__":
    main()
