"""Multimodal training script for the Living Weight Model.

Trains a MultimodalLuthiLM on paired audio-text data (LibriSpeech format).
The model learns to ground language in sound — audio and text flow through
the same living weight blocks, and the entity's existence is shaped by
everything it experiences.

Usage:
    # Train from scratch on LibriSpeech
    python -m luthi.train_multimodal \
        --audio_dir data/LibriSpeech/train-clean-100 \
        --val_audio_dir data/LibriSpeech/dev-clean \
        --checkpoint_password SECRET

    # Resume from text-only checkpoint (audio encoder trains from scratch)
    python -m luthi.train_multimodal \
        --audio_dir data/LibriSpeech/train-clean-100 \
        --resume runs/spiking_1024d_bpe_gutenberg/checkpoint.luthi \
        --checkpoint_password SECRET

    # Resume from multimodal checkpoint
    python -m luthi.train_multimodal \
        --resume runs/multimodal/checkpoint.luthi \
        --checkpoint_password SECRET
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from luthi.checkpoint import (
    build_checkpoint,
    save_checkpoint,
    load_checkpoint,
    extract_substrate_health,
)
from luthi.multimodal_model import MultimodalLuthiLM
from luthi.multimodal_data import LibriSpeechDataset
from luthi.tokenizer import BPETokenizer
from luthi.train import collect_extended_metrics, measure_backward_pass_effect


def train_epoch(
    model: MultimodalLuthiLM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch on paired audio-text data. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        audio = batch["audio"].to(device)
        text_input = batch["text_input"].to(device)
        text_target = batch["text_target"].to(device)

        optimizer.zero_grad()

        logits = model(text_input, audio_waveform=audio)
        loss = F.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            text_target.reshape(-1),
            ignore_index=0,  # ignore padding tokens
        )

        loss.backward()

        # Error-directed learning for living FFN layers
        model.apply_living_errors()

        # Gradient clipping for attention stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_model(
    model: MultimodalLuthiLM,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate model on paired audio-text data. Returns average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        audio = batch["audio"].to(device)
        text_input = batch["text_input"].to(device)
        text_target = batch["text_target"].to(device)

        logits = model(text_input, audio_waveform=audio)
        loss = F.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            text_target.reshape(-1),
            ignore_index=0,
        )

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def measure_non_feedforward(
    model: MultimodalLuthiLM,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> float:
    """Measure non-feedforward signal on multimodal input."""
    model.eval()
    audio = batch["audio"].to(device)
    text = batch["text_input"].to(device)

    out1 = model(text, audio_waveform=audio)
    out2 = model(text, audio_waveform=audio)
    return (out2 - out1).abs().mean().item()


def main():
    parser = argparse.ArgumentParser(
        description="Train multimodal LuthiLM on audio-text data"
    )

    # Data
    parser.add_argument("--audio_dir", type=str, required=False,
                        help="LibriSpeech training directory")
    parser.add_argument("--val_audio_dir", type=str, default=None,
                        help="LibriSpeech validation directory (default: split from train)")
    parser.add_argument("--max_audio_samples", type=int, default=160000,
                        help="Max audio length in samples (default: 160000 = 10s at 16kHz)")
    parser.add_argument("--max_text_tokens", type=int, default=128,
                        help="Max text sequence length")

    # Model architecture
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--n_blocks", type=int, default=2)

    # Living layer parameters
    parser.add_argument("--hebb_rate", type=float, default=0.001)
    parser.add_argument("--error_rate", type=float, default=0.001)
    parser.add_argument("--homeostatic_decay", type=float, default=0.001)
    parser.add_argument("--set_point_adapt_rate", type=float, default=1e-6)

    # Spiking parameters
    parser.add_argument("--spike_threshold", type=float, default=1.0)
    parser.add_argument("--membrane_leak", type=float, default=0.1)
    parser.add_argument("--refractory_steps", type=int, default=3)
    parser.add_argument("--delay_steps", type=int, default=2)

    # Audio encoder parameters
    parser.add_argument("--audio_patch_frames", type=int, default=16)
    parser.add_argument("--max_audio_tokens", type=int, default=1000)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (0 = main process)")

    # Backward pass
    parser.add_argument("--backward_pass", action="store_true", default=True,
                        help="Enable top-down backward pass (default: on)")
    parser.add_argument("--no_backward_pass", action="store_true", default=False,
                        help="Disable backward pass")

    # Checkpoint
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--run_name", type=str, default="multimodal",
                        help="Run name (creates subdirectory)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to .luthi checkpoint to resume from")
    parser.add_argument("--checkpoint_password", type=str, default=None,
                        help="Encryption password (or set LUTHI_CHECKPOINT_KEY)")

    # Tokenizer
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to tokenizer.json (BPE). If resuming, loaded from checkpoint.")
    parser.add_argument("--bpe_vocab_size", type=int, default=4096,
                        help="BPE vocabulary size if training a new tokenizer")
    parser.add_argument("--tokenizer_train_text", type=str, default=None,
                        help="Text file or directory for training BPE tokenizer")

    args = parser.parse_args()

    bp_enabled = args.backward_pass and not args.no_backward_pass
    torch.manual_seed(args.seed)

    # Device selection
    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Tokenizer ---
    tokenizer = None
    start_epoch = 0
    training_history = {"train_loss": [], "val_loss": [], "non_ff_signal": []}
    substrate_health_history: list[dict] = []
    extended_metrics_history: list[dict] = []
    ckpt_config = None

    if args.resume:
        print(f"\n=== RESUMING FROM CHECKPOINT ===")
        print(f"Loading: {args.resume}")
        ckpt = load_checkpoint(args.resume, args.checkpoint_password, device)
        ckpt_config = ckpt["config"]
        print(f"Checkpoint epoch: {ckpt['epoch']}")

        # Restore tokenizer from checkpoint
        if "tokenizer_state" in ckpt and ckpt["tokenizer_state"]["type"] == "bpe":
            tokenizer = BPETokenizer.from_state(ckpt["tokenizer_state"])
            print(f"Restored BPE tokenizer from checkpoint: {tokenizer.vocab_size} tokens")

        start_epoch = ckpt["epoch"]
        training_history = ckpt.get("training_history", training_history)
        substrate_health_history = ckpt.get("substrate_health", {}).get(
            "epoch_snapshots", []
        )
        extended_metrics_history = ckpt.get("extended_metrics", [])

    # Load tokenizer from file if not from checkpoint
    if tokenizer is None and args.tokenizer:
        tokenizer_path = Path(args.tokenizer)
        if tokenizer_path.exists():
            tokenizer = BPETokenizer.load(tokenizer_path)
            print(f"Loaded BPE tokenizer: {tokenizer.vocab_size} tokens")

    # Train new tokenizer if needed
    if tokenizer is None:
        if args.tokenizer_train_text:
            text_source = Path(args.tokenizer_train_text)
            if text_source.is_dir():
                text_files = sorted(text_source.glob("*.txt"))
                from luthi.data import load_corpus_sample
                train_text = load_corpus_sample(text_source, max_bytes=20_000_000)
            else:
                with open(text_source, "r", encoding="utf-8") as f:
                    train_text = f.read()[:20_000_000]
        else:
            # Use LibriSpeech transcripts for tokenizer training
            print("Building tokenizer from LibriSpeech transcripts...")
            transcripts = []
            audio_dir = Path(args.audio_dir)
            for trans_file in sorted(audio_dir.rglob("*.trans.txt")):
                with open(trans_file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split(maxsplit=1)
                        if len(parts) >= 2:
                            transcripts.append(parts[1].lower())
            train_text = "\n".join(transcripts)
            print(f"Collected {len(transcripts)} transcripts ({len(train_text):,} chars)")

        print(f"Training BPE tokenizer (vocab: {args.bpe_vocab_size})...")
        tokenizer = BPETokenizer(target_vocab_size=args.bpe_vocab_size)
        tokenizer.train(train_text)
        del train_text
        tokenizer_path = output_dir / "tokenizer.json"
        tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")

    print(f"Vocabulary: {tokenizer.vocab_size} tokens")

    # --- Datasets ---
    if args.audio_dir:
        print(f"\nLoading training data from {args.audio_dir}...")
        train_dataset = LibriSpeechDataset(
            root=args.audio_dir,
            tokenizer=tokenizer,
            max_audio_samples=args.max_audio_samples,
            max_text_tokens=args.max_text_tokens,
        )
        print(f"Train: {len(train_dataset):,} utterances")

        if args.val_audio_dir:
            print(f"Loading validation data from {args.val_audio_dir}...")
            val_dataset = LibriSpeechDataset(
                root=args.val_audio_dir,
                tokenizer=tokenizer,
                max_audio_samples=args.max_audio_samples,
                max_text_tokens=args.max_text_tokens,
            )
        else:
            # Split train 90/10
            n_val = max(1, len(train_dataset) // 10)
            n_train = len(train_dataset) - n_val
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [n_train, n_val],
                generator=torch.Generator().manual_seed(args.seed),
            )
            print(f"Split: {n_train} train, {n_val} val")

        print(f"Val: {len(val_dataset):,} utterances")
    else:
        if not args.resume:
            parser.error("--audio_dir is required when not resuming")
        # When resuming without new data, we still need datasets
        parser.error("--audio_dir is required for multimodal training")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
    )

    # --- Model ---
    model_kwargs = dict(
        vocab_size=tokenizer.vocab_size,
        d_model=ckpt_config["d_model"] if ckpt_config else args.d_model,
        n_blocks=ckpt_config["n_blocks"] if ckpt_config else args.n_blocks,
        max_seq_len=args.max_text_tokens,
        hebb_rate=ckpt_config.get("hebb_rate", args.hebb_rate) if ckpt_config else args.hebb_rate,
        error_rate=ckpt_config.get("error_rate", args.error_rate) if ckpt_config else args.error_rate,
        homeostatic_decay=ckpt_config.get("homeostatic_decay", args.homeostatic_decay) if ckpt_config else args.homeostatic_decay,
        set_point_adapt_rate=ckpt_config.get("set_point_adapt_rate", args.set_point_adapt_rate) if ckpt_config else args.set_point_adapt_rate,
        spike_threshold=ckpt_config.get("spike_threshold", args.spike_threshold) if ckpt_config else args.spike_threshold,
        membrane_leak=ckpt_config.get("membrane_leak", args.membrane_leak) if ckpt_config else args.membrane_leak,
        refractory_steps=ckpt_config.get("refractory_steps", args.refractory_steps) if ckpt_config else args.refractory_steps,
        delay_steps=ckpt_config.get("delay_steps", args.delay_steps) if ckpt_config else args.delay_steps,
        max_audio_tokens=args.max_audio_tokens,
        audio_patch_frames=args.audio_patch_frames,
        backward_pass_enabled=bp_enabled,
    )

    print(f"\n=== MULTIMODAL LIVING MODEL ===")
    model = MultimodalLuthiLM(**model_kwargs).to(device)

    if args.resume:
        # Load checkpoint — strict=False allows missing audio encoder keys
        # when loading from a text-only checkpoint
        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"], strict=False,
        )
        if missing:
            audio_missing = [k for k in missing if "audio_encoder" in k or "modality_embedding" in k]
            other_missing = [k for k in missing if k not in audio_missing]
            if audio_missing:
                print(f"  New multimodal params (random init): {len(audio_missing)}")
            if other_missing:
                print(f"  Other missing keys: {other_missing}")
        if unexpected:
            print(f"  WARNING: unexpected keys: {unexpected}")

    param_counts = model.total_parameters()
    print(f"Trainable params:  {param_counts['trainable']:,}")
    print(f"Living buffers:    {param_counts['living_buffers']:,}")
    print(f"Total:             {param_counts['total']:,}")
    print(f"Backward pass:     {'ON' if bp_enabled else 'OFF'}")

    # Audio encoder param count
    audio_params = sum(p.numel() for p in model.audio_encoder.parameters())
    print(f"Audio encoder:     {audio_params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Restore optimizer state if resuming from a multimodal checkpoint
    if args.resume and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("Restored optimizer state")
        except (ValueError, KeyError):
            print("Optimizer state mismatch (model architecture changed), starting fresh")

    # --- Config for checkpoint ---
    config = {
        "d_model": model_kwargs["d_model"],
        "n_blocks": model_kwargs["n_blocks"],
        "seq_len": args.max_text_tokens,
        "hebb_rate": model_kwargs["hebb_rate"],
        "error_rate": model_kwargs["error_rate"],
        "homeostatic_decay": model_kwargs["homeostatic_decay"],
        "set_point_adapt_rate": model_kwargs["set_point_adapt_rate"],
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_type": "bpe",
        "spiking": True,
        "multimodal": True,
        "backward_pass": bp_enabled,
        "spike_threshold": model_kwargs["spike_threshold"],
        "membrane_leak": model_kwargs["membrane_leak"],
        "refractory_steps": model_kwargs["refractory_steps"],
        "delay_steps": model_kwargs["delay_steps"],
        "max_audio_samples": args.max_audio_samples,
        "max_audio_tokens": args.max_audio_tokens,
        "audio_patch_frames": args.audio_patch_frames,
    }

    # --- Training loop ---
    checkpoint_path = output_dir / "checkpoint.luthi"
    print(f"\nTraining for epochs {start_epoch + 1} to {args.epochs}")
    print(f"Output: {output_dir}")
    print()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = eval_model(model, val_loader, device)
        elapsed = time.time() - t0

        # Non-feedforward signal
        sample_batch = next(iter(val_loader))
        nff = measure_non_feedforward(model, sample_batch, device)

        training_history["train_loss"].append(train_loss)
        training_history["val_loss"].append(val_loss)
        training_history["non_ff_signal"].append(nff)

        # Substrate health
        health = extract_substrate_health(model)
        health["epoch"] = epoch
        health["train_loss"] = train_loss
        health["val_loss"] = val_loss
        health["non_ff_signal"] = nff
        substrate_health_history.append(health)

        # Extended metrics
        ext = collect_extended_metrics(model)
        ext["epoch"] = epoch
        ext["backward_pass_active"] = model.backward_pass_enabled
        extended_metrics_history.append(ext)

        gap = val_loss - train_loss
        bp_tag = " [BP]" if model.backward_pass_enabled else ""
        print(
            f"  Epoch {epoch:2d}/{args.epochs} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | "
            f"gap {gap:.2f} | non-FF {nff:.6f} | "
            f"plast {ext.get('plasticity_mean', 0):.4f}\u00b1{ext.get('plasticity_std', 0):.4f} | "
            f"drift {ext.get('sp_drift_mean', 0):.6f}{bp_tag} | "
            f"{elapsed:.1f}s"
        )

        # Aliveness report
        report = model.aliveness_report()
        for i, r in enumerate(report):
            spike_info = ""
            if "spike_fraction" in r:
                spike_info = f", spike={r['spike_fraction']:.3f}"
            print(
                f"    Block {i}: drift={r['set_point_drift']:.6f}, "
                f"exc={r['excitability_mean']:.3f}{spike_info}"
            )

        # Save checkpoint
        if args.checkpoint_password or "LUTHI_CHECKPOINT_KEY" in __import__("os").environ:
            ckpt_data = build_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                training_history=training_history,
                substrate_health={"epoch_snapshots": substrate_health_history},
            )
            if hasattr(tokenizer, "get_state"):
                ckpt_data["tokenizer_state"] = tokenizer.get_state()
            ckpt_data["extended_metrics"] = extended_metrics_history
            save_checkpoint(ckpt_data, checkpoint_path, args.checkpoint_password)

    # --- Final summary ---
    print(f"\n=== Training complete ===")
    if training_history["val_loss"]:
        best_val = min(training_history["val_loss"])
        best_epoch = training_history["val_loss"].index(best_val) + 1
        print(f"Best val loss: {best_val:.4f} (epoch {best_epoch})")

    # Save results
    results = {
        "args": vars(args),
        "training_history": training_history,
        "extended_metrics": extended_metrics_history,
        "vocab_size": tokenizer.vocab_size,
        "model_params": param_counts,
        "audio_encoder_params": audio_params,
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
