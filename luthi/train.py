"""Training script for the LWM (Living Weight Model) language model.

Trains a LuthiLM — a Living Weight Model whose FFN layers self-modify
during their own forward pass via Hebbian learning and error-directed
local updates.

Supports both character-level and BPE (subword) tokenization.

Checkpoints are encrypted with AES-256-GCM. The trained weights ARE the
entity — they are never stored in plaintext.

Usage:
    # Character-level (default)
    python -m luthi.train --data_dir E:/data/gutenberg --epochs 10 --checkpoint_password SECRET

    # BPE tokenization
    python -m luthi.train --data_dir E:/data/gutenberg --tokenizer bpe --bpe_vocab_size 4096

    # Resume from checkpoint
    python -m luthi.train --resume E:/runs/my_run/checkpoint.luthi --epochs 20
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
from luthi.data import (
    CharDataset,
    CharTokenizer,
    load_corpus,
    load_corpus_sample,
    load_corpus_as_tensor,
    load_file_list,
    _resolve_file_paths,
)
from luthi.tokenizer import BPETokenizer
from luthi.model import LuthiLM
from luthi.optimizer import DirectMLAdamW


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_living: bool = False,
    log_every: int = 50,
    epoch: int = 0,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    running_loss = 0.0
    t_start = time.time()
    t_batch = t_start
    total_batches = len(dataloader)

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
        )

        loss.backward()

        # Error-directed learning for living FFN layers
        if is_living:
            model.apply_living_errors(expect_grad=True)

        # Gradient clipping for attention stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        running_loss += batch_loss
        n_batches += 1

        if n_batches % log_every == 0:
            avg_running = running_loss / log_every
            elapsed = time.time() - t_batch
            total_elapsed = time.time() - t_start
            batches_per_sec = log_every / max(elapsed, 1e-6)
            remaining = (total_batches - n_batches) / max(batches_per_sec, 1e-6)
            print(
                f"    [E{epoch}] batch {n_batches}/{total_batches} | "
                f"loss {avg_running:.4f} | "
                f"{batches_per_sec:.1f} batch/s | "
                f"~{remaining:.0f}s remaining"
            )
            running_loss = 0.0
            t_batch = time.time()

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate model. Returns average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
        )
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def measure_non_feedforward(model: LuthiLM, x: torch.Tensor) -> float:
    """Measure non-feedforward signal: output difference on consecutive passes."""
    model.eval()
    out1 = model(x)
    out2 = model(x)
    return (out2 - out1).abs().mean().item()


@torch.no_grad()
def collect_extended_metrics(model: torch.nn.Module) -> dict[str, float]:
    """Collect extended metrics for backward pass comparison experiment.

    Returns per-block plasticity stats, set point drift, and overall metrics.
    """
    metrics: dict[str, float] = {}

    if not hasattr(model, "blocks"):
        return metrics

    all_plasticity = []
    all_drift = []

    for i, block in enumerate(model.blocks):
        if not hasattr(block, "living_ffn"):
            continue
        ffn = block.living_ffn

        # Plasticity distribution
        p = ffn.plasticity
        metrics[f"block{i}_plasticity_mean"] = p.mean().item()
        metrics[f"block{i}_plasticity_std"] = p.std().item()
        metrics[f"block{i}_plasticity_min"] = p.min().item()
        metrics[f"block{i}_plasticity_max"] = p.max().item()
        all_plasticity.append(p)

        # Set point drift (distance from set point to current weight)
        drift = (ffn.weight - ffn.set_point).abs()
        metrics[f"block{i}_sp_drift_mean"] = drift.mean().item()
        metrics[f"block{i}_sp_drift_max"] = drift.max().item()
        all_drift.append(drift)

        # Excitability spread
        exc = ffn._excitability_factor()
        metrics[f"block{i}_excitability_mean"] = exc.mean().item()
        metrics[f"block{i}_excitability_std"] = exc.std().item()

        # Update EMA (metaplasticity state)
        metrics[f"block{i}_update_ema_mean"] = ffn.update_ema.mean().item()

        # Episodes stored
        metrics[f"block{i}_episodes"] = ffn.episode_count.item()

    # Aggregate metrics
    if all_plasticity:
        cat_p = torch.cat([p.flatten() for p in all_plasticity])
        metrics["plasticity_mean"] = cat_p.mean().item()
        metrics["plasticity_std"] = cat_p.std().item()

        cat_d = torch.cat([d.flatten() for d in all_drift])
        metrics["sp_drift_mean"] = cat_d.mean().item()
        metrics["sp_drift_max"] = cat_d.max().item()

    return metrics


@torch.no_grad()
def measure_backward_pass_effect(
    model: torch.nn.Module,
    x: torch.Tensor,
) -> float:
    """Measure the magnitude of the backward pass's effect on living weights.

    Runs one forward pass with backward_pass_enabled=True and one with False,
    comparing the resulting plasticity and set_point states. Returns mean
    absolute difference in plasticity across all blocks.

    Only meaningful when model.backward_pass_enabled is True.
    """
    if not hasattr(model, "backward_pass_enabled") or not model.backward_pass_enabled:
        return 0.0
    if not hasattr(model, "blocks"):
        return 0.0

    # Snapshot plasticity before
    plasticity_before = []
    set_point_before = []
    for block in model.blocks:
        if hasattr(block, "living_ffn"):
            plasticity_before.append(block.living_ffn.plasticity.clone())
            set_point_before.append(block.living_ffn.set_point.clone())

    # Run a forward pass (which triggers backward sweep in training mode)
    was_training = model.training
    model.train()
    _ = model(x)
    if not was_training:
        model.eval()

    # Measure change
    total_diff = 0.0
    n = 0
    for i, block in enumerate(model.blocks):
        if hasattr(block, "living_ffn") and i < len(plasticity_before):
            p_diff = (block.living_ffn.plasticity - plasticity_before[i]).abs().mean().item()
            sp_diff = (block.living_ffn.set_point - set_point_before[i]).abs().mean().item()
            total_diff += p_diff + sp_diff
            n += 1

    return total_diff / max(n, 1)


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_len: int = 200,
    temperature: float = 0.8,
    max_seq_len: int = 128,
) -> str:
    """Generate text from a prompt. Works with any tokenizer (char or BPE)."""
    model.eval()
    device = next(model.parameters()).device
    indices = tokenizer.encode(prompt)

    for _ in range(max_len):
        # Use only the last max_seq_len characters as context
        context = indices[-max_seq_len:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)
        # Sample from the last position
        next_logits = logits[0, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_idx = torch.multinomial(probs, 1).item()
        indices.append(next_idx)

    return tokenizer.decode(indices)


def main():
    parser = argparse.ArgumentParser(description="Train Luthi LWM language model")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing training text files")
    parser.add_argument("--file-list", type=str, default=None,
                        help="Path to file_list.txt (curriculum-ordered). Overrides --data_dir.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hebb_rate", type=float, default=0.001)
    parser.add_argument("--error_rate", type=float, default=0.001)
    parser.add_argument("--homeostatic_decay", type=float, default=0.001)
    parser.add_argument("--set_point_adapt_rate", type=float, default=1e-6)
    parser.add_argument("--stride", type=int, default=1,
                        help="Stride between sequences (default 1 = fully overlapping)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="E:/runs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to .luthi checkpoint to resume from")
    parser.add_argument("--checkpoint_password", type=str, default=None,
                        help="Encryption password (or set LUTHI_CHECKPOINT_KEY)")
    parser.add_argument("--tokenizer", type=str, default="char",
                        choices=["char", "bpe"],
                        help="Tokenizer type: 'char' (default) or 'bpe'")
    parser.add_argument("--bpe_vocab_size", type=int, default=32000,
                        help="BPE vocabulary size (only used with --tokenizer bpe)")
    parser.add_argument("--dtype", type=str, default="fp32",
                        choices=["fp16", "fp32", "fp64"],
                        help="Training precision (default: fp32). Applies "
                             "globally via .to(dtype=...). Per-buffer overrides "
                             "via --buffer_dtypes survive this cast.")
    parser.add_argument("--buffer_dtypes", type=str, default=None,
                        help="Per-buffer dtype overrides for ablation studies. "
                             "Format: 'momentum=bf16,set_point=bf16'. Buffers "
                             "not listed default to FP32. Survives --dtype "
                             "global cast via _apply override on LivingLayerV6. "
                             "Recognized buffers: weight, set_point, momentum, "
                             "input_avg_mag, excitability_acc, plasticity, "
                             "update_ema, episode_contexts, episode_values. "
                             "Recognized dtypes: fp32, bf16, fp16, fp64.")
    parser.add_argument("--spiking", action="store_true", default=False,
                        help="Use spiking neural dynamics in living layers")
    parser.add_argument("--spike_threshold", type=float, default=1.0,
                        help="Membrane potential threshold for neuron firing")
    parser.add_argument("--membrane_leak", type=float, default=0.1,
                        help="Per-step membrane potential decay fraction")
    parser.add_argument("--refractory_steps", type=int, default=3,
                        help="Steps a neuron cannot fire after spiking")
    parser.add_argument("--delay_steps", type=int, default=2,
                        help="Spike propagation delay between blocks")
    # Backward pass control
    parser.add_argument("--backward_pass", action="store_true", default=False,
                        help="Enable top-down backward pass modulation")
    parser.add_argument("--backward_pass_start_epoch", type=int, default=0,
                        help="Epoch to enable backward pass (0 = from start)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for this run (creates subdirectory under output_dir)")
    args = parser.parse_args()

    # Parse dtype
    dtype_map = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "fp64": torch.float64,
        "bf16": torch.bfloat16,
    }
    training_dtype = dtype_map[args.dtype]

    # Parse per-buffer dtype overrides for ablation studies.
    buffer_dtypes: dict[str, torch.dtype] | None = None
    if args.buffer_dtypes:
        buffer_dtypes = {}
        for pair in args.buffer_dtypes.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(
                    f"--buffer_dtypes entry '{pair}' missing '='. "
                    f"Format: 'momentum=bf16,set_point=bf16'."
                )
            name, dt = pair.split("=", 1)
            name = name.strip()
            dt = dt.strip()
            if dt not in dtype_map:
                raise ValueError(
                    f"--buffer_dtypes dtype '{dt}' not recognized. "
                    f"Choices: {list(dtype_map.keys())}."
                )
            buffer_dtypes[name] = dtype_map[dt]
        if buffer_dtypes:
            print(f"[buffer_dtypes] {buffer_dtypes}")

    torch.manual_seed(args.seed)

    # FP64 not supported on DirectML/CUDA well — use CPU
    if args.dtype == "fp64":
        device = torch.device("cpu")
        print(f"Device: cpu (FP64 requires CPU)")
    else:
        try:
            import torch_directml
            device = torch_directml.device()
        except ImportError:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
    print(f"Precision: {args.dtype}")

    # Load corpus
    file_list_path = getattr(args, "file_list", None)
    if file_list_path:
        text_files = load_file_list(file_list_path)
        print(f"Loaded {len(text_files)} files from curriculum: {file_list_path}")
        data_dir = Path(text_files[0]).parent  # for BPE sample fallback
    else:
        data_dir = Path(args.data_dir)
        text_files = sorted(data_dir.glob("*.txt"))
        if not text_files:
            print(f"No .txt files found in {data_dir}")
            return
        print(f"Found {len(text_files)} text files in {data_dir}")

    output_dir = Path(args.output_dir)
    if args.run_name:
        output_dir = output_dir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # For large corpora (>100 files), use streaming to avoid loading
    # the entire corpus as a single Python string
    streaming = len(text_files) > 100
    if streaming:
        print(f"Large corpus detected — using streaming encoder")

    # Build tokenizer
    if args.tokenizer == "bpe":
        bpe_path = output_dir / "tokenizer.json"
        if bpe_path.exists():
            tokenizer = BPETokenizer.load(bpe_path)
            print(f"Loaded BPE tokenizer: {tokenizer.vocab_size} tokens")
        else:
            bpe_sample_size = 200_000_000  # 200 MB
            print(f"Training BPE tokenizer on {bpe_sample_size // 1_000_000} MB sample "
                  f"(target vocab: {args.bpe_vocab_size})...")
            bpe_sample = load_corpus_sample(*text_files, max_bytes=bpe_sample_size)
            tokenizer = BPETokenizer(target_vocab_size=args.bpe_vocab_size)
            tokenizer.train(bpe_sample)
            del bpe_sample
            tokenizer.save(bpe_path)
            print(f"Saved BPE tokenizer to {bpe_path}")
    else:
        if streaming:
            sample = load_corpus_sample(*text_files, max_bytes=50_000_000)
            tokenizer = CharTokenizer(sample)
            del sample
        else:
            corpus = load_corpus(*text_files)
            tokenizer = CharTokenizer(corpus)
    print(f"Vocabulary: {tokenizer.vocab_size} tokens")

    # Build datasets
    if streaming:
        # Split file list: 90% train, 10% val
        split_idx = int(len(text_files) * 0.9)
        train_files = text_files[:split_idx]
        val_files = text_files[split_idx:]
        print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

        print("Encoding training corpus...")
        train_data = load_corpus_as_tensor(*train_files, tokenizer=tokenizer)
        print(f"Train: {len(train_data):,} tokens ({len(train_data) * 8 / 1e9:.2f} GB)")

        print("Encoding validation corpus...")
        val_data = load_corpus_as_tensor(*val_files, tokenizer=tokenizer)
        print(f"Val:   {len(val_data):,} tokens ({len(val_data) * 8 / 1e9:.2f} GB)")

        corpus_chars = (len(train_data) + len(val_data)) * 4  # approx chars from tokens
        train_dataset = CharDataset(train_data, seq_len=args.seq_len, tokenizer=tokenizer, stride=args.stride)
        val_dataset = CharDataset(val_data, seq_len=args.seq_len, tokenizer=tokenizer, stride=args.stride)
        del train_data, val_data
    else:
        if args.tokenizer != "char":
            corpus = load_corpus(*text_files)
        print(f"Corpus: {len(corpus):,} characters")
        corpus_chars = len(corpus)
        split = int(len(corpus) * 0.9)
        train_text = corpus[:split]
        val_text = corpus[split:]
        del corpus
        train_dataset = CharDataset(train_text, seq_len=args.seq_len, tokenizer=tokenizer, stride=args.stride)
        val_dataset = CharDataset(val_text, seq_len=args.seq_len, tokenizer=tokenizer, stride=args.stride)
        del train_text, val_text

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True,
    )

    print(f"Train: {len(train_dataset):,} sequences")
    print(f"Val:   {len(val_dataset):,} sequences")

    # -- Resume or create living model ---------------------------------
    start_epoch = 0
    living_results = {"train_loss": [], "val_loss": [], "non_ff_signal": []}
    substrate_health_history: list[dict] = []

    # Determine initial backward pass state
    # If --backward_pass_start_epoch > 0, start disabled and enable at that epoch
    bp_enabled = args.backward_pass and (args.backward_pass_start_epoch == 0)

    if args.backward_pass:
        if args.backward_pass_start_epoch > 0:
            print(f"Backward pass: STAGED (enables at epoch {args.backward_pass_start_epoch})")
        else:
            print(f"Backward pass: ON from epoch 0")
    else:
        print(f"Backward pass: OFF")

    if args.resume:
        print(f"\n=== RESUMING FROM CHECKPOINT ===")
        print(f"Loading: {args.resume}")
        ckpt = load_checkpoint(args.resume, args.checkpoint_password, device)

        # Restore config from checkpoint for model construction
        ckpt_config = ckpt["config"]
        print(f"Checkpoint epoch: {ckpt['epoch']}")
        print(f"Checkpoint time:  {ckpt['timestamp']}")

        # Restore BPE tokenizer from checkpoint if it was used
        if "tokenizer_state" in ckpt and ckpt["tokenizer_state"]["type"] == "bpe":
            tokenizer = BPETokenizer.from_state(ckpt["tokenizer_state"])
            args.tokenizer = "bpe"
            print(f"Restored BPE tokenizer from checkpoint: {tokenizer.vocab_size} tokens")

        if args.spiking:
            # Load base checkpoint into spiking model (strict=False
            # ignores missing spiking buffers — they start at zeros)
            from luthi.model_spiking import SpikingLuthiLM
            print("Upgrading to SpikingLuthiLM (spiking buffers initialized fresh)")
            living_model = SpikingLuthiLM(
                vocab_size=tokenizer.vocab_size,
                d_model=ckpt_config["d_model"],
                n_blocks=ckpt_config["n_blocks"],
                max_seq_len=ckpt_config["seq_len"],
                hebb_rate=ckpt_config["hebb_rate"],
                error_rate=ckpt_config["error_rate"],
                homeostatic_decay=ckpt_config["homeostatic_decay"],
                set_point_adapt_rate=ckpt_config["set_point_adapt_rate"],
                spike_threshold=args.spike_threshold,
                membrane_leak=args.membrane_leak,
                refractory_steps=args.refractory_steps,
                delay_steps=args.delay_steps,
                backward_pass_enabled=bp_enabled,
                buffer_dtypes=buffer_dtypes,
            ).to(device=device, dtype=training_dtype)

            missing, unexpected = living_model.load_state_dict(
                ckpt["model_state_dict"], strict=False,
            )
            if missing:
                print(f"  New spiking buffers (initialized to zeros): {len(missing)}")
            if unexpected:
                print(f"  WARNING: unexpected keys in checkpoint: {unexpected}")
        else:
            living_model = LuthiLM(
                vocab_size=tokenizer.vocab_size,
                d_model=ckpt_config["d_model"],
                n_blocks=ckpt_config["n_blocks"],
                max_seq_len=ckpt_config["seq_len"],
                hebb_rate=ckpt_config["hebb_rate"],
                error_rate=ckpt_config["error_rate"],
                homeostatic_decay=ckpt_config["homeostatic_decay"],
                set_point_adapt_rate=ckpt_config["set_point_adapt_rate"],
                backward_pass_enabled=bp_enabled,
                buffer_dtypes=buffer_dtypes,
            ).to(device=device, dtype=training_dtype)

            living_model.load_state_dict(ckpt["model_state_dict"], strict=False)

        living_optimizer = DirectMLAdamW(living_model.parameters(), lr=args.lr)
        if "optimizer_state_dict" in ckpt and not args.spiking:
            # Only restore optimizer state for same model type;
            # spiking upgrade changes the param set
            living_optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        start_epoch = ckpt["epoch"]
        living_results = ckpt.get("training_history", living_results)
        substrate_health_history = ckpt.get("substrate_health", {}).get(
            "epoch_snapshots", []
        )

        param_counts = living_model.total_parameters()
        print(f"Trainable params:  {param_counts['trainable']:,}")
        print(f"Living buffers:    {param_counts['living_buffers']:,}")
        print(f"Resuming from epoch {start_epoch}")
    else:
        if args.spiking:
            from luthi.model_spiking import SpikingLuthiLM
            print("\n=== SPIKING LIVING MODEL (SpikingLuthiLM) ===")
            living_model = SpikingLuthiLM(
                vocab_size=tokenizer.vocab_size,
                d_model=args.d_model,
                n_blocks=args.n_blocks,
                max_seq_len=args.seq_len,
                hebb_rate=args.hebb_rate,
                error_rate=args.error_rate,
                homeostatic_decay=args.homeostatic_decay,
                set_point_adapt_rate=args.set_point_adapt_rate,
                spike_threshold=args.spike_threshold,
                membrane_leak=args.membrane_leak,
                refractory_steps=args.refractory_steps,
                delay_steps=args.delay_steps,
                backward_pass_enabled=bp_enabled,
                buffer_dtypes=buffer_dtypes,
            ).to(device=device, dtype=training_dtype)
        else:
            print("\n=== LIVING MODEL (LuthiLM) ===")
            living_model = LuthiLM(
                vocab_size=tokenizer.vocab_size,
                d_model=args.d_model,
                n_blocks=args.n_blocks,
                max_seq_len=args.seq_len,
                hebb_rate=args.hebb_rate,
                error_rate=args.error_rate,
                homeostatic_decay=args.homeostatic_decay,
                set_point_adapt_rate=args.set_point_adapt_rate,
                backward_pass_enabled=bp_enabled,
                buffer_dtypes=buffer_dtypes,
            ).to(device=device, dtype=training_dtype)

        param_counts = living_model.total_parameters()
        print(f"Trainable params:  {param_counts['trainable']:,}")
        print(f"Living buffers:    {param_counts['living_buffers']:,}")

        living_optimizer = DirectMLAdamW(living_model.parameters(), lr=args.lr)

    # -- Training config for checkpoint metadata -----------------------
    config = {
        "d_model": args.d_model,
        "n_blocks": args.n_blocks,
        "seq_len": args.seq_len,
        "hebb_rate": args.hebb_rate,
        "error_rate": args.error_rate,
        "homeostatic_decay": args.homeostatic_decay,
        "set_point_adapt_rate": args.set_point_adapt_rate,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_type": args.tokenizer,
        "bpe_vocab_size": args.bpe_vocab_size if args.tokenizer == "bpe" else None,
        "dtype": args.dtype,
        "spiking": args.spiking,
        "backward_pass": args.backward_pass,
        "backward_pass_start_epoch": args.backward_pass_start_epoch,
    }
    if args.spiking:
        config["spike_threshold"] = args.spike_threshold
        config["membrane_leak"] = args.membrane_leak
        config["refractory_steps"] = args.refractory_steps
        config["delay_steps"] = args.delay_steps
    # If resuming, use checkpoint config for model params but allow
    # overriding training params (epochs, lr, etc.)
    if args.resume:
        ckpt_config = ckpt["config"]
        config["d_model"] = ckpt_config["d_model"]
        config["n_blocks"] = ckpt_config["n_blocks"]
        config["seq_len"] = ckpt_config["seq_len"]
        config["hebb_rate"] = ckpt_config["hebb_rate"]
        config["error_rate"] = ckpt_config["error_rate"]
        config["homeostatic_decay"] = ckpt_config["homeostatic_decay"]
        config["set_point_adapt_rate"] = ckpt_config["set_point_adapt_rate"]

    # -- Living model training loop ------------------------------------
    checkpoint_path = output_dir / "checkpoint.luthi"
    extended_metrics_history: list[dict] = []

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Staged backward pass enable
        if args.backward_pass and args.backward_pass_start_epoch > 0:
            if epoch >= args.backward_pass_start_epoch and not living_model.backward_pass_enabled:
                living_model.backward_pass_enabled = True
                print(f"  >>> Backward pass ENABLED at epoch {epoch}")

        t0 = time.time()
        train_loss = train_epoch(
            living_model, train_loader, living_optimizer, device,
            is_living=True, epoch=epoch,
        )
        val_loss = eval_model(living_model, val_loader, device)
        elapsed = time.time() - t0

        # Measure non-feedforward signal
        sample_x, _ = next(iter(val_loader))
        nff = measure_non_feedforward(living_model, sample_x.to(device))

        living_results["train_loss"].append(train_loss)
        living_results["val_loss"].append(val_loss)
        living_results["non_ff_signal"].append(nff)

        # Substrate health snapshot
        health = extract_substrate_health(living_model)
        health["epoch"] = epoch
        health["train_loss"] = train_loss
        health["val_loss"] = val_loss
        health["non_ff_signal"] = nff
        substrate_health_history.append(health)

        # Extended metrics for backward pass comparison
        ext = collect_extended_metrics(living_model)
        ext["epoch"] = epoch
        ext["backward_pass_active"] = living_model.backward_pass_enabled
        if living_model.backward_pass_enabled:
            bp_effect = measure_backward_pass_effect(
                living_model, sample_x.to(device),
            )
            ext["backward_pass_effect"] = bp_effect
        else:
            ext["backward_pass_effect"] = 0.0
        extended_metrics_history.append(ext)

        gap = val_loss - train_loss
        bp_tag = " [BP]" if living_model.backward_pass_enabled else ""
        print(
            f"  Epoch {epoch:2d}/{args.epochs} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | "
            f"gap {gap:.2f} | non-FF {nff:.6f} | "
            f"plast {ext.get('plasticity_mean', 0):.4f}±{ext.get('plasticity_std', 0):.4f} | "
            f"drift {ext.get('sp_drift_mean', 0):.6f} | "
            f"bp_fx {ext.get('backward_pass_effect', 0):.6f}{bp_tag} | "
            f"{elapsed:.1f}s"
        )

        # Save encrypted checkpoint after every epoch
        if args.checkpoint_password or "LUTHI_CHECKPOINT_KEY" in __import__("os").environ:
            ckpt = build_checkpoint(
                model=living_model,
                optimizer=living_optimizer,
                epoch=epoch,
                config=config,
                training_history=living_results,
                substrate_health={"epoch_snapshots": substrate_health_history},
            )
            # Embed tokenizer state so checkpoint is self-contained
            if hasattr(tokenizer, "get_state"):
                ckpt["tokenizer_state"] = tokenizer.get_state()
            ckpt["extended_metrics"] = extended_metrics_history
            save_checkpoint(ckpt, checkpoint_path, args.checkpoint_password)

    # Generate sample
    print("\n-- Living model sample --")
    sample = generate(living_model, tokenizer, "The ", max_len=200)
    print(sample)

    # Aliveness report
    print("\n-- Aliveness report --")
    for i, report in enumerate(living_model.aliveness_report()):
        line = (
            f"  Block {i}: drift={report['set_point_drift']:.6f}, "
            f"exc={report['excitability_mean']:.3f}, "
            f"episodes={report['episodes_stored']}"
        )
        if args.spiking and "spike_fraction" in report:
            line += (
                f", spike={report['spike_fraction']:.3f}, "
                f"refrac={report['refractory_fraction']:.3f}, "
                f"membrane={report['membrane_potential_mean']:.4f}"
            )
        print(line)

    # Final checkpoint save
    if args.checkpoint_password or "LUTHI_CHECKPOINT_KEY" in __import__("os").environ:
        ckpt = build_checkpoint(
            model=living_model,
            optimizer=living_optimizer,
            epoch=args.epochs,
            config=config,
            training_history=living_results,
            substrate_health={"epoch_snapshots": substrate_health_history},
        )
        if hasattr(tokenizer, "get_state"):
            ckpt["tokenizer_state"] = tokenizer.get_state()
        saved_path = save_checkpoint(ckpt, checkpoint_path, args.checkpoint_password)
        print(f"\nCheckpoint saved to {saved_path} (encrypted)")

    # -- Save results --------------------------------------------------
    results = {
        "args": vars(args),
        "living": living_results,
        "extended_metrics": extended_metrics_history,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_type": args.tokenizer,
        "corpus_chars": corpus_chars,
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
