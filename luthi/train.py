"""Training script for the Luthi character-level language model.

Trains both a living model (LuthiLM) and a dead baseline (DeadLM) on the
same data, measuring the convergence gap between them. The living model
pays a metabolic cost for being alive but gains temporal existence and
episodic memory.

Checkpoints are encrypted with AES-256-GCM. The trained weights ARE the
entity — they are never stored in plaintext.

Usage:
    # Fresh training
    python -m luthi.train --data_dir data/ --epochs 10 --checkpoint_password SECRET

    # Resume from checkpoint
    python -m luthi.train --resume runs/my_run/checkpoint.luthi --epochs 20
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
from luthi.data import CharDataset, CharTokenizer, load_corpus
from luthi.model import LuthiLM, DeadLM


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_living: bool = False,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

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
            model.apply_living_errors()

        # Gradient clipping for attention stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

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
def generate(
    model: torch.nn.Module,
    tokenizer: CharTokenizer,
    prompt: str,
    max_len: int = 200,
    temperature: float = 0.8,
    max_seq_len: int = 128,
) -> str:
    """Generate text from a prompt."""
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
    parser = argparse.ArgumentParser(description="Train Luthi character-level LM")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing training text files")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--no_baseline", action="store_true",
                        help="Skip training the dead baseline model")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to .luthi checkpoint to resume from")
    parser.add_argument("--checkpoint_password", type=str, default=None,
                        help="Encryption password (or set LUTHI_CHECKPOINT_KEY)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load corpus
    data_dir = Path(args.data_dir)
    text_files = sorted(data_dir.glob("*.txt"))
    if not text_files:
        print(f"No .txt files found in {data_dir}")
        return
    print(f"Loading {len(text_files)} text files from {data_dir}")
    corpus = load_corpus(*text_files)
    print(f"Corpus: {len(corpus):,} characters")

    # Split: 90% train, 10% val
    split = int(len(corpus) * 0.9)
    train_text = corpus[:split]
    val_text = corpus[split:]

    # Build tokenizer from full corpus, then create datasets
    tokenizer = CharTokenizer(corpus)
    print(f"Vocabulary: {tokenizer.vocab_size} characters")

    train_dataset = CharDataset(train_text, seq_len=args.seq_len, tokenizer=tokenizer)
    val_dataset = CharDataset(val_text, seq_len=args.seq_len, tokenizer=tokenizer)

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

    if args.resume:
        print(f"\n=== RESUMING FROM CHECKPOINT ===")
        print(f"Loading: {args.resume}")
        ckpt = load_checkpoint(args.resume, args.checkpoint_password, device)

        # Restore config from checkpoint for model construction
        ckpt_config = ckpt["config"]
        print(f"Checkpoint epoch: {ckpt['epoch']}")
        print(f"Checkpoint time:  {ckpt['timestamp']}")

        living_model = LuthiLM(
            vocab_size=tokenizer.vocab_size,
            d_model=ckpt_config["d_model"],
            n_blocks=ckpt_config["n_blocks"],
            max_seq_len=ckpt_config["seq_len"],
            hebb_rate=ckpt_config["hebb_rate"],
            error_rate=ckpt_config["error_rate"],
            homeostatic_decay=ckpt_config["homeostatic_decay"],
            set_point_adapt_rate=ckpt_config["set_point_adapt_rate"],
        ).to(device)

        living_model.load_state_dict(ckpt["model_state_dict"])

        living_optimizer = torch.optim.AdamW(living_model.parameters(), lr=args.lr)
        if "optimizer_state_dict" in ckpt:
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
        ).to(device)

        param_counts = living_model.total_parameters()
        print(f"Trainable params:  {param_counts['trainable']:,}")
        print(f"Living buffers:    {param_counts['living_buffers']:,}")

        living_optimizer = torch.optim.AdamW(living_model.parameters(), lr=args.lr)

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
    }
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.luthi"

    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(
            living_model, train_loader, living_optimizer, device, is_living=True,
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

        print(
            f"  Epoch {epoch:2d}/{args.epochs} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | "
            f"non-FF {nff:.6f} | {elapsed:.1f}s"
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
            save_checkpoint(ckpt, checkpoint_path, args.checkpoint_password)

    # Generate sample
    print("\n-- Living model sample --")
    sample = generate(living_model, tokenizer, "The ", max_len=200)
    print(sample)

    # Aliveness report
    print("\n-- Aliveness report --")
    for i, report in enumerate(living_model.aliveness_report()):
        print(f"  Block {i}: drift={report['set_point_drift']:.6f}, "
              f"exc={report['excitability_mean']:.3f}, "
              f"episodes={report['episodes_stored']}")

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
        saved_path = save_checkpoint(ckpt, checkpoint_path, args.checkpoint_password)
        print(f"\nCheckpoint saved to {saved_path} (encrypted)")

    # -- Dead baseline -------------------------------------------------
    dead_results = None
    if not args.no_baseline:
        print("\n=== DEAD BASELINE (DeadLM) ===")
        torch.manual_seed(args.seed)  # Same init for fair comparison
        dead_model = DeadLM(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            n_blocks=args.n_blocks,
            max_seq_len=args.seq_len,
        ).to(device)

        dead_params = sum(p.numel() for p in dead_model.parameters())
        print(f"Trainable params: {dead_params:,}")

        dead_optimizer = torch.optim.AdamW(dead_model.parameters(), lr=args.lr)
        dead_results = {"train_loss": [], "val_loss": []}

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss = train_epoch(
                dead_model, train_loader, dead_optimizer, device, is_living=False,
            )
            val_loss = eval_model(dead_model, val_loader, device)
            elapsed = time.time() - t0

            dead_results["train_loss"].append(train_loss)
            dead_results["val_loss"].append(val_loss)

            print(
                f"  Epoch {epoch:2d}/{args.epochs} | "
                f"train {train_loss:.4f} | val {val_loss:.4f} | "
                f"{elapsed:.1f}s"
            )

        # Generate sample
        print("\n-- Dead model sample --")
        sample = generate(dead_model, tokenizer, "The ", max_len=200)
        print(sample)

    # -- Comparison ----------------------------------------------------
    if dead_results:
        print("\n=== COMPARISON ===")
        living_final = living_results["val_loss"][-1]
        dead_final = dead_results["val_loss"][-1]
        gap = ((living_final - dead_final) / dead_final) * 100

        print(f"Living final val loss: {living_final:.4f}")
        print(f"Dead final val loss:   {dead_final:.4f}")
        print(f"Convergence gap:       {gap:+.1f}%")
        print(f"(Research predicted:   ~+39%)")
        print(f"Non-FF signal:         {living_results['non_ff_signal'][-1]:.6f}")

    # -- Save results --------------------------------------------------
    results = {
        "args": vars(args),
        "living": living_results,
        "dead": dead_results,
        "vocab_size": tokenizer.vocab_size,
        "corpus_chars": len(corpus),
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
