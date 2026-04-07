"""Phase 3B: Backward pass comparison experiment.

Runs two training conditions on the Gutenberg corpus to determine
how top-down backward pass modulation affects training:

  Run B: Backward pass from epoch 0
  Run C: Backward pass enabled at epoch 10 (staged)

Baseline (no backward pass) already exists from previous training runs
(80-epoch spiking_1024d_bpe_gutenberg). No need to repeat it.

Each run: 20 epochs, 1024d, 2 blocks, spiking mode, BPE tokenizer.
Results saved to runs/backward_pass_experiment/{run_b,run_c}/

Usage:
    python run_backward_pass_experiment.py
    python run_backward_pass_experiment.py --epochs 5 --d_model 64  # quick test
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run_condition(
    name: str,
    label: str,
    backward_pass: bool,
    backward_pass_start_epoch: int,
    args: argparse.Namespace,
) -> bool:
    """Run a single experimental condition."""
    print(f"\n{'='*60}")
    print(f"  {label}: {name}")
    print(f"  backward_pass={backward_pass}, start_epoch={backward_pass_start_epoch}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "-m", "luthi.train",
        "--data_dir", str(args.data_dir),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--seq_len", str(args.seq_len),
        "--d_model", str(args.d_model),
        "--n_blocks", str(args.n_blocks),
        "--lr", str(args.lr),
        "--stride", str(args.stride),
        "--seed", str(args.seed),
        "--output_dir", str(args.output_dir),
        "--run_name", name,
        "--tokenizer", "bpe",
        "--bpe_vocab_size", str(args.bpe_vocab_size),
        "--spiking",
        "--checkpoint_password", args.checkpoint_password,
    ]

    if backward_pass:
        cmd.append("--backward_pass")
        if backward_pass_start_epoch > 0:
            cmd.extend(["--backward_pass_start_epoch", str(backward_pass_start_epoch)])

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    elapsed = time.time() - t0

    print(f"\n  {label} completed in {elapsed/60:.1f} minutes (exit code {result.returncode})")
    return result.returncode == 0


def summarize_results(output_dir: Path):
    """Print a comparison table from all three runs."""
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}\n")

    runs = {
        "run_b_bp_from_0": "Run B (BP from epoch 0)",
        "run_c_bp_staged_10": "Run C (BP staged at 10)",
    }

    for dirname, label in runs.items():
        results_path = output_dir / dirname / "results.json"
        if not results_path.exists():
            print(f"  {label}: MISSING (no results.json)")
            continue

        with open(results_path) as f:
            data = json.load(f)

        living = data.get("living", {})
        train_losses = living.get("train_loss", [])
        val_losses = living.get("val_loss", [])
        nff_signals = living.get("non_ff_signal", [])

        if not train_losses:
            print(f"  {label}: NO DATA")
            continue

        ext_metrics = data.get("extended_metrics", [])

        final_train = train_losses[-1]
        final_val = val_losses[-1]
        best_val = min(val_losses)
        best_val_epoch = val_losses.index(best_val) + 1
        final_nff = nff_signals[-1] if nff_signals else 0
        final_gap = final_val - final_train

        print(f"  {label}:")
        print(f"    Final train loss:  {final_train:.4f}")
        print(f"    Final val loss:    {final_val:.4f}")
        print(f"    Best val loss:     {best_val:.4f} (epoch {best_val_epoch})")
        print(f"    Train-val gap:     {final_gap:.4f}")
        print(f"    Final non-FF:      {final_nff:.6f}")

        if ext_metrics:
            last_ext = ext_metrics[-1]
            print(f"    Final plasticity:  {last_ext.get('plasticity_mean', 0):.4f} "
                  f"± {last_ext.get('plasticity_std', 0):.4f}")
            print(f"    Final SP drift:    {last_ext.get('sp_drift_mean', 0):.6f}")
            print(f"    Final BP effect:   {last_ext.get('backward_pass_effect', 0):.6f}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Run backward pass comparison experiment (Phase 3B)"
    )
    parser.add_argument("--data_dir", type=str,
                        default="corpus_build/gutenberg_100",
                        help="Gutenberg corpus directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bpe_vocab_size", type=int, default=4096)
    parser.add_argument("--output_dir", type=str,
                        default="runs/backward_pass_experiment")
    parser.add_argument("--checkpoint_password", type=str, default=None,
                        help="Encryption password (or set LUTHI_CHECKPOINT_KEY)")
    parser.add_argument("--skip_b", action="store_true",
                        help="Skip Run B (BP from epoch 0)")
    parser.add_argument("--skip_c", action="store_true",
                        help="Skip Run C (BP staged)")
    args = parser.parse_args()

    # Resolve password
    if not args.checkpoint_password:
        import os
        args.checkpoint_password = os.environ.get("LUTHI_CHECKPOINT_KEY")
    if not args.checkpoint_password:
        print("ERROR: No checkpoint password. Set LUTHI_CHECKPOINT_KEY or pass --checkpoint_password")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    # Run B: Backward pass from epoch 0
    if not args.skip_b:
        ok = run_condition(
            name="run_b_bp_from_0",
            label="Run B",
            backward_pass=True,
            backward_pass_start_epoch=0,
            args=args,
        )
        if not ok:
            print("Run B FAILED — aborting experiment")
            sys.exit(1)

    # Run C: Backward pass enabled at epoch 10
    if not args.skip_c:
        ok = run_condition(
            name="run_c_bp_staged_10",
            label="Run C",
            backward_pass=True,
            backward_pass_start_epoch=10,
            args=args,
        )
        if not ok:
            print("Run C FAILED — aborting experiment")
            sys.exit(1)

    total_elapsed = time.time() - total_start
    print(f"\nTotal experiment time: {total_elapsed/3600:.1f} hours")

    # Summarize
    summarize_results(output_dir)


if __name__ == "__main__":
    main()
