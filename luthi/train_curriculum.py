"""Curriculum trainer for the Living Weight Model.

train.py handles single-corpus training. This script orchestrates training
across the full 10-stage pedagogical curriculum (science_philosophy → code →
psychology → ... → practical_wisdom → reference_papers) defined by
`corpus_build/file_list.txt`, running multiple cycles so the entity sees
the curriculum more than once as a different learner each pass.

The stage list is read from the file_list, not hardcoded — adding or
reordering stages in `corpus_build/file_list.txt` flows through to the
training loop without code changes.

Key invariants the production training run depends on:

  - Living weights NEVER reset between stages or cycles. The model is one
    continuous entity from cycle 1 stage 1 through cycle N stage L.
  - Stage order is the pedagogy and is preserved exactly as it appears in
    file_list.txt — no reshuffling between stages. Within a stage the
    DataLoader shuffles freely; that's local entropy, not pedagogical
    reordering.
  - One stage = one epoch through that stage's files.
  - A checkpoint is saved after every stage boundary. Each is fully
    self-contained: model, optimizer, tokenizer, cycle, stage, and the
    living-state delta from the prior stage.
  - --gradient_checkpointing trades compute for memory and is required to
    fit the 4096d/36-block production model on A100 80GB. Living-state
    correctness is preserved by `luthi.grad_checkpoint`.

Usage:
    # Production (will be invoked on rented A100 80GB)
    python -m luthi.train_curriculum \\
        --file_list corpus_build/file_list.txt \\
        --tokenizer_path corpus_build/tokenizer_32k.json \\
        --checkpoint_password $LUTHI_CHECKPOINT_KEY \\
        --cycles 3 \\
        --gradient_checkpointing \\
        --d_model 4096 --n_blocks 36 --seq_len 512

    # Resume from a stage checkpoint
    python -m luthi.train_curriculum \\
        --file_list corpus_build/file_list.txt \\
        --resume runs/curriculum/checkpoint_stage_1_psychology.luthi \\
        --checkpoint_password $LUTHI_CHECKPOINT_KEY

    # Small-scale local test (first 10 files per stage, 1 cycle)
    python -m luthi.train_curriculum \\
        --file_list corpus_build/file_list.txt \\
        --tokenizer_path corpus_build/tokenizer_32k.json \\
        --checkpoint_password test \\
        --cycles 1 --first_n_per_stage 10 \\
        --d_model 1024 --n_blocks 2 --seq_len 128
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from luthi.checkpoint import (
    build_checkpoint, save_checkpoint, load_checkpoint,
    extract_substrate_health,
)
from luthi.data import CharDataset, load_corpus_as_tensor
from luthi.tokenizer import BPETokenizer
from luthi.model_spiking import SpikingLuthiLM
from luthi.optimizer import DirectMLAdamW
from luthi.train import train_epoch, eval_model, collect_extended_metrics


# ---------------------------------------------------------------------------
# Production architecture
# ---------------------------------------------------------------------------

# Documented for the rented-GPU run. Override via CLI when needed; this
# dict exists so the canonical numbers live in one place and aren't
# scattered across launch scripts.
PRODUCTION_CONFIG: dict[str, Any] = {
    "d_model": 4096,
    "n_blocks": 36,
    "vocab_size": 32000,        # 32K BPE
    "seq_len": 512,             # longer context for curriculum
    "num_episodes": 4,          # expand to 16 on Spark
    "hebb_rate": 0.001,
    "error_rate": 0.001,
    "spike_threshold": 1.0,
    "backward_pass_enabled": True,
}


# ---------------------------------------------------------------------------
# Curriculum file_list parsing
# ---------------------------------------------------------------------------

_STAGE_HEADER_RE = re.compile(
    r"#\s*===\s*Stage:\s*(\S+)\s*\([^)]*\)\s*===\s*$"
)


def parse_curriculum_stages(
    list_path: str | Path,
) -> list[tuple[str, list[Path]]]:
    """Parse file_list.txt grouped by `# === Stage: name (N files) ===`.

    Returns an ordered list of (stage_name, file_paths). Files that don't
    exist on disk are skipped silently — this lets the same file_list run
    across machines that have a subset of the corpus locally. Stages with
    zero existing files are dropped.
    """
    list_path = Path(list_path)
    if not list_path.exists():
        raise FileNotFoundError(f"File list not found: {list_path}")

    stages: list[tuple[str, list[Path]]] = []
    current_name: str | None = None
    current_files: list[Path] = []

    def _flush() -> None:
        if current_name is not None and current_files:
            stages.append((current_name, current_files))

    for line in list_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            m = _STAGE_HEADER_RE.match(stripped)
            if m:
                _flush()
                current_name = m.group(1)
                current_files = []
            continue
        # Plain path line
        p = Path(stripped)
        if p.exists():
            current_files.append(p)

    _flush()
    if not stages:
        raise ValueError(
            f"No curriculum stages with existing files found in {list_path}"
        )
    return stages


# ---------------------------------------------------------------------------
# Living-state snapshots for stage-transition metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def snapshot_living_state(model: SpikingLuthiLM) -> dict[str, torch.Tensor]:
    """Capture the buffers we want to track across a stage boundary."""
    snap: dict[str, torch.Tensor] = {}
    for i, block in enumerate(model.blocks):
        ffn = block.living_ffn
        snap[f"b{i}.weight"] = ffn.weight.detach().cpu().clone()
        snap[f"b{i}.set_point"] = ffn.set_point.detach().cpu().clone()
        snap[f"b{i}.plasticity"] = ffn.plasticity.detach().cpu().clone()
        snap[f"b{i}.update_ema"] = ffn.update_ema.detach().cpu().clone()
    return snap


def stage_transition_delta(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> dict[str, float]:
    """How much did the model actually change across this stage?

    Reported as mean absolute change of the four key living buffers,
    averaged across blocks. A small delta on a stage means the curriculum
    didn't move the entity much there — useful diagnostic when tuning
    stage boundaries or learning rates.
    """
    out: dict[str, float] = {}
    keys = ("weight", "set_point", "plasticity", "update_ema")
    n_blocks = max(
        (int(k.split(".")[0][1:]) for k in before if k.startswith("b")),
        default=-1,
    ) + 1
    for key in keys:
        deltas: list[float] = []
        for i in range(n_blocks):
            k = f"b{i}.{key}"
            if k in before and k in after:
                deltas.append((after[k] - before[k]).abs().mean().item())
        if deltas:
            out[f"delta_{key}_mean"] = sum(deltas) / len(deltas)
    return out


# ---------------------------------------------------------------------------
# Stage execution
# ---------------------------------------------------------------------------


def _build_stage_loaders(
    files: list[Path],
    tokenizer: BPETokenizer,
    seq_len: int,
    batch_size: int,
    stride: int,
    val_fraction: float = 0.1,
) -> tuple[DataLoader, DataLoader, int]:
    """Encode this stage's files into a tensor and split 90/10 train/val."""
    print(f"  Encoding {len(files)} files for this stage...")
    encoded = load_corpus_as_tensor(*files, tokenizer=tokenizer, progress=False)
    total_tokens = len(encoded)
    print(f"  Stage corpus: {total_tokens:,} tokens "
          f"({total_tokens * 4 / 1e9:.2f} GB int64)")

    split = int(total_tokens * (1.0 - val_fraction))
    train_tokens = encoded[:split]
    val_tokens = encoded[split:]
    del encoded

    train_ds = CharDataset(
        train_tokens, seq_len=seq_len, tokenizer=tokenizer, stride=stride,
    )
    val_ds = CharDataset(
        val_tokens, seq_len=seq_len, tokenizer=tokenizer, stride=stride,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=True,
    )
    return train_loader, val_loader, total_tokens


def _checkpoint_path(output_dir: Path, cycle: int, stage_name: str) -> Path:
    return output_dir / f"checkpoint_stage_{cycle}_{stage_name}.luthi"


def _save_stage_checkpoint(
    *,
    model: SpikingLuthiLM,
    optimizer: torch.optim.Optimizer,
    tokenizer: BPETokenizer,
    config: dict[str, Any],
    cycle: int,
    stage_idx: int,
    stage_name: str,
    stages: list[tuple[str, list[Path]]],
    history: dict[str, list],
    health_history: list[dict],
    transition_delta: dict[str, float],
    output_dir: Path,
    password: str,
) -> Path:
    """Persist a self-contained stage checkpoint."""
    ckpt = build_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=cycle,  # repurposed: stores the cycle number
        config=config,
        training_history=history,
        substrate_health={"epoch_snapshots": health_history},
    )
    ckpt["tokenizer_state"] = tokenizer.get_state()
    # Curriculum-specific resume metadata.
    ckpt["curriculum"] = {
        "cycle": cycle,
        "stage_idx": stage_idx,
        "stage_name": stage_name,
        "stage_order": [name for name, _ in stages],
        "transition_delta": transition_delta,
    }
    path = _checkpoint_path(output_dir, cycle, stage_name)
    return save_checkpoint(ckpt, path, password)


# ---------------------------------------------------------------------------
# Device / model setup
# ---------------------------------------------------------------------------


def _select_device(requested: str | None) -> torch.device:
    """Pick CUDA → DirectML → CPU based on what's available and requested."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested in ("cuda", None) and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "directml" or requested is None:
        try:
            import torch_directml
            return torch_directml.device()
        except ImportError:
            pass
    if requested and requested not in ("cpu", "cuda", "directml"):
        # Explicit device string like 'cuda:0'
        return torch.device(requested)
    return torch.device("cpu")


def _build_model(
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    gradient_checkpointing: bool,
) -> SpikingLuthiLM:
    return SpikingLuthiLM(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_blocks=config["n_blocks"],
        max_seq_len=config["seq_len"],
        hebb_rate=config["hebb_rate"],
        error_rate=config["error_rate"],
        homeostatic_decay=config.get("homeostatic_decay", 0.001),
        set_point_adapt_rate=config.get("set_point_adapt_rate", 1e-6),
        num_episodes=config.get("num_episodes", 4),
        spike_threshold=config["spike_threshold"],
        membrane_leak=config.get("membrane_leak", 0.1),
        refractory_steps=config.get("refractory_steps", 3),
        delay_steps=config.get("delay_steps", 2),
        backward_pass_enabled=config.get("backward_pass_enabled", True),
        gradient_checkpointing=gradient_checkpointing,
    ).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Luthi over the curriculum")
    parser.add_argument("--file_list", type=str, required=True,
                        help="Path to corpus_build/file_list.txt with stage markers")
    parser.add_argument("--tokenizer_path", type=str,
                        default="corpus_build/tokenizer_32k.json")
    parser.add_argument("--output_dir", type=str, default="runs/curriculum")
    parser.add_argument("--checkpoint_password", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--cycles", type=int, default=3,
                        help="Number of full curriculum passes")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda | cpu | directml | cuda:N (auto-detect by default)")
    parser.add_argument("--dtype", type=str, default="fp32",
                        choices=["fp16", "fp32"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Required for the 4096d/36-block model on A100 80GB")
    # Architecture overrides — default to PRODUCTION_CONFIG.
    parser.add_argument("--d_model", type=int, default=PRODUCTION_CONFIG["d_model"])
    parser.add_argument("--n_blocks", type=int, default=PRODUCTION_CONFIG["n_blocks"])
    parser.add_argument("--num_episodes", type=int,
                        default=PRODUCTION_CONFIG["num_episodes"])
    parser.add_argument("--hebb_rate", type=float,
                        default=PRODUCTION_CONFIG["hebb_rate"])
    parser.add_argument("--error_rate", type=float,
                        default=PRODUCTION_CONFIG["error_rate"])
    parser.add_argument("--spike_threshold", type=float,
                        default=PRODUCTION_CONFIG["spike_threshold"])
    parser.add_argument("--no_backward_pass", action="store_true")
    # Testing knob: limit each stage to its first N files.
    parser.add_argument("--first_n_per_stage", type=int, default=0,
                        help="If >0, take only the first N files of each stage")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = _select_device(args.device)
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"Device: {device}    Precision: {args.dtype}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Curriculum stages ---------------------------------------------------
    stages = parse_curriculum_stages(args.file_list)
    if args.first_n_per_stage > 0:
        stages = [(name, files[:args.first_n_per_stage]) for name, files in stages]
    print(f"Curriculum: {len(stages)} stages")
    for name, files in stages:
        print(f"  - {name}: {len(files)} files")

    # ---- Tokenizer / model / optimizer ---------------------------------------
    tokenizer_path = Path(args.tokenizer_path)
    config: dict[str, Any] = {
        "d_model": args.d_model,
        "n_blocks": args.n_blocks,
        "vocab_size": PRODUCTION_CONFIG["vocab_size"],
        "seq_len": args.seq_len,
        "num_episodes": args.num_episodes,
        "hebb_rate": args.hebb_rate,
        "error_rate": args.error_rate,
        "homeostatic_decay": 0.001,
        "set_point_adapt_rate": 1e-6,
        "spike_threshold": args.spike_threshold,
        "membrane_leak": 0.1,
        "refractory_steps": 3,
        "delay_steps": 2,
        "backward_pass_enabled": not args.no_backward_pass,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "gradient_checkpointing": args.gradient_checkpointing,
    }

    start_cycle = 1
    start_stage_idx = 0
    history: dict[str, list] = {
        "stages": [],
    }
    health_history: list[dict] = []
    living_state_before: dict[str, torch.Tensor] | None = None

    if args.resume:
        print(f"\n=== Resuming from {args.resume} ===")
        ckpt = load_checkpoint(args.resume, args.checkpoint_password, device)
        # Reconstruct the tokenizer (must match the checkpoint).
        if "tokenizer_state" in ckpt:
            tokenizer = BPETokenizer.from_state(ckpt["tokenizer_state"])
        else:
            tokenizer = BPETokenizer.load(tokenizer_path)
        ckpt_config = ckpt["config"]
        # Trust the checkpoint's architecture; CLI overrides for arch are
        # ignored on resume to prevent silent shape mismatches.
        for k in (
            "d_model", "n_blocks", "vocab_size", "seq_len", "num_episodes",
            "hebb_rate", "error_rate", "homeostatic_decay",
            "set_point_adapt_rate", "spike_threshold", "membrane_leak",
            "refractory_steps", "delay_steps", "backward_pass_enabled",
        ):
            if k in ckpt_config:
                config[k] = ckpt_config[k]
        model = _build_model(config, device, dtype, args.gradient_checkpointing)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer = DirectMLAdamW(model.parameters(), lr=args.lr)
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except (ValueError, RuntimeError) as e:
                print(f"  optimizer state didn't restore cleanly: {e}")

        curriculum = ckpt.get("curriculum", {})
        last_cycle = int(curriculum.get("cycle", 1))
        last_stage_idx = int(curriculum.get("stage_idx", -1))
        history = ckpt.get("training_history", history)
        health_history = ckpt.get("substrate_health", {}).get(
            "epoch_snapshots", []
        )
        # Resume from the NEXT stage. If we just finished the last stage of
        # a cycle, advance the cycle counter and reset the stage index.
        if last_stage_idx + 1 < len(stages):
            start_cycle = last_cycle
            start_stage_idx = last_stage_idx + 1
        else:
            start_cycle = last_cycle + 1
            start_stage_idx = 0
        print(f"  Last completed: cycle {last_cycle}, stage "
              f"'{curriculum.get('stage_name', '?')}' (idx {last_stage_idx})")
        print(f"  Resuming at: cycle {start_cycle}, "
              f"stage '{stages[start_stage_idx][0]}'")
    else:
        print(f"\n=== Loading tokenizer from {tokenizer_path} ===")
        tokenizer = BPETokenizer.load(tokenizer_path)
        print(f"BPE vocab: {tokenizer.vocab_size}")
        config["vocab_size"] = tokenizer.vocab_size
        model = _build_model(config, device, dtype, args.gradient_checkpointing)
        optimizer = DirectMLAdamW(model.parameters(), lr=args.lr)
        n_params = model.total_parameters()
        print(f"Trainable params: {n_params['trainable']:,}")
        print(f"Living buffers:   {n_params['living_buffers']:,}")
        print(f"Gradient checkpointing: "
              f"{'ON' if args.gradient_checkpointing else 'OFF'}")

    living_state_before = snapshot_living_state(model)

    # ---- Curriculum loop -----------------------------------------------------
    for cycle in range(start_cycle, args.cycles + 1):
        print(f"\n{'=' * 60}")
        print(f"CYCLE {cycle} / {args.cycles}")
        print(f"{'=' * 60}")

        first_stage_in_cycle = (
            start_stage_idx if cycle == start_cycle else 0
        )
        for stage_idx in range(first_stage_in_cycle, len(stages)):
            stage_name, files = stages[stage_idx]
            t_stage = time.time()
            print(f"\n--- Cycle {cycle}, Stage {stage_idx + 1}/{len(stages)}: "
                  f"{stage_name} ({len(files)} files) ---")

            train_loader, val_loader, total_tokens = _build_stage_loaders(
                files, tokenizer, args.seq_len, args.batch_size, args.stride,
            )

            t_train = time.time()
            train_loss = train_epoch(
                model, train_loader, optimizer, device,
                is_living=True, epoch=cycle,
            )
            train_elapsed = time.time() - t_train

            val_loss = eval_model(model, val_loader, device)
            ext_metrics = collect_extended_metrics(model)

            living_state_after = snapshot_living_state(model)
            transition = stage_transition_delta(
                living_state_before or living_state_after,
                living_state_after,
            )
            living_state_before = living_state_after

            # Substrate health snapshot
            health = extract_substrate_health(model)
            health.update({
                "cycle": cycle,
                "stage": stage_name,
                "stage_idx": stage_idx,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })
            health_history.append(health)

            # Per-stage record
            stage_record = {
                "cycle": cycle,
                "stage_idx": stage_idx,
                "stage_name": stage_name,
                "n_files": len(files),
                "n_tokens": total_tokens,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "plasticity_mean": ext_metrics.get("plasticity_mean"),
                "plasticity_std": ext_metrics.get("plasticity_std"),
                "sp_drift_mean": ext_metrics.get("sp_drift_mean"),
                "transition_delta": transition,
                "elapsed_seconds": time.time() - t_stage,
            }
            history.setdefault("stages", []).append(stage_record)

            print(f"  train {train_loss:.4f} | val {val_loss:.4f} | "
                  f"gap {val_loss - train_loss:.3f} | "
                  f"plast {ext_metrics.get('plasticity_mean', 0):.4f}"
                  f"±{ext_metrics.get('plasticity_std', 0):.4f} | "
                  f"drift {ext_metrics.get('sp_drift_mean', 0):.5f} | "
                  f"dw {transition.get('delta_weight_mean', 0):.6f} | "
                  f"{train_elapsed:.0f}s")

            # Checkpoint after every stage (this is the resume granularity).
            if args.checkpoint_password or "LUTHI_CHECKPOINT_KEY" in __import__("os").environ:
                ckpt_path = _save_stage_checkpoint(
                    model=model, optimizer=optimizer, tokenizer=tokenizer,
                    config=config, cycle=cycle, stage_idx=stage_idx,
                    stage_name=stage_name, stages=stages,
                    history=history, health_history=health_history,
                    transition_delta=transition, output_dir=output_dir,
                    password=args.checkpoint_password,
                )
                print(f"  saved {ckpt_path.name}")

    # ---- Summary -------------------------------------------------------------
    summary_path = output_dir / "curriculum_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "config": config,
            "stages": [name for name, _ in stages],
            "cycles_completed": args.cycles,
            "history": history,
        }, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
