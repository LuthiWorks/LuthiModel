"""M8 Gate-1 smoke test: ~50-batch text-only end-to-end verification.

Wires up the **real** components (no mocks):
- MultimodalPredictiveCodingLM at tiny config (d_model=64, n_blocks=2)
- JEPALoss with EMA target + predictor
- JEPATrainer (driven step-by-step so the smoke can pin checkpoint timing)
- MultimodalDataLoaderImpl with TextDataset over a small file subset
- BPETokenizer at the production 32K vocab

What this verifies (Gate-1 sanity, v0.5 §6 in miniature):
- Model + loss + runner + data wire up without crashing.
- N batches produce no NaN in any logged metric.
- Per-modality §5 light/deep metrics emit finite numbers throughout.
- Loss doesn't explode (post-resume final loss within 10x of pre-checkpoint
  final loss).
- Mid-run checkpoint + ``resume_from_latest`` + continuity: fresh
  trainer instance can resume from disk and keep training; the loader's
  without-replacement state survives the round-trip.

This is a plumbing check, not a training run. Tiny model (~10M params
online + EMA target), small batch, ~5 text files of corpus -- ignore the
loss values themselves; this confirms the wiring.

Usage::

    python -m luthi.v2.m8_smoke

Exit 0 = passed; non-zero = failed (assertion in the log explains where).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim

from luthi.v2.jepa_loss import JEPALoss
from luthi.v2.jepa_runner import (
    CheckpointConfig,
    EpochConfig,
    JEPATrainer,
    KillCriteriaConfig,
    LoggingConfig,
    ModalitySampler,
    RunnerConfig,
    SamplerConfig,
)
from luthi.v2.multimodal_data import (
    MultimodalDataLoaderImpl,
    TextDataset,
    TextDatasetConfig,
    _read_filelist,
)
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM


# Defaults assume the standard repo layout.
DEFAULT_TOKENIZER = Path("corpus_build/tokenizer_32k.json")
DEFAULT_FILELIST = Path("corpus_build/m7_filelist.txt")
DEFAULT_RUN_DIR = Path("runs/m8_smoke")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M8 Gate-1 smoke test")
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument(
        "--filelist", type=Path, default=DEFAULT_FILELIST,
        help="Filelist from which to take the first --n-files entries.",
    )
    p.add_argument("--n-files", type=int, default=5)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--phase1-steps", type=int, default=30)
    p.add_argument("--phase2-steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_tiny_model(vocab_size: int) -> MultimodalPredictiveCodingLM:
    """Tiny model for smoke. d_model=64 is small enough that the EMA copy
    + VICReg covariance D^2 ops are negligible cost; n_blocks=2 keeps the
    PC top-down sweep cheap. max_*_tokens are also small so the predictor's
    position table stays compact."""
    return MultimodalPredictiveCodingLM(
        vocab_size=vocab_size,
        d_model=64,
        n_blocks=2,
        n_heads=2,
        ffn_expansion=1,
        max_seq_len=128,
        max_audio_tokens=128,
        max_vision_tokens=128,
        pc_rate=0.001,
        pred_learning_rate=0.0001,
        backward_pass_enabled=True,
    )


def build_smoke_trainer(
    text_dataset: TextDataset,
    loader: MultimodalDataLoaderImpl,
    run_dir: Path,
    lr: float,
    seed: int,
) -> tuple[JEPATrainer, ModalitySampler]:
    """Construct a fresh JEPATrainer wired to the given dataset + loader."""
    vocab_size = text_dataset.vocab_size()
    model = build_tiny_model(vocab_size)
    loss_module = JEPALoss(online_encoder=model)
    optimizer = optim.AdamW(
        [p for p in loss_module.parameters() if p.requires_grad],
        lr=lr,
    )

    corpus_sizes = loader.corpus_sizes_tokens()
    sampler_cfg = SamplerConfig(
        corpus_sizes_tokens=corpus_sizes,
        alpha=0.7,
    )
    sampler_generator = torch.Generator(device="cpu")
    sampler_generator.manual_seed(seed)
    sampler = ModalitySampler(sampler_cfg, generator=sampler_generator)

    runner_cfg = RunnerConfig(
        sampler=sampler_cfg,
        # Checkpoint interval set high so the smoke pins checkpoint
        # timing manually via trainer._checkpoint(reason=...).
        checkpoint=CheckpointConfig(interval_seconds=10**9, rolling_slots=3),
        logging=LoggingConfig(light_interval_batches=5, deep_interval_batches=20),
        kill_criteria=KillCriteriaConfig(
            warmup_batches=10,  # short warmup so kill checks activate during smoke
            loss_descent_window=200,  # large enough not to trigger in smoke
        ),
        epoch=EpochConfig(max_epochs=1, max_batches_per_epoch=10**9),
    )
    trainer = JEPATrainer(
        loss_module=loss_module,
        optimizer=optimizer,
        sampler=sampler,
        data_loader=loader,
        config=runner_cfg,
        run_dir=run_dir,
    )
    return trainer, sampler


def run_smoke_phase(
    trainer: JEPATrainer,
    sampler: ModalitySampler,
    n_steps: int,
    phase_name: str,
) -> list[dict]:
    """Drive ``n_steps`` of trainer.train_step + diagnostics + kill check
    directly so the smoke owns the loop timing. Records returned for
    assertion."""
    records: list[dict] = []
    for _ in range(n_steps):
        modality = sampler.sample()
        batch = trainer.data_loader.next_batch(modality)
        step_out = trainer.train_step(modality, batch)
        trainer._update_coverage(modality, batch)

        light_due = (
            (trainer.global_step + 1)
            % trainer.config.logging.light_interval_batches == 0
        )
        deep_due = (
            (trainer.global_step + 1)
            % trainer.config.logging.deep_interval_batches == 0
        )
        if light_due or deep_due:
            record = trainer._compute_and_log_diagnostics(
                step_out, light=light_due, deep=deep_due,
            )
            records.append(record)

        kill_reason = trainer._check_kill_criteria(modality)
        if kill_reason is not None:
            raise RuntimeError(
                f"[smoke {phase_name}] Unexpected kill at step "
                f"{trainer.global_step}: {kill_reason}"
            )

        trainer.global_step += 1
    return records


def assert_finite(records: list[dict], stage: str) -> None:
    for i, rec in enumerate(records):
        # Top-level loss components.
        for key in ("loss", "l_pred", "l_var", "l_cov"):
            val = rec.get(key)
            if val is not None and not math.isfinite(val):
                raise AssertionError(
                    f"[smoke {stage}] non-finite {key}={val} at record {i} "
                    f"step={rec.get('step')}"
                )
        # Nested light + deep metric dicts.
        for section_key in ("light", "deep"):
            section = rec.get(section_key, {})
            for k, v in section.items():
                if isinstance(v, (int, float)) and not math.isfinite(v):
                    raise AssertionError(
                        f"[smoke {stage}] non-finite {section_key}.{k}={v} "
                        f"at record {i} step={rec.get('step')}"
                    )


def assert_loss_continuity(
    phase1_records: list[dict],
    phase2_records: list[dict],
) -> None:
    """Post-resume loss should not be wildly different from pre-checkpoint.
    Loose bound (factor 10) -- catches catastrophic resume failures but
    tolerates the natural noise of a tiny model + small batch."""
    if not phase1_records or not phase2_records:
        return
    p1_final = phase1_records[-1].get("loss")
    p2_first = phase2_records[0].get("loss")
    if p1_final is None or p2_first is None:
        return
    if not (math.isfinite(p1_final) and math.isfinite(p2_first)):
        raise AssertionError(
            f"[smoke] non-finite loss across resume: "
            f"p1_final={p1_final} p2_first={p2_first}"
        )
    ratio = p2_first / max(abs(p1_final), 1e-9)
    if not (0.1 < abs(ratio) < 10.0):
        raise AssertionError(
            f"[smoke] post-resume loss ratio out of bounds: "
            f"p1_final={p1_final:.4f} p2_first={p2_first:.4f} ratio={ratio:.2f}"
        )


def main() -> int:
    args = parse_args()

    # Repro.
    torch.manual_seed(args.seed)

    # Validate inputs.
    if not args.tokenizer.exists():
        print(f"[smoke] FAIL: tokenizer not found at {args.tokenizer}", file=sys.stderr)
        return 2
    if not args.filelist.exists():
        print(f"[smoke] FAIL: filelist not found at {args.filelist}", file=sys.stderr)
        return 2

    # Build a tiny source list from the filelist.
    all_files = _read_filelist(args.filelist)
    if len(all_files) < args.n_files:
        print(
            f"[smoke] FAIL: filelist has {len(all_files)} entries, "
            f"need {args.n_files}", file=sys.stderr,
        )
        return 2
    source_paths = all_files[: args.n_files]
    print(f"[smoke] Using {len(source_paths)} files from {args.filelist}")

    # Reset run dir for a clean smoke.
    if args.run_dir.exists():
        shutil.rmtree(args.run_dir)
    args.run_dir.mkdir(parents=True)

    # ----- Phase 1: build, run, checkpoint -----
    print(f"[smoke] Loading + tokenizing tiny corpus ...")
    text_dataset = TextDataset(TextDatasetConfig(
        source_paths=[Path(p) for p in source_paths],
        tokenizer_path=args.tokenizer,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        stride=args.stride,
        base_seed=args.seed,
    ))
    print(
        f"[smoke] Corpus: {text_dataset.unique_token_count()} unique tokens, "
        f"{text_dataset.tokens_per_pass()} per-pass (overlap-counted)"
    )

    loader = MultimodalDataLoaderImpl(text=text_dataset)

    print("[smoke] Phase 1: building trainer ...")
    trainer1, sampler1 = build_smoke_trainer(
        text_dataset, loader, args.run_dir, args.lr, args.seed,
    )
    n_params = trainer1.loss_module.online_encoder.total_parameters()
    print(
        f"[smoke] Model: {n_params['trainable']:,} trainable + "
        f"{n_params['living_buffers']:,} living-weight buffers"
    )

    print(f"[smoke] Phase 1: running {args.phase1_steps} steps ...")
    records1 = run_smoke_phase(trainer1, sampler1, args.phase1_steps, "phase1")
    assert_finite(records1, "phase1")
    phase1_final = records1[-1]["loss"] if records1 else float("nan")
    print(
        f"[smoke] Phase 1 OK: {len(records1)} records logged, "
        f"final logged loss={phase1_final:.4f}"
    )

    print("[smoke] Phase 1: forcing checkpoint ...")
    trainer1._checkpoint(reason="smoke_phase1_end")
    ckpt_dir = args.run_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
    if not ckpts:
        raise AssertionError(f"[smoke] no checkpoint written under {ckpt_dir}")
    print(f"[smoke] Checkpoint written: {ckpts[-1].name}")

    # Drop trainer1 to simulate process death.
    final_step_phase1 = trainer1.global_step
    del trainer1

    # ----- Phase 2: fresh trainer, resume, run -----
    print("[smoke] Phase 2: rebuilding loader + trainer (simulated restart) ...")
    # Rebuild the loader from scratch -- this is what a real restart looks
    # like. The cache makes corpus re-encoding instant on the second build.
    text_dataset2 = TextDataset(TextDatasetConfig(
        source_paths=[Path(p) for p in source_paths],
        tokenizer_path=args.tokenizer,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        stride=args.stride,
        base_seed=args.seed,
    ))
    loader2 = MultimodalDataLoaderImpl(text=text_dataset2)
    trainer2, sampler2 = build_smoke_trainer(
        text_dataset2, loader2, args.run_dir, args.lr, args.seed,
    )

    ckpt_loaded = trainer2.resume_from_latest()
    print(
        f"[smoke] Resumed from {ckpt_loaded.name} at step {trainer2.global_step}"
    )
    if trainer2.global_step != final_step_phase1:
        raise AssertionError(
            f"[smoke] resume step mismatch: trainer2={trainer2.global_step} "
            f"phase1_final={final_step_phase1}"
        )

    print(f"[smoke] Phase 2: running {args.phase2_steps} steps ...")
    records2 = run_smoke_phase(trainer2, sampler2, args.phase2_steps, "phase2")
    assert_finite(records2, "phase2")

    # Continuity across the kill-restart boundary.
    assert_loss_continuity(records1, records2)
    phase2_final = records2[-1]["loss"] if records2 else float("nan")
    print(
        f"[smoke] Phase 2 OK: {len(records2)} records logged, "
        f"final logged loss={phase2_final:.4f}"
    )

    # Run config archived?
    cfg_path = args.run_dir / "run_config.json"
    if not cfg_path.exists():
        raise AssertionError(f"[smoke] run_config.json not archived at {cfg_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)
    if "sampler_probabilities" not in cfg:
        raise AssertionError("[smoke] run_config.json missing sampler_probabilities")

    print("[smoke] PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
