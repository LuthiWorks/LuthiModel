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
# Plumbing smoke is corpus-agnostic; pointing at m7_filelist.txt for
# convenience (any 5 text files would work). Does NOT imply this is the
# M8 baseline corpus -- v0.5 §2 specifies gutenberg_4gb, and that
# question is separate from this smoke's pass/fail.
DEFAULT_FILELIST = Path("corpus_build/m7_filelist.txt")
DEFAULT_RUN_DIR = Path("runs/m8_smoke")


def _param_l1_checksum(module: torch.nn.Module) -> float:
    """Sum of L1 norms across all parameters. Deterministic single float
    that changes if any parameter changes -- used to verify resume
    actually round-trips state (not just "loss didn't explode").
    """
    total = 0.0
    for p in module.parameters():
        total += float(p.detach().abs().sum().item())
    return total


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
            # Set warmup beyond the smoke length so kill criteria stay
            # inactive throughout. Early in training the EMA target is
            # ~identical to online (deepcopy + momentum 0.996 has barely
            # diverged), so encoder-asymmetry cosine reads ~1.0 -- not
            # because of collapse but because of initialization. Tripping
            # kill-5 in the smoke would be a false positive. Real-run
            # warmup is 5000; a synthetic-collapse test exercises the
            # kill path separately.
            warmup_batches=10**9,
            loss_descent_window=200,
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
        # train_step advances both global_step and modality_step[modality]
        # internally (per 4.8 review 2026-06-06 item A) -- don't increment
        # them again in this loop.
        trainer._update_coverage(modality, batch)

        # Cadence keys off the modality's own step counter so rare
        # modalities are instrumented on their own clock (item A).
        m_step = trainer.modality_step[modality]
        light_due = (
            m_step % trainer.config.logging.light_interval_batches == 0
        )
        deep_due = (
            m_step % trainer.config.logging.deep_interval_batches == 0
        )
        if light_due or deep_due:
            record = trainer._compute_and_log_diagnostics(
                step_out, light=light_due, deep=deep_due,
            )
            records.append(record)
            # Exercise the human-readable log path too (per 4.8 2026-06-06:
            # the prior smoke skipped this and training.log was never
            # written; we want to verify both log writers work).
            trainer._human_log_line(record)

        # Kill checks are run but only logged in the smoke -- with warmup
        # set beyond the smoke length, they should never trigger; if they
        # do, surface the fact rather than failing the smoke (kill-path
        # behavior gets its own dedicated test, not this plumbing run).
        kill_reason = trainer._check_kill_criteria(modality)
        if kill_reason is not None:
            print(
                f"[smoke {phase_name}] WARN: kill criterion fired during "
                f"smoke (warmup should have masked it): {kill_reason}"
            )

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

    # Snapshot state before kill so we can verify resume actually
    # round-trips, not just that loss didn't explode (per 4.8 review
    # 2026-06-06).
    pre_kill = {
        "online_checksum": _param_l1_checksum(trainer1.loss_module.online_encoder),
        "target_checksum": _param_l1_checksum(trainer1.loss_module.target_encoder),
        "predictor_checksum": _param_l1_checksum(trainer1.loss_module.predictor),
        "loader_state": trainer1.data_loader.state_dict(),
        "sampler_gen_state": trainer1.sampler.generator.get_state().clone(),
        "global_step": trainer1.global_step,
        "modality_step": dict(trainer1.modality_step),
        "tokens_consumed": dict(trainer1.tokens_consumed),
    }

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

    # Verify resume actually round-tripped state (per 4.8 review
    # 2026-06-06: not just "loss didn't explode" -- exact state match
    # so a regression where resume_from_latest silently re-inits would
    # fail the smoke).
    if trainer2.global_step != pre_kill["global_step"]:
        raise AssertionError(
            f"[smoke] resume global_step mismatch: trainer2={trainer2.global_step} "
            f"pre_kill={pre_kill['global_step']}"
        )
    if dict(trainer2.modality_step) != pre_kill["modality_step"]:
        raise AssertionError(
            f"[smoke] resume modality_step mismatch: "
            f"trainer2={dict(trainer2.modality_step)} "
            f"pre_kill={pre_kill['modality_step']}"
        )
    if dict(trainer2.tokens_consumed) != pre_kill["tokens_consumed"]:
        raise AssertionError(
            f"[smoke] resume tokens_consumed mismatch: "
            f"trainer2={dict(trainer2.tokens_consumed)} "
            f"pre_kill={pre_kill['tokens_consumed']}"
        )
    post = {
        "online_checksum": _param_l1_checksum(trainer2.loss_module.online_encoder),
        "target_checksum": _param_l1_checksum(trainer2.loss_module.target_encoder),
        "predictor_checksum": _param_l1_checksum(trainer2.loss_module.predictor),
    }
    for key in ("online_checksum", "target_checksum", "predictor_checksum"):
        if post[key] != pre_kill[key]:
            raise AssertionError(
                f"[smoke] resume {key} mismatch: "
                f"pre={pre_kill[key]:.6f} post={post[key]:.6f} "
                f"(loaded state != saved state -- resume_from_latest "
                f"is not actually restoring parameters)"
            )
    post_loader_state = trainer2.data_loader.state_dict()
    if post_loader_state != pre_kill["loader_state"]:
        raise AssertionError(
            f"[smoke] resume loader state mismatch: "
            f"pre={pre_kill['loader_state']} post={post_loader_state} "
            f"(without-replacement guarantee not surviving resume)"
        )
    post_sampler_gen = trainer2.sampler.generator.get_state()
    if not torch.equal(post_sampler_gen, pre_kill["sampler_gen_state"]):
        raise AssertionError(
            "[smoke] resume sampler generator state mismatch "
            "(sampler RNG not restored from checkpoint)"
        )
    print(
        "[smoke] Resume state round-trip OK: "
        "online/target/predictor params + loader shuffle position + "
        "sampler RNG all match pre-kill snapshot."
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

    # Both log writers exercised? (per 4.8 2026-06-06).
    jsonl_path = args.run_dir / "training_log.jsonl"
    human_log_path = args.run_dir / "training.log"
    if not jsonl_path.exists():
        raise AssertionError(f"[smoke] training_log.jsonl not written at {jsonl_path}")
    if not human_log_path.exists():
        raise AssertionError(
            f"[smoke] training.log not written at {human_log_path} -- "
            f"the smoke's loop must call trainer._human_log_line after "
            f"_compute_and_log_diagnostics to exercise that path."
        )
    if human_log_path.stat().st_size == 0:
        raise AssertionError(f"[smoke] training.log is empty at {human_log_path}")

    print("[smoke] PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
