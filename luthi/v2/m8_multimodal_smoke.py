"""M8 multimodal smoke: verify the full text+audio+vision path end-to-end.

Separate from ``m8_smoke.py`` (the text-only Gate-1 plumbing check, which
stays fast). This smoke exercises:

- ``MultimodalPredictiveCodingLM`` with all three perceptual encoders active.
- ``JEPALoss`` + ``JEPATrainer`` orchestrating per-modality calls.
- The real ``AudioDataset`` (LibriSpeech) and ``VisionDataset`` (COCO)
  yielding raw waveforms/images that drive the encoders' gradients.
- ``MultimodalDataLoaderImpl`` dispatching by modality.
- Per-modality cadence + diagnostics on a *genuinely* sparse modality
  (Audio at alpha=0.7), via the strict assertion 4.8 specified 2026-06-07:
  ``audio_records_count == modality_step["audio"] // light_interval``.

Two segments:

1. **Integration (alpha=0).** ~60 steps with uniform per-modality
   sampling so each modality gets enough samples to exercise its
   yield-raw gradient path, per-modality VICReg, diagnostics, and the
   EMA-target memory fit with three encoders.
2. **Sparsity (alpha=0.7, light_interval=2).** ~300 steps with the
   skewed sampler so audio is genuinely rare; the per-modality cadence
   fix is the thing under test. The deterministic seed pins audio's
   step count so the records-vs-modality-step equality is an exact
   assertion, not a statistical one.

Run::

    python -m luthi.v2.m8_multimodal_smoke

Exit 0 = passed; non-zero = failed (the assertion message names the gap).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

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
    AudioDataset,
    AudioDatasetConfig,
    MultimodalDataLoaderImpl,
    TextDataset,
    TextDatasetConfig,
    VisionDataset,
    VisionDatasetConfig,
    _read_filelist,
)
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM


# Defaults assume the standard repo layout + the disk paths Brian confirmed
# 2026-06-07.
DEFAULT_TOKENIZER = Path("corpus_build/tokenizer_32k.json")
DEFAULT_TEXT_FILELIST = Path("corpus_build/m7_filelist.txt")
DEFAULT_AUDIO_ROOT = Path("E:/data/LibriSpeech/dev-clean")
DEFAULT_VISION_ROOT = Path("E:/data/coco/val2017")
DEFAULT_RUN_DIR = Path("runs/m8_multimodal_smoke")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M8 multimodal smoke")
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--text-filelist", type=Path, default=DEFAULT_TEXT_FILELIST)
    p.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    p.add_argument("--vision-root", type=Path, default=DEFAULT_VISION_ROOT)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--n-text-files", type=int, default=5)
    p.add_argument("--text-batch-size", type=int, default=4)
    p.add_argument("--audio-batch-size", type=int, default=2)
    p.add_argument("--vision-batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--max-audio-samples", type=int, default=80000)
    p.add_argument("--integration-steps", type=int, default=60)
    p.add_argument("--sparsity-steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_tiny_multimodal_model(vocab_size: int) -> MultimodalPredictiveCodingLM:
    """Tiny model with all three encoders active. d_model=64 keeps the EMA
    target's three encoder copies small enough to fit at smoke scale."""
    return MultimodalPredictiveCodingLM(
        vocab_size=vocab_size,
        d_model=64,
        n_blocks=2,
        n_heads=2,
        ffn_expansion=1,
        max_seq_len=128,
        max_audio_tokens=256,
        max_vision_tokens=256,
        pc_rate=0.001,
        pred_learning_rate=0.0001,
        backward_pass_enabled=True,
    )


def build_loader(args: argparse.Namespace) -> MultimodalDataLoaderImpl:
    """Builds the three-modality loader, one tiny per-modality subset each."""
    # Tiny text subset.
    all_text = _read_filelist(args.text_filelist)
    if len(all_text) < args.n_text_files:
        raise ValueError(
            f"text filelist has {len(all_text)} entries, need {args.n_text_files}"
        )
    text_paths = [Path(p) for p in all_text[: args.n_text_files]]

    text = TextDataset(TextDatasetConfig(
        source_paths=text_paths,
        tokenizer_path=args.tokenizer,
        batch_size=args.text_batch_size,
        seq_len=args.seq_len,
        stride=args.stride,
        base_seed=args.seed,
    ))
    audio = AudioDataset(AudioDatasetConfig(
        root=args.audio_root,
        batch_size=args.audio_batch_size,
        max_audio_samples=args.max_audio_samples,
        base_seed=args.seed + 1,
    ))
    vision = VisionDataset(VisionDatasetConfig(
        root=args.vision_root,
        batch_size=args.vision_batch_size,
        base_seed=args.seed + 2,
    ))
    return MultimodalDataLoaderImpl(text=text, audio=audio, vision=vision)


def make_sampler(
    loader: MultimodalDataLoaderImpl,
    alpha: float,
    seed: int,
) -> tuple[ModalitySampler, SamplerConfig]:
    sampler_cfg = SamplerConfig(
        corpus_sizes_tokens=loader.corpus_sizes_tokens(),
        alpha=alpha,
    )
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return ModalitySampler(sampler_cfg, generator=gen), sampler_cfg


def build_trainer(
    loader: MultimodalDataLoaderImpl,
    text_dataset: TextDataset,
    sampler: ModalitySampler,
    sampler_cfg: SamplerConfig,
    run_dir: Path,
    lr: float,
    light_interval: int,
    deep_interval: int,
) -> JEPATrainer:
    vocab_size = text_dataset.vocab_size()
    model = build_tiny_multimodal_model(vocab_size)
    loss_module = JEPALoss(online_encoder=model)
    optimizer = optim.AdamW(
        [p for p in loss_module.parameters() if p.requires_grad],
        lr=lr,
    )
    runner_cfg = RunnerConfig(
        sampler=sampler_cfg,
        checkpoint=CheckpointConfig(interval_seconds=10**9, rolling_slots=3),
        logging=LoggingConfig(
            light_interval_batches=light_interval,
            deep_interval_batches=deep_interval,
        ),
        kill_criteria=KillCriteriaConfig(
            # Kills stay inactive throughout: rare modalities haven't
            # accumulated enough observations for thresholds to be
            # meaningful in a smoke-scale window. The integration check
            # is "metrics emit finite values," not "kills don't fire."
            warmup_batches=10**9,
            loss_descent_window=10**6,
        ),
        epoch=EpochConfig(max_epochs=1, max_batches_per_epoch=10**9),
    )
    return JEPATrainer(
        loss_module=loss_module,
        optimizer=optimizer,
        sampler=sampler,
        data_loader=loader,
        config=runner_cfg,
        run_dir=run_dir,
    )


def run_steps(
    trainer: JEPATrainer,
    sampler: ModalitySampler,
    n_steps: int,
    phase_name: str,
) -> list[dict]:
    """Drive train_step + diagnostics + kill check directly."""
    records: list[dict] = []
    for _ in range(n_steps):
        modality = sampler.sample()
        batch = trainer.data_loader.next_batch(modality)
        step_out = trainer.train_step(modality, batch)
        trainer._update_coverage(modality, batch)

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
            trainer._human_log_line(record)
            records.append(record)

        kill_reason = trainer._check_kill_criteria(modality)
        if kill_reason is not None:
            print(
                f"[smoke {phase_name}] WARN: kill criterion fired "
                f"(warmup should have masked it): {kill_reason}"
            )

    return records


def assert_finite(records: list[dict], stage: str) -> None:
    for i, rec in enumerate(records):
        for key in ("loss", "l_pred", "l_var", "l_cov"):
            val = rec.get(key)
            if val is not None and not math.isfinite(val):
                raise AssertionError(
                    f"[{stage}] non-finite {key}={val} at record {i}"
                )
        for section_key in ("light", "deep"):
            section = rec.get(section_key, {})
            for k, v in section.items():
                if isinstance(v, (int, float)) and not math.isfinite(v):
                    raise AssertionError(
                        f"[{stage}] non-finite {section_key}.{k}={v} "
                        f"at record {i}"
                    )


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    if not args.tokenizer.exists():
        print(f"[smoke] FAIL: tokenizer not found at {args.tokenizer}", file=sys.stderr)
        return 2
    if not args.text_filelist.exists():
        print(f"[smoke] FAIL: text filelist not found at {args.text_filelist}", file=sys.stderr)
        return 2
    if not args.audio_root.exists():
        print(f"[smoke] FAIL: audio root not found at {args.audio_root}", file=sys.stderr)
        return 2
    if not args.vision_root.exists():
        print(f"[smoke] FAIL: vision root not found at {args.vision_root}", file=sys.stderr)
        return 2

    if args.run_dir.exists():
        shutil.rmtree(args.run_dir)
    args.run_dir.mkdir(parents=True)

    print("[smoke] Building multimodal loader ...")
    loader = build_loader(args)
    sizes = loader.corpus_sizes_tokens()
    print(f"[smoke] Per-modality tokens_per_pass: {sizes}")

    # We use the same text dataset reference for vocab_size lookup.
    text_dataset: TextDataset = loader._datasets["text"]  # type: ignore[assignment]

    # ----- Phase 1: integration (alpha=0) -----
    print("[smoke] Phase 1 (integration, alpha=0): building trainer ...")
    sampler1, sampler_cfg1 = make_sampler(loader, alpha=0.0, seed=args.seed)
    print(f"[smoke]   per-modality probs: {sampler1.per_modality_probs()}")

    trainer = build_trainer(
        loader=loader,
        text_dataset=text_dataset,
        sampler=sampler1,
        sampler_cfg=sampler_cfg1,
        run_dir=args.run_dir,
        lr=args.lr,
        light_interval=5,
        deep_interval=20,
    )
    n_params = trainer.loss_module.online_encoder.total_parameters()
    print(
        f"[smoke]   Model: {n_params['trainable']:,} trainable + "
        f"{n_params['living_buffers']:,} living-weight buffers"
    )

    print(f"[smoke] Phase 1: running {args.integration_steps} steps ...")
    records1 = run_steps(trainer, sampler1, args.integration_steps, "phase1")
    assert_finite(records1, "phase1")

    # Every modality must have been sampled and produced at least one record.
    modality_step_after_p1 = dict(trainer.modality_step)
    print(f"[smoke]   modality_step after phase 1: {modality_step_after_p1}")
    per_modality_record_counts_p1: dict[str, int] = {"text": 0, "audio": 0, "vision": 0}
    for r in records1:
        per_modality_record_counts_p1[r["modality"]] = (
            per_modality_record_counts_p1.get(r["modality"], 0) + 1
        )
    print(f"[smoke]   per-modality records: {per_modality_record_counts_p1}")

    for m in ("text", "audio", "vision"):
        if modality_step_after_p1[m] == 0:
            raise AssertionError(
                f"[smoke phase1] modality {m!r} never sampled "
                f"(alpha=0 should give it ~33% probability)"
            )
        if per_modality_record_counts_p1[m] == 0 and modality_step_after_p1[m] >= 5:
            raise AssertionError(
                f"[smoke phase1] modality {m!r} reached "
                f"modality_step={modality_step_after_p1[m]} but produced "
                f"zero records (light_interval=5; expected >= 1)"
            )

    # ----- Phase 2: sparsity (alpha=0.7) -----
    # Swap sampler in place; trainer holds the new one for run_steps.
    print("[smoke] Phase 2 (sparsity, alpha=0.7): swapping sampler ...")
    sampler2, sampler_cfg2 = make_sampler(loader, alpha=0.7, seed=args.seed + 100)
    trainer.sampler = sampler2
    # Switch to light_interval=2 (per-modality units). With alpha=0.7,
    # audio occupies ~1% of steps; over the sparsity-steps window it
    # will accumulate enough audio_steps that floor(modality_step['audio'] / 2)
    # is what we strictly assert against.
    trainer.config.logging.light_interval_batches = 2
    trainer.config.logging.deep_interval_batches = 50
    print(f"[smoke]   per-modality probs: {sampler2.per_modality_probs()}")

    pre_p2 = dict(trainer.modality_step)
    pre_p2_audio_records = per_modality_record_counts_p1["audio"]

    print(f"[smoke] Phase 2: running {args.sparsity_steps} steps ...")
    records2 = run_steps(trainer, sampler2, args.sparsity_steps, "phase2")
    assert_finite(records2, "phase2")

    modality_step_after_p2 = dict(trainer.modality_step)
    print(f"[smoke]   modality_step after phase 2: {modality_step_after_p2}")

    # Count audio records produced *only during phase 2* by reading the
    # JSONL (the runner persisted everything from both phases there).
    all_jsonl_records = read_jsonl(args.run_dir / "training_log.jsonl")
    audio_records_total = sum(1 for r in all_jsonl_records if r["modality"] == "audio")
    audio_records_p2 = audio_records_total - pre_p2_audio_records

    # Expected count, derived from per-modality cadence math:
    #   audio_records_in_p2 = floor(audio_step_after_p2 / interval)
    #                       - floor(audio_step_after_p1 / interval_p1)
    # In phase 1 interval was 5; in phase 2 it's 2. The audio records
    # *during phase 2* should equal:
    #   floor(audio_step_after_p2 / 2) - floor(audio_step_after_p1 / 2)
    # ...because the cadence at the boundary changes mid-stream. But
    # actually the runner fires diagnostics based on the *current* interval
    # value, so the relevant divisor for the boundary check is light_interval
    # at the moment each step ran. Easier: assert that the total audio
    # records logged equals the number of integer multiples of the
    # respective intervals hit, summed by phase.
    audio_step_after_p1 = pre_p2["audio"]
    audio_step_after_p2 = modality_step_after_p2["audio"]
    expected_p1_audio_records = audio_step_after_p1 // 5
    expected_p2_audio_records = (
        (audio_step_after_p2 // 2) - (audio_step_after_p1 // 2)
    )
    expected_total_audio_records = (
        expected_p1_audio_records + expected_p2_audio_records
    )

    print(
        f"[smoke]   audio: step_after_p1={audio_step_after_p1} "
        f"step_after_p2={audio_step_after_p2} "
        f"records_total={audio_records_total} "
        f"expected_total={expected_total_audio_records}"
    )

    if audio_records_total != expected_total_audio_records:
        raise AssertionError(
            f"[smoke] sparse-cadence assertion failed: audio_records_total="
            f"{audio_records_total} but expected "
            f"{expected_total_audio_records} "
            f"(audio_step_p1={audio_step_after_p1} interval_p1=5, "
            f"audio_step_p2={audio_step_after_p2} interval_p2=2). "
            f"This is the exact test 4.8 specified 2026-06-07: per-"
            f"modality cadence must mean audio's records track its own "
            f"step count, not global cadence. A regression to global-"
            f"step gating would break this equality."
        )

    # And the sparse modality must have been instrumented despite being
    # globally rare (the headline behavior the cadence fix exists for).
    if modality_step_after_p2["audio"] == 0:
        raise AssertionError(
            f"[smoke phase2] audio was never sampled in {args.sparsity_steps} "
            f"steps at alpha=0.7 -- sampling RNG may be misconfigured"
        )
    if audio_records_total == 0 and modality_step_after_p2["audio"] >= 2:
        raise AssertionError(
            f"[smoke phase2] audio reached modality_step="
            f"{modality_step_after_p2['audio']} but logged zero records "
            f"in either phase. The per-modality cadence fix should have "
            f"diagnosed the sparse modality."
        )

    # Run-config archived + both log files written.
    cfg_path = args.run_dir / "run_config.json"
    jsonl_path = args.run_dir / "training_log.jsonl"
    human_log_path = args.run_dir / "training.log"
    for p in (cfg_path, jsonl_path, human_log_path):
        if not p.exists() or p.stat().st_size == 0:
            raise AssertionError(f"[smoke] missing/empty artifact: {p}")

    print(f"[smoke] All artifacts present and non-empty")
    print(f"[smoke] PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
