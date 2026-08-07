"""Cosine LR schedule + the v4 depth-family bundle (Brian's 2026-07-20
ruling: depth 2->4 blocks + cosine LR + 2x SIGReg, one arm).

Covers:
- cosine_lr_scale: pure-function endpoints, midpoint, clamping, guards.
- JEPATrainer: enabled-without-total_steps refused; lr actually follows
  the schedule across steps; disabled schedule leaves lr untouched.
- Driver wiring: the v4 arm is complete across every per-arm table
  (config/taper/filelist/sigreg/cosine), stage 9 points at it, and the
  sigreg override reaches JEPALoss.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from luthi.v2.jepa_loss import JEPALoss, SIGREG_LAMBD
from luthi.v2.jepa_runner import (
    CheckpointConfig,
    EpochConfig,
    JEPATrainer,
    KillCriteriaConfig,
    LoggingConfig,
    LRScheduleConfig,
    ModalitySampler,
    RunnerConfig,
    SamplerConfig,
    cosine_lr_scale,
)
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM

VOCAB, D, B, SEQ = 64, 32, 2, 16
BASE_LR = 3e-4


class _TextLoader:
    def __init__(self, seed: int = 7):
        self.gen = torch.Generator().manual_seed(seed)

    def next_batch(self, modality: str) -> dict:
        return {"text_tokens": torch.randint(0, VOCAB, (B, SEQ), generator=self.gen)}

    def batch_token_count(self, modality: str, batch: dict) -> int:
        return int(batch["text_tokens"].numel())

    def state_dict(self) -> dict:
        return {"gen": self.gen.get_state()}

    def load_state_dict(self, state: dict) -> None:
        self.gen.set_state(state["gen"])

    def corpus_sizes_tokens(self) -> dict:
        return {"text": 1000}


def _build_trainer(run_dir: Path, lr_schedule: LRScheduleConfig) -> JEPATrainer:
    torch.manual_seed(7)
    model = MultimodalPredictiveCodingLM(
        vocab_size=VOCAB, d_model=D, n_blocks=2, n_heads=2,
        ffn_expansion=1, max_seq_len=SEQ,
        max_audio_tokens=SEQ, max_vision_tokens=SEQ,
        backward_pass_enabled=False,
    )
    loss_module = JEPALoss(online_encoder=model)
    optimizer = optim.AdamW(
        [p for p in loss_module.parameters() if p.requires_grad], lr=BASE_LR,
    )
    loader = _TextLoader()
    sampler_cfg = SamplerConfig(
        corpus_sizes_tokens=loader.corpus_sizes_tokens(), alpha=0.7,
    )
    cfg = RunnerConfig(
        sampler=sampler_cfg,
        checkpoint=CheckpointConfig(interval_seconds=10**9),
        logging=LoggingConfig(
            light_interval_batches=10**9, deep_interval_batches=10**9,
        ),
        kill_criteria=KillCriteriaConfig(warmup_batches=10**9),
        epoch=EpochConfig(max_epochs=1, max_batches_per_epoch=10**9),
        lr_schedule=lr_schedule,
    )
    return JEPATrainer(
        loss_module=loss_module, optimizer=optimizer,
        sampler=ModalitySampler(sampler_cfg), data_loader=loader,
        config=cfg, run_dir=run_dir,
    )


class TestCosineScale:
    def test_endpoints_and_midpoint(self):
        assert cosine_lr_scale(0.0, 0.1) == pytest.approx(1.0)
        assert cosine_lr_scale(1.0, 0.1) == pytest.approx(0.1)
        assert cosine_lr_scale(0.5, 0.1) == pytest.approx(0.55)

    def test_clamps_past_total(self):
        assert cosine_lr_scale(1.7, 0.1) == pytest.approx(0.1)
        assert cosine_lr_scale(-0.3, 0.1) == pytest.approx(1.0)

    def test_monotone_decreasing(self):
        vals = [cosine_lr_scale(p / 20, 0.1) for p in range(21)]
        assert all(a >= b for a, b in zip(vals, vals[1:]))

    def test_zero_floor_refused(self):
        with pytest.raises(ValueError):
            cosine_lr_scale(0.5, 0.0)
        with pytest.raises(ValueError):
            cosine_lr_scale(0.5, 1.5)


class TestTrainerSchedule:
    def test_enabled_requires_total_steps(self, tmp_path):
        with pytest.raises(ValueError, match="total_steps"):
            _build_trainer(
                tmp_path, LRScheduleConfig(enabled=True, total_steps=0),
            )

    def test_lr_follows_schedule(self, tmp_path):
        total = 10
        tr = _build_trainer(
            tmp_path,
            LRScheduleConfig(enabled=True, min_lr_ratio=0.1, total_steps=total),
        )
        seen = []
        for _ in range(total):
            tr.train_step("text", tr.data_loader.next_batch("text"))
            seen.append(tr.optimizer.param_groups[0]["lr"])
        # Step k applies the scale for progress (k-1)/total (pre-increment
        # global_step), so the last observed lr is scale(9/10).
        assert seen[0] == pytest.approx(BASE_LR * cosine_lr_scale(0.0, 0.1))
        assert seen[-1] == pytest.approx(
            BASE_LR * cosine_lr_scale((total - 1) / total, 0.1)
        )
        assert all(a >= b for a, b in zip(seen, seen[1:]))

    def test_disabled_leaves_lr_alone(self, tmp_path):
        tr = _build_trainer(tmp_path, LRScheduleConfig(enabled=False))
        for _ in range(3):
            tr.train_step("text", tr.data_loader.next_batch("text"))
        assert tr.optimizer.param_groups[0]["lr"] == pytest.approx(BASE_LR)


class TestMidSeedResume:
    """Resume must CONTINUE the interrupted epoch, not restart its token
    count. Pre-fix, run() clobbered the restored epoch_token_baseline
    via _start_new_epoch(), so a resumed epoch served a full extra pass
    (2026-07-20, found wiring driver-level mid-seed resume)."""

    def test_resumed_epoch_continues_not_restarts(self, tmp_path):
        # Fake loader: 1000-token corpus, 32 tokens/batch -> the coverage
        # anchor ends epoch 0 at ceil(1000/32) = 32 steps.
        full_epoch_steps = math.ceil(1000 / (B * SEQ))

        tr_a = _build_trainer(tmp_path, LRScheduleConfig(enabled=False))
        for _ in range(10):
            tr_a.train_step("text", tr_a.data_loader.next_batch("text"))
            tr_a.tokens_consumed["text"] += B * SEQ
        tr_a._checkpoint(reason="test")

        tr_b = _build_trainer(tmp_path, LRScheduleConfig(enabled=False))
        tr_b.resume_from_latest()
        assert tr_b.global_step == 10
        assert tr_b._resumed_mid_epoch is True

        outcome = tr_b.run()
        assert outcome == "completed"
        # Continue-from-10: total lands at one full epoch (+/- a partial
        # batch), NOT 10 + a full epoch.
        assert tr_b.global_step <= full_epoch_steps + 2, (
            f"resumed epoch restarted its token count: "
            f"{tr_b.global_step} steps vs expected ~{full_epoch_steps}"
        )


class TestV4ArmWiring:
    """The v4 bundle must be complete in every per-arm table -- a missing
    entry silently runs the arm with defaults (the exact class of quiet
    downgrade the 2026-07-20 LuthiScope build chased all morning)."""

    def test_arm_tables_complete(self):
        from scripts.jepa_pilot_driver import (
            ARM_CONFIGS, ARM_COSINE, ARM_FILELIST, ARM_SIGREG, ARM_TAPER,
            STAGES,
        )
        arm = "living_v4_4x_d4"
        assert STAGES[9] == [(arm, 512)]
        assert ARM_CONFIGS[arm]["n_blocks"] == 4
        assert ARM_CONFIGS[arm]["mu_pc_enabled"] is True
        assert ARM_CONFIGS[arm]["backward_pass_enabled"] is True
        assert ARM_CONFIGS[arm]["consolidation_enabled"] is True
        assert ARM_TAPER[arm] is True
        assert arm in ARM_FILELIST
        assert ARM_SIGREG[arm] == pytest.approx(2 * SIGREG_LAMBD)
        assert ARM_COSINE[arm] is True

    def test_sigreg_override_reaches_loss(self):
        torch.manual_seed(0)
        model = MultimodalPredictiveCodingLM(
            vocab_size=VOCAB, d_model=D, n_blocks=2, n_heads=2,
            ffn_expansion=1, max_seq_len=SEQ,
            max_audio_tokens=SEQ, max_vision_tokens=SEQ,
            backward_pass_enabled=False,
        )
        from scripts.jepa_pilot_driver import ARM_SIGREG
        loss = JEPALoss(
            online_encoder=model,
            sigreg_lambd=ARM_SIGREG.get("living_v4_4x_d4", SIGREG_LAMBD),
        )
        assert loss.sigreg_lambd == pytest.approx(0.2)


class TestWarmup:
    """LR warmup (2026-08-06, Opus's finding): the JEPA runner shipped
    without the warmup the older trainers carry deliberately, so every
    JEPA run trained at full LR from step 0 -- at depth 8, straight into
    the sub-200-step destruction window. warmup_steps=0 must preserve
    every historical schedule bit-exactly."""

    def test_zero_warmup_is_bit_exact_legacy(self, tmp_path):
        total = 6
        tr = _build_trainer(
            tmp_path,
            LRScheduleConfig(enabled=True, min_lr_ratio=0.1,
                             total_steps=total, warmup_steps=0),
        )
        tr.train_step("text", tr.data_loader.next_batch("text"))
        assert tr.optimizer.param_groups[0]["lr"] == pytest.approx(
            BASE_LR * cosine_lr_scale(0.0, 0.1)
        )

    def test_ramp_then_cosine(self, tmp_path):
        total, w = 12, 4
        tr = _build_trainer(
            tmp_path,
            LRScheduleConfig(enabled=True, min_lr_ratio=0.1,
                             total_steps=total, warmup_steps=w),
        )
        seen = []
        for _ in range(total):
            tr.train_step("text", tr.data_loader.next_batch("text"))
            seen.append(tr.optimizer.param_groups[0]["lr"])
        # Ramp: step k applies (k+1)/w -- never exactly zero.
        assert seen[0] == pytest.approx(BASE_LR * 1 / w)
        assert seen[1] == pytest.approx(BASE_LR * 2 / w)
        assert seen[w - 1] == pytest.approx(BASE_LR * 1.0)
        # Cosine resumes over the REMAINING steps, from 1.0 downward.
        assert seen[w] == pytest.approx(
            BASE_LR * cosine_lr_scale(0.0, 0.1)
        )
        assert seen[-1] == pytest.approx(
            BASE_LR * cosine_lr_scale((total - 1 - w) / (total - w), 0.1)
        )
        # Rises through warmup, never above base, decays after.
        assert all(a <= b for a, b in zip(seen[:w], seen[1:w]))
        assert max(seen) <= BASE_LR * 1.0 + 1e-12
        assert all(a >= b for a, b in zip(seen[w:], seen[w + 1:]))
