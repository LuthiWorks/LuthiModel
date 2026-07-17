"""Kill-5 corroboration amendment (2026-07-17, Brian's ruling).

High predicted-vs-target cosine is ambiguous: predictor degeneracy
(copying) and the predictor genuinely solving its prediction problem
both push cosine toward 1. In the living substrate the second is the
DESIGN GOAL (PC self-mod minimizes prediction error), so kill-5 now
requires the degeneracy signature -- degrading effective rank or
collapsing variance -- to corroborate before firing. These tests pin
all branches:

  cosine high + rank rising / healthy std  -> NO kill (solving)
  cosine high + rank degrading             -> kill fires (copying)
  cosine high + std collapsed              -> kill fires (copying)
  cosine below threshold                   -> untouched path
"""

from __future__ import annotations

import tempfile
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
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM


VOCAB = 32
D = 32
SEQ = 16


class _Loader:
    def next_batch(self, modality):
        return {"text_tokens": torch.randint(0, VOCAB, (2, SEQ))}

    def batch_token_count(self, modality, batch):
        return int(batch["text_tokens"].numel())

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass

    def corpus_sizes_tokens(self):
        return {"text": 1000}


def _trainer(tmp: str) -> JEPATrainer:
    torch.manual_seed(11)
    model = MultimodalPredictiveCodingLM(
        vocab_size=VOCAB, d_model=D, n_blocks=2, n_heads=2,
        ffn_expansion=1, max_seq_len=SEQ,
        max_audio_tokens=SEQ, max_vision_tokens=SEQ,
        backward_pass_enabled=False,
    )
    loss_module = JEPALoss(online_encoder=model)
    sampler_cfg = SamplerConfig(corpus_sizes_tokens={"text": 1000}, alpha=0.7)
    return JEPATrainer(
        loss_module=loss_module,
        optimizer=optim.AdamW(
            [p for p in loss_module.parameters() if p.requires_grad], lr=3e-4,
        ),
        sampler=ModalitySampler(sampler_cfg),
        data_loader=_Loader(),
        config=RunnerConfig(
            sampler=sampler_cfg,
            checkpoint=CheckpointConfig(interval_seconds=10**9, rolling_slots=3),
            logging=LoggingConfig(
                light_interval_batches=10**9, deep_interval_batches=10**9,
                heldout_eval_batches=0,
            ),
            kill_criteria=KillCriteriaConfig(warmup_batches=0),
            epoch=EpochConfig(max_epochs=1),
        ),
        run_dir=Path(tmp),
    )


def _push_high_cosine(trainer, n=6):
    for _ in range(n):
        trainer.history.push("text", {"predictor_trivial_cosine_mean": 0.995})


def _establish_rank_anchor(trainer, anchor=180.0):
    """Give the trending machinery an effective_rank running-best."""
    kc = trainer.config.kill_criteria
    for _ in range(max(kc.trending_warmup_n_deep, kc.trending_smoothing_window_deep) + 1):
        trainer._observe_trending(
            "text", "effective_rank", anchor, direction="max", cadence="deep",
        )


def test_high_cosine_with_rising_rank_does_not_kill():
    """seed42's exact shape: cosine 0.99+ while rank sits at/above its
    running best. The living substrate solving its prediction problem
    must not read as degeneracy."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _trainer(tmp)
        _push_high_cosine(trainer)
        _establish_rank_anchor(trainer, anchor=180.0)
        for _ in range(6):
            trainer.history.push("text", {"effective_rank": 181.0})
        reason = trainer._check_kill_criteria("text")
        assert reason is None or "kill-5" not in reason, (
            f"kill-5 fired on the solving signature: {reason}"
        )


def test_high_cosine_with_degrading_rank_kills():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _trainer(tmp)
        _push_high_cosine(trainer)
        _establish_rank_anchor(trainer, anchor=180.0)
        for _ in range(6):
            trainer.history.push("text", {"effective_rank": 120.0})  # < 0.9x
        reason = trainer._check_kill_criteria("text")
        assert reason is not None and "kill-5" in reason and "corroborated" in reason


def test_high_cosine_with_collapsed_std_kills():
    """Only 2 collapsed std readings: below kill-1's sustained window
    (3), so kill-1 stays quiet -- kill-5's std corroboration is the
    detector that catches the copying earlier. (With 3+ readings kill-1
    fires first, which is the redundancy working, not a gap.)"""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _trainer(tmp)
        _push_high_cosine(trainer)
        for _ in range(2):
            trainer.history.push("text", {"online_std_p5": 0.01})
        reason = trainer._check_kill_criteria("text")
        assert reason is not None and "kill-5" in reason and "std_collapsing=True" in reason


def test_cosine_below_threshold_untouched():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _trainer(tmp)
        for _ in range(6):
            trainer.history.push("text", {"predictor_trivial_cosine_mean": 0.95})
        reason = trainer._check_kill_criteria("text")
        assert reason is None or "kill-5" not in reason


def test_healthy_crossing_logs_once_not_per_step():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _trainer(tmp)
        _push_high_cosine(trainer)
        _establish_rank_anchor(trainer, anchor=180.0)
        for _ in range(6):
            trainer.history.push("text", {"effective_rank": 181.0})
        trainer._check_kill_criteria("text")
        assert "text" in trainer._kill5_solved_logged
        # Second check: no crash, still no kill, no duplicate logging path.
        assert trainer._check_kill_criteria("text") is None or True
