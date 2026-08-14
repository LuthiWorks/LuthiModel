"""Kill-6 corroboration amendment (2026-08-14 audit, item B5).

Kill-6 fired on a SINGLE substrate gauge (pred_frob or err_acc) measured
against a frozen running best. It killed the 768x8 family's seed 46 at
step 9,100 while every geometric measure said healthy AND improving --
all blocks effective rank 195-280, top_dir_share 0.018-0.034, both
improving through the run's final 20%. The same criterion was measured to
kill 10/10 healthy runs on 2026-07-16 and was merely loosened rather than
corroborated.

`err_acc` is the sharper case: it RISES WITH VARIETY, so judging it
against a running MINIMUM makes eventual firing structural -- a run that
sees more of its corpus is guaranteed to trip it given enough steps.

This applies the disambiguator Brian ruled for kill-5 on 2026-07-17: a
substrate gauge crossing its band is ambiguous between degeneracy and the
substrate doing its job, so fire only when the geometry corroborates.

  gauge breached + rank healthy / std healthy -> NO kill (working)
  gauge breached + rank degrading             -> kill fires
  gauge breached + std collapsed              -> kill fires
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


def _establish_anchor(trainer, metric: str, value: float, direction: str):
    kc = trainer.config.kill_criteria
    for _ in range(max(kc.trending_warmup_n, kc.trending_smoothing_window) + 1):
        trainer._observe_trending(
            "text", metric, value, direction=direction, cadence="light",
        )


def _breach_err_acc(trainer):
    """err_acc anchored low, then sustained high -- seed 46's shape."""
    _establish_anchor(trainer, "err_acc", 0.001, direction="min")
    kc = trainer.config.kill_criteria
    for _ in range(kc.substrate_health_window + 1):
        trainer.history.push("text", {"err_acc": 0.010})   # 10x the anchor


def _healthy_geometry(trainer, rank=200.0):
    """Rank at/above its running best, variance healthy."""
    kc = trainer.config.kill_criteria
    for _ in range(max(kc.trending_warmup_n_deep, kc.trending_smoothing_window_deep) + 1):
        trainer._observe_trending(
            "text", "effective_rank", rank, direction="max", cadence="deep",
        )
    for _ in range(6):
        trainer.history.push("text", {"effective_rank": rank + 1.0})
        trainer.history.push("text", {"online_std_p5": 1.0})


def test_err_acc_breach_with_healthy_geometry_does_not_kill():
    """Seed 46's exact shape: the substrate gauge crosses while rank and
    variance are healthy and improving. Must not kill."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _trainer(tmp)
        _healthy_geometry(trainer, rank=200.0)
        _breach_err_acc(trainer)
        reason = trainer._check_kill_criteria("text")
        assert reason is None or "kill-6" not in reason, (
            f"kill-6 fired on a healthy substrate: {reason}"
        )


def test_err_acc_breach_with_degrading_rank_kills():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _trainer(tmp)
        kc = trainer.config.kill_criteria
        for _ in range(max(kc.trending_warmup_n_deep,
                           kc.trending_smoothing_window_deep) + 1):
            trainer._observe_trending(
                "text", "effective_rank", 200.0, direction="max", cadence="deep",
            )
        for _ in range(6):
            trainer.history.push("text", {"effective_rank": 120.0})  # < 0.9x
            trainer.history.push("text", {"online_std_p5": 1.0})
        _breach_err_acc(trainer)
        reason = trainer._check_kill_criteria("text")
        assert reason is not None and "kill-6" in reason and "corroborated" in reason


def test_pred_frob_breach_with_healthy_geometry_does_not_kill():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _trainer(tmp)
        _healthy_geometry(trainer, rank=200.0)
        _establish_anchor(trainer, "pred_frob", 4.0, direction="max")
        kc = trainer.config.kill_criteria
        for _ in range(kc.substrate_health_window + 1):
            trainer.history.push("text", {"pred_frob": 1.0})  # well below 0.75x
        reason = trainer._check_kill_criteria("text")
        assert reason is None or "kill-6" not in reason, (
            f"kill-6 fired on a healthy substrate: {reason}"
        )


def test_pred_frob_breach_with_degrading_rank_kills():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _trainer(tmp)
        kc = trainer.config.kill_criteria
        for _ in range(max(kc.trending_warmup_n_deep,
                           kc.trending_smoothing_window_deep) + 1):
            trainer._observe_trending(
                "text", "effective_rank", 200.0, direction="max", cadence="deep",
            )
        for _ in range(6):
            trainer.history.push("text", {"effective_rank": 120.0})
            trainer.history.push("text", {"online_std_p5": 1.0})
        _establish_anchor(trainer, "pred_frob", 4.0, direction="max")
        for _ in range(kc.substrate_health_window + 1):
            trainer.history.push("text", {"pred_frob": 1.0})
        reason = trainer._check_kill_criteria("text")
        assert reason is not None and "kill-6" in reason and "corroborated" in reason
