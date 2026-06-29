"""Item #6 (Plan §3) test 4: the catastrophic-forgetting retention gate.

These are the TRAINER-level §3 tests (the M9Trainer's corpus-replay interleave
+ retention gate + rollback). They live here rather than in
``test_catastrophic_forgetting.py`` -- which is the LAYER-level living-weight
consolidation suite (single PredictiveCodingLayer, weight-drift) -- because
they need the full M9Trainer scaffolding and concern a different mechanism.

Covered:
  - the gate FIRES and rolls back when corpus retention exceeds the ceiling
    (the instrument works, and rollback restores last-good core theta);
  - retention HOLDS (no breach) under corpus-replay + low lived LR;
  - lived ``l_pred`` trends DOWN under repeated lived updates (so "retention
    held" can't be an artifact of the model simply not learning anything).
"""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path

import torch
import torch.optim as optim

from luthi.sanctuary_interface import encode_state
from luthi.v2.jepa_loss import JEPALoss
from luthi.v2.jepa_runner import (
    CheckpointConfig, EpochConfig, KillCriteriaConfig, LoggingConfig,
    ModalitySampler, RunnerConfig, SamplerConfig,
)
from luthi.v2.m9.runner import M9Config, M9Trainer
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM


VOCAB, D, SEQ, B = 32, 32, 16, 2


class _TextLoader:
    def __init__(self, seed=0):
        self.gen = torch.Generator().manual_seed(seed)

    def next_batch(self, modality):
        assert modality == "text"
        return {"text_tokens": torch.randint(0, VOCAB, (B, SEQ), generator=self.gen)}

    def batch_token_count(self, modality, batch):
        return int(batch["text_tokens"].numel())

    def corpus_sizes_tokens(self):
        return {"text": 1000}


@contextlib.contextmanager
def _trainer(*, replay_ratio=0.0, lived_lr=1e-4, check_every=2, eps=0.5, seed=7):
    with tempfile.TemporaryDirectory() as tmp:
        torch.manual_seed(seed)
        model = MultimodalPredictiveCodingLM(
            vocab_size=VOCAB, d_model=D, n_blocks=2, n_heads=2,
            ffn_expansion=1, max_seq_len=SEQ,
            max_audio_tokens=SEQ, max_vision_tokens=SEQ,
            backward_pass_enabled=False,
        )
        loss_module = JEPALoss(online_encoder=model)
        loader = _TextLoader(seed=seed)
        sampler_cfg = SamplerConfig(
            corpus_sizes_tokens=loader.corpus_sizes_tokens(), alpha=0.7,
        )
        t = M9Trainer(
            loss_module=loss_module,
            optimizer=optim.AdamW(
                [p for p in loss_module.parameters() if p.requires_grad], lr=1e-3,
            ),
            sampler=ModalitySampler(sampler_cfg), data_loader=loader,
            config=RunnerConfig(
                sampler=sampler_cfg,
                checkpoint=CheckpointConfig(interval_seconds=10**9, rolling_slots=3),
                logging=LoggingConfig(
                    light_interval_batches=10**9, deep_interval_batches=10**9,
                ),
                kill_criteria=KillCriteriaConfig(warmup_batches=10**9),
                epoch=EpochConfig(max_epochs=1, max_batches_per_epoch=10**9),
            ),
            run_dir=Path(tmp),
            m9_config=M9Config(
                mcts_budget_per_cycle=4, lived_lr=lived_lr,
                corpus_replay_ratio=replay_ratio,
                retention_check_every=check_every, retention_floor_eps=eps,
            ),
        )
        try:
            yield t
        finally:
            t.action_log.close()


def _transition(trainer):
    enc = trainer.loss_module.online_encoder
    s_t = encode_state(enc, text_tokens=torch.randint(0, VOCAB, (1, SEQ)), pool=True)
    s_next = encode_state(enc, text_tokens=torch.randint(0, VOCAB, (1, SEQ)), pool=True)
    a_t = torch.randn(1, D)
    context_obs = {"text_tokens": torch.randint(0, VOCAB, (1, SEQ))}
    return s_t, a_t, s_next, context_obs


def _ctx(context_obs):
    return {"context_obs": context_obs, "plan_snapshot": None}


def test_retention_baseline_and_probe_are_passive():
    """corpus_retention() is a passive probe: calling it does not mutate the
    living substrate (frozen + BN-eval + no_grad)."""
    with _trainer() as trainer:
        enc = trainer.loss_module.online_encoder
        living = [
            m.weight.detach().clone()
            for m in enc.modules() if hasattr(m, "pc_rate")
        ]
        r1 = trainer.corpus_retention()
        r2 = trainer.corpus_retention()
        # Deterministic (no self-mod, no BN drift) and non-mutating.
        assert abs(r1 - r2) < 1e-6
        after = [m.weight for m in enc.modules() if hasattr(m, "pc_rate")]
        for before, now in zip(living, after):
            assert torch.equal(before, now), "corpus_retention mutated substrate"


def test_retention_gate_fires_and_rolls_back():
    """Force the gate: with the baseline driven near zero, any real retention
    exceeds the ceiling, so the next check must fire and roll the core back
    to the last-good snapshot."""
    with _trainer(replay_ratio=0.0, lived_lr=1e-2, check_every=1, eps=0.01) as trainer:
        s_t, a_t, s_next, context_obs = _transition(trainer)
        # Snapshot the pre-lived core as last-good, and force a tiny baseline.
        last_good = [p.detach().clone() for p in trainer._core_params()]
        trainer._last_good_core_theta = [p.clone() for p in last_good]
        trainer._retention_baseline = 1e-9

        metrics = trainer.observe_transition(s_t, a_t, s_next, ctx=_ctx(context_obs))

        assert metrics.get("retention_breached") is True, (
            "gate did not fire despite retention >> ceiling"
        )
        assert metrics.get("retention_rolled_back") is True
        # Rollback restored the core to the pre-lived last-good snapshot.
        for p, good in zip(trainer._core_params(), last_good):
            assert torch.equal(p, good), "core theta not restored on rollback"


def test_retention_holds_with_replay_and_low_lr():
    """Healthy path: corpus replay + a low lived LR keep retention under the
    ceiling across a run -- no breach fires."""
    with _trainer(replay_ratio=1.0, lived_lr=1e-4, check_every=2, eps=0.5) as trainer:
        breached_any = False
        for _ in range(8):
            s_t, a_t, s_next, context_obs = _transition(trainer)
            metrics = trainer.observe_transition(s_t, a_t, s_next, ctx=_ctx(context_obs))
            if metrics.get("retention_breached"):
                breached_any = True
            # Replay actually ran (developmental diet active).
            assert metrics.get("corpus_replay_steps", 0) >= 1
        assert not breached_any, (
            "retention gate fired under healthy replay + low-LR conditions"
        )


def test_lived_l_pred_trends_down_on_fixed_transition():
    """Guards 'retention held via no learning': repeated lived updates on a
    FIXED transition must reduce the lived prediction error -- the world
    model is actually learning the transition, not standing still."""
    with _trainer(replay_ratio=0.0, lived_lr=1e-2, check_every=10**9) as trainer:
        s_t, a_t, s_next, context_obs = _transition(trainer)
        losses = []
        for _ in range(15):
            m = trainer.observe_transition(s_t, a_t, s_next, ctx=_ctx(context_obs))
            losses.append(m["lived_l_pred"])
        first = losses[0]
        last_mean = sum(losses[-3:]) / 3.0
        assert last_mean < first, (
            f"lived l_pred did not decrease: first={first:.4f}, "
            f"last3_mean={last_mean:.4f}"
        )
