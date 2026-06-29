"""Item #6 (Plan §1) test 1: ``observe_transition`` with ``context_obs`` in
ctx runs the lived JEPA update, moving the world-model core from realized
experience.

The lived update re-encodes the raw context under frozen plasticity, scores
the predictor's pooled forecast against the realized pooled ``s_next``, and
steps ``lived_optimizer`` over the encoder's backprop params + predictor.
These tests isolate the lived gradient (corpus_replay_ratio=0 so nothing else
moves the core) and confirm the loss is finite and target-sensitive.
"""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path

import pytest
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
    def __init__(self, seed: int = 0):
        self.gen = torch.Generator().manual_seed(seed)

    def next_batch(self, modality: str) -> dict:
        assert modality == "text"
        return {
            "text_tokens": torch.randint(
                0, VOCAB, (B, SEQ), generator=self.gen,
            )
        }

    def batch_token_count(self, modality, batch):
        return int(batch["text_tokens"].numel())

    def corpus_sizes_tokens(self):
        return {"text": 1000}


def _build_trainer(run_dir: Path, *, replay_ratio: float = 0.0, seed: int = 7):
    torch.manual_seed(seed)
    model = MultimodalPredictiveCodingLM(
        vocab_size=VOCAB, d_model=D, n_blocks=2, n_heads=2,
        ffn_expansion=1, max_seq_len=SEQ,
        max_audio_tokens=SEQ, max_vision_tokens=SEQ,
        backward_pass_enabled=False,
    )
    loss_module = JEPALoss(online_encoder=model)
    optimizer = optim.AdamW(
        [p for p in loss_module.parameters() if p.requires_grad], lr=1e-3,
    )
    loader = _TextLoader(seed=seed)
    sampler_cfg = SamplerConfig(
        corpus_sizes_tokens=loader.corpus_sizes_tokens(), alpha=0.7,
    )
    cfg = RunnerConfig(
        sampler=sampler_cfg,
        checkpoint=CheckpointConfig(interval_seconds=10**9, rolling_slots=3),
        logging=LoggingConfig(
            light_interval_batches=10**9, deep_interval_batches=10**9,
        ),
        kill_criteria=KillCriteriaConfig(warmup_batches=10**9),
        epoch=EpochConfig(max_epochs=1, max_batches_per_epoch=10**9),
    )
    return M9Trainer(
        loss_module=loss_module, optimizer=optimizer,
        sampler=ModalitySampler(sampler_cfg), data_loader=loader,
        config=cfg, run_dir=run_dir,
        # Isolate the lived gradient: lift the lived LR so the move is
        # measurable in one step, and turn replay OFF so ONLY the lived
        # step touches the core.
        m9_config=M9Config(
            mcts_budget_per_cycle=4, lived_lr=1e-2,
            corpus_replay_ratio=replay_ratio,
        ),
    )


@contextlib.contextmanager
def _trainer(*, replay_ratio: float = 0.0, seed: int = 7):
    """Build a trainer in a temp run_dir and guarantee the action-log file
    handle is closed before the temp dir is torn down (Windows rmtree fails
    on an open handle)."""
    with tempfile.TemporaryDirectory() as tmp:
        t = _build_trainer(Path(tmp), replay_ratio=replay_ratio, seed=seed)
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


def _first_attention_weight(trainer):
    # A backprop Parameter on the text path that the lived gradient reaches.
    return trainer.loss_module.online_encoder.blocks[0].attention.q_proj.weight


def test_lived_update_moves_encoder_and_predictor():
    with _trainer() as trainer:
        s_t, a_t, s_next, context_obs = _transition(trainer)

        enc_before = _first_attention_weight(trainer).detach().clone()
        pred_before = trainer.loss_module.predictor.output_norm.weight.detach().clone()

        metrics = trainer.observe_transition(
            s_t, a_t, s_next,
            ctx={"context_obs": context_obs, "plan_snapshot": None},
        )

        # The lived metrics are present and finite.
        assert "lived_l_pred" in metrics
        assert metrics["lived_l_pred"] == metrics["lived_l_pred"]  # not NaN
        assert metrics["lived_l_pred"] >= 0.0
        assert metrics["lived_delta_theta_norm"] > 0.0

        # The encoder's backprop params AND the predictor moved -- purely
        # from the lived gradient (replay is off).
        assert not torch.equal(
            _first_attention_weight(trainer), enc_before
        ), "encoder backprop param did not move on the lived update"
        assert not torch.equal(
            trainer.loss_module.predictor.output_norm.weight, pred_before
        ), "predictor did not move on the lived update"


def test_lived_update_advances_theta_version():
    with _trainer() as trainer:
        s_t, a_t, s_next, context_obs = _transition(trainer)
        before = trainer.staleness.theta_version
        trainer.observe_transition(
            s_t, a_t, s_next,
            ctx={"context_obs": context_obs, "plan_snapshot": None},
        )
        assert trainer.staleness.theta_version > before, (
            "lived update must tick theta_version via observe_drift"
        )


def test_no_context_obs_leaves_core_untouched():
    """Back-compat: without context_obs, observe_transition is head-only --
    the world-model core does not move and no lived metrics appear."""
    with _trainer() as trainer:
        s_t, a_t, s_next, _ = _transition(trainer)
        enc_before = _first_attention_weight(trainer).detach().clone()
        theta_before = trainer.staleness.theta_version

        metrics = trainer.observe_transition(s_t, a_t, s_next, ctx={})

        assert "lived_l_pred" not in metrics
        assert torch.equal(_first_attention_weight(trainer), enc_before), (
            "core moved without context_obs -- the lived gate leaked"
        )
        assert trainer.staleness.theta_version == theta_before


def test_empty_context_obs_fails_loud():
    """An empty context_obs must raise before encode(**{}) crashes
    cryptically -- the lived path is opt-in, so a committed context must be
    encodable (Window A audit)."""
    with _trainer() as trainer:
        _, a_t, _, _ = _transition(trainer)
        with pytest.raises(ValueError, match="non-empty context_obs"):
            trainer.loss_module.compute_lived_loss({}, a_t, torch.randn(1, D))


def test_gradient_checkpointing_encoder_is_refused():
    """The frozen re-encode is unsafe under encoder gradient checkpointing
    (replay runs the normal branch in backward); the lived loss must refuse
    it rather than silently corrupt living state (Window A audit)."""
    with _trainer() as trainer:
        _, a_t, _, context_obs = _transition(trainer)
        # The smoke encoder has no such flag; simulate a checkpointing
        # encoder by setting the conventional attribute.
        trainer.loss_module.online_encoder.gradient_checkpointing = True
        with pytest.raises(RuntimeError, match="gradient checkpointing"):
            trainer.loss_module.compute_lived_loss(
                context_obs, a_t, torch.randn(1, D),
            )


def test_lived_loss_is_target_sensitive():
    """compute_lived_loss on a FIXED model: a different realized next state
    gives a different prediction loss; the same one is reproducible."""
    with _trainer() as trainer:
        trainer.loss_module.eval()  # determinism: no BN/dropout drift
        _, a_t, _, context_obs = _transition(trainer)
        target_a = torch.randn(1, D)
        target_b = torch.randn(1, D) * 5.0

        la = trainer.loss_module.compute_lived_loss(context_obs, a_t, target_a)
        la2 = trainer.loss_module.compute_lived_loss(context_obs, a_t, target_a)
        lb = trainer.loss_module.compute_lived_loss(context_obs, a_t, target_b)

        # Same target -> same loss (the re-encode is deterministic under
        # freeze: no self-mod, so the model state is identical across calls).
        assert abs(float(la["l_pred"]) - float(la2["l_pred"])) < 1e-5
        # Different target -> different loss.
        assert abs(float(la["l_pred"]) - float(lb["l_pred"])) > 1e-5
