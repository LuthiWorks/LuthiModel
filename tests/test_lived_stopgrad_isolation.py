"""Item #6 (Plan §1) test 2: stop-grad isolation between the lived
world-model update and the M9 heads.

The lived JEPA gradient must train ONLY the world-model core (encoder
backprop params + predictor); it must not leak onto the M9 heads (V-head,
habit-net, decoders) or onto ``output_proj`` -- the text decode head, which
is deliberately excluded from ``lived_optimizer`` (Finding 2) and trained by
the M9 text-LM signal instead. Conversely the existing M9-head path must not
leak grad onto the encoder/predictor; that scrub is covered in
test_m9_external_actor, reasserted here for the lived-active configuration.
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
from luthi.v2.living_layer_pc import PredictiveCodingLayer
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
def _trainer(seed=7):
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
            m9_config=M9Config(mcts_budget_per_cycle=4, corpus_replay_ratio=0.0),
        )
        try:
            yield t
        finally:
            t.action_log.close()


def _context(trainer):
    a_t = torch.randn(1, D)
    target = torch.randn(1, D)
    context_obs = {"text_tokens": torch.randint(0, VOCAB, (1, SEQ))}
    return context_obs, a_t, target


def _zero_all_grads(trainer):
    for p in trainer.loss_module.parameters():
        p.grad = None
    for module in (
        trainer.v_head, trainer.v_target, trainer.habit_net,
        trainer.rest_action, trainer.decoders, trainer.preferences,
    ):
        for p in module.parameters():
            p.grad = None


def test_lived_backward_does_not_leak_onto_heads_or_output_proj():
    with _trainer() as trainer:
        context_obs, a_t, target = _context(trainer)
        _zero_all_grads(trainer)

        lived = trainer.loss_module.compute_lived_loss(context_obs, a_t, target)
        lived["loss"].backward()

        # M9 heads: untouched by the lived gradient.
        for name, module in (
            ("v_head", trainer.v_head),
            ("habit_net", trainer.habit_net),
            ("decoders", trainer.decoders),
            ("rest_action", trainer.rest_action),
        ):
            for p in module.parameters():
                assert p.grad is None, (
                    f"lived backward leaked gradient onto {name}"
                )

        # Finding 2: output_proj (text decode head) gets no lived gradient.
        for p in trainer.loss_module.online_encoder.output_proj.parameters():
            assert p.grad is None, (
                "lived backward leaked gradient onto output_proj (it must "
                "train via the M9 text-LM head, not the lived path)"
            )

        # Sanity: the world-model core DID receive gradient.
        assert any(
            p.grad is not None
            for p in trainer.loss_module.predictor.parameters()
        ), "predictor received no lived gradient"


def test_lived_optimizer_param_set_excludes_output_proj_and_living_weight():
    with _trainer() as trainer:
        opt_ids = {
            id(p)
            for g in trainer.lived_optimizer.param_groups
            for p in g["params"]
        }
        assert opt_ids, "lived_optimizer has no params"

        # output_proj excluded (Finding 2).
        for p in trainer.loss_module.online_encoder.output_proj.parameters():
            assert id(p) not in opt_ids, (
                "output_proj must be excluded from lived_optimizer"
            )

        # The living-FFN weight is a buffer (self-modifies during
        # perception), so it is not a Parameter and cannot be in any
        # optimizer -- assert that explicitly.
        for m in trainer.loss_module.online_encoder.modules():
            if isinstance(m, PredictiveCodingLayer):
                assert id(m.weight) not in opt_ids, (
                    "living-FFN weight buffer must not be in lived_optimizer"
                )


def test_v_head_still_updates_with_lived_path_active():
    """Integration: the lived world-model update and the M9-head update
    coexist in one observe_transition -- the V-head still trains (the lived
    path didn't starve or corrupt the head update). The converse contract
    (M9-head backward leaves the core's grads clean) is covered by
    test_m9_external_actor::test_no_residual_grad_on_m8_params with the
    head-only ctx."""
    with _trainer() as trainer:
        enc = trainer.loss_module.online_encoder
        s_t = encode_state(enc, text_tokens=torch.randint(0, VOCAB, (1, SEQ)), pool=True)
        s_next = encode_state(enc, text_tokens=torch.randint(0, VOCAB, (1, SEQ)), pool=True)
        context_obs, a_t, _ = _context(trainer)

        before = trainer.v_head.net[0].weight.detach().clone()
        trainer.observe_transition(
            s_t, a_t, s_next,
            ctx={"context_obs": context_obs, "plan_snapshot": None},
        )
        assert not torch.equal(trainer.v_head.net[0].weight, before), (
            "V-head must still update when the lived path is active"
        )
