"""Tests for EMIT_BATCH_1 (LuthiScope/docs/EMIT_BATCH_1.md).

Covers the new training_log.jsonl emissions:

- Top-level: ``grad_norm`` (float), ``lr`` (float), ``nonfinite`` (bool).
- ``substrate{}`` extras: ``set_point_drift``, ``update_ema_mean``,
  ``precision_mean``.
- Deep-cadence ``substrate_blocks`` array.

The two tests the spec calls out:
  1. Logged record carries the new keys with finite values on a tiny
     smoke run.
  2. An injected NaN loss sets ``nonfinite=True`` on the record.

Field names are locked to LuthiScope's auto-keyed UI -- a typo in the
implementation leaves a panel silently empty in the dashboard, so the
test asserts the exact strings.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
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
B = 2


class _TextLoader:
    """Minimal text-only loader satisfying the MultimodalDataLoader Protocol.

    Same shape as the loader in ``tests/m9/test_runner.py`` -- a
    JEPATrainer needs ``next_batch``/``batch_token_count``/``state_dict``/
    ``load_state_dict``/``corpus_sizes_tokens`` to function.
    """

    def __init__(self, vocab: int, batch: int, seq_len: int, seed: int = 0):
        self.vocab = vocab
        self.batch = batch
        self.seq_len = seq_len
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self._cursor = 0

    def next_batch(self, modality: str) -> dict:
        tokens = torch.randint(
            0, self.vocab, (self.batch, self.seq_len), generator=self.gen,
        )
        self._cursor += 1
        return {"text_tokens": tokens}

    def batch_token_count(self, modality: str, batch: dict) -> int:
        return int(batch["text_tokens"].numel())

    def state_dict(self) -> dict:
        return {"cursor": self._cursor, "gen": self.gen.get_state()}

    def load_state_dict(self, state: dict) -> None:
        self._cursor = state["cursor"]
        self.gen.set_state(state["gen"])

    def corpus_sizes_tokens(self) -> dict:
        return {"text": 1000}


def _build_trainer(run_dir: Path, seed: int = 7) -> JEPATrainer:
    torch.manual_seed(seed)
    model = MultimodalPredictiveCodingLM(
        vocab_size=VOCAB, d_model=D, n_blocks=2, n_heads=2,
        ffn_expansion=1, max_seq_len=SEQ,
        max_audio_tokens=SEQ, max_vision_tokens=SEQ,
        backward_pass_enabled=False,
    )
    loss_module = JEPALoss(online_encoder=model)
    optimizer = optim.AdamW(
        [p for p in loss_module.parameters() if p.requires_grad], lr=3e-4,
    )
    loader = _TextLoader(VOCAB, B, SEQ, seed=seed)
    sampler_cfg = SamplerConfig(
        corpus_sizes_tokens=loader.corpus_sizes_tokens(), alpha=0.7,
    )
    sampler = ModalitySampler(sampler_cfg)
    runner_cfg = RunnerConfig(
        sampler=sampler_cfg,
        checkpoint=CheckpointConfig(interval_seconds=10**9, rolling_slots=3),
        logging=LoggingConfig(
            light_interval_batches=10**9, deep_interval_batches=10**9,
        ),
        kill_criteria=KillCriteriaConfig(warmup_batches=10**9),
        epoch=EpochConfig(max_epochs=1, max_batches_per_epoch=10**9),
    )
    return JEPATrainer(
        loss_module=loss_module, optimizer=optimizer, sampler=sampler,
        data_loader=loader, config=runner_cfg, run_dir=run_dir,
    )


def _step_and_log(trainer: JEPATrainer) -> dict:
    """Run one train_step(will_log=True), then one full diagnostics
    firing (light+deep). Returns the persisted record dict (also returned
    by _compute_and_log_diagnostics)."""
    batch = trainer.data_loader.next_batch("text")
    step_out = trainer.train_step("text", batch, will_log=True)
    record = trainer._compute_and_log_diagnostics(
        step_out, light=True, deep=True,
    )
    return record


# ---------------------------------------------------------------------------
# Spec test #1: new keys present + finite on a tiny smoke run.
# ---------------------------------------------------------------------------


class TestEmitBatch1NewKeys:
    """The contract LuthiScope auto-keys on. Spec-locked field names."""

    def test_top_level_grad_norm_present_and_finite(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(Path(tmp))
            record = _step_and_log(trainer)

            assert "grad_norm" in record, (
                "missing top-level grad_norm; LuthiScope's GRADIENT NORM "
                "panel auto-keys on this exact field"
            )
            assert isinstance(record["grad_norm"], float)
            assert torch.isfinite(torch.tensor(record["grad_norm"])).item()
            # Non-trivial: a tiny PC model with random tokens produces
            # non-zero grads on the optimizer-trained params.
            assert record["grad_norm"] > 0.0

    def test_top_level_lr_present_and_finite(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(Path(tmp))
            record = _step_and_log(trainer)

            assert "lr" in record
            assert isinstance(record["lr"], float)
            assert torch.isfinite(torch.tensor(record["lr"])).item()
            # Matches the optimizer's actual rate.
            assert record["lr"] == pytest.approx(3e-4)

    def test_top_level_nonfinite_present_and_bool(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(Path(tmp))
            record = _step_and_log(trainer)

            assert "nonfinite" in record
            assert isinstance(record["nonfinite"], bool)
            # Healthy smoke run -> no NaN/Inf in loss or grads.
            assert record["nonfinite"] is False

    def test_substrate_extras_present_and_finite(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(Path(tmp))
            record = _step_and_log(trainer)

            assert "substrate" in record
            for key in ("set_point_drift", "update_ema_mean", "precision_mean"):
                assert key in record["substrate"], (
                    f"missing substrate.{key}; LuthiScope's "
                    f"DRIFT & PLASTICITY / PRECISION panels auto-key on "
                    f"the exact string"
                )
                val = record["substrate"][key]
                assert isinstance(val, float)
                assert torch.isfinite(torch.tensor(val)).item(), (
                    f"substrate.{key} = {val} (non-finite on smoke run)"
                )

    def test_substrate_legacy_keys_preserved(self):
        """The pre-EMIT_BATCH_1 substrate keys (pred_frob, err_acc) must
        keep emitting -- kill-6 + pilot-set machinery read them."""
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(Path(tmp))
            record = _step_and_log(trainer)

            assert "pred_frob" in record["substrate"]
            assert "err_acc" in record["substrate"]

    def test_substrate_blocks_deep_only(self):
        """substrate_blocks is emitted at deep cadence; absent on
        light-only firings."""
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(Path(tmp))
            batch = trainer.data_loader.next_batch("text")
            step_out = trainer.train_step("text", batch, will_log=True)

            # Light-only firing -> no substrate_blocks.
            record_light = trainer._compute_and_log_diagnostics(
                step_out, light=True, deep=False,
            )
            assert "substrate_blocks" not in record_light

            # Deep firing -> substrate_blocks present.
            batch2 = trainer.data_loader.next_batch("text")
            step_out2 = trainer.train_step("text", batch2, will_log=True)
            record_deep = trainer._compute_and_log_diagnostics(
                step_out2, light=True, deep=True,
            )
            assert "substrate_blocks" in record_deep
            assert isinstance(record_deep["substrate_blocks"], list)

    def test_substrate_blocks_one_entry_per_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(Path(tmp))
            record = _step_and_log(trainer)

            assert len(record["substrate_blocks"]) == 2  # n_blocks
            for entry in record["substrate_blocks"]:
                for key in (
                    "set_point_drift", "update_ema_mean", "precision_mean",
                    "prediction_norm", "error_acc_mean",
                ):
                    assert key in entry, (
                        f"missing substrate_blocks[*].{key}"
                    )
                    val = entry[key]
                    assert isinstance(val, float)
                    assert torch.isfinite(torch.tensor(val)).item()

    def test_record_persisted_to_jsonl(self):
        """The record actually lands in training_log.jsonl, not just the
        in-memory return value. LuthiScope reads from disk."""
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(Path(tmp))
            record = _step_and_log(trainer)

            log_path = trainer.metric_log_path
            assert log_path.exists()
            with open(log_path, "r", encoding="utf-8") as f:
                lines = [line for line in f if line.strip()]
            assert len(lines) == 1
            on_disk = json.loads(lines[0])
            # New top-level keys made it through json.dumps.
            assert on_disk["grad_norm"] == record["grad_norm"]
            assert on_disk["lr"] == record["lr"]
            assert on_disk["nonfinite"] == record["nonfinite"]
            assert "substrate_blocks" in on_disk


# ---------------------------------------------------------------------------
# Spec test #2: injected NaN sets nonfinite=True.
# ---------------------------------------------------------------------------


class TestInjectedNaNSetsNonfinite:
    """The fail-loud guard. nonfinite tracks both NaN loss and NaN grads --
    either triggers the flag. This batch only LOGS nonfinite; the kill
    on sustained nonfinite is a separate follow-up (per spec, 2026-06-20)."""

    def test_nan_loss_sets_nonfinite_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(Path(tmp))

            # Patch the loss module to return NaN. compute_modality_loss
            # is what train_step calls; replacing it surfaces NaN at the
            # loss level (which then propagates through .backward() so
            # grads are also NaN -- both halves of the nonfinite check
            # exercised).
            original = trainer.loss_module.compute_modality_loss

            def _nan_loss(modality, batch, **kwargs):
                result = original(modality, batch, **kwargs)
                # Replace with a NaN loss that's still on the same graph
                # so backward() runs and populates .grad attrs (which
                # then carry NaN through the chain).
                result["loss"] = result["loss"] * float("nan")
                return result

            trainer.loss_module.compute_modality_loss = _nan_loss

            batch = trainer.data_loader.next_batch("text")
            trainer.train_step("text", batch, will_log=True)

            assert trainer._last_nonfinite is True, (
                "expected _last_nonfinite=True after NaN loss injection; "
                "the fail-loud guard didn't fire"
            )

    def test_nonfinite_surfaces_in_record(self):
        """The flag set in train_step makes it through to the record."""
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(Path(tmp))

            original = trainer.loss_module.compute_modality_loss

            def _nan_loss(modality, batch, **kwargs):
                result = original(modality, batch, **kwargs)
                result["loss"] = result["loss"] * float("nan")
                return result

            trainer.loss_module.compute_modality_loss = _nan_loss

            batch = trainer.data_loader.next_batch("text")
            step_out = trainer.train_step("text", batch, will_log=True)
            record = trainer._compute_and_log_diagnostics(
                step_out, light=True, deep=False,
            )

            assert record["nonfinite"] is True

    def test_healthy_step_keeps_nonfinite_false(self):
        """Sanity counterpoint to the NaN injection: clean step -> False."""
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(Path(tmp))
            batch = trainer.data_loader.next_batch("text")
            trainer.train_step("text", batch, will_log=True)
            assert trainer._last_nonfinite is False


# ---------------------------------------------------------------------------
# will_log gating: hot-path overhead protection.
# ---------------------------------------------------------------------------


class TestWillLogGating:
    """Grads exist only between backward() and step() -- compute has to
    live in train_step. Looping all params on every non-logging step is
    overhead, so the spec gates the compute on will_log."""

    def test_no_grad_norm_compute_when_will_log_false(self):
        """When will_log=False, _last_grad_norm stays at its initial NaN
        sentinel (no overwrite)."""
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(Path(tmp))
            # Fresh trainer: sentinel NaN.
            import math
            assert math.isnan(trainer._last_grad_norm)

            batch = trainer.data_loader.next_batch("text")
            trainer.train_step("text", batch, will_log=False)

            # No log -> sentinel preserved.
            assert math.isnan(trainer._last_grad_norm)
            assert trainer._last_nonfinite is False

    def test_grad_norm_computed_when_will_log_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(Path(tmp))
            batch = trainer.data_loader.next_batch("text")
            trainer.train_step("text", batch, will_log=True)

            assert trainer._last_grad_norm > 0.0  # populated
            assert trainer._last_nonfinite is False

    def test_grad_norm_scoped_to_optimizer_params_only(self):
        """grad_norm covers optimizer.param_groups (backprop-trained):
        encoders, attention, embeddings, predictor, projection heads.
        Living-weight buffers are NOT folded in -- they update via the
        PC mechanism, not autograd. update_ema_mean is the separate
        substrate-change signal."""
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(Path(tmp))
            batch = trainer.data_loader.next_batch("text")
            trainer.train_step("text", batch, will_log=True)

            # Compute the expected grad_norm manually from optimizer params.
            total_sq = 0.0
            for group in trainer.optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        total_sq += float(p.grad.detach().norm().item()) ** 2
            expected = total_sq ** 0.5
            assert trainer._last_grad_norm == pytest.approx(expected, rel=1e-5)

            # And a sanity check that this differs from any naïve
            # all-buffers norm -- the substrate has buffers (precision,
            # error_acc, etc.) that are NOT in optimizer.param_groups
            # but DO have non-zero norms. Including them would inflate
            # grad_norm. We don't enforce a specific value here, just
            # confirm the scope is restricted as intended.
            living_buffer_total = 0.0
            for block in trainer.loss_module.online_encoder.blocks:
                for _, buf in block.living_ffn.named_buffers():
                    # .norm() requires float/complex; episode_count and
                    # similar integer buffers don't carry a meaningful
                    # norm anyway. Cast or skip.
                    if not buf.is_floating_point():
                        continue
                    living_buffer_total += float(buf.norm().item()) ** 2
            assert living_buffer_total > 0, (
                "test precondition: living buffers should have non-zero "
                "norm; otherwise scope-test is degenerate"
            )
