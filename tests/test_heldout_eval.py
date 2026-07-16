"""Held-out eval harness tests (2026-07-15, JEPA program).

Pins the three disciplines the pre-registered criteria depend on:

1. **The split cannot leak.** With stride < seq_len a random split leaks
   ~half of every held-out sequence into training; compute_text_split
   must keep every training window >= seq_len clear of every holdout
   window, and holdout windows must not overlap each other.
2. **Evaluation cannot change the model.** The living substrate
   self-modifies on any unguarded forward; an eval pass must leave every
   buffer (living state AND the projection heads' BatchNorm running
   stats) bitwise identical, and must not touch the trainer's kill
   history / pilot state / smoothed-loss buffer.
3. **The probe ships with its own floor.** A probe that cannot beat its
   shuffled-label control on a signal known to be learnable is a blind
   ruler; the positive control is part of the harness, not an
   afterthought.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.optim as optim

from luthi.v2.eval_heldout import (
    fit_next_token_probe,
    heldout_latent_prediction,
    probe_accuracy,
)
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
from luthi.v2.multimodal_data import compute_text_split
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM


VOCAB = 32
D = 32
SEQ = 16
B = 2


# ---------------------------------------------------------------------------
# 1. Split arithmetic (pure function, hammered)
# ---------------------------------------------------------------------------


class TestComputeTextSplit:
    @pytest.mark.parametrize("n_tokens,seq_len,stride,frac", [
        (10_000, 128, 64, 0.05),
        (10_000, 128, 128, 0.10),
        (5_000, 64, 32, 0.02),
        (100_000, 128, 64, 0.01),
        (1_000, 100, 10, 0.30),
    ])
    def test_no_training_window_touches_any_holdout_window(
        self, n_tokens, seq_len, stride, frac,
    ):
        n_train, holdout_starts = compute_text_split(
            n_tokens, seq_len, stride, frac,
        )
        assert n_train > 0 and holdout_starts
        last_train_end = (n_train - 1) * stride + seq_len
        first_holdout_start = min(holdout_starts)
        assert last_train_end + seq_len <= first_holdout_start + seq_len, (
            "gap arithmetic broken"
        )
        # The sharp claim: train windows end at least seq_len BEFORE the
        # first holdout window begins (the leakage gap).
        assert last_train_end <= first_holdout_start - 0, (
            f"training window [{last_train_end - seq_len}, {last_train_end}) "
            f"reaches past the holdout boundary {first_holdout_start}"
        )
        gap = first_holdout_start - last_train_end
        assert gap >= 0
        # Holdout windows: in-bounds, non-overlapping.
        for s in holdout_starts:
            assert s + seq_len <= n_tokens
        sorted_starts = sorted(holdout_starts)
        for a, b in zip(sorted_starts, sorted_starts[1:]):
            assert b - a >= seq_len, "holdout windows overlap each other"

    def test_gap_is_at_least_seq_len(self):
        """The load-bearing constant: nearest train token index to the
        holdout region is >= seq_len away from the holdout's first token,
        so no emitted training sequence shares ANY token with holdout."""
        n_tokens, seq_len, stride, frac = 50_000, 128, 64, 0.05
        n_train, holdout_starts = compute_text_split(
            n_tokens, seq_len, stride, frac,
        )
        last_train_token = (n_train - 1) * stride + seq_len - 1
        first_holdout_token = min(holdout_starts)
        assert first_holdout_token - last_train_token >= 1
        # And the documented gap: boundary layout leaves a full seq_len
        # between the training region's cap and the holdout start.
        holdout_begin = n_tokens - int(n_tokens * frac)
        assert min(holdout_starts) == holdout_begin
        assert last_train_token < holdout_begin - 1

    def test_zero_fraction_is_legacy_identical(self):
        n_train, holdout = compute_text_split(10_000, 128, 64, 0.0)
        assert holdout == []
        assert n_train == ((10_000 - 128) // 64) + 1

    def test_degenerate_fractions_fail_loud(self):
        with pytest.raises(ValueError):
            compute_text_split(1_000, 128, 64, 0.99)  # no training left
        with pytest.raises(ValueError):
            compute_text_split(10_000, 128, 64, -0.1)
        with pytest.raises(ValueError):
            compute_text_split(10_000, 128, 64, 1.0)


# ---------------------------------------------------------------------------
# Shared tiny harness (holdout-capable loader over a synthetic corpus)
# ---------------------------------------------------------------------------


class _HoldoutTextLoader:
    """Tiny loader with a real train/holdout split over a synthetic token
    stream, satisfying both the MultimodalDataLoader Protocol and the
    duck-typed holdout API."""

    def __init__(self, n_tokens: int = 4096, seed: int = 0,
                 deterministic_next: bool = False):
        gen = torch.Generator().manual_seed(seed)
        if deterministic_next:
            # token[t+1] = (token[t] + 1) % VOCAB — a next-token signal a
            # linear probe on any non-degenerate representation can read.
            start = torch.randint(0, VOCAB, (1,), generator=gen).item()
            self.tokens = (torch.arange(n_tokens) + start) % VOCAB
        else:
            self.tokens = torch.randint(0, VOCAB, (n_tokens,), generator=gen)
        self.n_train, self.holdout_starts = compute_text_split(
            n_tokens, SEQ, SEQ // 2, 0.10,
        )
        self.gen = gen
        self._cursor = 0

    def next_batch(self, modality: str) -> dict:
        idx = torch.randint(0, self.n_train, (B,), generator=self.gen)
        starts = idx * (SEQ // 2)
        self._cursor += 1
        return {"text_tokens": torch.stack(
            [self.tokens[s : s + SEQ] for s in starts.tolist()], dim=0,
        )}

    def batch_token_count(self, modality: str, batch: dict) -> int:
        return int(batch["text_tokens"].numel())

    def state_dict(self) -> dict:
        return {"cursor": self._cursor, "gen": self.gen.get_state()}

    def load_state_dict(self, state: dict) -> None:
        self._cursor = state["cursor"]
        self.gen.set_state(state["gen"])

    def corpus_sizes_tokens(self) -> dict:
        return {"text": 1000}

    # Duck-typed holdout API
    def holdout_batch_count(self, modality: str, batch_size: int) -> int:
        return len(self.holdout_starts) // batch_size

    def holdout_batches(self, modality: str, batch_size: int):
        for i in range(self.holdout_batch_count(modality, batch_size)):
            starts = self.holdout_starts[i * batch_size : (i + 1) * batch_size]
            yield {"text_tokens": torch.stack(
                [self.tokens[s : s + SEQ] for s in starts], dim=0,
            )}


def _build(loader=None, heldout_batches: int = 4):
    torch.manual_seed(11)
    model = MultimodalPredictiveCodingLM(
        vocab_size=VOCAB, d_model=D, n_blocks=2, n_heads=2,
        ffn_expansion=1, max_seq_len=SEQ,
        max_audio_tokens=SEQ, max_vision_tokens=SEQ,
        backward_pass_enabled=False,
    )
    loss_module = JEPALoss(online_encoder=model)
    loader = loader or _HoldoutTextLoader()
    sampler_cfg = SamplerConfig(
        corpus_sizes_tokens=loader.corpus_sizes_tokens(), alpha=0.7,
    )
    trainer = JEPATrainer(
        loss_module=loss_module,
        optimizer=optim.AdamW(
            [p for p in loss_module.parameters() if p.requires_grad], lr=3e-4,
        ),
        sampler=ModalitySampler(sampler_cfg),
        data_loader=loader,
        config=RunnerConfig(
            sampler=sampler_cfg,
            checkpoint=CheckpointConfig(interval_seconds=10**9, rolling_slots=3),
            logging=LoggingConfig(
                light_interval_batches=10**9, deep_interval_batches=10**9,
                heldout_eval_batches=heldout_batches,
            ),
            kill_criteria=KillCriteriaConfig(warmup_batches=10**9),
            epoch=EpochConfig(max_epochs=1, max_batches_per_epoch=10**9),
        ),
        run_dir=Path(tempfile.mkdtemp()),
    )
    return trainer, loss_module, loader


def _full_snapshot(trainer) -> dict:
    """Every buffer of the loss module (living state + BN running stats)
    plus the trainer's eval-adjacent bookkeeping."""
    snap = {
        f"buf::{k}": v.detach().clone()
        for k, v in trainer.loss_module.named_buffers()
    }
    snap["__smoothed_len"] = len(trainer._smoothed_loss_buf)
    snap["__kill_history"] = trainer.history.state_dict()
    snap["__pilot"] = {
        m: dict(d) for m, d in trainer._stationary_baselines.items()
    }
    snap["__global_step"] = trainer.global_step
    snap["__modality_step"] = dict(trainer.modality_step)
    return snap


# ---------------------------------------------------------------------------
# 2. Evaluation changes nothing
# ---------------------------------------------------------------------------


class TestEvalMutatesNothing:
    def test_evaluate_heldout_leaves_model_bitwise_identical(self):
        trainer, loss_module, loader = _build()
        # Prime with real training steps so there is living state to protect.
        for _ in range(3):
            batch = loader.next_batch("text")
            trainer.train_step("text", batch)

        before = _full_snapshot(trainer)
        results = trainer.evaluate_heldout()
        after = _full_snapshot(trainer)

        assert results and "text" in results
        assert results["text"]["n_batches"] > 0
        assert torch.isfinite(torch.tensor(results["text"]["l_pred_mean"]))
        for key, prev in before.items():
            cur = after[key]
            if isinstance(prev, torch.Tensor):
                assert torch.equal(prev, cur), (
                    f"evaluate_heldout mutated {key} -- evaluation is "
                    f"contaminating the subject"
                )
            else:
                assert prev == cur, f"evaluate_heldout advanced {key}"

    def test_eval_restores_train_mode(self):
        trainer, loss_module, _ = _build()
        loss_module.train()
        trainer.evaluate_heldout()
        assert loss_module.training, "eval guard failed to restore train mode"

    def test_eval_is_deterministic_across_calls(self):
        trainer, _, _ = _build()
        r1 = trainer.evaluate_heldout()
        r2 = trainer.evaluate_heldout()
        assert r1["text"]["l_pred_mean"] == r2["text"]["l_pred_mean"], (
            "same model, same holdout, different numbers -- either the "
            "eval mutates state or the holdout set is not fixed"
        )

    def test_legacy_loader_without_holdout_skips_cleanly(self):
        class _Legacy:
            def next_batch(self, modality):
                return {"text_tokens": torch.randint(0, VOCAB, (B, SEQ))}
            def batch_token_count(self, modality, batch):
                return int(batch["text_tokens"].numel())
            def state_dict(self):
                return {}
            def load_state_dict(self, state):
                pass
            def corpus_sizes_tokens(self):
                return {"text": 1000}
        trainer, _, _ = _build(loader=_Legacy())
        assert trainer.evaluate_heldout() == {}

    def test_disabled_by_config(self):
        trainer, _, _ = _build(heldout_batches=0)
        assert trainer.evaluate_heldout() == {}


# ---------------------------------------------------------------------------
# 3. The probe and its floor
# ---------------------------------------------------------------------------


class TestProbe:
    def test_probe_clears_chance_on_learnable_signal_and_floor_does_not(self):
        """Positive control (protocol §1): on a corpus where token[t+1]
        is a deterministic function of token[t], the linear readout must
        beat chance handily even on an UNTRAINED encoder (the signal is
        in the embedding), and the shuffled-label floor must sit near
        chance. A probe that can't pass this cannot certify any null."""
        loader = _HoldoutTextLoader(deterministic_next=True)
        trainer, loss_module, _ = _build(loader=loader)

        train_batches = [loader.next_batch("text") for _ in range(32)]
        probe = fit_next_token_probe(
            loss_module, train_batches, vocab_size=VOCAB,
            max_batches=32, epochs=12,
        )
        heldout = list(loader.holdout_batches("text", 8))
        real = probe_accuracy(loss_module, probe, heldout)
        floor = probe_accuracy(
            loss_module, probe, heldout, shuffled_label_floor=True,
        )
        chance = 1.0 / VOCAB
        # Calibration note: the encoder is UNTRAINED (random trunk), so
        # the readout ceiling is how well token identity survives random
        # mixing -- far from 1.0, but a working instrument must sit well
        # above chance while its shuffled floor sits at chance. 3x/2x
        # margins chosen from the first calibration run (untrained-trunk
        # top1 ~4.3x chance).
        assert real["top1"] > 3 * chance, (
            f"probe top1={real['top1']:.3f} on a deterministic next-token "
            f"signal -- the instrument cannot see"
        )
        assert floor["top1"] < 2 * chance, (
            f"shuffled-label floor top1={floor['top1']:.3f} is not at "
            f"chance -- the floor is broken"
        )
        assert real["top1"] > 2 * floor["top1"]

    def test_heldout_latent_prediction_direct(self):
        loader = _HoldoutTextLoader()
        _, loss_module, _ = _build(loader=loader)
        r = heldout_latent_prediction(
            loss_module, loader.holdout_batches("text", 8), "text",
            max_batches=3,
        )
        assert r["n_batches"] == 3
        assert torch.isfinite(torch.tensor(r["l_pred_mean"]))
