"""Light-cadence per-block activation scalars (2026-10-06).

Stage 1 of the activation-monitoring plan. Why these tests exist: the
deep-cadence per-block tape (substrate_blocks) carries SVD gauges that
cost too much to run at light cadence, so individual blocks were
invisible between deep firings -- a block could die (or saturate) with
no light-cadence trace. These scalars (mean/std/rms, p50/p99 of |h|,
dead_frac, sat_frac) are O(elements) reductions computed eagerly in
encode(), so they ride the ~10x light cadence without retaining any
tensors. Field names are spec-locked to the stream contract: LuthiScope
will auto-key a blocks-x-time heatmap on them, so a typo here leaves a
panel silently empty.
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
from luthi.v2.multimodal_model_pc import (
    _block_activation_scalars,
    _DEAD_ACT_EPS,
    _SAT_ACT_THRESH,
    MultimodalPredictiveCodingLM,
)


B, T, D = 4, 8, 16

_SCALAR_KEYS = (
    "act_mean", "act_std", "act_rms",
    "act_p50", "act_p99", "dead_frac", "sat_frac",
)


def _rich(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, D, generator=g)


class TestBlockActivationScalars:
    def test_keys_and_plain_floats(self):
        s = _block_activation_scalars(_rich())
        assert tuple(s.keys()) == _SCALAR_KEYS
        for k, v in s.items():
            assert isinstance(v, float), f"{k} is {type(v)}, not float"
        # The whole point of eager scalars: they must survive JSON.
        json.dumps(s)

    def test_values_match_torch(self):
        h = _rich(1)
        s = _block_activation_scalars(h)
        assert s["act_mean"] == pytest.approx(h.mean().item())
        assert s["act_std"] == pytest.approx(h.std(unbiased=False).item())
        assert s["act_rms"] == pytest.approx(h.pow(2).mean().sqrt().item())
        af = h.abs().flatten()
        assert s["act_p50"] == pytest.approx(af.quantile(0.50).item())
        assert s["act_p99"] == pytest.approx(af.quantile(0.99).item())

    def test_dead_tensor_reads_dead(self):
        s = _block_activation_scalars(torch.zeros(B, T, D))
        assert s["dead_frac"] == pytest.approx(1.0)
        assert s["sat_frac"] == pytest.approx(0.0)
        assert s["act_std"] == pytest.approx(0.0)
        assert s["act_rms"] == pytest.approx(0.0)

    def test_saturated_tensor_reads_saturated(self):
        s = _block_activation_scalars(torch.full((B, T, D), 10.0))
        assert s["sat_frac"] == pytest.approx(1.0)
        assert s["dead_frac"] == pytest.approx(0.0)

    def test_healthy_tensor_reads_neither(self):
        s = _block_activation_scalars(_rich(2))
        assert s["dead_frac"] == pytest.approx(0.0)
        assert s["sat_frac"] == pytest.approx(0.0)

    def test_thresholds_are_pinned(self):
        """The dead/saturation thresholds are arbitrary-but-fixed
        diagnostic constants. If someone changes them, the time series
        breaks silently -- pin them here so the change is deliberate."""
        assert _DEAD_ACT_EPS == 1e-5
        assert _SAT_ACT_THRESH == 5.0

    def test_does_not_hold_graph(self):
        h = _rich(3).requires_grad_(True)
        s = _block_activation_scalars(h)
        assert h.requires_grad  # input untouched...
        assert h.grad is None  # ...and the helper detached: nothing retained
        assert all(isinstance(v, float) for v in s.values())


# ---------------------------------------------------------------------------
# Smoke: wiring through loss -> runner -> record at light cadence.
# ---------------------------------------------------------------------------


class _TextLoader:
    """Minimal text-only loader (same shape as the EMIT_BATCH_1 one)."""

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
        vocab_size=32, d_model=32, n_blocks=2, n_heads=2,
        ffn_expansion=1, max_seq_len=16,
        max_audio_tokens=16, max_vision_tokens=16,
        backward_pass_enabled=False,
    )
    loss_module = JEPALoss(online_encoder=model)
    optimizer = optim.AdamW(
        [p for p in loss_module.parameters() if p.requires_grad], lr=3e-4,
    )
    loader = _TextLoader(32, 2, 16, seed=seed)
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


class TestLightBlocksEmission:
    def test_emitted_on_light_cadence_one_entry_per_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(Path(tmp))
            batch = trainer.data_loader.next_batch("text")
            step_out = trainer.train_step(
                "text", batch, will_log=True, will_light=True,
            )
            record = trainer._compute_and_log_diagnostics(
                step_out, light=True, deep=False,
            )

            blocks = record["light_blocks"]
            assert isinstance(blocks, list)
            assert len(blocks) == 2, (
                f"expected one entry per block (n_blocks=2), got {len(blocks)}"
            )
            for i, b in enumerate(blocks):
                assert tuple(b.keys()) == _SCALAR_KEYS, (
                    f"block {i}: field names are spec-locked"
                )
                for k, v in b.items():
                    assert isinstance(v, float)
                    assert torch.isfinite(torch.tensor(v)).item(), (
                        f"block {i}.{k} = {v} (non-finite on smoke run)"
                    )

    def test_absent_without_flag(self):
        """Additive field: without will_light the record carries None,
        not a crash and not a shape change."""
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(Path(tmp))
            batch = trainer.data_loader.next_batch("text")
            step_out = trainer.train_step("text", batch, will_log=True)
            record = trainer._compute_and_log_diagnostics(
                step_out, light=True, deep=False,
            )
            assert record["light_blocks"] is None

    def test_deep_path_unaffected(self):
        """The deep-cadence tape still works when both flags fire."""
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(Path(tmp))
            batch = trainer.data_loader.next_batch("text")
            step_out = trainer.train_step(
                "text", batch, will_log=True, will_light=True, will_deep=True,
            )
            record = trainer._compute_and_log_diagnostics(
                step_out, light=True, deep=True,
            )
            assert record["light_blocks"] is not None
            assert len(record["light_blocks"]) == 2
            assert "substrate_blocks" in record
            assert len(record["substrate_blocks"]) == 2


# ---------------------------------------------------------------------------
# 2026-10-08 review fixes: the guard the docstring promised, and bounded
# quantile cost at scale.
# ---------------------------------------------------------------------------


class TestReviewFixes:
    def test_helper_failure_does_not_kill_training_step(self, monkeypatch):
        """Diagnostics must never kill a run. If the helper raises, encode()
        records an all-NaN entry per block and training proceeds."""
        import luthi.v2.multimodal_model_pc as mm

        def _boom(h):
            raise RuntimeError("synthetic diagnostic failure")

        monkeypatch.setattr(mm, "_block_activation_scalars", _boom)
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(Path(tmp))
            batch = trainer.data_loader.next_batch("text")
            step_out = trainer.train_step(
                "text", batch, will_log=True, will_light=True,
            )
            record = trainer._compute_and_log_diagnostics(
                step_out, light=True, deep=False,
            )
            blocks = record["light_blocks"]
            assert len(blocks) == 2
            for b in blocks:
                assert tuple(b.keys()) == _SCALAR_KEYS
                assert all(v != v for v in b.values())  # all NaN

    def test_large_tensor_quantiles_subsampled_and_close(self):
        """Above the cap, p50/p99 come from a strided subsample: still
        accurate, and exact stats (mean/std/dead/sat) cover every element."""
        from luthi.v2.multimodal_model_pc import _QUANTILE_MAX_ELEMS

        g = torch.Generator().manual_seed(11)
        n = _QUANTILE_MAX_ELEMS * 3 + 7
        h = torch.randn(1, 1, n, generator=g)
        s = _block_activation_scalars(h)
        af = h.abs().flatten()
        # |N(0,1)| median ~0.674, p99 ~2.576
        assert s["act_p50"] == pytest.approx(0.6745, abs=0.01)
        assert s["act_p99"] == pytest.approx(2.576, abs=0.03)
        assert s["act_mean"] == pytest.approx(h.mean().item())
        assert s["dead_frac"] == pytest.approx(
            (af < _DEAD_ACT_EPS).float().mean().item()
        )
