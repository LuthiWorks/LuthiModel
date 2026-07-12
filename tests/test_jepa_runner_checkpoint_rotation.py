"""Checkpoint-rotation regression tests (Fable 5 verification pass, 2026-07-12).

The bug these tests pin down: ``JEPATrainer._checkpoint`` enforced the
rolling cap with ``existing[:excess]`` where ``excess`` can be NEGATIVE
(fewer checkpoints than ``rolling_slots``). A negative slice deletes
from the FRONT of the list -- so with 2 checkpoints and 3 slots,
``existing[:-1]`` unlinked the older one. Steady state was ONE
checkpoint on disk, never three, silently defeating the
fallback-to-older-slots durability design (v0.5 s4 / B6) that exists
because M7 died to a power loss at 24.5% of epoch 1.

Invisible to every prior test and smoke: they all set
``interval_seconds=10**9`` so interval checkpointing never fires, and a
single surviving checkpoint still resumes fine -- the classic
silent-success shape.
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
B = 2


class _TextLoader:
    """Minimal text-only loader satisfying the MultimodalDataLoader
    Protocol (same shape as the loader in
    ``tests/test_jepa_runner_emit_batch_1.py``)."""

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


def _build_trainer(run_dir: Path, rolling_slots: int = 3, seed: int = 7) -> JEPATrainer:
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
        checkpoint=CheckpointConfig(
            interval_seconds=10**9, rolling_slots=rolling_slots,
        ),
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


def _ckpt_names(run_dir: Path) -> list[str]:
    return sorted(p.name for p in (run_dir / "checkpoints").glob("ckpt_*.pt"))


class TestRollingCapUnderfill:
    """Fewer checkpoints than slots must delete NOTHING.

    This is the regression: pre-fix, the second write deleted the
    first (negative-slice front deletion), so slot accumulation never
    got past one file.
    """

    def test_two_checkpoints_three_slots_keeps_both(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(Path(tmp), rolling_slots=3)
            trainer._checkpoint(reason="test-1")
            trainer.global_step += 1
            trainer._checkpoint(reason="test-2")

            names = _ckpt_names(Path(tmp))
            assert names == ["ckpt_00000000.pt", "ckpt_00000001.pt"], (
                f"under-capacity write must not delete anything; on disk: {names}. "
                "existing[:negative_excess] deletes from the FRONT -- the "
                "rolling cap must only unlink when excess > 0."
            )

    def test_slots_actually_fill_to_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(Path(tmp), rolling_slots=3)
            for i in range(3):
                trainer.global_step = i
                trainer._checkpoint(reason=f"test-{i}")

            names = _ckpt_names(Path(tmp))
            assert len(names) == 3, (
                f"three writes with three slots must leave three files "
                f"(the durability design's whole premise); on disk: {names}"
            )


class TestRollingCapOverfill:
    """More checkpoints than slots must keep exactly the newest N."""

    def test_five_checkpoints_three_slots_keeps_newest_three(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(Path(tmp), rolling_slots=3)
            for i in range(5):
                trainer.global_step = i
                trainer._checkpoint(reason=f"test-{i}")

            names = _ckpt_names(Path(tmp))
            assert names == [
                "ckpt_00000002.pt",
                "ckpt_00000003.pt",
                "ckpt_00000004.pt",
            ], f"expected the newest three slots to survive; on disk: {names}"


class TestFallbackToOlderSlot:
    """The durability behavior the slots exist for: a corrupt newest
    checkpoint must fall back to the next-older slot on resume.

    Pre-fix this could not work in practice -- there WAS no older slot
    on disk to fall back to.
    """

    def test_resume_from_latest_skips_corrupt_newest(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            trainer = _build_trainer(run_dir, rolling_slots=3)
            trainer.global_step = 10
            trainer._checkpoint(reason="good")
            trainer.global_step = 20
            trainer._checkpoint(reason="to-be-corrupted")

            ckpts = sorted((run_dir / "checkpoints").glob("ckpt_*.pt"))
            assert len(ckpts) == 2, (
                f"need two slots on disk for the fallback test; got "
                f"{[p.name for p in ckpts]} -- if only one survives, the "
                f"rotation bug has regressed"
            )
            # Simulate the power-loss shape resume_from_latest documents:
            # newest-by-filename exists but its bytes are garbage.
            ckpts[-1].write_bytes(b"not a checkpoint")

            fresh = _build_trainer(run_dir / "fresh", rolling_slots=3)
            loaded = fresh.resume_from_latest(run_dir / "checkpoints")
            assert loaded == ckpts[0], (
                f"resume_from_latest must fall back to the older slot; "
                f"loaded {loaded}"
            )
            assert fresh.global_step == 10
