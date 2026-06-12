"""Unit tests for M9Trainer (the JEPATrainer composition that wires
the M9 head training loop).

Run from project root:
    python -m luthi.v2.m9.test_runner

Tests cover the first-slice scope:
- Construction: all M9 components owned, two optimizers in place,
  V_target frozen and synced to V at init.
- train_step: returns the M8 result schema plus an `m9` sub-dict
  with V / habit / decoder sublosses. M8 dynamics preserved (the
  inherited core step runs unmodified; M9 phase is additive).
- Checkpoint round-trip: save, build a fresh trainer, resume,
  verify M9 component params match bit-exact.
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
    KillCriteriaConfig,
    LoggingConfig,
    ModalitySampler,
    RunnerConfig,
    SamplerConfig,
)
from luthi.v2.m9.runner import M9Config, M9Trainer
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM


VOCAB = 32
D = 32
SEQ = 16
B = 2


class _TextLoader:
    """Minimal text-only loader that satisfies the MultimodalDataLoader
    protocol. Yields synthetic random token batches; no real corpus.
    Sufficient for runner-skeleton plumbing tests where the point is
    that train_step / checkpoint mechanics hold.
    """

    def __init__(self, vocab: int, batch: int, seq_len: int, seed: int = 0):
        self.vocab = vocab
        self.batch = batch
        self.seq_len = seq_len
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self._cursor = 0

    def next_batch(self, modality: str) -> dict:
        assert modality == "text"
        tokens = torch.randint(
            0, self.vocab, (self.batch, self.seq_len),
            generator=self.gen,
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


def _build_trainer(run_dir: Path, seed: int = 7) -> M9Trainer:
    torch.manual_seed(seed)
    model = MultimodalPredictiveCodingLM(
        vocab_size=VOCAB, d_model=D, n_blocks=2, n_heads=2,
        ffn_expansion=1, max_seq_len=SEQ,
        max_audio_tokens=SEQ, max_vision_tokens=SEQ,
        backward_pass_enabled=False,  # cheaper for smoke
    )
    loss_module = JEPALoss(online_encoder=model)
    optimizer = optim.AdamW(
        [p for p in loss_module.parameters() if p.requires_grad],
        lr=1e-3,
    )
    loader = _TextLoader(VOCAB, B, SEQ, seed=seed)
    sampler_cfg = SamplerConfig(
        corpus_sizes_tokens=loader.corpus_sizes_tokens(),
        alpha=0.7,
    )
    sampler = ModalitySampler(sampler_cfg)
    runner_cfg = RunnerConfig(
        sampler=sampler_cfg,
        checkpoint=CheckpointConfig(interval_seconds=10**9, rolling_slots=3),
        logging=LoggingConfig(light_interval_batches=10**9, deep_interval_batches=10**9),
        kill_criteria=KillCriteriaConfig(warmup_batches=10**9),
        epoch=EpochConfig(max_epochs=1, max_batches_per_epoch=10**9),
    )
    return M9Trainer(
        loss_module=loss_module,
        optimizer=optimizer,
        sampler=sampler,
        data_loader=loader,
        config=runner_cfg,
        run_dir=run_dir,
        m9_config=M9Config(),
    )


# ---------- Construction ----------

def test_constructs_with_all_m9_components():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _build_trainer(Path(tmp))
        # All component bag entries are present.
        for attr in (
            "v_head", "v_target", "habit_net", "decoders",
            "preferences", "activity_bands", "delta_s_module",
            "delta_s_band", "efe", "gamma", "m9_kills",
            "staleness", "mi_probe", "action_log",
        ):
            assert hasattr(trainer, attr), f"missing M9 component: {attr}"
        # Two optimizers in place: M8 core + M9 heads.
        assert trainer.optimizer is not trainer.m9_optimizer
        # V_target is a frozen Polyak copy synced to V at init.
        for p in trainer.v_target.parameters():
            assert not p.requires_grad
        for p, p_t in zip(trainer.v_head.parameters(), trainer.v_target.parameters()):
            assert torch.equal(p, p_t)
        trainer.action_log.close()


# ---------- Two-phase train_step ----------

def test_train_step_returns_m8_and_m9_sublosses():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _build_trainer(Path(tmp))
        batch = trainer.data_loader.next_batch("text")
        out = trainer.train_step("text", batch)

        # M8 schema preserved.
        assert "loss" in out
        assert "modality" in out and out["modality"] == "text"
        assert "raw" in out
        # M9 phase added.
        assert "m9" in out
        for k in ("v_loss", "habit_loss", "decoder_loss", "total"):
            assert k in out["m9"], f"missing M9 subloss: {k}"
        # All finite.
        for k, v in out["m9"].items():
            assert v == v, f"NaN in m9.{k}"
        trainer.action_log.close()


def test_train_step_advances_step_counters():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _build_trainer(Path(tmp))
        before = trainer.global_step
        batch = trainer.data_loader.next_batch("text")
        trainer.train_step("text", batch)
        assert trainer.global_step == before + 1
        assert trainer.modality_step["text"] == 1
        trainer.action_log.close()


def test_v_target_drifts_toward_v():
    """Polyak: after one step, V_target params are between init and V."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _build_trainer(Path(tmp))
        # Snapshot V_target init = V init.
        v_init = [p.detach().clone() for p in trainer.v_head.parameters()]
        batch = trainer.data_loader.next_batch("text")
        trainer.train_step("text", batch)
        # V_target should have moved toward updated V (which has been
        # adjusted by m9_optimizer).
        any_moved = False
        for p_init, p_t in zip(v_init, trainer.v_target.parameters()):
            if not torch.equal(p_init, p_t):
                any_moved = True
                break
        assert any_moved, "V_target params should drift via Polyak update"
        trainer.action_log.close()


# ---------- Checkpoint round-trip ----------

def test_checkpoint_roundtrip_preserves_m9_state():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        t1 = _build_trainer(run_dir, seed=11)

        # Run a few steps to make state diverge from init.
        for _ in range(3):
            batch = t1.data_loader.next_batch("text")
            t1.train_step("text", batch)

        t1._checkpoint(reason="test")
        t1.action_log.close()
        ckpt_path = sorted((run_dir / "checkpoints").glob("ckpt_*.pt"))[-1]

        # Capture state for comparison.
        before = {
            "v_head": [p.detach().clone() for p in t1.v_head.parameters()],
            "v_target": [p.detach().clone() for p in t1.v_target.parameters()],
            "habit_mean_weight": t1.habit_net.mean_head.weight.detach().clone(),
            "decoder_attention_gate_weight": t1.decoders.attention.gate_head.weight.detach().clone(),
            "decoder_memory_salience_weight": t1.decoders.memory.salience_head.weight.detach().clone(),
            "gamma": float(t1.gamma.gamma.detach().item()),
            "preference_engagement_weight": t1.preferences.engagement_weight.detach().clone(),
            "delta_s_internal_mask": t1.delta_s_module.internal_mask.detach().clone(),
        }

        # Build a fresh trainer (different seed -> different inits) and resume.
        t2 = _build_trainer(run_dir, seed=99)
        t2.resume(ckpt_path)
        # Compare.
        for p_b, p_a in zip(before["v_head"], t2.v_head.parameters()):
            assert torch.equal(p_b, p_a), "v_head mismatch after resume"
        for p_b, p_a in zip(before["v_target"], t2.v_target.parameters()):
            assert torch.equal(p_b, p_a), "v_target mismatch after resume"
        assert torch.equal(
            before["habit_mean_weight"], t2.habit_net.mean_head.weight,
        )
        assert torch.equal(
            before["decoder_attention_gate_weight"],
            t2.decoders.attention.gate_head.weight,
        )
        assert torch.equal(
            before["decoder_memory_salience_weight"],
            t2.decoders.memory.salience_head.weight,
        )
        assert abs(before["gamma"] - float(t2.gamma.gamma.detach().item())) < 1e-6
        assert torch.equal(
            before["preference_engagement_weight"],
            t2.preferences.engagement_weight,
        )
        assert torch.equal(
            before["delta_s_internal_mask"],
            t2.delta_s_module.internal_mask,
        )
        t2.action_log.close()


def main() -> int:
    tests = [
        test_constructs_with_all_m9_components,
        test_train_step_returns_m8_and_m9_sublosses,
        test_train_step_advances_step_counters,
        test_v_target_drifts_toward_v,
        test_checkpoint_roundtrip_preserves_m9_state,
    ]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed.append((t.__name__, str(e)))
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    if failed:
        print(f"\n{len(failed)} test(s) failed")
        return 1
    print(f"\nAll {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
