"""Kill-7 plateau semantics (M1 fix, 2026-07-15).

Pre-fix, criterion 7 ("objective unlearnable") compared window halves
forever, so a healthily-converged plateau read as unlearnable and killed
a multi-day run at its best moment (same failure family as the K-M9-7
false-halt, 2026-07-05). The fix: armed until first sustained descent
(> kill7_descent_margin relative), then permanently disarmed. These
tests pin all three regimes and the checkpoint round-trip of the latch:

  flat from the start      -> kill fires (the semantic kill-7 exists for)
  descend, then plateau    -> NO kill (the false-kill, fixed)
  ambiguous shallow window -> neither kill nor disarm (keep watching)
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
WINDOW = 40  # small descent window so tests fill it instantly


class _Loader:
    def __init__(self):
        self.gen = torch.Generator().manual_seed(0)

    def next_batch(self, modality):
        return {"text_tokens": torch.randint(
            0, VOCAB, (2, SEQ), generator=self.gen,
        )}

    def batch_token_count(self, modality, batch):
        return int(batch["text_tokens"].numel())

    def state_dict(self):
        return {"gen": self.gen.get_state()}

    def load_state_dict(self, state):
        self.gen.set_state(state["gen"])

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
            kill_criteria=KillCriteriaConfig(
                warmup_batches=0,  # arm immediately; we drive the buffer by hand
                loss_descent_window=WINDOW,
            ),
            epoch=EpochConfig(max_epochs=1),
        ),
        run_dir=Path(tmp),
    )


def _fill(trainer: JEPATrainer, values: list[float]) -> None:
    trainer._smoothed_loss_buf.clear()
    for v in values:
        trainer._smoothed_loss_buf.append(v)


def test_flat_from_start_kills():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _trainer(tmp)
        _fill(trainer, [1.0] * WINDOW)
        reason = trainer._check_kill_criteria("text")
        assert reason is not None and "kill-7" in reason
        assert not trainer._kill7_descent_established


def test_descend_then_plateau_does_not_kill():
    """The false-kill, fixed: a run that descended and then converged
    must NOT be killed as unlearnable."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _trainer(tmp)
        # Phase 1: clear descent fills the window -> latch sets.
        descent = [2.0 - 0.02 * i for i in range(WINDOW)]
        _fill(trainer, descent)
        assert trainer._check_kill_criteria("text") is None
        assert trainer._kill7_descent_established, (
            "sustained descent must disarm kill-7"
        )
        # Phase 2: hard plateau -- pre-fix this returned kill-7.
        _fill(trainer, [1.2] * WINDOW)
        reason = trainer._check_kill_criteria("text")
        assert reason is None or "kill-7" not in reason, (
            f"kill-7 fired on a healthy converged plateau: {reason}"
        )


def test_ambiguous_shallow_window_neither_kills_nor_disarms():
    """Descending, but under the establish margin: keep watching."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _trainer(tmp)
        # ~0.4% relative descent across halves < 1% margin, but second
        # half strictly below first half so the kill branch stays quiet.
        shallow = [1.0 - 0.0001 * i for i in range(WINDOW)]
        _fill(trainer, shallow)
        reason = trainer._check_kill_criteria("text")
        assert reason is None or "kill-7" not in reason
        assert not trainer._kill7_descent_established


def test_latch_survives_checkpoint_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _trainer(tmp)
        _fill(trainer, [2.0 - 0.02 * i for i in range(WINDOW)])
        trainer._check_kill_criteria("text")
        assert trainer._kill7_descent_established
        trainer._checkpoint(reason="test")

        fresh = _trainer(tmp + "_fresh")
        loaded = fresh.resume_from_latest(Path(tmp) / "checkpoints")
        assert loaded is not None
        assert fresh._kill7_descent_established, (
            "a resumed run re-armed kill-7 against its own later plateau"
        )
