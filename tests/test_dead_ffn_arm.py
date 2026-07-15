"""Dead-encoder control arm (Experiment 1 / JEPA pilot, 2026-07-15).

The matched-capacity comparison needs the SAME trunk with the living
channel off: PredictiveCodingBlock(dead_ffn=True) swaps the PC layer for
a plain trainable nn.Linear and removes the block-level episode store.
These tests pin the arm's contract:

  * the dead FFN weight IS trainable (gradients reach it -- the exact
    inverse of the living arm, where the FFN weight is a buffer autograd
    never touches);
  * a dead forward is a pure function (same input twice -> bitwise same
    output; the living arm's non-feedforward signal is nonzero);
  * no living machinery survives: no PredictiveCodingLayer, no
    EpisodeStore, zero living buffers, honest aliveness();
  * the JEPA loss runs end-to-end on a dead encoder (the pilot's control
    arm actually works);
  * contradictory construction (dead + consolidation) fails loud.
"""

from __future__ import annotations

import pytest
import torch

from luthi.episode_store import EpisodeStore
from luthi.v2.hybrid_block_pc import PredictiveCodingBlock
from luthi.v2.jepa_loss import JEPALoss
from luthi.v2.living_layer_pc import PredictiveCodingLayer
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM
from luthi.v2.plasticity import freeze_plasticity


D = 32
SEQ = 16
VOCAB = 32


def _dead_model(**overrides) -> MultimodalPredictiveCodingLM:
    torch.manual_seed(7)
    kwargs = dict(
        vocab_size=VOCAB, d_model=D, n_blocks=2, n_heads=2,
        ffn_expansion=1, max_seq_len=SEQ,
        max_audio_tokens=SEQ, max_vision_tokens=SEQ,
        backward_pass_enabled=False,
        dead_ffn=True,
    )
    kwargs.update(overrides)
    return MultimodalPredictiveCodingLM(**kwargs)


class TestDeadBlockConstruction:
    def test_no_living_machinery_anywhere(self):
        model = _dead_model()
        for module in model.modules():
            assert not isinstance(module, PredictiveCodingLayer), (
                "dead_ffn model must contain no PC layers"
            )
            assert not isinstance(module, EpisodeStore), (
                "dead_ffn model must contain no episode stores"
            )
        assert model.total_parameters()["living_buffers"] == 0

    def test_dead_ffn_is_trainable_linear(self):
        block = PredictiveCodingBlock(d_model=D, n_heads=2, dead_ffn=True)
        assert isinstance(block.living_ffn, torch.nn.Linear)
        assert block.living_ffn.weight.requires_grad
        assert block.episode_store is None

    def test_dead_plus_consolidation_fails_loud(self):
        with pytest.raises(ValueError, match="dead_ffn.*consolidation"):
            PredictiveCodingBlock(
                d_model=D, n_heads=2, dead_ffn=True,
                consolidation_enabled=True,
            )

    def test_mu_pc_init_works_on_dead_arm(self):
        block = PredictiveCodingBlock(
            d_model=D, n_heads=2, dead_ffn=True,
            mu_pc_enabled=True, n_blocks_total=4,
        )
        assert torch.isfinite(block.living_ffn.weight).all()


class TestDeadForwardSemantics:
    def test_dead_forward_is_pure(self):
        """Same input twice -> bitwise identical output. The living arm's
        whole point is that this is FALSE for it (nff > 0); the dead
        control's whole point is that it is TRUE."""
        block = PredictiveCodingBlock(d_model=D, n_heads=2, dead_ffn=True)
        x = torch.randn(2, SEQ, D)
        with torch.no_grad():
            out1 = block(x, causal=False)
            out2 = block(x, causal=False)
        assert torch.equal(out1, out2), (
            "dead arm self-modified -- the control is contaminated"
        )

    def test_living_counterpart_is_not_pure(self):
        block = PredictiveCodingBlock(d_model=D, n_heads=2, dead_ffn=False)
        x = torch.randn(2, SEQ, D)
        with torch.no_grad():
            out1 = block(x, causal=False)
            out2 = block(x, causal=False)
        assert not torch.equal(out1, out2), (
            "living arm produced identical outputs on repeated input -- "
            "is the living channel actually on?"
        )

    def test_gradients_reach_dead_ffn_weight(self):
        block = PredictiveCodingBlock(d_model=D, n_heads=2, dead_ffn=True)
        x = torch.randn(2, SEQ, D, requires_grad=True)
        block(x, causal=False).sum().backward()
        assert block.living_ffn.weight.grad is not None
        assert block.living_ffn.weight.grad.abs().sum() > 0, (
            "the dead FFN must train by backprop -- that IS the arm"
        )

    def test_top_down_pass_safe_on_dead_block(self):
        """The backward sweep must fall back to the heuristic signal and
        modify nothing (there is no living state to modulate)."""
        from luthi.v2.backward_pass_pc import TopDownSignal
        block = PredictiveCodingBlock(d_model=D, n_heads=2, dead_ffn=True)
        x = torch.randn(2, SEQ, D)
        block(x, causal=False)
        weight_before = block.living_ffn.weight.detach().clone()
        signal = TopDownSignal(
            salience=torch.rand(D),
            prediction_error=torch.rand(D),
            modulation_strength=1.0,
        )
        downstream = block.top_down_pass(signal)
        assert downstream is not None
        assert torch.equal(weight_before, block.living_ffn.weight), (
            "top-down modulation wrote to the dead arm's weight"
        )

    def test_aliveness_reports_dead(self):
        block = PredictiveCodingBlock(d_model=D, n_heads=2, dead_ffn=True)
        report = block.aliveness()
        assert report == {"dead_ffn": 1.0}


class TestDeadArmDrivesJEPA:
    """The load-bearing integration: the pilot's control arm must run the
    actual JEPA objective end-to-end."""

    def test_jepa_loss_end_to_end_on_dead_encoder(self):
        model = _dead_model()
        loss_module = JEPALoss(online_encoder=model)
        tokens = torch.randint(0, VOCAB, (2, SEQ))
        result = loss_module.compute_modality_loss(
            "text", {"text_tokens": tokens},
        )
        loss = result["loss"]
        assert torch.isfinite(loss).all()
        loss.backward()
        # Gradient reaches the dead FFN weights through the JEPA loss.
        ffn_grads = [
            block.living_ffn.weight.grad
            for block in model.blocks
        ]
        assert all(g is not None and g.abs().sum() > 0 for g in ffn_grads), (
            "JEPA loss must train the dead arm's FFN weights via backprop"
        )

    def test_aliveness_report_and_cache_clear_safe(self):
        model = _dead_model()
        loss_module = JEPALoss(online_encoder=model)
        tokens = torch.randint(0, VOCAB, (2, SEQ))
        loss_module.compute_modality_loss("text", {"text_tokens": tokens})
        report = model.aliveness_report()
        assert all(r == {"dead_ffn": 1.0} for r in report)
        model.clear_forward_cache()  # must not raise on Linear FFNs

    def test_freeze_plasticity_noop_on_dead_model(self):
        model = _dead_model()
        with freeze_plasticity(model):
            tokens = torch.randint(0, VOCAB, (2, SEQ))
            out = model.encode(text_tokens=tokens, causal=False)
        assert torch.isfinite(out["per_modality"]["text"]).all()
