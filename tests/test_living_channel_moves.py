"""The living channel moves through the real training path.

Claim under test: "the act of processing changes the processor" -- a
``compute_modality_loss`` training forward must leave the living weight
buffers changed, by more than float noise, with no optimizer step taken.

Why this test exists (fresh-eyes audit, 2026-09-29): unit tests cover
``pc_self_modify`` math and the layer forward in isolation, and
``test_frozen_plasticity_reencode`` covers the freeze through a raw
encode -- but nothing asserted end-to-end that the JEPA training path
itself moves living weights. A silently-frozen channel (wrong flag
default, an over-broad freeze leaking into the training path, eval-mode
gating) would pass every existing test while the substrate sat still.
That is the healthy-but-inert failure class this repo keeps finding,
so it gets pinned at the integration level.

The freeze control runs the same path under ``freeze_plasticity`` and
asserts EXACTLY zero movement -- proving the movement the first test
sees comes from the living channel, not from the optimizer (no step is
taken) or from test scaffolding.
"""

from __future__ import annotations

import torch

from luthi.v2.jepa_loss import JEPALoss
from luthi.v2.living_layer_pc import PredictiveCodingLayer
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM
from luthi.v2.plasticity import freeze_plasticity


VOCAB = 32
D = 32
SEQ = 16
B = 2

# Movement must exceed this many ULPs of the weight scale to count as
# real. Below ~2 ULP an update is arithmetically dead (lost to float
# rounding); 10 is a conservative floor -- the healthy path measures
# in the millions (audit probe, 2026-09-29).
MIN_ULP = 10.0


def _build_loss(seed: int = 7) -> JEPALoss:
    torch.manual_seed(seed)
    model = MultimodalPredictiveCodingLM(
        vocab_size=VOCAB, d_model=D, n_blocks=2, n_heads=2,
        ffn_expansion=1, max_seq_len=SEQ,
        max_audio_tokens=SEQ, max_vision_tokens=SEQ,
        backward_pass_enabled=False,
    )
    loss = JEPALoss(online_encoder=model)
    loss.train()
    return loss


def _living_layers(loss: JEPALoss) -> list[PredictiveCodingLayer]:
    layers = [
        m for m in loss.online_encoder.modules()
        if isinstance(m, PredictiveCodingLayer)
    ]
    assert layers, "no PredictiveCodingLayer in trunk -- test is vacuous"
    return layers


def _batch() -> dict:
    return {"text_tokens": torch.randint(0, VOCAB, (B, SEQ))}


def _max_ulp(before: torch.Tensor, after: torch.Tensor) -> float:
    delta = (after - before).abs()
    scale = before.abs().clamp(min=1e-30)
    ulp = delta / (torch.finfo(before.dtype).eps * scale)
    return float(ulp.max().item())


class TestLivingChannelMoves:
    def test_training_forward_moves_every_living_layer(self):
        """One compute_modality_loss (no optimizer step) must move each
        living layer's weight buffer by a real, above-noise amount."""
        loss = _build_loss()
        layers = _living_layers(loss)
        # The living weight is a BUFFER, never a Parameter: the optimizer
        # cannot move it, so any change here is the living channel alone.
        for layer in layers:
            assert not any(
                p is layer.weight for p in loss.parameters()
            ), "living weight became a Parameter -- optimizer could move it"

        before = [l.weight.detach().clone() for l in layers]
        skips_before = [int(l.nonfinite_forward_skips.item()) for l in layers]
        result = loss.compute_modality_loss("text", _batch())
        assert torch.isfinite(result["loss"]), "loss itself went non-finite"

        for i, layer in enumerate(layers):
            max_d = float((layer.weight.detach() - before[i]).abs().max())
            ulp = _max_ulp(before[i], layer.weight.detach())
            assert max_d > 0.0, (
                f"layer {i}: living weight did not move on a training "
                f"forward -- the living channel is silent on the path "
                f"that is supposed to embody 'processing changes the "
                f"processor'"
            )
            assert ulp > MIN_ULP, (
                f"layer {i}: max|dw|={max_d:.3e} is {ulp:.1f} ULP of the "
                f"weight scale -- arithmetically dead, not learning"
            )
            assert int(layer.nonfinite_forward_skips.item()) == skips_before[i], (
                f"layer {i}: the NaN guard skipped self-modification -- "
                f"movement measured on other layers is not the full story"
            )

    def test_freeze_control_is_exactly_zero(self):
        """Same path under freeze_plasticity: bitwise zero movement.

        Pairs with the test above -- together they prove the freeze is
        the thing making the difference, not a dead code path, and that
        the movement measured above comes from pc_self_modify alone.
        """
        loss = _build_loss()
        layers = _living_layers(loss)
        before = [l.weight.detach().clone() for l in layers]
        with freeze_plasticity(loss.online_encoder):
            loss.compute_modality_loss("text", _batch())
        for i, layer in enumerate(layers):
            assert torch.equal(layer.weight.detach(), before[i]), (
                f"layer {i}: frozen path mutated living weight -- "
                f"freeze_plasticity is leaking on the training path"
            )
