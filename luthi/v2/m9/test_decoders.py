"""Unit tests for the M9 launch decoder set.

Run from project root:
    python -m luthi.v2.m9.test_decoders

Spec-correctness properties:
- Each decoder produces shapes that match its modality and an
  intensity scalar in [0, 1].
- re_encode produces a [B, D] reconstructed action.
- Cycle-consistency residual is non-negative and finite.
- DecoderRegistry.external_stasis returns True iff all modality
  intensities fall below their thresholds (K-M9-5 input).
- readable_summary produces one record per batch element with the
  expected keys (instrumentation §11.i).
- Gradients flow from decode through re_encode (cycle-consistency
  is differentiable so P4 truthfulness can train decoders).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from luthi.v2.m9.decoders import (
    AttentionDecoder,
    DecoderRegistry,
    MemoryDecoder,
    TextDecoder,
)


D = 16
V = 32
B = 4


def _build_registry(
    intensity_thresholds: dict | None = None,
) -> DecoderRegistry:
    text = TextDecoder(
        output_proj=nn.Linear(D, V),
        d_model=D,
        vocab_size=V,
    )
    attn = AttentionDecoder(d_model=D, n_modalities=3)
    mem = MemoryDecoder(d_model=D)
    return DecoderRegistry(
        text=text,
        attention=attn,
        memory=mem,
        intensity_thresholds=intensity_thresholds,
    )


# ---------- TextDecoder ----------

def test_text_decoder_shapes():
    dec = TextDecoder(nn.Linear(D, V), D, V)
    a = torch.randn(B, D)
    out = dec.decode(a)
    assert out["logits"].shape == (B, V)
    assert out["intensity"].shape == (B,)
    assert (out["intensity"] >= 0).all()
    assert (out["intensity"] <= 1).all()
    rec = dec.re_encode(out)
    assert rec.shape == (B, D)


# ---------- AttentionDecoder ----------

def test_attention_decoder_shapes():
    dec = AttentionDecoder(d_model=D, n_modalities=3)
    a = torch.randn(B, D)
    out = dec.decode(a)
    assert out["gates"].shape == (B, 3)
    assert (out["gates"] >= 0).all() and (out["gates"] <= 1).all()
    assert out["intensity"].shape == (B,)
    rec = dec.re_encode(out)
    assert rec.shape == (B, D)


# ---------- MemoryDecoder ----------

def test_memory_decoder_shapes():
    dec = MemoryDecoder(d_model=D)
    a = torch.randn(B, D)
    out = dec.decode(a)
    assert out["salience"].shape == (B,)
    assert out["intensity"].shape == (B,)
    assert (out["salience"] >= 0).all() and (out["salience"] <= 1).all()
    rec = dec.re_encode(out)
    assert rec.shape == (B, D)


# ---------- Cycle consistency ----------

def test_cycle_consistency_residual_nonnegative_and_finite():
    reg = _build_registry()
    a = torch.randn(B, D)
    cc = reg.cycle_consistency(a)
    for name, res in cc["per_modality_residual"].items():
        assert res.shape == (B,)
        assert torch.all(res >= 0), f"{name} residual must be non-negative"
        assert torch.all(torch.isfinite(res))


def test_cycle_consistency_differentiable():
    """P4 truthfulness needs to train decoders -- gradient must flow
    from the cycle-consistency residual back into decoder params.
    """
    reg = _build_registry()
    a = torch.randn(B, D)
    cc = reg.cycle_consistency(a)
    loss = sum(r.mean() for r in cc["per_modality_residual"].values())
    loss.backward()
    # At least one of each decoder's params has a non-zero gradient.
    for mod in (reg.text, reg.attention, reg.memory):
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in mod.parameters()
        )
        assert has_grad, f"{mod.__class__.__name__} should receive gradient"


# ---------- External stasis (K-M9-5 input) ----------

def test_external_stasis_fires_when_all_intensities_low():
    """All decoder intensities below threshold -> True."""
    reg = _build_registry(
        intensity_thresholds={"text": 0.5, "attention": 0.5, "memory": 0.5}
    )
    # Manufacture outputs with explicit intensities below threshold.
    fake_outs = {
        "text": {
            "logits": torch.zeros(B, V),
            "intensity": torch.full((B,), 0.1),
        },
        "attention": {
            "gates": torch.zeros(B, 3),
            "intensity": torch.full((B,), 0.1),
        },
        "memory": {
            "salience": torch.zeros(B),
            "intensity": torch.full((B,), 0.1),
        },
    }
    stasis = reg.external_stasis(fake_outs)
    assert stasis.shape == (B,)
    assert torch.all(stasis), "all-low should yield stasis True"


def test_external_stasis_off_when_any_intensity_high():
    """Even ONE modality above threshold breaks stasis."""
    reg = _build_registry(
        intensity_thresholds={"text": 0.5, "attention": 0.5, "memory": 0.5}
    )
    # Text intensity high, others low.
    fake_outs = {
        "text": {
            "logits": torch.zeros(B, V),
            "intensity": torch.full((B,), 0.9),  # above threshold
        },
        "attention": {
            "gates": torch.zeros(B, 3),
            "intensity": torch.full((B,), 0.1),
        },
        "memory": {
            "salience": torch.zeros(B),
            "intensity": torch.full((B,), 0.1),
        },
    }
    stasis = reg.external_stasis(fake_outs)
    assert not torch.any(stasis), (
        "stasis must be False when any modality is above threshold"
    )


# ---------- Readable summary ----------

def test_readable_summary_per_batch():
    reg = _build_registry()
    a = torch.randn(B, D)
    summary = reg.readable_summary(a)
    assert len(summary) == B
    for r in summary:
        for key in (
            "text_token_argmax", "text_intensity",
            "attention_gates", "attention_intensity",
            "memory_salience", "memory_intensity",
            "all_silent", "highest_intensity_modality",
        ):
            assert key in r, f"missing key {key}"
        assert isinstance(r["attention_gates"], dict)
        assert len(r["attention_gates"]) == 3
        assert r["highest_intensity_modality"] in (
            "text", "attention", "memory", "rest"
        )


def test_readable_summary_top_modality_matches_max():
    """When all intensities present, top_modality should be argmax."""
    reg = _build_registry(
        intensity_thresholds={"text": 0.0, "attention": 0.0, "memory": 0.0}
    )
    # Force-set intensity bias so memory wins clearly.
    with torch.no_grad():
        reg.text.intensity_head.bias.fill_(-5.0)         # sigmoid -> ~0
        reg.attention.intensity_head.bias.fill_(-5.0)
        reg.memory.intensity_head.bias.fill_(5.0)        # sigmoid -> ~1
        for layer in (
            reg.text.intensity_head, reg.attention.intensity_head,
            reg.memory.intensity_head,
        ):
            layer.weight.zero_()
    a = torch.randn(B, D)
    summary = reg.readable_summary(a)
    for r in summary:
        assert r["highest_intensity_modality"] == "memory", (
            f"expected memory dominant, got {r}"
        )


def test_readable_summary_rest_when_all_silent():
    reg = _build_registry(
        intensity_thresholds={"text": 0.5, "attention": 0.5, "memory": 0.5}
    )
    # Force all intensities below threshold.
    with torch.no_grad():
        for layer in (
            reg.text.intensity_head, reg.attention.intensity_head,
            reg.memory.intensity_head,
        ):
            layer.bias.fill_(-5.0)
            layer.weight.zero_()
    a = torch.randn(B, D)
    summary = reg.readable_summary(a)
    for r in summary:
        assert r["all_silent"]
        assert r["highest_intensity_modality"] == "rest"


def main() -> int:
    tests = [
        test_text_decoder_shapes,
        test_attention_decoder_shapes,
        test_memory_decoder_shapes,
        test_cycle_consistency_residual_nonnegative_and_finite,
        test_cycle_consistency_differentiable,
        test_external_stasis_fires_when_all_intensities_low,
        test_external_stasis_off_when_any_intensity_high,
        test_readable_summary_per_batch,
        test_readable_summary_top_modality_matches_max,
        test_readable_summary_rest_when_all_silent,
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
