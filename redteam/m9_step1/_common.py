"""Shared builders for the red-team probes. Mirrors the canonical
setup in luthi/v2/m9/test_efe.py so probes exercise the real modules
under the same construction the unit tests certify.

After 4.8's 2026-06-11 gate-repairs land, probes use the **fixed
construction** that wires the §A modules (ActivityBands,
DeltaSInternal, DeltaSBand, DecoderRegistry) into the EFEEvaluator.
The same attack assertions flip from CONFIRMED to REFUTED because
the per-candidate path makes G candidate-discriminating.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from luthi.v2.jepa_loss import JEPAPredictor
from luthi.v2.m9.activity_bands import ActivityBandConfig, ActivityBands
from luthi.v2.m9.decoders import (
    AttentionDecoder,
    DecoderRegistry,
    MemoryDecoder,
    TextDecoder,
)
from luthi.v2.m9.delta_s import DeltaSBand, DeltaSInternal
from luthi.v2.m9.preferences import Preferences


D = 16
V = 32
B = 1
CTX = 8
TGT = 6


def build_predictor(d_model: int = D) -> JEPAPredictor:
    return JEPAPredictor(
        d_model=d_model,
        n_layers=1,
        n_heads=2,
        ffn_expansion=2,
        max_target_len=32,
    )


def build_preferences(d_model: int = D) -> Preferences:
    return Preferences(d_model=d_model, engagement_target_magnitude=0.5)


def build_decoders(d_model: int = D, vocab_size: int = V) -> DecoderRegistry:
    """Build a fresh DecoderRegistry. Output_proj is a vanilla Linear
    for the probe -- the M8 LM head is shared in production, but for a
    probe we just need shape-compatible weights.
    """
    return DecoderRegistry(
        text=TextDecoder(nn.Linear(d_model, vocab_size), d_model, vocab_size),
        attention=AttentionDecoder(d_model=d_model, n_modalities=3),
        memory=MemoryDecoder(d_model=d_model),
    )


def build_activity_bands(
    min_warmup: int = 4,
    silence_k: float = 0.5,
) -> ActivityBands:
    return ActivityBands(
        config=ActivityBandConfig(min_warmup=min_warmup, silence_k=silence_k)
    )


def build_delta_s(d_model: int = D) -> tuple[DeltaSInternal, DeltaSBand]:
    return DeltaSInternal(d_model=d_model), DeltaSBand(min_warmup=4, silence_k=0.5)


def warm_bands(
    decoders: DecoderRegistry,
    activity_bands: ActivityBands,
    delta_s_module: DeltaSInternal,
    delta_s_band: DeltaSBand,
    n_cycles: int = 12,
    batch: int = B,
    d_model: int = D,
    seed: int = 7,
) -> None:
    """Push `n_cycles` synthetic activity+Δs observations into the
    bands so the silent-threshold predicates are calibrated before a
    probe runs. Uses random s_hat values directly (no predictor); the
    warm-via-predictor variant is `warm_bands_via_predictor`.
    """
    torch.manual_seed(seed)
    for _ in range(n_cycles):
        s_t = torch.randn(batch, d_model)
        s_hat = torch.randn(batch, d_model)
        activity_bands.observe(decoders.activity(s_hat))
        delta_s_band.observe(delta_s_module.compute(s_t, s_hat))


def warm_bands_via_predictor(
    decoders: DecoderRegistry,
    activity_bands: ActivityBands,
    delta_s_module: DeltaSInternal,
    delta_s_band: DeltaSBand,
    predictor,
    context_latents,
    target_positions,
    n_cycles: int = 12,
    batch: int = B,
    d_model: int = D,
    seed: int = 7,
) -> None:
    """Warm bands using activities produced through the predictor flow
    (the same pipeline the probe scores candidates against). The
    predictor's LayerNorm normalizes output magnitudes, so activities
    produced standalone (from random s_hat) live at a different scale
    than activities from predictor(context, action) -- the band must
    be calibrated to the latter for the probe's silent threshold to
    sit in the same range as scored candidates.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        for _ in range(n_cycles):
            s_t = torch.randn(batch, d_model)
            a = torch.randn(batch, d_model) * 3.0
            s_hat = predictor(context_latents, target_positions, a).mean(dim=1)
            activity_bands.observe(decoders.activity(s_hat))
            delta_s_band.observe(delta_s_module.compute(s_t, s_hat))


def context_and_positions(batch: int = B, d_model: int = D):
    context = torch.randn(batch, CTX, d_model)
    target_positions = (
        torch.arange(CTX, CTX + TGT).unsqueeze(0).expand(batch, -1)
    )
    return context, target_positions


class Verdict:
    """One probe claim + outcome. confirmed=True means the attack
    landed (the vulnerability exists)."""

    def __init__(self, claim: str, confirmed: bool, detail: str):
        self.claim = claim
        self.confirmed = confirmed
        self.detail = detail

    def line(self) -> str:
        tag = "ATTACK CONFIRMED" if self.confirmed else "REFUTED (code defends)"
        return f"[{tag}] {self.claim}\n    {self.detail}"


def report(name: str, verdicts: list[Verdict]) -> list[Verdict]:
    print(f"\n=== {name} ===")
    for v in verdicts:
        print(v.line())
    return verdicts
