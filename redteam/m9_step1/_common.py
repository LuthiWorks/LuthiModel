"""Shared builders for the red-team probes. Mirrors the canonical
setup in luthi/v2/m9/test_efe.py so probes exercise the real modules
under the same construction the unit tests certify.
"""

from __future__ import annotations

import torch

from luthi.v2.jepa_loss import JEPAPredictor
from luthi.v2.m9.preferences import Preferences


D = 16
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
