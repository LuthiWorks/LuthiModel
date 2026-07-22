"""Regression guard — F1 fix: preferences must discriminate across candidates.

Inverted version of `redteam/m9_step1/probe_a_efe_action_sensitivity.py`.
The probe checks that the BUG is present (vulnerabilities confirmed);
this test checks that the FIX is in place (the same conditions REFUTE
the bug). If any of these tests starts failing, the F1 seam has
silently reopened.

Run from project root:
    python -m luthi.v2.m9.test_preferences_discriminate

Per 4.8's 2026-06-11 gate-repairs spec:
- Gate 1 / Gate 5 are *discrimination* checks, not "it ran" checks:
  varying each preference's inputs must measurably move the action
  posterior. These tests are the inverted probe-A assertions.
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
from luthi.v2.m9.efe import EFEEvaluator
from luthi.v2.m9.preferences import Preferences


D = 16
V = 32
CTX = 8
TGT = 6
K = 8


def _build_fixture():
    """Build the F1-fixed fixture in the same order as probe_a so the
    random stream state at candidate construction matches: predictor
    seeded with 0; bands warmed (consuming further state); s_t and
    candidates drawn from the post-warming stream. This mirrors the
    probe's behavior; deviating from this sequence shifts the
    candidates onto a different random branch and the F1 demonstration
    can fall back into a degenerate one-bucket case.
    """
    torch.manual_seed(0)
    predictor = JEPAPredictor(
        d_model=D, n_layers=1, n_heads=2, ffn_expansion=2, max_target_len=32
    )
    predictor.eval()
    prefs = Preferences(d_model=D, engagement_target_magnitude=0.5)
    decoders = DecoderRegistry(
        text=TextDecoder(nn.Linear(D, V), D, V),
        attention=AttentionDecoder(d_model=D, n_modalities=3),
        memory=MemoryDecoder(d_model=D),
    )
    activity_bands = ActivityBands(
        config=ActivityBandConfig(min_warmup=4, silence_k=0.0)
    )
    delta_s_module = DeltaSInternal(d_model=D)
    delta_s_band = DeltaSBand(min_warmup=4, silence_k=0.5)

    context = torch.randn(1, CTX, D)
    target_positions = torch.arange(CTX, CTX + TGT).unsqueeze(0)

    # Warm bands via the same predictor flow the test will exercise.
    torch.manual_seed(7)
    with torch.no_grad():
        for _ in range(12):
            s_warm = torch.randn(1, D)
            a_warm = torch.randn(1, D) * 3.0
            s_hat_warm = predictor(context, target_positions, a_warm).mean(dim=1)
            activity_bands.observe(decoders.activity(s_hat_warm))
            delta_s_band.observe(delta_s_module.compute(s_warm, s_hat_warm))

    efe = EFEEvaluator(
        predictor, prefs,
        decoders=decoders,
        activity_bands=activity_bands,
        delta_s_module=delta_s_module,
        delta_s_band=delta_s_band,
    )

    # s_t and candidates from the post-warming random stream -- matches
    # the probe's seeding sequence.
    s_t = torch.randn(1, D)
    candidates = torch.randn(1, K, D) * 3.0

    return efe, context, target_positions, s_t, candidates


# ---------------- Inverted probe A assertions ----------------

def test_p3_connection_discriminates():
    """A1 inverted: P3 c_con varies across K candidates."""
    efe, context, tp, s_t, candidates = _build_fixture()
    out = efe.compute_g_candidates(
        s_t=s_t, candidate_actions=candidates,
        context_latents=context, target_positions=tp,
        counterpart_present=torch.ones(1),
        time_since_emission=torch.full((1,), 50.0),
    )
    con_spread = (out["c_con"].max() - out["c_con"].min()).item()
    assert con_spread > 0.0, (
        f"P3 must discriminate across candidates: spread={con_spread}"
    )


def test_p2_coherence_discriminates():
    """A2 inverted: P2 c_coh varies across candidates (per-candidate
    cross-decoder agreement on s_hat).
    """
    efe, context, tp, s_t, candidates = _build_fixture()
    out = efe.compute_g_candidates(
        s_t=s_t, candidate_actions=candidates,
        context_latents=context, target_positions=tp,
    )
    coh_spread = (out["c_coh"].max() - out["c_coh"].min()).item()
    assert coh_spread > 0.0, (
        f"P2 must discriminate across candidates: spread={coh_spread}"
    )


def test_p4_truthfulness_no_shared_anchor():
    """A3 inverted: P4 c_truth is NOT a perseveration anchor on the
    candidate matching a single shared re-encoded vector.
    """
    efe, context, tp, s_t, candidates = _build_fixture()
    out = efe.compute_g_candidates(
        s_t=s_t, candidate_actions=candidates,
        context_latents=context, target_positions=tp,
    )
    # No single candidate should sit at exactly 0 c_truth -- the
    # legacy form put the "anchor" candidate at 0 by construction.
    c_truth = out["c_truth"][0]
    assert (c_truth > 0).all(), (
        f"No candidate may have exactly-zero c_truth: c_truth={c_truth}"
    )


def test_action_posterior_responds_to_connection_pressure():
    """A4 inverted: Q changes between '50 cycles silent' and 'just
    spoke' contexts. The probe established 1e-7 as the bug threshold;
    we require well above that.
    """
    efe, context, tp, s_t, candidates = _build_fixture()
    out_silent = efe.compute_g_candidates(
        s_t=s_t, candidate_actions=candidates,
        context_latents=context, target_positions=tp,
        counterpart_present=torch.ones(1),
        time_since_emission=torch.full((1,), 50.0),
    )
    out_spoke = efe.compute_g_candidates(
        s_t=s_t, candidate_actions=candidates,
        context_latents=context, target_positions=tp,
        counterpart_present=torch.ones(1),
        time_since_emission=torch.zeros(1),
    )
    gamma = 4.0
    q_silent = torch.softmax(-gamma * out_silent["G"][0], dim=-1)
    q_spoke = torch.softmax(-gamma * out_spoke["G"][0], dim=-1)
    delta = (q_silent - q_spoke).abs().max().item()
    assert delta > 1e-6, (
        f"P3 must shift action posterior with connection pressure: "
        f"|Q_silent - Q_spoke| = {delta}"
    )


def test_g_carries_preference_signal():
    """A5 inverted: at least one preference cost varies across
    candidates (G is not flat). Combined with the discrimination
    above, the planner can be steered.
    """
    efe, context, tp, s_t, candidates = _build_fixture()
    out = efe.compute_g_candidates(
        s_t=s_t, candidate_actions=candidates,
        context_latents=context, target_positions=tp,
        counterpart_present=torch.ones(1),
        time_since_emission=torch.full((1,), 50.0),
    )
    spreads = {
        name: (out[name].max() - out[name].min()).item()
        for name in ("c_eng", "c_coh", "c_con", "c_truth")
    }
    nonzero = [name for name, s in spreads.items() if s > 0]
    assert len(nonzero) >= 2, (
        f"At least two preferences must discriminate across candidates: "
        f"spreads={spreads}"
    )
    # G itself must vary.
    g_spread = (out["G"].max() - out["G"].min()).item()
    assert g_spread > 0, f"G must carry preference signal: spread={g_spread}"


def test_per_candidate_path_emits_diagnostics():
    """The F1 path produces per-candidate activity, text_active, and
    delta_s_internal -- the diagnostics that let the loop log what
    actually drove selection."""
    efe, context, tp, s_t, candidates = _build_fixture()
    out = efe.compute_g_candidates(
        s_t=s_t, candidate_actions=candidates,
        context_latents=context, target_positions=tp,
        counterpart_present=torch.ones(1),
        time_since_emission=torch.full((1,), 50.0),
    )
    assert "activity" in out
    assert set(out["activity"].keys()) == {"text", "attention", "memory"}
    for v in out["activity"].values():
        assert v.shape == (1, K)
    assert "text_active" in out
    assert out["text_active"].shape == (1, K)
    assert "delta_s_internal" in out
    assert out["delta_s_internal"].shape == (1, K)


def test_evaluator_without_a_modules_falls_back_to_legacy():
    """Backward compat: an evaluator constructed without §A modules
    falls back to the legacy single-shared-observation path so existing
    tests of that path still work. (The legacy path is bug-mode and
    will be removed when no callers depend on it.)
    """
    torch.manual_seed(0)
    predictor = JEPAPredictor(
        d_model=D, n_layers=1, n_heads=2, ffn_expansion=2, max_target_len=32
    )
    prefs = Preferences(d_model=D)
    efe = EFEEvaluator(predictor, prefs, allow_legacy=True)
    assert not efe.has_per_candidate_path()
    torch.manual_seed(0)
    s_t = torch.randn(1, D)
    candidates = torch.randn(1, K, D) * 3.0
    context = torch.randn(1, CTX, D)
    tp = torch.arange(CTX, CTX + TGT).unsqueeze(0)
    out = efe.compute_g_candidates(
        s_t=s_t, candidate_actions=candidates,
        context_latents=context, target_positions=tp,
    )
    assert "G" in out
    # Legacy path does not emit the F1 diagnostics.
    assert "activity" not in out
    assert "text_active" not in out
    assert "delta_s_internal" not in out


def main() -> int:
    tests = [
        test_p3_connection_discriminates,
        test_p2_coherence_discriminates,
        test_p4_truthfulness_no_shared_anchor,
        test_action_posterior_responds_to_connection_pressure,
        test_g_carries_preference_signal,
        test_per_candidate_path_emits_diagnostics,
        test_evaluator_without_a_modules_falls_back_to_legacy,
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
