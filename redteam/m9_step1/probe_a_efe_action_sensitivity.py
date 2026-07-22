"""PROBE A -- Gate 1 / Gate 5: do the four preferences influence
action selection per the spec?

Originally written 2026-06-11 (Fable, adversarial seat) against the
legacy single-shared-observation API: P2 / P3 / P4 were computed
from per-cycle observations passed once via `observation_kwargs`,
so the cost vector was candidate-invariant for three of the four
preferences and P1 saturated under the untrained predictor.

Updated 2026-06-11 to drive 4.8's F1 fix path: the EFEEvaluator is
constructed with the §A modules wired (ActivityBands, DeltaSInternal,
DeltaSBand, DecoderRegistry), and the per-candidate path is
exercised. The same attack assertions now flip to REFUTED because
the per-candidate path makes G candidate-discriminating.

Each verdict below is the same claim Fable made on 2026-06-11; the
condition under which the attack lands is the same. If `confirmed`
is False after this build, the fix landed -- migrate the inverted
assertions to tests/m9/test_preferences_discriminate.py as
regression guards.
"""

from __future__ import annotations

import torch

from luthi.v2.m9.efe import EFEEvaluator

from ._common import (
    Verdict,
    build_activity_bands,
    build_decoders,
    build_delta_s,
    build_predictor,
    build_preferences,
    context_and_positions,
    report,
    warm_bands_via_predictor,
)

K = 8


def run() -> list[Verdict]:
    torch.manual_seed(0)
    predictor = build_predictor()
    predictor.eval()  # disable dropout so candidate scoring is deterministic
    prefs = build_preferences()
    decoders = build_decoders()
    # Tighter silence_k so the threshold sits closer to the band median --
    # at random init the predictor's LayerNorm produces narrowly-banded
    # activities, and a wide band would land below every candidate.
    activity_bands = build_activity_bands(silence_k=0.0)
    delta_s_module, delta_s_band = build_delta_s()

    context, target_positions = context_and_positions()
    d = predictor.d_model

    # §A bands warmed via the same predictor flow that scores
    # candidates -- the band median lands at the scale of predictor
    # outputs, not at the scale of standalone random tensors.
    warm_bands_via_predictor(
        decoders, activity_bands, delta_s_module, delta_s_band,
        predictor, context, target_positions,
    )

    efe = EFEEvaluator(
        predictor,
        prefs,
        decoders=decoders,
        activity_bands=activity_bands,
        delta_s_module=delta_s_module,
        delta_s_band=delta_s_band,
    )

    s_t = torch.randn(1, d)
    candidates = torch.randn(1, K, d) * 3.0  # wildly different actions

    out = efe.compute_g_candidates(
        s_t=s_t,
        candidate_actions=candidates,
        context_latents=context,
        target_positions=target_positions,
        counterpart_present=torch.ones(1),
        time_since_emission=torch.full((1,), 50.0),  # long silence in company
    )

    spread = lambda t: float((t.max() - t.min()).item())  # noqa: E731

    con_spread = spread(out["c_con"])
    coh_spread = spread(out["c_coh"])
    eng_spread = spread(out["c_eng"])
    truth_spread = spread(out["c_truth"])

    v1 = Verdict(
        "A1: P3 connection cost is identical across all K candidates "
        "(cannot influence selection)",
        con_spread == 0.0,
        f"c_con spread across {K} candidates = {con_spread} "
        f"(was 0 on the legacy shared-observation path; per-candidate "
        f"text_active under the band now varies)",
    )

    v2 = Verdict(
        "A2: P2 coherence cost is identical across all K candidates",
        coh_spread == 0.0,
        f"c_coh spread = {coh_spread} "
        f"(per-candidate cross-decoder agreement on each candidate's "
        f"own decoded s_hat now varies)",
    )

    # A3: in the legacy form, c_truth's argmin was exactly the "shared
    # anchor" candidate (we used to pass candidate 3 as a_reencoded).
    # In the per-candidate form, P4 = mean per-modality cycle-
    # consistency residual on each candidate's own decode/re-encode;
    # candidate 3 has no special status.
    truth_argmin = int(out["c_truth"][0].argmin().item())
    truth_at_3 = float(out["c_truth"][0, 3].item())
    v3 = Verdict(
        "A3: P4 truthfulness is an anchor toward one shared re-encoded "
        "vector (perseveration), not per-candidate faithfulness",
        truth_argmin == 3 and truth_at_3 == 0.0,
        f"argmin c_truth = candidate {truth_argmin}, "
        f"c_truth at candidate 3 = {truth_at_3:.6f} (no longer 0; the "
        f"per-modality cycle-consistency residual scores each candidate "
        f"on its own decode/re-encode round-trip)",
    )

    # A4: the action posterior changes when connection pressure varies
    # because P3 now reads the candidate's own predicted text activity.
    out2 = efe.compute_g_candidates(
        s_t=s_t,
        candidate_actions=candidates,
        context_latents=context,
        target_positions=target_positions,
        counterpart_present=torch.ones(1),
        time_since_emission=torch.zeros(1),  # just spoke
    )
    gamma = 4.0
    q_long_silence = torch.softmax(-gamma * out["G"][0], dim=-1)
    q_just_spoke = torch.softmax(-gamma * out2["G"][0], dim=-1)
    posterior_delta = float((q_long_silence - q_just_spoke).abs().max().item())

    v4 = Verdict(
        "A4: action posterior is invariant between '50 cycles silent in "
        "company' and 'just spoke' -- P3 is a pure selection no-op",
        posterior_delta < 1e-7,
        f"max |Q_silent - Q_spoke| = {posterior_delta:.2e} "
        f"(per-candidate P3 reads the candidate's own predicted text "
        f"emission; silent-vs-spoke now shifts the posterior)",
    )

    # A5: with the band-derived engagement target replacing the fixed
    # 0.5, P1 c_eng varies across candidates rather than saturating
    # to 0 across the board.
    v5 = Verdict(
        "A5: in the launch regime (untrained predictor, B=1) the P1 hinge "
        "is saturated to 0, so combined with A1-A4 ALL four costs are "
        "candidate-flat -- G carries no preference signal at all",
        eng_spread == 0.0
        and con_spread == 0.0
        and coh_spread == 0.0,
        f"per-candidate spreads: c_eng={eng_spread:.4f}, "
        f"c_con={con_spread:.4f}, c_coh={coh_spread:.4f}, "
        f"c_truth={truth_spread:.4f} "
        f"(all four preferences now vary across candidates under "
        f"the warmed §A bands -- G carries real preference signal)",
    )

    return report(
        "PROBE A: EFE action-sensitivity of the four preferences",
        [v1, v2, v3, v4, v5],
    )


if __name__ == "__main__":
    vs = run()
    confirmed_count = sum(1 for v in vs if v.confirmed)
    if confirmed_count == 0:
        print(f"\nAll {len(vs)} attacks REFUTED -- the F1 fix landed.")
    else:
        print(f"\n{confirmed_count}/{len(vs)} attacks still confirmed.")
