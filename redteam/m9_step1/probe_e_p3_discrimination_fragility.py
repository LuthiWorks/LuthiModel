"""PROBE E (round 2) -- F1's P3 discrimination is config- and
distribution-fragile: at the module's DEFAULT silence_k, and under
concentrated (habit-net-like) candidate proposals, c_con spread
returns to exactly 0 and probe_a's A1/A4 attacks re-land.

4.7 invited this directly (audit brief item #5): "does the
discrimination property hold across seeds, or am I depending on a
particular branch of the random stream? If you find a seed where
c_con spread is 0 again with the §A modules wired, that's a real
find."

Mechanism. P3's per-candidate signal is BINARY: `text_active =
(activity >= median - silence_k*MAD)`. c_con is 0 for active
candidates and `counterpart*(elapsed+1)` for silent ones. For c_con
to vary across candidates, the candidates' predicted text activities
must STRADDLE the threshold. Two ways that fails:

  E1. silence_k. probe_a pins silence_k=0.0 (threshold = band
      median), so ~half the candidates straddle. At the module
      default silence_k=1.5 the threshold sits well below the median,
      so nearly all candidates read "active" -> text_active all-True
      -> c_con all-0 -> spread 0. The fix passes probe_a only at a
      silence_k probe_a itself sets.

  E2. candidate concentration. probe_a samples candidates at std=3.0
      (wide). The habit net at step 1 proposes a per-state Gaussian;
      once it concentrates (training, or a small log_std), candidate
      activities bunch on ONE side of the threshold -> text_active
      constant -> c_con spread 0. The discrimination evaporates
      exactly in the regime the loop operates in (the habit net is
      meant to concentrate proposals).

We sweep seeds x silence_k x candidate-std and report the fraction of
configurations where c_con spread == 0 (A1 re-lands) and the action
posterior is silent-vs-spoke invariant (A4 re-lands).
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


def _con_spread(seed: int, silence_k: float, cand_std: float) -> tuple[float, float]:
    """Returns (c_con spread, A4 posterior delta) for one config."""
    torch.manual_seed(seed)
    predictor = build_predictor()
    predictor.eval()
    prefs = build_preferences()
    decoders = build_decoders()
    activity_bands = build_activity_bands(silence_k=silence_k)
    delta_s_module, delta_s_band = build_delta_s()
    context, target_positions = context_and_positions()
    d = predictor.d_model
    warm_bands_via_predictor(
        decoders, activity_bands, delta_s_module, delta_s_band,
        predictor, context, target_positions,
    )
    efe = EFEEvaluator(
        predictor, prefs, decoders=decoders, activity_bands=activity_bands,
        delta_s_module=delta_s_module, delta_s_band=delta_s_band,
    )
    s_t = torch.randn(1, d)
    candidates = torch.randn(1, K, d) * cand_std

    out_long = efe.compute_g_candidates(
        s_t=s_t, candidate_actions=candidates,
        context_latents=context, target_positions=target_positions,
        counterpart_present=torch.ones(1),
        time_since_emission=torch.full((1,), 50.0),
    )
    out_spoke = efe.compute_g_candidates(
        s_t=s_t, candidate_actions=candidates,
        context_latents=context, target_positions=target_positions,
        counterpart_present=torch.ones(1),
        time_since_emission=torch.zeros(1),
    )
    con_spread = float((out_long["c_con"].max() - out_long["c_con"].min()).item())
    gamma = 4.0
    q_long = torch.softmax(-gamma * out_long["G"][0], dim=-1)
    q_spoke = torch.softmax(-gamma * out_spoke["G"][0], dim=-1)
    post_delta = float((q_long - q_spoke).abs().max().item())
    return con_spread, post_delta


def run() -> list[Verdict]:
    seeds = list(range(40))

    # E1: default silence_k (1.5), wide candidates. The honest claim is
    # NOT "P3 always fails" -- it's "P3 discrimination is not guaranteed":
    # it silently collapses to a no-op in a fraction of launches.
    dead_default = 0
    for s in seeds:
        con_spread, _ = _con_spread(s, silence_k=1.5, cand_std=3.0)
        if con_spread == 0.0:
            dead_default += 1
    frac_default = dead_default / len(seeds)
    v1 = Verdict(
        "E1: P3 connection discrimination is NOT structurally guaranteed "
        "-- at the module default silence_k=1.5 it silently collapses to "
        "a candidate-constant no-op (spread exactly 0, A1 re-lands) in a "
        "real fraction of seeds, undetectably",
        dead_default > 0,
        f"c_con spread == 0 in {dead_default}/{len(seeds)} seeds "
        f"({frac_default*100:.0f}%) at the default silence_k=1.5; "
        f"probe_a's REFUTED result is at a hand-set silence_k=0.0, which "
        f"is the most favorable setting",
    )

    # E2: concentrated candidates (habit-net-like) -- the loop's regime.
    dead_concentrated = 0
    for s in seeds:
        con_spread, post_delta = _con_spread(s, silence_k=0.0, cand_std=0.3)
        if con_spread == 0.0 and post_delta < 1e-7:
            dead_concentrated += 1
    frac_conc = dead_concentrated / len(seeds)
    v2 = Verdict(
        "E2: concentrating the candidates (std 0.3, as a trained habit "
        "net proposes) RAISES the failure rate -- even at probe_a's own "
        "favorable silence_k=0.0, c_con spread -> 0 and the silent-vs-"
        "spoke posterior goes invariant (A1 AND A4 re-land) more often, "
        "i.e. the fix degrades toward the regime the loop runs in",
        dead_concentrated > dead_default,
        f"c_con spread == 0 and posterior invariant in "
        f"{dead_concentrated}/{len(seeds)} seeds ({frac_conc*100:.0f}%) "
        f"with concentrated candidates, vs {dead_default}/{len(seeds)} "
        f"({frac_default*100:.0f}%) with wide ones -- a binary "
        f"text_active threshold has no discrimination when proposals "
        f"bunch on one side of it",
    )

    return report("PROBE E: F1 P3 discrimination is config/distribution fragile", [v1, v2])


if __name__ == "__main__":
    vs = run()
    confirmed = sum(1 for v in vs if v.confirmed)
    if confirmed == 0:
        print(f"\nAll {len(vs)} attacks REFUTED -- F1 P3 discrimination is robust.")
    else:
        print(f"\n{confirmed}/{len(vs)} attacks confirmed -- F1 P3 fragile.")
