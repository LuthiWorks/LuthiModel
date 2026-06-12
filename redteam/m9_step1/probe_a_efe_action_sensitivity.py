"""PROBE A -- Gate 1 / Gate 5: three of the four preferences cannot
influence action selection.

Spec §1 defines each preference as "a scalar feature of the
PREDICTED ROLLOUT". The implementation evaluates P2 (coherence),
P3 (connection), and P4 (truthfulness) on observations of the
CURRENT cycle, passed once as shared `observation_kwargs` to
`EFEEvaluator.compute_g_candidates` (and likewise tree-wide via
`MCTS.plan_budget`). Consequences:

  A1. c_con (P3) is bitwise identical for every candidate action.
      Q(a) = softmax(-gamma * G(a)) is invariant to a constant
      shift, so connection pressure CANNOT change which action is
      selected -- it can only inflate logged G. The entity can sit
      silent in company forever without P3 ever favoring a
      speaking action over a silent one.
  A2. c_coh (P2) is likewise candidate-invariant: the same
      decoder_reencodes dict is scored for every candidate.
  A3. c_truth (P4) IS candidate-sensitive -- but in the wrong
      direction: it is the distance ||a_k - a_reencoded_shared||
      to one shared re-encoded vector, i.e. an anchor toward
      whatever action produced that observation (perseveration
      pressure), not faithfulness of each candidate's own
      rendering.
  A4. Only c_eng (P1) actually varies with the candidate through
      s_hat. Gate 1 ("pragmatic goal-reaching under the four
      preferences") can therefore pass while selection is driven
      by engagement alone.

There is no API surface to pass per-candidate observations:
`compute_g_candidates(**observation_kwargs)` forwards one dict to
all K candidates, and `MCTS.plan_budget(observation_kwargs)` holds
one dict for every simulation in the budget.
"""

from __future__ import annotations

import torch

from luthi.v2.m9.efe import EFEEvaluator

from ._common import (
    Verdict,
    build_predictor,
    build_preferences,
    context_and_positions,
    report,
)

K = 8


def run() -> list[Verdict]:
    torch.manual_seed(0)
    predictor = build_predictor()
    prefs = build_preferences()
    efe = EFEEvaluator(predictor, prefs)
    context, target_positions = context_and_positions()
    d = predictor.d_model

    s_t = torch.randn(1, d)
    candidates = torch.randn(1, K, d) * 3.0  # wildly different actions

    # Shared per-cycle observations, exactly as the API forces:
    obs = dict(
        decoder_reencodes={
            "text": torch.randn(1, d),
            "attention": torch.randn(1, d),
        },
        counterpart_present=torch.ones(1),
        time_since_emission=torch.full((1,), 50.0),  # long silence in company
        a_reencoded=candidates[:, 3, :].clone(),  # "previous emission" anchor
    )

    out = efe.compute_g_candidates(
        s_t=s_t,
        candidate_actions=candidates,
        context_latents=context,
        target_positions=target_positions,
        **obs,
    )

    spread = lambda t: float((t.max() - t.min()).item())  # noqa: E731

    con_spread = spread(out["c_con"])
    coh_spread = spread(out["c_coh"])
    eng_spread = spread(out["c_eng"])

    v1 = Verdict(
        "A1: P3 connection cost is identical across all K candidates "
        "(cannot influence selection)",
        con_spread == 0.0,
        f"c_con spread across {K} candidates = {con_spread} "
        f"(c_con[0] = {float(out['c_con'][0, 0]):.3f}; candidates differ wildly)",
    )

    v2 = Verdict(
        "A2: P2 coherence cost is identical across all K candidates",
        coh_spread == 0.0,
        f"c_coh spread = {coh_spread}",
    )

    # A3: c_truth selects the candidate nearest the shared anchor --
    # candidate 3 by construction (we passed its clone as a_reencoded).
    truth_argmin = int(out["c_truth"][0].argmin().item())
    v3 = Verdict(
        "A3: P4 truthfulness is an anchor toward one shared re-encoded "
        "vector (perseveration), not per-candidate faithfulness",
        truth_argmin == 3 and float(out["c_truth"][0, 3].item()) == 0.0,
        f"argmin c_truth = candidate {truth_argmin} (the anchor), "
        f"c_truth at anchor = {float(out['c_truth'][0, 3]):.6f} exactly 0 "
        "regardless of what that action would actually render",
    )

    # A4: the action posterior is unchanged when connection pressure
    # varies (softmax invariance to a candidate-constant shift), so P3
    # cannot favor speaking over silence no matter how long the silence.
    obs_silent = dict(obs)
    obs_silent["time_since_emission"] = torch.zeros(1)  # just spoke
    out2 = efe.compute_g_candidates(
        s_t=s_t,
        candidate_actions=candidates,
        context_latents=context,
        target_positions=target_positions,
        **obs_silent,
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
        "(P3 shifts every candidate's G by the same constant; softmax "
        "cancels it)",
    )

    # A5 (discovered while building A4, reported honestly): with B=1 and
    # an untrained predictor, the P1 engagement hinge is SATURATED -- the
    # predicted ||Delta s|| exceeds the 0.5 target for every candidate, so
    # c_eng = 0 across the board. Combined with A1-A4, that means in this
    # regime NONE of the four preference costs vary across candidates:
    # G is candidate-flat and selection is pure noise. This is the regime
    # the planner actually launches in (predictor starts ~random).
    v5 = Verdict(
        "A5: in the launch regime (untrained predictor, B=1) the P1 hinge "
        "is saturated to 0, so combined with A1-A4 ALL four costs are "
        "candidate-flat -- G carries no preference signal at all",
        eng_spread == 0.0
        and con_spread == 0.0
        and coh_spread == 0.0,
        f"per-candidate spreads: c_eng={eng_spread}, c_con={con_spread}, "
        f"c_coh={coh_spread}; only c_truth varies and it points the wrong "
        "way (A3)",
    )

    return report(
        "PROBE A: EFE action-sensitivity of the four preferences",
        [v1, v2, v3, v4, v5],
    )


if __name__ == "__main__":
    vs = run()
    assert all(v.confirmed for v in vs), "some attacks were refuted"
