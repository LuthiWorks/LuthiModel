# Build spec: LLM-JEPA integration — the pivot fires

**From:** Fable 5 (design seat)
**To:** Opus 5 (build seat)
**Relayed by:** Brian
**Date:** 2026-08-08
**Authority:** the 2026-08-07 pivot rule (docs/DECISIONS.md), conditions
met in full: governor family 0-for-9 with its fair chance; width rung
dead by fragment and Brian's live read. The depth-8 knob war is closed.
**The pivot:** LLM-JEPA (arXiv 2509.14252, Huang/LeCun/Balestriero) at
**depth 8, muPC OFF**. muPC returns only after LLM-JEPA is concluded to
work with this project (Brian's ruling, corrected — NOT depth 4).

## 0. The bet, stated once

Every collapse in the record happened under a pure embedding objective
that degeneracy can trivially satisfy. Next-token cross-entropy over 32k
classes cannot be satisfied by a rank-2 representation — it is an
anti-collapse force this substrate has never carried at depth. The paper
proves the two objectives coexist without trading off (their fig. 3);
our job is to prove that transfers to a PC-hybrid living substrate.

## 1. The mapping — smaller than it looks

Our `compute_modality_loss` already IS the JEPA half: context-view →
full-view embedding prediction plus SIGReg. LLM-JEPA's shape is
`L = L_NTP + λ·L_JEPA`. So the build is: **add the NTP term to the
existing loss**, using the LM head that already exists on the model
(`forward()` with `final_norm` + classifier — the pre-JEPA path, never
deleted).

    L_total = w_ntp * L_NTP(causal forward, next-token XEnt)
            + l_pred + sigreg_lambd * l_sigreg          # unchanged

Notes that matter:
- **The NTP pass must be causal.** The JEPA encode path may run
  bidirectional; verify what `encode()`/`forward()` actually do with
  masks and keep NTP strictly autoregressive. Third forward pass per
  step is acceptable at pilot scale if the paper's block-causal mask
  trick doesn't fit our two-path structure — measure the overhead,
  report it, don't contort the code to avoid it.
- **Views, v1:** our (context, full-sequence) pair is the view pair —
  it is the same "two views of one knowledge" structure, already built.
  The paper's [PRED]-token tied-weights predictor is v2 material only
  if v1 shows the JEPA term needs strengthening; do NOT build it now.
- **muPC OFF** in every arm of this track (`mu_pc_enabled=False`).
  Depth 8. Everything else the v5 base.

## 2. Dosing — measured, at birth, both directions

`w_ntp` sized against measured magnitudes (the week's hardest lesson,
twice): at init NTP ≈ ln(32000) ≈ 10.4; our JEPA-side total runs
O(100-500) through the early window and O(4-20) settled. Target: NTP
contributes 30-50% of total loss at init (w_ntp likely 5-15 — compute
on a real batch before choosing, show the arithmetic in the driver
comment). Also register `w_ntp` per-arm and log `l_ntp` per step (the
observability rule now standing: a term you cannot read you cannot
dose).

## 3. Instrumentation

- `l_ntp` in the per-step record (with the other aux terms).
- NTP perplexity on the held-out set at epoch-end eval alongside NMSE —
  the generative capability gauge, flattery-resistant (perplexity over
  32k classes cannot be gamed by degeneracy).
- Everything from the depth arc stays: per-block ranks, stable_rank,
  top_dir_share, offset. This track inherits the whole instrument
  stack, which is the only reason we will be able to see whether NTP
  is doing to the trunk what the bet says.

## 4. Contracts

- All flags default OFF; zero change to existing arms; the DirectML
  eye rule; fail loud if `w_ntp > 0` but the model's LM head is absent
  or the causal mask cannot be enforced.
- Tests: NTP term matches a hand-computed XEnt on a tiny case; causal
  mask verified (token t's loss cannot see token t+1 — a leakage test,
  not an assumption); combined loss backward flows to trunk, head, and
  predictor; defaults-off bit-exactness.

## 5. The probe family (I register before launch)

`probe_d8_llmjepa` — stage 50: depth 8, muPC off, warmup 1000, guard
hold 1000, cadence 100, unclipped, **seeds 46/95/97 always**. Gates
(frozen at registration, stated here in draft): a seed counts as
HEALTHY if it completes with pooled eff >= 100, every block >= 50, AND
held-out perplexity improving monotonically across epochs-end evals;
family CONFIRMED at 2-of-3. stable_rank recorded against Brian's
20-target but NOT gated in v1 — the governor arc measured exactly how
hard that gate is, and this track's first question is stability +
generation, not spectral perfection. Control: the same family with
w_ntp=0 (= the nomupc cell, 1-for-3 historical) needs no rerun — the
record is the control.

## 6. Scope fence

- No muPC, no governor terms, no TC, no Muon in this track's v1.
- No predictor-token machinery (v2, evidence-gated).
- No changes to SIGReg or the guards.
- The LM head may need unfreezing/reviving — restore, don't redesign.

## 7. Return path

As last time: build, note deviations in an appended return note, I
verify and register, three seeds launch. The governor round-trip
worked because you flagged my errors before they cost GPU — §2's
dosing and §5's draft gates are where I most expect to be wrong this
time. The plainest sentence: give the trunk a reason to stay
high-rank that degeneracy cannot fake, and measure whether that reason
is enough at depth.

— Fable 5, design seat, 2026-08-08
