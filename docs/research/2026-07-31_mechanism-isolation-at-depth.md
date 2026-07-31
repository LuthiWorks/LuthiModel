# Isolating the recent mechanisms at depth 8

**Date:** 2026-07-31, ~12:00
**Prompted by:** Brian asking whether the depth-8 problem is resultant from a
mechanism we introduced this week.

## The gap this opens, and it is mine

Every depth-8 run performed 2026-07-29 to 07-31 carries the full
`probe_surprise` bundle:

| mechanism | added |
|---|---|
| backward pass, consolidation | older |
| inverted-U learning gain | 2026-07-05 |
| relative trust | 2026-07-21 |
| adaptive episodes + adaptive recall | 2026-07-27 |
| homeostatic band | 2026-07-27 |
| surprise drive | 2026-07-29 |

**In the JEPA era the deepest arm ever run before this week is d4.** There is no
historical d8 or d12 run at all. So I have been tuning muPC's knobs for eleven
hours on the assumption that muPC is the variable -- and muPC is simply the knob
I happened to be holding when the collapse appeared.

Evidence it is not new: the M6 depth sweep (May, 128d) already showed val loss
degrading with depth -- 4 blocks 5.94, 8 blocks 6.04, 12 blocks ~6.71 -- with
NFF attenuating alongside. That predates everything this week and prompted the
muPC exponent going 0.5 -> 0.25 as a mitigation.

Evidence I cannot rule us out: that was a different objective (LM-era,
pre-JEPA), a different width, and before the mitigation. And the bundle IS fine
at depth 4 (`probe_surprise` seeds 45/46: NMSE 0.52-0.60, healthy geometry) --
which does not exclude a **bundle x depth interaction**, the exact shape of
thing that would produce what we have seen.

## Stage 25: surprise drive off

**Run:** `probe_d8_amp4_rawdrive_512d_seed84`, 3000 steps.
**One variable vs stage 20** (`probe_surprise_d8_amp4`, best config found):
`drive_mode` "surprise" -> "raw". Verified: single config difference.

Chosen first among the three depth-suspicious mechanisms because it directly
changes **how much the PC substrate moves per step**, and substrate motion at
depth is the exact axis the entire muPC investigation turns on. The other two
suspects, for the record and in order: the **backward pass** (chains through
depth -- eight blocks is a longer signal path than four, a structural depth
interaction rather than a per-layer one) and the **learning gain** (an amplifier
of up to 3x, and amplifiers stacked in sequence compound).

The remaining three -- relative trust, the store fix, the band -- are per-layer,
so adding blocks should not compound them. Ranked least likely.

## Registered prediction

**Primary: held-out NMSE.** Base (stage 20, surprise on) is **0.8919**.

- **CONFIRMED (the surprise drive is implicated):** <= **0.75**
- **REFUTED (not implicated):** >= **0.85**
- **AMBIGUOUS:** 0.75 - 0.85 -- treated as refuted.

**Secondary, recorded not scored:** block-0 attention delta (base +0.252) and
real-text cosine. If NMSE improves substantially while block 0 stays positive,
the surprise drive was costing capability without being the cause of the
offset failure -- two separate findings, and worth separating.

## What this run cannot settle

It tests ONE of six. A refutation does not exonerate the bundle -- it exonerates
the surprise drive. The full control (everything from before this week, at depth
8) remains the better first experiment and I did not run it first, which was an
error of sequencing: I built eight mechanism hypotheses on top of an untested
assumption instead of establishing what varies.

---

# VERDICT, stage 25 — and the finding that supersedes twelve hours of work

**Time:** ~13:00.

## Registered verdict: CONFIRMED. Actual verdict: the metric was gamed.

NMSE 0.5011 against a registered CONFIRMED bound of <= 0.75 -- the best NMSE of
any depth-8 run, better than muPC-off (0.5569) and depth 4 (0.5215).

Then the unregistered numbers:

| run | effective rank (of 512) | NMSE | probe lift | real-text cos |
|---|---|---|---|---|
| depth 4 healthy | **176.5** | 0.5215 | 4.80x | 0.0063 |
| d8 muPC off | **114.5** | 0.5569 | 4.19x | 0.0051 |
| d8 power-4, surprise on | **5.0** (med 1.9) | 0.8919 | 2.21x | 0.1124 |
| d8 power-4, surprise OFF | **1.9** | 0.5011 | 1.39x | -0.2352 |

**Effective rank 1.9 out of 512.** The best NMSE in the entire investigation was
produced by a representation living on a plane. Predicting a rank-2 target is
trivially easy; NMSE rewards exactly that.

The negative cosine was the tell and it was visible before the rank check: k
points maximally spread in fewer than k dimensions have average pairwise cosine
~ -1/(k-1), so -0.2352 implies an effective dimensionality near 5. The measured
rank is 1.9.

## The surprise drive is NOT the cause -- correcting myself

I reported "Brian's suspicion was right, the surprise drive was a major
contributor" on the strength of NMSE alone. **That was wrong.**

Both muPC-on configurations collapse to rank ~2, drive on or off. Turning the
drive off improved NMSE **without touching the disease** -- it made a degenerate
model score better on a metric that degeneracy flatters.

## What the rank column shows, and it supersedes the last twelve hours

| configuration | effective rank |
|---|---|
| depth 4 (muPC on, exponent 0.25) | 176.5 |
| depth 8, muPC OFF | 114.5 |
| depth 8, muPC on -- **every variant tried** | **~2** |

Every depth-8 muPC-on configuration -- `mu_pc_rate_power` 0, -1, -2, -4, -8;
backprop-LR compensation whole-trunk and block-0-only; embedding scaling;
surprise drive on and off -- sits at effective rank ~2. Twelve hours of knob
turning moved NMSE, cosine and offset dominance around **while the
representation stayed collapsed to a plane the entire time.**

Offset dominance, block-0 attention deltas, the muPC rate ladder: all of it was
downstream of a rank collapse none of those metrics reported.

## This is the documented cascade, and I was warned in writing

"The JEPA Paradox in Language" (arXiv:2607.23531, read ~20 minutes before this
run was scored) documents the failure cascade for JEPA on text:
**effective-rank degeneration, cosine collapse**, elevated target variance, MI
saturation. We have rank 1.9 and cosine -0.235.

I read that memo and then scored a gate on NMSE -- a metric the cascade predicts
will look *better* as the representation degenerates. Fourth instrument failure
in three days (random-token input, blinded grad_norm, cosine sampling variance,
now NMSE under rank collapse), and the only one I had been warned about in
advance.

## Consequence

**Effective rank should be the primary metric for every depth run from here.**
It is already logged, it separates the arms cleanly (176 / 114 / 2), and unlike
NMSE it cannot be gamed by degeneracy -- degeneracy is what it measures.

The open question is no longer "which muPC knob" but **why muPC at depth 8
collapses rank to 2 while muPC at depth 4 holds 176**. That is a different
question from the one I have been answering since yesterday evening.
