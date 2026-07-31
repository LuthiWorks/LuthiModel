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
