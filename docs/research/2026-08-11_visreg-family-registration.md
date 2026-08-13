# VISReg replacement family — registration (BEFORE launch)

**Author:** Fable 5, in the build seat by Brian's assignment ("Since you
have the context here I want you to build VISReg into the project"),
2026-08-11. Design rulings this executes:
`docs/reviews/2026-08-10_pruning-and-visreg-brief-for-opus.md`.
**Status:** REGISTERED, not yet launched. Gates frozen here before the
first GPU step, per house protocol.

## The family

- **Arm:** `probe_d8_visreg`, stage 54. 512d x 8 blocks, muPC OFF,
  warmup 1000, guard hold 1000, deep cadence 100, unclipped,
  seeds 46/95/97, **6000 steps** (the late-collapse window must be in
  frame: the v2 deaths came at 3700-5700).
- **The single variable vs the v5-d8 record:** SIGReg (Epps-Pulley CF,
  additive `l_pred + 0.2*l_sigreg`, through a Linear head) is REPLACED
  by VISReg (arXiv 2606.02572: scale + sliced-Wasserstein shape +
  center, convex `0.4*l_pred + 0.6*L_Reg`, on TRUNK latents directly).
  NTP is OUT (w_ntp=0): the aux-theorem read (2509.12249) shows its
  protective condition is unavailable at pilot scale, and the cleanest
  swap beats a compound one.
  - Honest accounting: this "single variable" is one MECHANISM but
    three entangled changes (statistic, additive->convex form, head
    removed from the reg path). If the family's result is interesting,
    the decomposition arms exist; they are not pre-committed.
- **Doses: the paper's, adopted as published.** lambda = 0.6 (their
  small-dataset value), component weights 1.0/1.0/1.0, K = 1024
  (their K = C*D, C > 1, at D = 512), eps = 1e-4. No re-derived
  dosing — the week's costliest errors were exactly that.

## The bet, stated once

Every depth-8 death shares one anatomy: the transit comes (universal),
and the rescue never does. The paper's stated motivation is that the
Epps-Pulley gradient VANISHES as the embedding collapses — a mechanism
for "depth breaks the rescue." VISReg's sorted-quantile objective keeps
gradient at the floor (pinned in code:
`test_collapse_gradient_nonvanishing`). The bet is that the rescue
healthy d4 performs 5-for-5 becomes available at depth 8 when the
anti-collapse force still has a gradient down there.

Secondary bet: the **center term** (`||mu||^2`, first-class) attacks
offset dominance — the measured first act of every collapse in the
record — directly, where SIGReg attacked it only through the CF match.

## Measured init magnitude (smoke, recorded so nobody re-derives mid-family)

CPU smoke through the real driver path, 20 steps: heldout
`l_visreg_mean ~ 7.8e5`. The "O(1) by construction" claim in the
feasibility read is **refuted as built** for trunk latents at init:
the shape term divides by per-dim sigma under stop-grad, so the
untrained trunk's offset-over-sigma ratio (the disease itself) is
amplified into the standardized values, and the summed-over-N form
scales it up. Synthetic decomposition confirms the class: offset is
the driver (shape 171, center 16 vs shape 3.5 on clean latents at
matched scale). This is the regularizer seeing the disease loudly,
not a dosing error; expected to fall steeply as the offset dies. The
per-term tape (`l_vis_scale/shape/center`, logged from birth) is the
check — if `l_visreg` has not fallen by orders of magnitude by step
1000, the convex mix is starving `l_pred` and the family reads as
mis-dosed, not as "VISReg fails."

## Frozen predictions (priors in the record, even when wrong — 0-for-4 last week)

1. The early transit still comes (~step 200-600). VISReg does not
   prevent the fall; nothing does, including health.
2. Offset fraction falls faster and further than in any prior d8 run
   by step 1000 (the center term is direct).
3. At least one seed shows a genuine rescue — a recovery to pooled
   eff >= 100 sustained 500+ steps after a sub-50 transit — which NO
   depth-8 configuration has ever shown. This is the property bet; if
   all three seeds transit and stay at the floor with VISReg's gradient
   demonstrably nonzero (l_vis_shape moving, weights changing, rank
   not), the "vanishing gradient" mechanism is refuted as THE cause
   and the floor attractor is deeper than the objective.

## Gates (frozen)

A seed is **HEALTHY** iff it completes 6000 steps un-killed with
pooled effective rank >= 100 at end AND every block >= 50 at end.
A seed is **RESCUE-POSITIVE** iff after any sub-50 transit it recovers
to pooled eff >= 100 and holds it 500+ steps (rescue is the thing
depth broke; a rescue is a result even if the end state degrades).
**Family CONFIRMED at 2-of-3 HEALTHY.** `chorus_stable_rank` recorded
against Brian's 20-target but not gated (the governor arc measured
that gate's difficulty; first question is rescue, not spectral
perfection). Precision front-back divergence tracked per seed (the
standing late-collapse lead from the sweep verdict).

## Guard-calibration notes, registered in advance this time

- Early NMSE will be astronomical (untrained trunk + regularizer-
  dominant loss; smoke read 1662 at step 20). Guard hold 1000 covers
  the ramp. If a seed dies at ~step 1000-1100 with rank still
  CLIMBING, that is the established guard-artifact class — Brian's
  standing extension rule (2026-08-06) applies: analyze, and re-run
  with a longer hold if the tape shows recovery signs.
- No ppl veto (no NTP term); the plain NMSE divergence rule and the
  loss guard are the arbiters, and NMSE's floor-inversion pathology
  (three demonstrations) means any NMSE-trip verdict gets the rank
  tape read before it is believed.

## Launch

    python scripts/jepa_pilot_driver.py --stage 54 --seeds 46,95,97 \
        --epochs 1 --max-batches-per-epoch 6000 --heldout-batches 5

(First launch attempt failed on `--seeds 46 95 97` -- the flag takes a
comma-separated string. Corrected here; no GPU time spent.)

Not launched at registration time; awaiting Brian's go.

---

## VERDICT — scored 2026-08-11, family complete

| seed | outcome | final eff | chorus | stable | tds | heldout NMSE | probe lift |
|------|---------------------------|-------|-------|-------|-------|-------|--------|
| 46 | completed, 6000 | 128.9 | 16.58 | 16.3 | 0.026 | 1.002 | 1.78x |
| 95 | completed, 6000 | 136.8 | 19.73 | 19.9 | 0.023 | 0.504 | 3.75x |
| 97 | killed:nmse=2.71 @4400 | 118.0 | 15.49 | — | 0.031 | — | 1.41x* |

*97's probe ran post-kill on the wrecked-transient state; its step-4000
l_pred (0.074) was the best in the family.

**FAMILY: CONFIRMED, 2-of-3 HEALTHY** (gates as frozen: complete
un-killed, pooled eff >= 100, every block >= 50 — seed 46 blocks
129-203, seed 95 blocks 137-210).

### Frozen predictions, scored

1. "The early transit still comes" — **REFUTED, 3-for-3.** No seed
   ever fell below eff ~99 at ANY step. The universal transit — present
   in every prior run at every depth including healthy d4 — did not
   occur. VISReg did not enable a rescue; it abolished the fall.
2. "Offset falls faster and further" — **CONFIRMED.** top_dir_share
   born at ~0.07 and monotone down to 0.023-0.031; the soloist never
   formed in any seed. The center term killed the first act in the crib.
3. "At least one seed shows a genuine rescue" — **MOOT** (nothing to
   rescue). The property bet (gradient survives collapse) was never
   exercised because collapse never started; the mechanism's value
   showed up as prevention, not rescue.

### New phenomenon, named: SCALE BREATHING

All three seeds show transient std excursions (std50 jumping 0.4 <->
2.4 within ~100 steps) that VISReg pulls back each time — the convex
negotiation working. Seed 97 survived two (steps 2000, 3200) and died
when a third (4400) coincided with a guard checkpoint: nmse 2.71
against limit 2.0, with eff 118 / chorus 15.5 / l_pred 0.074-at-4000 —
healthy geometry killed mid-breath. Per the registered rank-tape rule,
97 reads as a **guard-timing artifact on a recoverable transient**, not
a collapse death. Consequences: (a) the seed-97 class argues for a
transient-tolerant divergence rule (e.g. two consecutive over-limit
checks) — a REGISTERED CHANGE for the next family, not a mid-family
patch; (b) scale breathing's cause (plasticity events? LR-scale
resonance?) is an open forensic.

### Also learned

- Two distinct healthy equilibria: 46 quiet-sharp (std ~0.8, l_pred
  0.28), 95 loud-blunt (std ~1.9, l_pred 1.15, but the BEST heldout
  NMSE 0.504 and probe lift 3.75x). Loudness cost training-loss and
  bought generalization; worth its own read someday.
- Probe lift 3.75x with NO NTP term — the LLM-JEPA family's 4.7x now
  looks mostly geometric, not linguistic.
- The guard-cadence NMSE was invisible in LuthiScope while deciding
  runs' lives (seed-97 lesson); quick evals are now logged and marked
  (`quick: true`) as of this commit.

### Standing

Depth-8 collapse: **SOLVED at 512d, provisionally** (2-of-3, this
family). Per the ledger definition, "concluded working" requires
replication at the ruled **768x8**. That is the next family. NTP
reintroduction (capability layer over stable geometry) and the
scale-breathing forensic queue behind it.

---

## SUPERSESSION NOTICE — 2026-08-13 (Opus 5, post-768 audit)

**This family's scored predictions describe a 6,000-step window, and the
phenomenon they declare absent begins at ~10,000 steps.** The reads below
are not withdrawn — they are correct about what they measured — but they
do not support the conclusions drawn from them, and the Standing above
must be read with this attached. Full evidence: AMENDMENT 3 in
`2026-08-11_768x8-family-spec.md`, Finding (A).

What the 768 tape shows, at this family's own wire step:

| step 6,000, seed 97 (768) | value | reads as |
|---|---|---|
| b0 effective rank | 264.1 | healthy |
| b0 `top_dir_share` | 0.039 | "the soloist never formed" |

Identical in character to this family's 3-for-3 read. Then, in the same
run: `top_dir_share` begins climbing at ~9,600, crosses 0.20 at 13,100,
and ends at **0.919**; effective rank ends at **2.0**. And at step 29,800
that run took a transit (eff 360 → 204, chorus 34 → 5.4) and recovered by
30,100.

Consequences for the scored predictions:

1. **Prediction 1 — "VISReg abolished the fall" does not transfer.** The
   768 run transited at 29,800, ~5x beyond this family's horizon. What
   this family established is that no transit occurs *in the first 6,000
   steps*. "Abolished" was a claim about all time and the evidence covers
   6,000 steps.
2. **Prediction 2 — "the soloist never formed in any seed. The center
   term killed the first act in the crib." SUPERSEDED.** The soloist
   forms; it forms late. It was not killed, it had not been born yet.
   The center term's actual measured contribution is ~0.55% of the
   regularizer at the end of a long run (768x8 spec, Finding (B)), and
   `offset_dominance` is 0.995+ in every block except the one where the
   term is applied.
3. **Prediction 3 — MOOT stands**, for the reason given, but note the 768
   run *did* exercise the property bet at 29,800 and it held: the
   surviving gradient made rescue possible. That is this family's
   mechanism claim vindicated on someone else's tape.

**Standing, amended:** depth-8 collapse is **not** solved at 512d. It is
**unobserved at 512d within 6,000 steps**, which is a different and much
weaker statement. The registered obligation this creates: one 512 VISReg
seed run to 25–30k steps (~5h) before any conclusion about VISReg and
depth-8 collapse is carried forward. Whether the 768 outcome is a width
effect or a run-length effect cannot be attributed until that control
exists.

Recorded per the house rule that supersession notices are cheap and
documented conclusions read as more final than they were.
