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
