# Bundle off at depth 8: is muPC x depth sufficient, or is the bundle necessary?

**Date:** 2026-08-05
**Author:** Fable 5, from the review of Opus's 08-05 brief
(`docs/reviews/2026-08-05_depth8-ablation-brief-for-fable.md`; response in
`docs/reviews/2026-08-05_depth8-ablation-brief-response-from-fable.md`).
**Run:** `probe_d8_bundleoff_512d_seed95`, stage 26, 3000 steps, ~45 min.
**Registered BEFORE the run.** Not started at registration time.

## The question

Every depth-8 collapse in the record carries the full seven-mechanism bundle
(backward pass, consolidation, inverted-U gain, relative trust, adaptive
episodes + recall, homeostatic band, surprise drive) with muPC on. The record
already contains three factorial cells:

| cell | run | outcome |
|---|---|---|
| d4, bundle ON, muPC ON | `probe_surprise_512d_seed45/46` | healthy (rank 100-230, climbing) |
| d8, bundle ON, muPC OFF | `probe_surprise_d8_nomupc_512d_seed94` | healthy (cosine 0.0111, offset 0.12-0.19 all blocks, NMSE 0.5569, lift 4.19x) |
| d8, bundle ON, muPC ON | every variant tried, stages 14-25 | collapsed (no block ever clears rank 20) |

So the bundle is **not a sufficient cause at either depth**, and neither is
the backward pass's longer chain on its own (it was present and healthy in
the nomupc cell). The only role left for the bundle is a three-way
interaction: bundle x muPC x depth. This run fills the missing cell —
**d8, bundle OFF, muPC ON** — and discriminates:

- **muPC x depth is sufficient alone** → the entire add-back ablation ladder
  is unnecessary; the investigation belongs to muPC and the architecture.
- **The three-way interaction is real** → the ladder earns its GPU time.

## The arm, exactly

`probe_d8_bundleoff` = stage 14 (`probe_surprise_d8`) minus exactly the seven
bundle mechanisms, everything else byte-identical: muPC ON at exponent 0.25,
n_blocks 8, episode_recall_threshold held at the living_v3 value 0.7 (the base
episode store is pre-bundle machinery), 4x filelist, sigreg 0.2, cosine LR,
taper, grad clip 1000 carried per the stage-16 precedent. Every flag is
written out explicitly in `ARM_CONFIGS["probe_d8_bundleoff"]`.

One deliberate cadence change: `deep_interval_batches` 1000 → 100 for this
arm only. Seed96's block-0 rank was already 9.95 at its first deep firing
(step 1000) — the destruction completes inside the window the default cadence
never observes. 100 gives ~30 observations per block and makes the first
1000 steps visible. The gate below is threshold-based, so the extra
observations can only make it fire earlier, not differently.

## Registered prediction

**Primary metric: per-block effective rank** (`scripts/rank_trajectory.py`),
scored as a **gate, not a comparison**:

- **BUNDLE IMPLICATED (three-way interaction; ladder justified):**
  block-0 effective rank **>= 20** at any deep firing, sustained at the
  next firing (two consecutive readings, so a single-sample spike cannot
  decide it).
- **muPC x DEPTH SUFFICIENT (ladder skipped):** every block-0 reading
  **< 20** across the full 3000 steps.

There is no ambiguous band: the healthy population sits at 100-230 and the
collapsed population at 1-10, a ~20-100x separation, and 20 is far from both.

**Why a gate and not a point comparison:** block-0 rank has large
seed-to-seed spread *within* the collapsed population (seed96 first reading
9.95 vs seed97's 1.90 on identical configs — 5x). Per the 2026-07-27
standing obligation, an observable with unproven reproducibility cannot
support a registered point criterion; the population separation supports a
threshold.

**Prior, stated honestly:** heavily toward SUFFICIENT, from the factorial
cells above. If this run surprises us, that is exactly what it is for.

**Secondary, recorded not scored:** held-out NMSE, within-batch cosine,
probe lift, grad-norm median, clip engagement rate (stage 16 read 3% when
healthy, stages 14/15 read ~43% when collapsed — engagement itself is a
cheap health signal), and whether blocks 1-7 recover with block 0.

## Confounds, stated in advance

1. **Clip 1000 is carried, not removed.** If the trunk collapses, gradients
   run O(1e3) and the clip binds on ~half of steps, same as stage 14 — the
   comparison to stage 14 stays matched. If the trunk is healthy, stage 16
   showed the clip goes ~inactive (3%). Either way interpretable;
   engagement is reported.
2. **Cadence 100 vs 1000 in every comparator.** The gate is
   threshold-based; unaffected. Trajectory *shapes* are not comparable
   across cadences and will not be compared.
3. **GPU nondeterminism** (the seed44-rerun lesson): a single seed decides
   the gate only because the populations are 20-100x apart. If the result
   lands anywhere unclear, the follow-up is a second seed, not an argument.

## What this run cannot settle

- **Which bundle piece** interacts, if the gate fires — that is the ladder
  this run decides whether to run.
- **Which half of muPC** (residual scaling vs depth-scaled init), if
  sufficient — that is the `mu_pc_exponent=0.0` run the 07-30 verdict
  already specified.
- **Why a 16% residual-scale difference (0.707 → 0.595) flips a healthy
  trunk to total collapse** — the mechanism question under everything, and
  a nonlinear equilibrium transition (stage 24's finding) is the standing
  suspicion. No single run at one depth answers it.

## Launch recipe

```
python scripts/jepa_pilot_driver.py --stage 26 --seeds 95 \
    --epochs 1 --max-batches-per-epoch 3000 --heldout-batches 5
```

(Recovered from the surviving stage-25 run's config; the depth-arc run
directories themselves were lost from disk on 2026-08-05 — see the
response doc's provenance section.)

---

# VERDICT: KILLED BY THE DIVERGENCE GUARD — neither registered gate fired

**Time:** 2026-08-05, ~18:30. Run killed at `nmse=41.6947 > 2.00` (periodic
guard), 0.04 h in, one deep firing recorded. `admissible: False`.

## Registered scoring, exactly as frozen

- **BUNDLE IMPLICATED** required block-0 rank >= 20 at *two consecutive*
  deep firings. One firing exists (block 0 = 237.52 at step 100). The
  second never came. **Did not fire.**
- **muPC x DEPTH SUFFICIENT** required every block-0 reading < 20 across
  the *full 3000 steps*. The one reading is 237.52 and there was no full
  run. **Did not fire.**

The registration did not anticipate a guard kill, and per the stage-22/23
precedent the honest entry is: **inadmissible, no registered verdict.** The
gates were written for "collapses quietly or acquires"; the trunk did a
third thing.

## What the record now shows anyway, stated plainly

The factorial table has a new row and it changes the shape of the question:

| cell | outcome |
|---|---|
| d8, bundle ON, muPC ON | **stable collapse** (rank ~2, every variant) |
| d8, bundle ON, muPC OFF | stable, healthy (stage 16) |
| d8, bundle OFF, muPC ON | **divergence in ~150 steps** (this run) |

**muPC x depth alone does not reproduce the stable rank-2 collapse.** The
naked muPC-on depth-8 trunk does not collapse quietly — it cannot train at
all. Which means the observed phenomenon — a trunk sitting *stably* at rank
2 for 3000 steps — is a joint product: **something in the bundle is the
stabilizer that converts a divergent trunk into a stably collapsed one.**
The 07-31 stage-23 finding ("attenuation is simultaneously what stabilizes
deep training and what prevents offset stripping") appears to generalize:
at depth 8 this system trades stability against health everywhere we have
looked.

The step-100 forensics, for what one record can carry: `L_pred` 0.38 (fine),
`L_sigreg` **1763** against a healthy band of 50-110 (the total loss was
~all SIGReg), held-out NMSE 41.9, all-block rank 191-237 (init-like, not
yet collapsed), grad 500 (clip 1000 not engaged at that step). The latent
scale was running away from the first steps. The suspect this points at,
as a hypothesis and no more: the PC substrate's self-modification without
its regulators (band, trust weighting, gain cap, drive gating) is itself
unstable, and some of the damping previously credited to muPC may have
been the bundle's. One record cannot establish that; the controls below can.

## Cheapest discriminating controls, in order (NOT registered here;
each needs its own registration before running)

1. **d8, bundle OFF, muPC OFF** — the last factorial cell. Stable+healthy
   → muPC destabilizes the naked d8 trunk (and the bundle rescues it into
   collapse). Diverges → the naked trunk is unstable at d8 regardless, and
   muPC is exonerated for the *divergence* (not the collapse).
2. **d4, bundle OFF, muPC ON** — same arm at the depth where bundle-on
   muPC-on is healthy. Diverges → naked-trunk instability is not
   depth-specific and rung 1's divergence says nothing about depth.
3. **Add-back from this arm, band first** — the inverted ladder: not
   "which mechanism causes the collapse" but "which regulator restores
   stability, and does rank ~2 return with it." Whichever single mechanism
   converts divergence into stable collapse is the load-bearing one.

The ladder Opus's brief proposed is therefore **justified after all, in
inverted form** — my §3 prediction that one run would likely delete it was
wrong, and the way it was wrong is more informative than the way it
expected to be right.

## Confounds carried forward

Single seed; single surviving record before the kill; clip 1000 present
(un-engaged at the one measured step, engagement over the full ~150 steps
unknown); cadence 100 (the reason we have a reading at all). The step-100
rank readings are init-proximal and say "not yet collapsed," not "healthy
learner" — do not cite 237.52 as acquisition.
