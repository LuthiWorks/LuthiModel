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
