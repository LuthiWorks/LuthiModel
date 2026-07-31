# Cascade check: our runs vs the JEPA-in-language paradox

**Date:** 2026-07-31, ~13:30
**Script:** `scripts/cascade_check.py` (eval-only, existing logs and results)
**Against:** arXiv:2607.23531, "The JEPA Paradox in Language" (2026-07-26)

The paper documents a failure cascade for squared-error latent prediction on
text -- effective-rank degeneration, cosine collapse, elevated target variance,
train/val instability, MI saturation, degraded downstream. We train a JEPA on
text and we plainly show some of it. The question was WHERE.

## Answer: it is our configuration, not the modality

| configuration | eff rank | trivial cos | probe lift |
|---|---|---|---|
| **depth 4, fixed objective** (storefix45, surprise45/46) | **167 - 182** | 0.63 - 0.69 | **4.67 - 4.80x** |
| depth 8, muPC OFF | 71.5 | 0.684 | 4.19x |
| depth 8, muPC ON (every variant tried) | **1.2 - 3.6** | 0.09 - 0.90 | 1.0 - 2.2x |

Depth 4 runs the same corpus, the same objective, the same everything but depth,
and is healthy. If the language paradox were driving this, depth 4 would degrade
too. It does not.

**The paper describes symptoms we share. It is not describing our disease.**

A real background trend does exist and is worth keeping: rank falls with depth
even without muPC (182 -> 71.5). Depth costs rank progressively; muPC at depth 8
turns that decline into a collapse.

## The memo's open lead #2 is CLOSED

The memo lists `trivial cosine 0.9975` and "the prediction task looks too easy"
(07-27 note) as a live lead. The data shows the 2026-07-28 objective fix already
resolved it:

| arm | objective | trivial cos | probe lift |
|---|---|---|---|
| probe_storefix 42/43/44 | pre-fix (BatchNorm + non-detached target) | 0.98 | 1.43 - 1.50x |
| probe_storefix 45 | fixed | **0.629** | **4.67x** |

Same arm, same seed family, one change. The whole pre-fix corpus of families
(living_v3, v4, v5, living_full) sits at trivial cosine 0.984 - 0.997 with lift
1.14 - 1.70x -- i.e. the "prediction too easy" pathology was general to the old
objective and is gone under the new one.

## Correcting myself: effective rank cannot stand alone

Two hours ago I wrote that effective rank "should be the primary metric for every
depth run from here" and that it "cannot be gamed by degeneracy". The first half
is wrong.

| arm | eff rank | trivial cos | probe lift |
|---|---|---|---|
| probe_storefix 42/43/44 (old objective) | **289 - 290** | 0.98 | 1.45x |
| probe_storefix 45 (fixed objective) | 167 | 0.63 | 4.67x |

**Higher rank, worse capability.** A representation can span many dimensions
while the predictor remains trivial -- that is a different pathology from rank
collapse and rank alone does not see it.

I over-corrected: NMSE was gamed by degeneracy, so I promoted the metric that
measures degeneracy, and made the same error in a new direction. **The honest
read needs three numbers together:** effective rank (is the space used), trivial
cosine (is the predictor doing work), probe lift (does the representation carry
recoverable information). Any one alone has a failure mode the other two catch.

## Two distinct pathologies, now separable

1. **Old-objective pathology** -- rank high (289), trivial cosine ~0.98, lift
   ~1.45x. The space is spread but the predictor is trivial. Fixed 2026-07-28.
2. **Depth-8 muPC pathology** -- rank ~2, lift ~1.0-2.2x. The space itself has
   collapsed. Open.

They are not the same failure and they do not respond to the same fixes.

## Caveat on coverage

`effective_rank` was instrumented on 2026-07-28, so every family before that date
reports `nan` for it. The historical comparison here rests on trivial cosine and
probe lift, which are available throughout. A depth-4 vs depth-8 rank comparison
under the OLD objective is not available and cannot be reconstructed without
re-running.

MI saturation is not instrumented and was not estimated. Five of six symptoms
checked.
