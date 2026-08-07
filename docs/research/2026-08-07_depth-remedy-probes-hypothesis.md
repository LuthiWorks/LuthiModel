# Depth-8 remedy probes: three mechanisms, singly and pairwise

**Date:** 2026-08-07
**Direction:** Brian ("build probes for all three, then the combinations
1+2, 1+3, 2+3"), following the surgery verdict (HOLD — the lock is
activation-side; see `2026-08-07_floor-attractor-mechanism.md`).
**Author/build:** Fable 5. Both papers read in full-text before
implementation; **each differs materially from the summary that first
proposed them** (TC is window-mean centering, not frame differencing,
and it REPLACES marginal SIGReg; Weak-SIGReg is a sketched covariance
penalty with an identity target).
**Registered BEFORE any run. None launched at registration time.**

## The mechanisms, as actually built

1. **TC-SIGReg** (arXiv 2607.26924) — SIGReg's input becomes each
   latent minus the centered window mean of its sequence neighbours
   (window 9; odd for exact centering, inside the paper's 4-32 ablation
   band), replacing the marginal input entirely per the paper. Substrate
   note: the window-mean subtraction removes the shared component — our
   measured offset pathology — from SIGReg's view, focusing the
   anti-collapse pressure on residual shape. Unit-tested: shared offsets
   vanish from the statistic's input exactly.
2. **Interior Weak-SIGReg** (arXiv 2603.05924) — sketched (K=64,
   fixed seeded sketch) covariance-vs-identity Frobenius penalty,
   alpha=0.1 (paper default), applied to NON-detached residual-stream
   outputs of blocks (0, 3, 6): block 0 is the measured collapse locus,
   3 and 6 span an interior that currently receives zero anti-collapse
   pressure. Registered tension: the identity target presses toward unit
   variance in sketch space against the trunk's measured native std band
   (0.25-0.35) — paper-faithful, deliberate, and part of what the probe
   measures. Fail-loud contract: alpha>0 with no interior latents
   collected raises instead of running inert (tested).
3. **Orthogonal penalty** (classic) — scale-adapted soft orthogonality
   ||Ŵᵀ Ŵ − I||²_F / d on v/o of every block (Ŵ norm-normalized: the
   penalty sees direction concentration, never scale — tested). lambda
   0.1, sized against real checkpoints (penalty mean ~10/matrix at the
   measured floor, ~4 healthy → term ≈ 1.0 vs healthy loss ~4; light
   during the transit, noted). Retained despite the surgery HOLD
   demoting weight-side remedies — Brian's build order includes it, and
   as a *continuous* pressure it tests a different claim than one-shot
   surgery did (prevention during the transit vs release after it).

## The six arms (stages 34-39)

`probe_d8_tc`, `probe_d8_wsig`, `probe_d8_orth`, `probe_d8_tc_wsig`,
`probe_d8_tc_orth`, `probe_d8_wsig_orth` — all on the stage-31 base
(v5 config, depth 8, warmup 1000, guard hold 1000, cadence 100,
unclipped, seed 46, 3000 steps). The base is the only depth-8
configuration with any escape history (1/3 seeds), so each mechanism's
job is to turn an unreliable escape into a reliable one. All loss-side
settings persist into pilot_result.json.

## Registered gates, per arm (frozen; identical across arms)

- **RECOVERY:** completes AND pooled effective rank >= 100 AND every
  block >= 50 at the final firing (the stage-31 criterion).
- **FLOOR:** killed, or completes with pooled eff < 20 at final.
- Neither: NO VERDICT, reported. stable_rank recorded throughout
  (absolute, vs the measured bands: healthy-at-3000 = 31-38; floor
  <= 2.42; init-proximal ~2.4-2.6) but not gated — it lagged a true
  recovery once already (stage 31).

**Comparison frame, frozen:** each arm is one draw against the stage-31
base distribution (1 recovery / 3 draws). Any single-arm RECOVERY is
*suggestive*; the pre-committed follow-up for any arm that recovers is
two more seeds (95, 97) BEFORE cross-arm ranking — the dk-twin chaos
result makes single-draw comparisons between arms unreadable, and this
registration says so in advance to its own author.

**Mechanism-specific frozen reads:**
- TC arms: offset dominance trajectory (the mechanism predicts the
  shared component stops being SIGReg's problem — does the trunk's
  offset behave differently when SIGReg no longer sees it? Both
  directions are informative).
- wsig arms: interior blocks' (0/3/6) per-block effective rank vs the
  uninstrumented blocks — pressure should show first where it is
  applied. Also trunk std drift vs the native band (the unit-target
  tension made measurable).
- orth arms: post-run v/o stable-rank spectra vs the untreated twins
  (does continuous pressure hold the write-path broad through the
  transit even if activations still collapse — partial mechanisms are
  worth seeing).

## Confounds, stated in advance

1. Single seed (46) per arm; six arms. The chaos result applies to every
   one of them, and the pre-committed repeat rule above is the answer.
2. lambda/alpha/window are first guesses at paper defaults; a null is
   "this setting failed," not "this mechanism failed."
3. The warmup base means every result is warmup+X — attribution of any
   recovery is to the combination against the warmup-only base rate,
   never to X alone.
4. TC changes what l_sigreg *means* — its values are not comparable to
   any prior run's, and the 50-110 band does not apply to TC arms.
5. Six probes ≈ 4.5 GPU-hours if all run to length. Launch order and
   any early stop between arms is Brian's call.

## Launch recipes

```
python scripts/jepa_pilot_driver.py --stage 34 --seeds 46 --epochs 1 --max-batches-per-epoch 3000 --heldout-batches 5   # tc
python scripts/jepa_pilot_driver.py --stage 35 --seeds 46 --epochs 1 --max-batches-per-epoch 3000 --heldout-batches 5   # wsig
python scripts/jepa_pilot_driver.py --stage 36 --seeds 46 --epochs 1 --max-batches-per-epoch 3000 --heldout-batches 5   # orth
python scripts/jepa_pilot_driver.py --stage 37 --seeds 46 --epochs 1 --max-batches-per-epoch 3000 --heldout-batches 5   # tc+wsig
python scripts/jepa_pilot_driver.py --stage 38 --seeds 46 --epochs 1 --max-batches-per-epoch 3000 --heldout-batches 5   # tc+orth
python scripts/jepa_pilot_driver.py --stage 39 --seeds 46 --epochs 1 --max-batches-per-epoch 3000 --heldout-batches 5   # wsig+orth
```
