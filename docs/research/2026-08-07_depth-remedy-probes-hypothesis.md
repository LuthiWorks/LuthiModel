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

---

# SINGLES VERDICT (stages 34-36): three FLOORs, one real clue, two under-doses

| arm | outcome | gate | the mechanism-specific read |
|---|---|---|---|
| tc | killed @ ~1000, nmse 2.72 | FLOOR | **Block 0 held rank 209-239 the entire run — first time in 13 depth-8 runs.** Interior oscillated violently (min block 2.6→119→185→1.4; SIGReg 20↔1600) and died. With the offset removed from SIGReg's view, block 0 stops being crushed — the July offset-fight mechanism confirmed from the reverse direction. |
| wsig | killed @ ~2100, nmse 3.82 | FLOOR | **Differential null:** pressured blocks 0/3/6 collapsed in lockstep with unpressured. Transit delayed (400-700) and 1000+ live-guard steps survived in the healed-offset floor state. |
| orth | killed @ ~1100, nmse 2.38 | FLOOR | Carve happened anyway: post-run block-0 v/o at 5-6 despite continuous pressure. |

**Sizing self-correction, owned:** wsig's alpha=0.1 was taken on paper
faith (tuned there for supervised losses of order 1); against our
transit-era losses of 100-500 the penalty contributed ~1 — plausibly
100x under-dosed, predicting exactly the observed null. Orth's term was
likewise ~1 during the transit. The singles tested the mechanisms at
homeopathic doses; only TC (which redirects the existing, full-strength
SIGReg rather than adding a small term) was dosed at strength — and it
produced the day's one structural change.

**tc_wsig conditional (Brian's):** NOT launched — the frozen criterion
(wsig differential) read null. Superseded by the dose ladder below, per
Brian's ruling to explore settings before retiring mechanisms.

---

# DOSE LADDER (stages 40-42), registered before launch

Same base, same gates, same frozen reads as the singles. One change per
arm, sized against measured loss magnitudes rather than paper defaults:

- **Stage 40 `probe_d8_wsig1`:** alpha = 1.0 (penalty ~10 vs healthy
  loss ~4; still small in transit).
- **Stage 41 `probe_d8_wsig10`:** alpha = 10.0 (penalty ~100 —
  commensurate with transit-era loss; the properly-sized dose under the
  sizing argument).
- **Stage 42 `probe_d8_orth1`:** lambda = 1.0 (term ~10 at the measured
  floor spectra).

**Frozen decision rules:** wsig differential (pressured vs unpressured
block rank) appearing at any dose → the mechanism grips → tc_wsig runs
at that dose under Brian's standing conditional. Differential absent
even at alpha=10 → earned retirement of interior-Weak-SIGReg for this
substrate (a dose that co-dominates the loss and still cannot separate
the blocks it directly pressures has been given its full chance). Orth
read: post-run write-path spectra vs the lambda=0.1 twin. All single
seed (46); the repeat rule from the singles registration carries.

---

# LADDER RUNG 2 (wsig alpha=10): the collapse ARRESTED — recovery-class, NO VERDICT by 0.9

**Outcome:** completed, guards live from 1000. **The deep transit never
fully happened** — first run in the record: min block rank bottomed at
~22-33 (vs 1-2 in all 16 prior d8 runs), then climbed 2500 steps to a
final state of all blocks 62-142, offset 0.135, SIGReg ~12-22, probe
lift **3.67x** (vs stage 31's 4.33x, stage 16's 4.19x — real signal,
not flattery). Scored against the frozen gates exactly: **NO VERDICT** —
pooled eff read 99.1 at the final firing against the gate's >= 100,
having sat at 111-120 for the prior thousand steps; and the
**differential read is null again** — unpressured blocks rose in step
with pressured ones. The mechanism grips globally (three blocks'
gradients reach the whole trunk through attention), not locally.

**Correction to this doc's own narrative:** the "every counter-force
gets absorbed" reading from the singles was a dosing artifact. At
loss-commensurate dose, activation-side pressure *prevented* the deep
collapse rather than being absorbed. Prevention, not escape — a new
outcome class.

**Consequence (Brian's standing conditional):** tc_wsig runs at the
gripping dose. Stage 43 `probe_d8_tc_wsig10` = TC (window 9, replacing
marginal) + interior wsig alpha=10 — the block-0 protector plus the
whole-trunk arrestor, both at strength. Same base, same gates, seed 46.
Queued behind the in-flight pipeline (orth1, then the 0.1-dose pairs).
Pre-committed: if it recovers, seeds 95/97 before any conclusion
hardens; and wsig10 itself owes the same repeats regardless.

---

# REFERENCE: RoBlock (ICLR 2026 submission) — the real source behind "R1UN"

Verified via Brian's browser (OpenReview challenge-walls automated
access): **"RoBlock: Wide and Deep Scaling of Recommenders via Embedding
Collapse Mitigation"** (openreview.net/forum?id=Tuxg7dcg3a; anonymous
code at anonymous.4open.science/r/RoBlock-2F8A). Provenance note: an
earlier search summary presented its "rank-1 update normalization"
component remixed into this project's vocabulary (transformer blocks,
SIGReg) — the citation was real, the framing was not. Domain is
recommender-system embedding tables, but the core concern (depth-wise
collapse intensifying with model depth, unfixed by input-layer remedies)
is structurally ours.

Transfer candidates for the design read (Opus):
1. **R1UN spectrum rebalancing** — cheap approximate spectral fix, no
   per-step SVD; answers the cost objection to activation-spectrum
   rebalancing at block boundaries.
2. **HSIC-guided decoupling** — independence, strictly stronger than the
   covariance decorrelation of our wsig penalty, which at alpha=10 is
   now measured to arrest the collapse. A principled upgrade path from
   a mechanism that already grips.
3. Field-wise multi-head router — rec-sys-specific; likely no transfer.

Sits on the design shortlist alongside TC-family objective reshaping and
trunk-interior decorrelation (of which HSIC is the strong form), with
Muon as the odds-shifter class behind them.

---

# PAIRS + tc_wsig10 VERDICT: weak pairs all FLOOR; the strength pair ANTI-COMPOSES

**Weak-dose pairs (Brian's original build, stages 37-39):** tc_wsig
killed nmse 403 / probe 0.000; tc_orth killed nmse 43.5; wsig_orth
killed nmse 2.86. All FLOOR, as the dosing analysis predicted.

**tc_wsig10 (stage 43, both winners at strength): FLOOR — fastest
collapse in the record.** Rank 2.8 at the FIRST firing (every other run
starts ~215 init-proximal); killed at first live check, nmse 284. The
combination is not additive — it is destructive. Working hypothesis for
the record (untested): wsig10's arrest depended on co-pressure with
marginal SIGReg; TC removes marginal SIGReg by construction, leaving
the interior identity-target to fight prediction alone. Whatever the
mechanism, the measured lesson is loud: **mechanisms that work alone do
not compose by default. Every combination is its own experiment.**

**Day's standing tally (all seed 46, single draws):** arrest-class:
wsig10 alone. Recovery-class: warmup alone (1/3 seeds). Partial
protection: tc alone (block 0). Floors: everything else including all
four combinations tried. Next per the battery protocol: repeats of
wsig10 (seeds 95/97), the scheduled-muPC arm (running), then the
width-ratio rung (Brian's aspect-ratio hypothesis: d8@512 halves
width-per-depth vs both the healthy d4 cell AND the production
4096x36 shape — d8@768 or @1024 tests shape-vs-depth).
