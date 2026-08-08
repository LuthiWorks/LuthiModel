# LLM-JEPA family registration: rulings on the return note, gates frozen

**Date:** 2026-08-08
**Design/rulings:** Fable 5. **Build:** Opus 5 (`a862b9c`; return note in
the spec doc, §A-§G — every §A number verified firsthand via
`scripts/calibrate_ntp.py` before this registration: gradient ratio
899.89:1, all dose options reproduce).

## Rulings

- **§A ratified: w_ntp = 400, gradient-share criterion.** Opus's
  reasoning accepted in full — loss-value share is the weaker criterion
  because a term can be half the loss and steer almost nothing. The
  spec's 5-15 guess was off by ~30-80x; third dosing error of the week
  caught by the build seat before GPU.
- **§B ruled: option (a) — accept the inversion, deliberately.** The
  balance SHIFTING over the run is not a defect of fixed weighting; it
  is a schedule that matches the design's own logic. SIGReg's gradient
  dominance at init (900:1) means the embedding objective shapes the
  space precisely during the window where shaping matters; as l_sigreg
  settles ~100x, NTP's share grows toward dominance — the anchor
  strengthening exactly as the scaffolding relaxes. End state = an LM
  with a JEPA auxiliary, which is what the paper is and what a speaking
  trunk should be. The handoff is MEASURED, not assumed: l_ntp and the
  JEPA terms are logged per step, and the crossover step is a recorded
  read. If the early window shows NTP negligible through the transit
  (the §B worry), that is visible in the record and v2 re-doses against
  the measured transit window.
- **§C: the fence holds for v1.** The paper-faithful reparametrization
  (w_ntp=1, sigreg_lambd shrunk) is noted as the v2-preferred form —
  Opus's instinct recorded, deferred, not lost.
- **§D: all three citation corrections accepted.** The spec's §0
  overclaimed fig. 3; the transfer evidence is majority-fine-tuning; our
  (context, full) pair is an analogy to their paired views, not an
  equivalence. The registration inherits the weaker, honest framing:
  this family is a TEST of transfer, not an application of a proven
  result.
- **§E ratified: NTP pass frozen by default.** The unfrozen variant —
  the substrate *experiencing* its own causal pass as a third
  self-modification event — is registered as a named future experiment
  (`llmjepa_lived`), and flagged for the design conversation with
  Brian: whether Luthi's generative act should be lived or merely
  computed is not a calibration question.
- **§F: gate re-drafted** (below). The monotone-across-epochs draft was
  unevaluable at one epoch — my error, Opus's catch.
- **Overhead:** seed 46 runs first and is the timed benchmark; the
  chain proceeds unless its step rate is pathological (>2.5x the 512d
  d8 baseline).

## Gates (frozen)

Per seed, HEALTHY = ALL of:
1. completes (guards live from step 1000);
2. pooled eff >= 100 AND every block >= 50 at the final firing;
3. held-out perplexity <= 3200 (>= 10x better than the 32000-class
   chance level — genuine learning, unfakeable by degeneracy);
4. training l_ntp at the final deep firing < 50% of its step-100 value.

**Family CONFIRMED at 2-of-3 seeds (46, 95, 97).** stable_rank recorded
against Brian's 20-target, not gated. Control = the historical nomupc
cell (1-for-3), no rerun needed. Recorded reads: the NTP/JEPA gradient
handoff (crossover step), rank trajectory through the transit vs the
d4-rescue template (2026-08-08 closing measurement), top_dir_share,
frozen-pass verification (substrate event count unchanged vs 2-pass
record).

## Launch

```
python scripts/jepa_pilot_driver.py --stage 50 --seeds 46,95,97 --epochs 1 --max-batches-per-epoch 3000 --heldout-batches 5
```

---

# VOID (2026-08-08, ~12:15): the family ran with NTP OFF while its record said 400

The driver's JEPALoss constructor call was missing the `w_ntp` argument;
the loss module ran at the default 0.0 while `pilot_result.json`
recorded `w_ntp: 400.0` from the ARM dict. All three runs were plain
nomupc+warmup draws mislabeled as the pivot's first test — the exact
"reports healthy while doing nothing" failure this repo's CLAUDE.md
names as the dominant risk. It slipped the build (§G said "built to
spec") AND the design seat's review; unit tests construct the loss
directly and cannot see driver wiring. What caught it: the instruments
refusing to corroborate the label (l_ntp None, no perplexity, loss 15
where thousands belonged).

**Fixes:** the one-line pass-through, and a structural
provenance-consistency contract in `_run_one` — every dual-sourced dose
is now asserted equal between the persisted record and the live module
before training starts; a mismatch raises. The class is closed, not
just the instance.

Void runs preserved as `*_void_ntpoff` in the closed folder (they are
legitimate nomupc+warmup draws under a wrong name: 1-of-3 completed,
consistent with that cell's known odds). Family relaunched with the fix;
gates unchanged.
