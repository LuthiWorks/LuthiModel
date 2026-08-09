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

---

# FAMILY VERDICT (real run, NTP live): NOT CONFIRMED at 3000 steps — and the first REPRODUCED capability-positive depth-8 result in the record

Gates: 0-of-3 HEALTHY (rank gate missed in all; l_ntp halving narrowly
missed at 0.57 on seed 46 — note the field lives at record top level,
not in `light`; my first scoring script read -1 and is corrected here).

The substance, which the gates were deliberately too strict to flatter:
**seeds 46 and 95 reproduce each other** — completed, eff 64-67 climbing
monotonically from the transit floor to end-of-run, perplexity 294/259
(chance = 32000), probe lift 4.77x/4.74x (healthy-d4 territory, the two
highest ever at depth 8), soloist share pressed to 0.19/0.18. Seed 97
died late and marginal (2.13) with lift 3.40x even so. This is the
first time ANY capability-positive depth-8 signature has reproduced
across seeds. The bet's mechanism reads exactly as designed: NTP holds a
rescue path open; the climb is real, monotone, and runway-limited — the
3000-step horizon ends every tape mid-recovery.

**Registered follow-up (the runway family):** `probe_d8_llmjepa6k`
(stage 51) — identical arm, 6000 steps, same four gates evaluated at the
6000-step final firing, seeds 46/95/97. If the monotone climb is real,
doubling the runway should carry eff past 100 and the blocks past 50;
if it plateaus sub-gate, the recovery has a ceiling and v2 goes to the
dose/parametrization questions (§B/§C) with that measured fact.

---

# RUNWAY FAMILY VERDICT + THE TWO-GAUGE AMENDMENT (2026-08-08, afternoon)

**Stage 51, first attempt: 0-for-3 — all executed at marginal NMSE
(2.41/2.08/2.54) before step 1900, carrying lifts 4.23x/2.62x/3.67x at
death.** The runway question was never answered; the family was killed
by a guard whose founding claim ("NMSE > 2.0 is broken regardless of
objective") is falsified by its own victims: seed 46 died at 2.41
holding a 4.23x probe lift. Fourth instrument of the arc caught judging
a new regime by an old regime's physics.

**Amendment (registered, arm-gated, tested):** the two-gauge execution
rule — for arms with w_ntp > 0, a marginal NMSE trip is VETOED, loudly,
when the same divergence probe reads held-out perplexity below 8000
(a quarter of the 32k chance level). Genuinely broken generation does
not veto; nonfinite NMSE is never vetoed; the independent 10x-loss guard
still kills catastrophic runaways unilaterally; every non-NTP arm keeps
the old rule bit-exactly. This is not "the guard stops saying things we
dislike" — it is "execution requires both objectives' gauges to agree
the run is dead," and it was built only after the old rule produced a
measured counterexample to its own justification.

Pre-veto runs preserved as *_preveto in the closed folder. Family
relaunched under the amended rule; gates unchanged.

---

# CORRECTION (minutes after relaunch): the veto as BUILT is broader than as REGISTERED

The amendment text says "a MARGINAL NMSE trip is vetoed." The code
implements no marginality: ANY trip is vetoed below the perplexity
bound, and the first live veto paroled nmse=279.8 (ppl 2039) — a
mid-transit JEPA free-fall, not a marginal disagreement. The label/
behavior gap is the same class the provenance assert closes, here in
the safety amendment itself, written at speed. Owned by the design seat.

**Disposition (chosen over killing a live python mid-run):** the family
runs to completion under the as-built rule and is RECLASSIFIED as the
combined-objective's uncensored observation run — the first time the
new objective's NMSE trajectory will ever be seen past a trip. The
veto-line stream is the measurement: if paroled runs recover, the old
guard was executing recoverable transits; if they decay until the
independent loss guard shoots them, NMSE was an honest early warning.
The v2 two-gauge rule (with a marginality ceiling, ~2x the limit) gets
written from these measured joint trajectories, not from intent.
Completion-dependent gate readings from this family carry an asterisk:
"completes" under the as-built rule is weaker than under the registered
rule. Rank/perplexity gate components are unaffected.

---

# RUNWAY FAMILY VERDICT (observation-grade, two-gauge rule): 3-for-3 completions, 0-for-3 gates — and the diagnosis

First depth-8 family ever to complete all seeds. Final states:
46: eff 65 / min 54 / ppl 1050 / lift 4.23x (plateaued-sagged);
95: eff 110 / min 13 / ppl 167 / lift 4.81x (late surge, chorus 5.75
rising 4x in the final 200 steps at cutoff);
97: eff 79 / min 27 / ppl 204 / lift 4.77x.
Each seed missed a DIFFERENT gate criterion; all passed completion and
perplexity. The two-gauge rule paroled seed 46 through NMSE spikes to
347 and delivered three complete tapes; no paroled run decayed into the
LM-with-dead-JEPA end state (all recovered geometry to eff 65-110).

**The measured diagnosis (Brian's "are the knobs right?" — they were
not):** NTP held 78-85% of the loss at the FIRST firing and 98-100%
from step 1100 on. The registered "graceful handoff" never existed;
l_sigreg fell its hundred-fold within the first few hundred steps. The
family ran as language models with a vestigial (1-2%) embedding
objective — hence capability robust in every draw while geometry
limped. Opus's SB warning was correct and understated; the design
seat's option-(a) ruling is refuted by its own recorded read.

# V2 REGISTRATION: JEPA leads, NTP anchors (the original intent, first time actually run)

`probe_d8_llmjepa_v2` (stage 52): identical arm, **w_ntp = 3** — sized
from the measured settled magnitudes (l_ntp ~5, JEPA side ~20 →
NTP ~35-40% settled share; ~2% through the early window, which belongs
to SIGReg's shaping per the original design). Same gates, same
two-gauge rule, seeds 46/95/97, 6000 steps. Confounds: the veto's
as-built breadth carries (marginality ceiling deferred until this
family's uncensored trajectories are analyzed); single dose point — if
v2 under-anchors (capability collapses back toward pure-JEPA outcomes),
the truth lies between 3 and 400 and a bisection family follows.
