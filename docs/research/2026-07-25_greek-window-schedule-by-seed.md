# Greek-page serving schedule, per seed (v5 family analysis aid)

**Date:** 2026-07-25
**Author:** Fable 5 (cross-line seat), at Brian's request
**Status:** Descriptive companion to the 2026-07-24 amendment's frozen seed44
prediction. Nothing here alters the registered read; it locates the probe
events the registered read will be laid against.

## Method

Exact loader replay, reconstructed from `luthi/v2/multimodal_data.py`:

- Corpus: the cached 4x tensor (`b623e9aa…​.pt`, 50,183,452 tokens, 482 files).
- Greek detection in token space: 563 of 32K vocab pieces decode to text
  containing Greek/Coptic or Greek-Extended codepoints (U+0370–03FF,
  U+1F00–1FFF); per-window density = Greek pieces per 128-token window.
- Split arithmetic: `compute_text_split` with seq_len 128, stride 64,
  holdout 0.02 → **n_train_sequences = 768,431** (~24,013 steps/epoch at
  batch 32; 3 epochs ≈ the 72,042-step run).
- Shuffle: `perm_e = randperm(768431, gen)` with
  `gen.manual_seed((seed ^ ((e·0x9E3779B97F4A7C15) & 2⁶⁴-1)) & 2⁶³-1)`,
  starts = perm·stride; batch *k* serves concatenated-stream slots
  `[32k, 32k+32)`; step = slot//32 + 1. (Replicates the loader's operator
  precedence exactly: `&` binds before `^`.)

**Validation anchor:** the replay must reproduce v4 seed44's documented
Greek-page serving at **step 58650** (attributed 2026-07-24 to PG11130
"Greek in a Nutshell"). It does, at every density tier examined — an
isolated single-sequence serving, epoch 2, step 58650.

## The probe

The extreme tier (window ≥ 80/128 Greek pieces) is **12 servable sequences**,
all inside PG11130's polytonic grammar core, tokens [38,864,128 .. 38,869,440]
— peak density 93/128. Each is served once per epoch → **36 extreme servings
per 72K-step run**, schedule fixed by seed number alone (identical for v4 and
v5 runs of the same seed).

## Extreme-tier (≥80/128) serving steps, per seed

- **seed 42:** 942, 4220, 4592, 8323, 9535, 10997, 13839, 13890, 15851,
  20138, 22100, 23015, 24881, 29090, 29986, 32028, 32619, 32676, 36139,
  36865, 39179, 41653, 42315, 42565, 50095, 51580, 53992, 54350, 55053,
  59145, 61356, 61627, 63439, 65390, 65779, 66512
- **seed 43:** 828, 3717, 3942, 6886, 7613, 11158, 12866, 13670, 14920,
  17871, 22263, 22354, 28686, 29695, 33044, 34105, 35929, 37643, 38605,
  38970, 41713, 44353, 46605, 46822, 48305, 49261, 50464, 51371, 51429,
  58793, 59618, 63796, 66516, 67264, 69345, 69787
- **seed 44:** 640, 1834, 2183, 6769, 9331, 11800, 14458, 14477, 20968,
  21102, 23103, 23990, 24339, 28688, 35160, 36827, 39186, 39523, 40601,
  40788, 42090, 42681, 46223, 47740, 48237, **58650**, 63670, 64387, 65538,
  67856, 68232, 68366, 68939, 69331, 69505, 71602
- **seed 45:** 1950, 8493, 8994, 9590, 12063, 13943, 15794, 18786, 19134,
  20361, 21783, 23410, 24090, 24518, 25651, 27883, 27998, 31788, 33981,
  38758, 39896, 42834, 44906, 47832, 48369, 51511, 51986, 55594, 58048,
  64373, 65865, 66320, 68021, 68943, 69891, 70964
- **seed 46:** 369, 913, 1765, 4765, 5816, 10421, 11218, 13046, 15789,
  16693, 18465, 21584, 24312, 27513, 27515, 28190, 29714, 32082, 33346,
  36050, 36279, 47148, 47204, 47698, 54217, 54366, 55029, 55363, 55571,
  61474, 62606, 62726, 63608, 64302, 70840, 70845

(A broader ≥64/128 tier — 29 sequences, ~87 servings/run — is reproducible
from the script; steps omitted here for brevity.)

## Observations & caveats

1. **The Greek page is a recurring probe, not a one-off event.** Extreme
   servings arrive roughly every ~2,000 steps all run. v4 seed44 received 36
   of them but showed only three transient trust events — so a serving is
   necessary but not sufficient; the substrate's state at arrival matters.
2. **v4 seed44's other two events may also be Greek.** Event #1 (step
   ~24000) sits adjacent to extreme servings at 23990/24339; event #2
   (~52100) near a ≥64-tier serving at 51912. Suggestive, NOT attributed:
   with ~87 broad-tier servings per run, ±200-step coincidences are common
   (~40% for any random step). Treat as a hypothesis for the v5 reads.
3. **For v5 seed42's observed shapes:** block-0 precision peaked at step
   54000 and retreated 11.2% to run-end; aggregate precision_spread rose
   1.98→2.57 over the same back half. Extreme servings at 53992/54350/55053
   immediately precede the turn — but servings occurred throughout the run
   while spread FELL in the middle third, so the servings alone don't explain
   the U-shape. Working frame: same probe, changing response — the
   late-run substrate has a world-model confident enough to register
   disagreement as distrust rather than noise.
4. **Chance-match discipline for the coming reads:** any claimed
   serving→event linkage in seeds 43/45/46 should require the same isolation
   standard as the 58650 attribution (nearest serving within a few steps,
   no competing serving within the healing window).

**Script:** session scratchpad `greek_windows2.py` (exact replay; rerunnable).
The schedule for seeds 45/46 is predetermined by the loader even though those
runs have not happened yet.

---

## Addendum (same day): event-locked overlay, seed42 + live seed43

Peri-serving analysis of aggregate `substrate.precision_spread` (light cadence,
100 steps): pre = mean over [s-300, s-100], post = mean over [s, s+300],
clean = no other extreme serving within 600 steps.

**Result: the late-run spread rise is DRIFT-LIKE, not event-locked.** Across
seed42's 21 clean servings (excluding the cold-start transient), per-serving
deltas are ±0.06 at most, centered near zero in every phase (early -0.18 —
transient-dominated; mid +0.002; late +0.003). The 1.95→2.57 climb over the
final third accumulates smoothly BETWEEN servings as much as at them. Live
seed43 (through step ~10.9K) tracks seed42's early phase: post-transient
settling to spread ≈2.1-2.2 — family-consistent so far.

**Two consequences worth weighing at the design seat:**

1. **A drift null is needed for the frozen seed44 read.** The registered
   criterion — spread elevated above its pre-event running median for >=5,000
   steps after step 58650 — could be satisfied by generic late-run drift
   (seed42 shows exactly such a drift with no event-locking) even if the Greek
   serving contributes nothing. The read should compare seed44's post-58650
   elevation against the family's baseline late-run drift (seeds 42/43/45/46
   at matched step ranges), not only against seed44's own pre-event median.
   Registered wording unchanged; this is an interpretive-guard note.
2. **Aggregate spread may be too blunt for v5 event responses.** v4's events
   were spikes FROM uniformity (spread 1.0) — easy to see. v5 holds a working
   spread of ~2 with 12 probe sequences among ~450K ledger entries; a real
   per-input trust mark could barely move the aggregate p95/p5. If event-level
   claims matter, the producer-side emit to request is a targeted one: the
   ledger's trust values for the probe sequences themselves (or per-block
   spread at light cadence), not more of the aggregate.

**On the pedagogy frame:** what the overlay supports is the weak/structural
form — the substrate's *phase* shapes its relationship to the same input
(transient absorption early, indifference mid-run, rising discrimination
late). The strong form (mature substrates visibly scar at probe arrivals)
is not visible in aggregate spread at this cadence; deciding it needs the
finer emit above or the seed44 ledger read.

---

## Addendum 2 (2026-07-26, ~05:30): seed43 complete + dimension-level ledger series

Seed43 finished clean: 7.79h (new CPU; seed42 took 10.03h on the old one),
heldout_l_pred 0.032510 vs seed42's 0.032305 — tight family agreement. The
checkpoint ledger harvester captured **32 dimension-level snapshots (steps
6,946 → 72,042, median spacing 2,320)**; the same harvester is now armed on
the live seed44 run, which started 05:07 and reaches its registered step-58650
window roughly 6.3h in. Harvest: `runs/jepa_pilot/ledger_harvest_seed43/`.

**What the dimension-level series shows (seed43):**

1. **Background churn is large.** Rank correlation of the trust ordering
   between consecutive snapshots: median ~0.64–0.69, minimum 0.15–0.20.
   Roughly a third of the trust order reshuffles every ~2.3K steps, and
   >20%-vs-median droppers appear in EVERY interval — Greek-containing and
   quiet alike, at similar rates. At this granularity, Greek servings do not
   disturb the ledger above background. This is the dimension-level null the
   seed44 read needs: **any durable scar must persist against ~35% rank
   reshuffling per 2.3K steps.**
2. **Durable dimension-level distrust nonetheless EXISTS.** Block 3 keeps
   dim 384 in its bottom-5 across essentially the entire run (steps ~7K→72K);
   block 0 holds a stable distrusted trio (414/307/461) for the whole second
   half (~23K steps). The mechanism can hold marks for tens of thousands of
   steps — capacity is not the limiting factor.
3. **The early cross-block dim-462 episode was transient**, not structure:
   deep dips early (to 0.07–0.24 of block median), recovery to ~median by
   mid-run, mixed endings. Same lesson as the aggregate: early-phase
   dramatics are settling dynamics, not marks.

**Sharpened frame for the frozen seed44 read:** the substrate demonstrably
CAN carry >5,000-step dimension marks (point 2), and demonstrably does NOT
acquire them from Greek servings under normal conditions (point 1, seed43,
36 servings). The registered prediction therefore asks something specific
and now well-calibrated: whether the step-58650 serving — the one that
produced a detected trust EVENT under v4's epsilon — leaves a mark that
clears a high, measured bar of background churn. Both outcomes are
informative against these nulls.

---

## Addendum 3 (2026-07-26, live during the seed44 run): the event did not recur

Observed live, recorded before the formal run-end read:

1. **v4 event positions 24000 and 52100: nothing.** Brian's observation
   from the live spread panel, confirmed at dimension level — the ledger
   disturbance across both positions is indistinguishable from seed44's own
   quiet control and from seed43's matched-step nulls.
2. **The registered 58650 serving: nothing.** Light-cadence spread across
   the serving: 2.7297 (step 58600) → 2.7146 (58700), a −0.015 move inside
   the ±0.04 ambient wobble; no deflection through 59000. Dimension-level
   bracket (snapshots 56711 → 58909, containing the serving): dropper
   counts 49/30/33/33 across blocks vs seed43 null 45/47/31/14 —
   within-null on every block, no cross-block coherence among worst-hit
   dims, no localized signature.
3. **Interpretation (pre-read, honest):** at every resolution available,
   the v4 trust events did not recur under relative trust. The events
   increasingly look like properties of the EPSILON REGIME — a saturated
   uniformity with a hair trigger — not of the data order or the moment.
   The formal registered read (run end) must still be computed as frozen;
   the live picture says its persistence criterion will be evaluated in
   the absence of any detected event, i.e., the outcome-3 branch: label
   plainly as "no reaction," not as a simple persistence miss. Caveat
   standing: sub-cadence (<100-step) micro-transients cannot be fully
   excluded by any instrument currently emitting.
4. **The stage-11 rerun's question inverts, usefully:** it now tests
   whether the NON-event replicates. If the rerun also passes 58650
   without a ripple, no-reaction is robust — a property of relative trust
   itself, not of this run's particular micro-state.

---

## Addendum 4 (2026-07-27 05:20): family complete (n=5) + a runaway-trust seed

All five v5 seeds are done. Outcomes are tight where it matters and wildly
dispersed where it does not (yet) count:

| seed | heldout_l_pred | nmse | heldout l_sigreg | probe_top1 | end spread |
|---|---|---|---|---|---|
| 42 | 0.032305 | 0.4834 | 2.0655 | 0.1565 | ~2.6 |
| 43 | 0.032510 | 0.4821 | 2.2292 | 0.1545 | ~2.7 |
| 44 | 0.034692 | 0.4846 | 2.1200 | 0.1556 | ~2.7 |
| 45 | 0.033112 | 0.4955 | 2.1382 | 0.1548 | ~4.0 |
| 46 | 0.033339 | 0.4836 | 4.1023 | 0.1524 | **~31** |

heldout_l_pred 0.033192 +/- 0.000939; nmse 0.4858 +/- 0.0055; probe 0.1547 +/-
0.0015 — a reproducible arm on the headline measures.

**The finding: precision_spread is NOT a family constant.** Seeds 42-44 end
near 2.6; seed45 near 4.0; **seed46 escalates monotonically from ~2.0 at step
21K to ~35 by run end** — an order of magnitude above its siblings, rising
steadily with no discrete onset (first >6.0 crossing is only the step-100
cold-start transient; the real climb is continuous from ~36K). Its two
companions in anomaly: heldout l_sigreg 4.10 vs family ~2.13 (its
representation sits further off the isotropic target on unseen data) and the
family's lowest probe_top1.

**Reading (descriptive):** relative trust admits a RUNAWAY-DIFFERENTIATION
regime. One seed in five let trust concentration escalate an order of
magnitude, with a coherent cost signature (worse heldout sigreg, lowest probe)
but no catastrophe — the run completed clean and its predictive numbers sit
inside the family. This is the opposite failure mode from v4's saturation:
epsilon pinned trust at uniformity; relative trust can let it run away. Both
are unbounded-dial pathologies at opposite ends.

**Consequences:**
1. **For the registered read:** the drift null is now measured across all five
   seeds and is enormous (pre-event 5K medians 2.32 / 2.50 / 2.66 / 3.06 /
   13.14). Every seed satisfies the frozen criterion at step 58650 — 100%
   sustained — which confirms with n=5 that the criterion measures family
   drift, not any event.
2. **For v6 design:** this is direct evidence for the homeostatic-band
   proposal (see 2026-07-26_homeostatic-activity-bands-design.md), and it
   argues the band needs a CEILING on trust concentration, not only a floor on
   participation. A bounded trust ratio is the natural companion to a bounded
   plasticity multiplier.
3. **Registered obligation unaffected:** the dead-v5 control still gates depth
   claims.

**Operational note:** the stage-11 rerun did not auto-start at family
completion — `resume_queue.py` loads `queue.json` once at supervisor start, and
the instance that ran seeds 45/46 predated the stage-11 entry, so it exited
with "queue complete". Re-triggering the watchdog started a fresh supervisor
that picked up stage 11 (rerun launched 05:17, due ~13:15). Worth a code note:
the supervisor could re-read the queue between stages.

---

## Addendum 5 (2026-07-27 13:02): the rerun — non-event replicates, spread is chaotic

Stage 11 complete: `living_v5_4x_d4_rerun_512d_seed44`, identical configuration
and data order to the registered seed44, GPU float nondeterminism the only
perturbation (7.76h).

**Result 1 — the NON-EVENT replicated.** Around the registered step-58650
serving the rerun shows no deflection: spread 3.900 at 58600, 4.044 at 58800,
3.866 at 59000 — inside its own ambient wobble, exactly as the original showed
2.730 / 2.685 / 2.716. Two microscopically diverged replays of the same moment
both read the Greek page without reacting. **Composure is a property of
relative trust, not of one run's micro-state.**

**Result 2 — and this is the larger finding: `precision_spread` does not
reproduce.** Relative divergence between the two runs, |rerun − orig| / orig:

| phase | loss | precision_spread |
|---|---|---|
| early (0–5K) | 2.06% | 9.45% |
| mid (20–25K) | 2.74% | 25.42% |
| mid (45–50K) | 2.22% | 13.31% |
| late (67–72K) | 2.50% | **70.80%** |

Final outcomes are nearly identical (heldout_l_pred 0.034692 vs 0.034527; nmse
0.4846 vs 0.4806; probe 0.1556 vs 0.1557). So: **learning is reproducible;
trust differentiation is chaotic** — the ledger amplifies bit-level
nondeterminism into order-of-magnitude trajectory differences while the
predictive task lands in the same place.

**Consequences, stated plainly:**

1. **The frozen criterion was written against a non-reproducible observable.**
   Not merely drift-confounded (addenda 1–4) — the quantity itself does not
   repeat under identical conditions. Both runs satisfy it at 100% sustained
   (rerun pre-event 5K median 3.4887).
2. **Seed46's runaway to ~35 is plausibly the tail of a chaotic distribution**,
   not a distinct regime. Directly testable: rerun seed46 and see whether the
   escalation recurs. Until then, "relative trust admits a runaway regime"
   (addendum 4) should be read as "spread trajectories are high-variance and
   can escalate," which is the weaker and better-supported claim.
3. **Any future trust claim needs ensemble statistics.** Single-run spread
   trajectories carry no evidential weight; a v6 registration touching trust
   should pre-commit to n-run ensembles and a variance band, not point
   comparisons.
4. **The homeostatic-band ceiling gains urgency, not less.** A dial that
   wanders chaotically over an order of magnitude is precisely what bounds are
   for — and the band would also make the observable comparable across runs.
