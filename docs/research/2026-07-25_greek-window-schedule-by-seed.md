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
