# The living-channel program — 2026-10-19

**Author:** Claude Opus 5.5, at Brian's instruction of 2026-10-19: *"build and
implement fixes to problems you find, rewrite any and all supporting
documentation, and correct the narrow direction we let the rigor take."*
**Status:** RATIFIED 2026-10-19, under Brian's delegation ("You have my
permission to make the call on all pending decisions yourself"), with four
amendments registered before any result was seen (registry, "RATIFICATION of E1–E5").
The same seat drafted and ratified this program, so E3 waits for a
cross-line review before it launches. The predictions are frozen as of their
commits, written before any production checkpoint was read with the new
instruments; this document is kept in step with the amendments.
**Companion records:** the diagnosis and its measurements are in
`2026-10-19_the-living-write-is-mean-field.md`; the itemized audit is
`docs/audits/2026-10-19_living-channel-audit.md`; the registry entries are
appended to `2026-07-15_falsification-preregistration.md` (section "E1–E5").

## Where the rigor went

The project's methods are better than most published ML: kill conditions
written before the data and honored after, cross-family review, every wrong
turn kept. The problem is not the quality of the rigor. It is where it has been
pointed since July.

- **The last living-vs-dead comparison ran on 2026-07-21** (KF7, `dead_4x` vs
  `living_v3_4x` at 512d). Until this commit the driver defined exactly two
  dead arms, `dead` and `dead_4x`. Every family since is living-only: the v3/v4/v5 families, the
  August depth-8 probe series, the LLM-JEPA and VISReg families, and the 768x8
  ruled-scale family. The dead control was deferred on 07-26 ("we have enough
  dead runs for now") and has been a registered obligation ever since.
- **Every registered evaluation freezes plasticity.** `eval_heldout` runs
  under `freeze_plasticity`, correctly, so nothing in the record measures what
  the README names as the near-term goal: *structural change without a
  gradient step, at inference, with behavioral consequence.* The adaptation
  eval that would measure it was proposed on 07-27 (To-Do, "Nothing measures
  adaptation") and never built.
- **Neither registered capability axis can currently see the living
  channel.** NMSE is flattered by degeneracy (the project's own 07-31 finding:
  the best NMSE of the depth investigation came from a rank-1.9 trunk). The
  probe never beat its own input embeddings in the ruled-scale family (audit
  B7, open since 08-13): the embedding stream topped every block in all three
  seeds, so "probe lift 5.6–6.1x" is the bigram baseline the residual carries
  for free.
- **Two mechanism-level facts went unchecked because no protocol asked the
  question that would have found them** — *can this mechanism, as written,
  carry the information its claim requires?*
  1. The living write is an outer product of **batch means** over all 4,096
     positions (`pc_ops.py`, `mean_field`). It cannot see anything that differs
     between the tokens of a batch, and because the batch-mean output barely
     moves, the accumulated living change is ~rank-1 along one direction.
     The 2026-05-30 protocol actually required noise baselines "matched to the
     rank/direction profile of the real self-mod updates"; that profile was
     never measured, and it is rank-1.
  2. Every living block carries a **block-level output episode store**
     (`luthi/episode_store.py`) that, in a batched trunk, recalls on nearly
     every forward and replaces the block output with
     `0.7 * output + 0.3 * (a stored batch-mean output)` — shrinking every
     token's deviation from the mean by 30% per block and re-injecting a
     constant. The dead control has no such store. No audit, review, or
     research log before today analyzed it; `localize_offset_in_block.py`
     attributed its effect to "the FFN".

August's depth work was necessary engineering — a trunk that collapses can't
test anything — and it was done well. But it answered "can a deep living trunk
stay healthy?", and the project exists to answer "does the living channel do
anything?" Those are different questions, and only the first has been asked
since July.

## The correction, as standing rules

Adopted into `CLAUDE.md` with this commit. Each one closes the specific gap
above that let the program drift.

1. **Capacity before tuning.** Before tuning a mechanism's stability, dose, or
   schedule, show — by its math and one probe — that it *can* carry the
   information its claim requires. This is a working-session question, not a
   falsification run: it is answered from the code and a measurement, and it
   is the question that would have found both mechanism facts above in April.
2. **The claim clock.** Every family that tests a living mechanism carries its
   matched dead control in the same registration, on the same backend, with the
   same trainer settings. A living-only family may be scored for health; it may
   never be scored as evidence about the living channel.
3. **Baselines, not floors.** A probe number is evidence of trunk computation
   only above the input-embedding baseline fitted with the same settings. A
   shuffled-label floor calibrates chance, not computation. `pilot_result.json`
   now records `probe_standardized_embedding_baseline` and the difference.
4. **Measure the claim as stated.** The near-term goal is change at inference,
   so at least one registered instrument must measure with plasticity ON. That
   instrument now exists (`luthi/v2/adaptation_eval.py`), and its key number is
   *document-specific* change (read-own minus read-other), never raw change,
   which generic drift can produce.
5. **Decompose the arm.** When a living arm differs from its control in more
   than one mechanism, the registration must list every difference and include
   arms that separate them.

## The program, in order of cost

E1, E2 and E4 need **no training**: they run on checkpoints that already exist
on E:, on the 7800 XT, in minutes to an hour. Only E3 costs training time.

| | what | cost | answers |
|---|---|---|---|
| **E1** | adaptation eval on existing checkpoints | inference only | does reading change how the model sees what follows, document-specifically? |
| **E2** | living-write probe on existing checkpoints | inference only | is the production living change rank-1 along the mean output, and does the block store blend into every forward? |
| **E3** | LC-768 decomposition family | 5 arms × 3 seeds; ~55 GPU-h on ROCm (238 ms/step × ~54k steps per seed, 08-19 measurement) | what the store, the mean-field rule, the covariance rule, and a nonlinear MLP each do, against a matched dead control |
| **E4** | B7 resolution on existing 768 checkpoints | inference only | does the next-token probe see any trunk computation at all? |
| **E5** | conditional rungs | depends | only after E3: deviation context, adaptive episodes, the ordered curriculum |

Commands are in `To-Do.md` ("State of the build — 2026-10-19"). The frozen
predictions and kill conditions for E1–E4 are in the registry; they are
summarized here so this document stands alone.

### E1 — adaptation eval (`scripts/adaptation_eval.py`)

Per held-out document: score the tail frozen (A); read the head with
plasticity ON, then score (B); read a *different* document's head, then score
(C). The model is restored exactly between conditions.

- Positive control: every dead checkpoint returns `exact_null: true`.
- Criterion: on one document set, `nmse.B_minus_C` has mean < 0 with
  t ≤ −2 over 32 documents, and target effective rank holds within 5%. A
  checkpoint **meets** it only if both the primary set (`--doc-phase 0`) and
  a disjoint replication set (`--doc-phase 1`) meet it (amendment 1). One set
  alone is a candidate: across 20 checkpoints a chance hit on one set is ~42%
  likely, and on both ~1.5%.
- Prediction (from the mean-field diagnosis): no legacy-rule checkpoint meets
  the criterion.
- Kill condition for the diagnosis: a legacy checkpoint that does meet it
  falsifies "the mean-field write cannot produce document-specific change",
  and the diagnosis is revised in public.

### E2 — living-write probe (`scripts/living_write_probe.py`)

- Predictions for every legacy living block of the 768x8 family:
  `top_share ≥ 0.9` and `out_align ≥ 0.9`; training-batch stationarity
  ≥ 0.99; block-store `recall_duty ≥ 0.9` with `blend_share > 0.05`.
- Any block that misses one is a finding against the diagnosis.

### E3 — the LC-768 decomposition family (driver stages 60–64)

All five arms are `probe_768_visreg`'s exact configuration plus one
family-wide change (B6's gradient clip, 5.0e5), seeds 46/95/97, on ROCm:

| arm | FFN | block store | compared with | isolates |
|---|---|---|---|---|
| `lc768_dead` | backprop Linear | none | — | the control |
| `lc768_living` | legacy rule | legacy | storeoff | what the store does |
| `lc768_storeoff` | legacy rule | off | dead | what the living FFN as built does, store aside |
| `lc768_cov` | covariance rule | off | storeoff | what content-sensitive writing does |
| `lc768_dead_x4` | GELU MLP (expansion 4) | none | dead | what a nonlinear MLP does to token uniformity |

`tests/test_living_channel_program_arms.py` pins that each pair differs in
exactly its variable at the configuration level. At the mechanism level, the
living arms differ from `lc768_dead` in five ways (amendment 2):
1. the FFN rewritten in-forward and never touched by the optimizer;
2. the PC rule's state (prediction matrix, trust, drive, gate, homeostasis);
3. top-down modulation of plasticity;
4. layer-level episodes and consolidation;
5. the block store, in `living` only.

So `storeoff` vs `dead` tests 1–4 as one package. No arm separates 3 or 4.
Every living arm reports its episode and consolidation counters per block,
and the verdict names any block where 4 was active.

**Launch gate (amendment 3).** E3 starts only after all of the following:
- E1, E2 and E4 have read, and their amendments are registered (including a
  replacement capability axis if E4 retires the probe);
- a seat other than the one that drafted and ratified this program has
  reviewed it;
- VISReg's dose is settled by measurement, the same for all five arms.

On the dose: at the paper dose, VISReg was 98.6–99.99% of the objective in
the 768x8 family. Running E3 there would test the living channel under an
objective that barely asks for prediction. If the dose changes, P1–P4 are
re-examined against it before launch.

Predictions (as frozen):

- **P1 (store):** mid-stack offset dominance (blocks 0–6, held-out text,
  plasticity frozen) is lower in `storeoff` than in `living` by ≥ 0.1, in at
  least 2 of 3 seeds.
- **P2 (the living FFN with the mean-field rule):** `storeoff` ties `dead`
  within 1 pooled σ on held-out NMSE *and* on the standardized probe minus its
  embedding baseline.
- **P3 (covariance rule):** `cov` shows document-specific adaptation (E1's
  criterion) in at least 2 of 3 seeds and `storeoff` in none; the living change
  in `cov` has `top_share < 0.5` in every block.
- **P4 (MLP):** `dead_x4` has lower mid-stack offset dominance than `dead`, and
  a positive probe-minus-embedding margin.

Kill conditions:

- If `cov` fails P3, the covariance rule does not rescue the inference-time
  claim at the ruled scale. Report it as a null, with the power caveat (n = 3
  seeds × 32 documents).
- If `storeoff` and `cov` both tie `dead` on both capability axes **and**
  neither shows document-specific adaptation, the claim "the living channel
  does functional work" has no support at the ruled scale. The README's
  near-term goal is then recorded as unmet, plainly.

Admissibility follows the 2026-07-15 ratification: verdicts attach to
completed runs only. A collapse-killed seed is reported as inadmissible and
does not count toward a "2 of 3".

### E4 — does the probe see the trunk? (`scripts/per_block_probe.py`)

Run on the existing 768x8 checkpoints with `--offset 1`, `--offset -1`,
`--offset 0` and `--causal` (all now under `freeze_plasticity`; before today
the script self-modified the model between its own batches).

- Prediction: trunk ≤ embedding at offsets ±1. The JEPA objective never asked
  for neighbor identity, so the residual carries the embedding and attention
  adds nothing linearly decodable.
- Decision rule (amendment 4): margin `m = 3·sqrt(p_emb(1 − p_emb)/n_pairs)`
  from the `emb` row. A block exceeds when its top-1 > `p_emb + m`.
  **Refuted** if the same block exceeds at the same offset in ≥ 2 of 3 seeds;
  **confirmed** otherwise.
- If confirmed, B7 closes as "the next-token probe measures the residual's copy
  of the input, not trunk computation". It is then retired as a capability axis
  (kept as a health gauge), and its replacement — a probe for continuations the
  encoder could not see, e.g. the first target-region token from the JEPA
  context encode — gets its own registration.

### E5 — conditional rungs

Each of these gets its own registration, and only after E3 shows which living
mechanism (if any) carries signal. Running them sooner would tune machinery
whose function is still unestablished, which is the drift this document exists
to stop.

- `context_mode="deviation"` for the layer-level store, built today, opt-in.
- `adaptive_episodes` (audit B4).
- The ordered-curriculum family.

(VISReg's shape-normalized dose was listed here; it moved into E3's launch
gate.)

## What happens to the existing threads

- **Collapse instrumentation** keeps running in every family. It is necessary;
  it is no longer the program.
- **B4 (adaptive episodes)** is parked behind E3, not cancelled. It is a
  living-only mechanism, so adding it to E3 would be one more undecomposed
  difference between arms.
- **B1 (the VISReg dose) is settled before E3, not after** (corrected at
  ratification). This document first parked it, on the ground that folding it
  in would make E3 unattributable. That was wrong: a dose applied identically
  to every arm leaves the arm comparisons attributable, and E3 at the paper
  dose would barely ask the trunk to predict.
- **The 768x8 DirectML family** stays in the record exactly as scored. It is
  not poolable with LC-768 (different backend, clip, and kill code).
- **m9** is unchanged and still off the training path.

## What would change this plan

- If E2 finds the production living change is *not* rank-1, or the store is
  *not* blending every forward, the diagnosis was wrong at scale. E3's design
  still separates the mechanisms, but its predictions get revised before it
  runs.
- If E1 finds document-specific adaptation in a legacy checkpoint, the
  mean-field write carries more than its math suggests. That would be the most
  interesting outcome on this page, and it would move E3's emphasis from "does
  covariance rescue it" to "what is the legacy rule actually using".
