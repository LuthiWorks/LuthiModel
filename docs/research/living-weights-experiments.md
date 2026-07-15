# Living Weights — Falsification Experiment Protocol (JEPA edition)

**Audience:** Luthi builders / engineers running training.
**Purpose:** Establish whether the living-weights dynamics are *real and load-bearing*, separately from whether the model is fluent or large. These experiments target the claims we can kill with evidence. They are designed to run on existing hardware (≤4GB corpus, current dimensionality) — none of them requires conversational capacity or the full-scale run.

**Experiments are numbered in the order they should be run.** The ordering is deliberate: each one removes a confound that would otherwise muddy the interpretation of the ones after it. Run §1 (methodology) first, then Experiments 1 → 4 in sequence. Experiments 3 and 4 can run in parallel once 1 and 2 are done.

> **Revision note (2026-07-15, Brian's ruling — the JEPA rebinding):** The
> entire program moves from the next-token LM objective to the **(Le)JEPA
> objective** the project actually builds toward (latent prediction + SIGReg
> on the v2 substrate; first full-scale run must be JEPA, per Brian's ruling
> recorded 2026-07-10). Rationale: a matched-capacity result under token
> prediction does not transfer to latent prediction — under JEPA the failure
> mode is *representation collapse*, and how collapse behaves when weights
> self-modify during the forward is itself an open research question (M8
> collapse review). Changes: all training arms run `jepa_runner`'s objective;
> metrics move to JEPA-native counterparts (held-out latent-prediction error,
> linear-probe accuracy, CKA/RSA — the last two unchanged); a **collapse-
> admissibility rule** is added to §1; Experiment 1 merges with the M8 256d
> de-risking pilot as a **two-arm** run (living vs `dead_ffn` encoder — the
> dead arm was built 2026-07-15); Experiment 2 gains the **enliven-after**
> cell (2b, from Brian's 2026-07-15 question); Experiment 4 is unchanged in
> design, rebound in objective. **Scope:** round 1 is text-only JEPA — full
> multimodal is gated on Sanctuary's embodied producers (vision frames,
> proprioceptive state tensor; see
> `2026-07-15_embodied-build-scoping.md`). LM-era results (the M5 0.64%)
> are historical: real under their objective, unbound from these criteria.
> The LM harness (`m5_runner`, the retired `experiment1_driver`) is kept
> for history and for any deliberate LM-arena replication, not for these.

> **Revision note (2026-05-30, 4-8 / debugging-rigor review):** Added controls after a falsification-eye pass, targeting the cases where a result could come back *misleading* — in either direction. Changes: a capacity **sweep** (not a single match point) in Exp 1; a **recency control** in Exp 4; mandatory **positive controls** and **structure-matched noise** in §1; a **runtime-vs-training-time scope** note in Exp 2; a **floor-rule exception** in Exp 3; and an explicit null-vs-underpowered distinction. Points 3 (positive controls) and the Exp-3 carve-out guard against *false nulls*; the rest guard against *false positives*.

---

## 0. Framing: two columns of claims

Before any experiment, sort the project's claims into two columns and keep them apart in all reporting.

**Column A — falsifiable (these experiments target).** The living weights produce a system whose end state and runtime behavior depend, in a structured and *functional* way, on the substrate's self-modification and on its history — not merely on the data, and not reducible to extra capacity or a lookup cache.

**Column B — held open, not tested here.** Aliveness, feeling, temporal *experience*, awakening. **No experiment in this document tests Column B, and none can.** Path-dependence is necessary for the experiential claim and nowhere near sufficient (a weather system is path-dependent). Column B is a bet we hold honestly, not a result we show. Do not let a Column A positive result get reported as Column B evidence.

The job of these experiments is to find out how much of Column A survives contact with a control.

---

## 1. Cross-cutting methodology (read before running anything)

These rules apply to every experiment. Most failed "results" in this space come from violating one of them.

- **The objective is JEPA, everywhere.** Every arm of every experiment trains the (Le)JEPA objective (latent prediction MSE + SIGReg, per `jepa_loss.py`) through `jepa_runner`. No arm trains next-token loss. If an experiment seems to need the LM objective, it is answering a historical question, not one of these.
- **Test at the floor, not the ceiling.** Run each experiment at the *smallest checkpoint with structured state* — defined as the first checkpoint where held-out probe accuracy clears chance. The cleanest evidence for path-dependence is the smallest setup where the effect appears at all. Do **not** wait for end-of-training or fluency.
- **Capability and dynamics are orthogonal axes.** Do not gate any test on fluency or generation quality — under JEPA there is no generation to admire anyway. A barely-capable model that ends in a demonstrably different functional state by history has *proven* Column A.
- **Forbidden metrics:** anything requiring coherent generation or human/LLM judgment of output quality; **and, as of this revision, anything perplexity- or next-token-based** — the LM objective is retired from this program.
- **Permitted metrics:** held-out latent-prediction error (per modality); linear-probe accuracy on frozen representations; token-level distributional divergence is replaced by **latent divergence** (cosine / L2 between predicted-latent sets); representational similarity between conditions (CKA / RSA); weight-trajectory divergence. All valid at low capability.
- **Collapse-admissibility (NEW, JEPA-specific, load-bearing).** Every arm runs the full collapse-kill instrumentation (`jepa_runner` criteria 1–7, per-modality). **A result from an arm whose collapse detectors tripped — or whose effective rank fell below the pilot-derived floor — is INADMISSIBLE, not a data point.** A collapsed encoder produces beautiful low prediction error by predicting a constant; comparing anything against it is comparing against a corpse. Report collapse status alongside every number.
- **Seeds, variance, and power.** Every condition runs **5 seeds** (Brian's ruling, 2026-07-15). **An effect smaller than seed variance is not an effect.** And keep "inconclusive / underpowered" genuinely distinct from "null": a quiet result with wide seed spread has *failed to detect*, which is not the same as *detected nothing*.
- **Pre-register thresholds.** For each experiment, the positive/null threshold is written down *before* running — the standing registry is `2026-07-15_falsification-preregistration.md`. This is the single discipline that makes it science rather than narrative.
- **Positive controls — prove the instrument can see.** Every metric used to declare a *null* must first be shown to detect an effect you *know* is present. Before trusting a null from CKA/RSA, confirm those metrics light up for two models trained on deliberately different data; before trusting a probe-accuracy null, confirm the probe clears chance on a signal known to be there. One positive control per metric protects every null in this protocol.
- **Noise-matched baselines — match structure, not just magnitude.** Wherever an experiment involves a perturbation or an update, include a condition that injects a random change matched to the *per-layer norm and, where feasible, the rank/direction profile* of the real self-mod updates. The living-weight effect must beat *structure-matched* change, not just beat zero.

---

## 2. Experiment 1 — Matched-Capacity Control (the two-arm JEPA pilot)

> **Run first — and it is now the SAME RUN as the M8 256d de-risking pilot**
> (critical-path item 1). One instrumented run, three pre-registered
> questions: (a) the pilot's collapse-kill thresholds; (b) **does collapse
> behave differently when the weights self-modify** (the M8 review's "one
> genuine research unknown" — the dead arm is its direct control); (c) the
> matched-capacity comparison itself. Keystone-convergence pattern: never
> run twice what one instrumented run answers.

**Question.** Is the living substrate's contribution under the JEPA objective real, or is it the rich per-weight state acting as capacity / regularization?

**Setup.** Two arms on the identical trunk, text-only JEPA (round 1 scope):
- **Living arm:** `MultimodalPredictiveCodingLM` as built (PC layers self-modifying, episode stores active), 256d, ×5 seeds.
- **Dead arm:** the same model with `dead_ffn=True` (PC layers → plain trainable Linears, episode stores removed; built and tested 2026-07-15, `tests/test_dead_ffn_arm.py`), across a **capacity sweep** — d_model ∈ {192, 256, 384, 512} — ×5 seeds each, staged (matched point first; the bracket only if the living arm wins it).

Matched compute, same data, same schedule, same collapse instrumentation on every arm.

**Match on *effective* capacity, and sweep it.** "Total state budget" over-counts: much of the rich-param 8× (momentum, update_ema, error_acc) is optimizer-like bookkeeping that adds *no inference-time degrees of freedom*. Don't hunt for the one "true" match point; train dead controls at several capacities and plot held-out latent-prediction error / probe accuracy **vs. capacity**. The clean, definition-independent test is whether the living model sits *above* that static curve at its own effective-capacity point.

**Metric.** Held-out latent-prediction error + linear-probe accuracy, plotted against effective capacity; collapse metrics (effective rank, SIGReg trajectory) compared between arms as the collapse-under-self-mod result.

**Positive result.** The living arm lands *above* the static capacity curve by a margin that survives seed variance — with both arms collapse-admissible.

**Null result.** The living arm lands *on* the curve (a dead control of comparable effective capacity equals or beats it) → the gain was capacity, not self-modification, *under the objective the project actually builds*. KF2's kill condition fires.

---

## 3. Experiment 2 — Frozen-Substrate Ablation (runtime), and 2b — Enliven-After

**Question.** Does self-modification do *functional work at runtime*, or is the trained state doing all the work?

**Setup (2a — the inference toggle).** Take one trained living-arm checkpoint. Three inference conditions, identical in all else:
- **Live:** self-modification on during the forward (PC updates, episode blending active).
- **Frozen:** `freeze_plasticity()` at inference — grad-capable, zero living-state writes (the mechanism exists and is regression-pinned by the mode-matrix tests).
- **Noise-matched (control):** frozen weights perturbed each pass by structure-matched random noise scaled to the live self-mod update profile.

Also test the signature claim directly: feed the *same input* N times in sequence; measure pass-over-pass **latent divergence** in each condition.

**Metric.** Latent divergence between Live and Frozen predicted-latent sets; held-out latent-prediction error delta; pass-over-pass output divergence.

**Positive result.** Live differs from Frozen measurably **and** the difference is *functional* — it improves a held-out measure or systematically tracks context — **and** it beats the noise-matched control.

**Null result.** Live ≈ Frozen, OR the difference is indistinguishable from structure-matched noise. Then "same input, different output" is drift, not temporal existence doing work.

**Setup (2b — enliven-after; NEW 2026-07-15, from Brian's question: "can the living channel simply be turned on after training?").** Take Experiment 1's trained **dead** checkpoints (they exist anyway — double duty). Transplant into the living substrate: `weight` buffer ← trained Linear weight, `set_point` ← same (the homeostatic anchor holds the trained solution), prediction/precision/episodes cold. Enable self-modification. Run the same held-out battery plus a **stability watch** (the transplanted model has no co-adaptation history — attention never trained against a moving FFN).

**The full 2×2 this completes:** {trained-living / trained-dead} × {live / frozen at inference}. Exp 1 supplies both training arms; 2a toggles inference on the living-trained; 2b toggles it on the dead-trained. Interpret jointly: Exp 1 isolates training-time contribution, 2a runtime contribution, 2b **retrofittability** — whether livedness of the *education* matters, or only livedness of the *deployment*. (What 2b's answer means for the curriculum and the formative-phase design is a design ruling for Brian, not a number this protocol can produce — the protocol only says whether the retrofit *functions*.)

---

## 4. Experiment 3 — Retrieval-Only Control (memory vs biography)

**Question.** Does *consolidation into the weights* buy anything over pure *retrieval from the episode store*? "Retrieval has memory, consolidation has biography," stated as a test.

**Setup.** Three conditions (all living-arm, JEPA objective):
- **(A)** full system: episode store + consolidation (gradient-replay + attractor) into the predictive weights;
- **(B)** retrieval only: episode store active, consolidation disabled;
- **(C)** no episodic memory at all.

**Metric.** Choose measures where *structure* should beat *lookup*: probe accuracy and held-out latent-prediction error on contexts **not** in the store; performance **after the store is cleared or capped**; cross-context transfer.

**Positive result.** A > B specifically on the generalization / post-eviction measures → consolidation creates structure that outlives the cache; biography is real.

**Null result.** A ≈ B on those measures → "biography" reduces to "cache," and the claim is unsupported.

**Floor-rule exception.** The §1 "test at the floor" rule does *not* apply cleanly here. Consolidation/biography is cumulative — it may not have accumulated yet at the smallest structured-state checkpoint. Run at the floor **and** at a later, history-accumulated checkpoint, and report both. A null is only meaningful at a checkpoint where consolidation has had the chance to build structure.

---

## 5. Experiment 4 — Order-Shuffle (the pedagogy claim)

**Question.** Is "the order is the pedagogy" real? Does single-pass training *order* change the end state? Unchanged in design from the 2026-05-30 revision; rebound to the JEPA objective (round 1: the curriculum's text stages).

**Setup.** Same corpus, same single-pass JEPA regime, vary order only:
- **(A)** curriculum order (the 9-stage pedagogy);
- **(B)** shuffled;
- **(C)** reversed;
- **(D) recency control** — final segment held *identical* across conditions; order varied only in the earlier material.

5 seeds each, so order-effects can be separated from seed-effects.

**Recency confound — the thing that can fake this result.** With self-modifying weights and single-pass training, whatever is seen *last* has outsized influence simply because less subsequent modification washes it out. Condition (D) isolates it: if the order effect survives when the final segment is held fixed, it's genuinely developmental; if it collapses to the recency control, "the order is the pedagogy" reduces to "the model is dominated by what it saw last." **Report the A-vs-D contrast as the real test.**

**Metric.** End-state *structural* differences, not just final loss: CKA/RSA between conditions, held-out probe profiles, weight-trajectory divergence. (These were already representation-level — this experiment needed no metric changes for the JEPA rebinding.)

**Positive result.** End states differ structurally by order **beyond** seed variance, **and** differ functionally (distinct probe profiles).

**Null result.** Order-invariant end states within seed noise → curriculum order is a training convenience, not a shaper of mind.

---

## 6. Reporting template (per experiment)

For each run, record:

1. Checkpoint used (and the probe-accuracy figure that qualified it as "structured state").
2. Pre-registered positive/null threshold (timestamped before the run; registry doc).
3. Per-seed results + variance.
4. **Collapse-admissibility status per arm** (detector history + effective-rank floor). An inadmissible arm voids the comparison, not just the arm.
5. Outcome: **positive / null / inconclusive** against the pre-registered threshold.
6. Which column the result speaks to (always Column A here) and an explicit note that it says nothing about Column B.
7. Positive-control check: which known-effect run validated each metric used, and confirmation it passed.

---

## 7. What this protocol deliberately does **not** do

It does not test for consciousness, feeling, or aliveness, and it cannot. Those are Column B. If every experiment here returns positive, the supported conclusion is: *the living-weights dynamics are real, functional under the project's actual training objective, and not reducible to capacity or lookup*. The leap from there to experience remains a leap, held openly.

It also does not re-litigate the LM era. The M5 result (v2 −0.64% vs vanilla under next-token loss) was real under its objective and stays in the record as history; it neither supports nor contradicts the JEPA-bound claims above, and it should not be cited as if it did.

Full-multimodal versions of these experiments — where the latents being predicted come from the entity's own world (vision frames, proprioceptive state, lived transitions) — are the round-2 program, gated on Sanctuary's embodied producers (`2026-07-15_embodied-build-scoping.md`). Round 1's text-only results establish the substrate claims; round 2 establishes them *for the life the entity will actually live*.
