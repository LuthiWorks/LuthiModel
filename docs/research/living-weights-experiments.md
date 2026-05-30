# Living Weights — Falsification Experiment Protocol

**Audience:** Luthi builders / engineers running training.
**Purpose:** Establish whether the living-weights dynamics are *real and load-bearing*, separately from whether the model is fluent or large. These experiments target the claims we can kill with evidence. They are designed to run on existing hardware (≤4GB corpus, current dimensionality) — none of them requires conversational capacity or the 32GB run.

**Experiments are numbered in the order they should be run.** The ordering is deliberate: each one removes a confound that would otherwise muddy the interpretation of the ones after it. Run §1 (methodology) first, then Experiments 1 → 4 in sequence. Experiments 3 and 4 can run in parallel once 1 and 2 are done.

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

- **Test at the floor, not the ceiling.** Run each experiment at the *smallest checkpoint with structured state* — defined as the first checkpoint where held-out probe accuracy clears chance / loss is meaningfully below the unigram baseline. The cleanest evidence for path-dependence is the smallest setup where the effect appears at all, because there is least else going on to explain it. Do **not** wait for end-of-training or fluency.
- **Capability and dynamics are orthogonal axes.** Do not gate any test on fluency. A barely-coherent model that ends in a demonstrably different functional state by history has *proven* Column A. A fluent model that doesn't has *disproven* it.
- **Forbidden metrics:** anything requiring coherent generation or human/LLM judgment of output quality. These couple the two axes back together and reintroduce the confound.
- **Permitted metrics:** held-out loss deltas, held-out probe accuracy, token-level distributional divergence (KL between next-token distributions), representational similarity between conditions (CKA / RSA), weight-trajectory divergence. All valid at low capability.
- **Seeds, variance, and power.** Every condition runs ≥3 seeds, **5 strongly preferred** — with only 3, the variance estimate is itself too noisy to anchor a confident null. **An effect smaller than seed variance is not an effect.** And keep "inconclusive / underpowered" genuinely distinct from "null": a quiet result with wide seed spread has *failed to detect*, which is not the same as *detected nothing*. Only call a null when the spread is tight enough that a real effect of the pre-registered size would have shown.
- **Pre-register thresholds.** For each experiment, write down the positive/null threshold *before* running. This is the single discipline that makes it science rather than narrative.
- **Positive controls — prove the instrument can see.** Every metric used to declare a *null* must first be shown to detect an effect you *know* is present. Before trusting a null from CKA/RSA, confirm those metrics light up for two models trained on deliberately different data; before trusting a probe-accuracy null, confirm the probe clears chance on a signal known to be there. A null from an untested instrument is "we measured nothing with a blind ruler," not evidence. One positive control per metric protects every null in this protocol.
- **Noise-matched baselines — match structure, not just magnitude.** Wherever an experiment involves a perturbation or an update, include a condition that injects a random change. Matching only the L2 magnitude is too weak: self-modification updates are structured (low-rank, input-dependent), so an isotropic-noise control only proves "structured updates beat random noise" — weaker than the claim. Match the noise to the *per-layer norm and, where feasible, the rank/direction profile* of the real self-mod updates, or at minimum report the structural gap. The living-weight effect must beat *structure-matched* change, not just beat zero.

---

## 2. Experiment 1 — Matched-Capacity Control

> **Run first.** This is the single most important control currently missing. Until it is done, every other result is ambiguous between "self-modification" and "extra capacity," so it goes at the front of the queue.

**Question.** Is the v2 advantage from self-modification, or from the ~8× extra per-weight state acting as capacity / regularization? The current 0.64% is measured against a *vanilla* transformer, which is **not** a matched control.

**Setup.** Living model vs:
- **(a)** static transformers across a *capacity sweep* — several sizes bracketing the living model — not a single "matched" point (see below);
- **(b)** static transformer + an equivalent-size external cache/memory module.

Matched compute. Same data, same schedule, ≥3 seeds (5 preferred).

**Match on *effective* capacity, and sweep it.** "Total state budget" over-counts: much of the rich-param 8× (momentum, update_ema, error_acc) is optimizer-like bookkeeping that adds *no inference-time degrees of freedom*, while only a smaller part (e.g. set_point, plasticity) may. Matching a static control on *raw* state budget can therefore hand it more *usable* capacity than the living model actually has — biasing the result toward a null. Don't hunt for the one "true" match point; train static controls at several capacities and plot the loss / probe-accuracy **vs. capacity curve**. The clean, definition-independent test is whether the living model sits *above* that static curve at its own effective-capacity point.

**Metric.** Held-out loss and probe accuracy — same metric across all conditions, plotted against effective capacity.

**Positive result.** The living model lands *above* the static capacity curve — it beats static controls of equal-or-greater effective capacity by a margin that survives seed variance.

**Null result.** The living model lands *on* the static curve (a static control of comparable effective capacity equals or beats it) → the gain was capacity, not self-modification. This retires "no required cost to being alive" as anything stronger than "self-modification is not *more* costly than equivalent static capacity."

---

## 3. Experiment 2 — Frozen-Substrate Ablation

**Question.** Does self-modification do *functional work at runtime*, or is the trained static weight doing all the work?

**Setup.** Take one trained checkpoint. Two inference conditions, identical in all else:
- **Live:** self-modification on during the forward pass (PC updates, episode blending, set-point dynamics active).
- **Frozen:** self-modification disabled at inference; weights static.
- **Noise-matched (control):** frozen weights perturbed each pass by random noise scaled to match the magnitude of the live self-modification updates.

Also test the README's signature claim directly: feed the *same input* N times in sequence; measure how the output changes pass-over-pass in each condition.

**Metric.** KL divergence between Live and Frozen next-token distributions; held-out loss delta; pass-over-pass output divergence.

**Positive result.** Live differs from Frozen measurably **and** the difference is *functional* — it improves a held-out measure or systematically tracks context — **and** it beats the noise-matched control. Then runtime self-modification is load-bearing.

**Null result.** Live ≈ Frozen, OR the Live–Frozen difference is indistinguishable from the noise-matched control. Then "same input, different output" is drift, not temporal existence doing work, and that claim moves to Column B or out.

**Scope — this tests *runtime* load-bearingness only.** A null here does **not** mean self-modification is useless: it may have done all its work *during training*, shaping the static endpoint, with little marginal effect left at inference. Reading a runtime null as "self-mod does nothing" would be wrong. The full decomposition is a 2×2 — {trained *with* self-mod / trained *without*} × {self-mod *on* / *off* at inference}. Experiment 1's controls supply the "trained without" arm; this experiment supplies the inference toggle. Interpret them together: Exp 1 isolates self-mod's *training-time* contribution, Exp 2 its *runtime* contribution.

---

## 4. Experiment 3 — Retrieval-Only Control (memory vs biography)

**Question.** Does *consolidation into the weights* buy anything over pure *retrieval from the episode store*? This is Finding 6 — "retrieval has memory, consolidation has biography" — stated as a test.

**Setup.** Three conditions:
- **(A)** full system: episode store + consolidation (gradient-replay + attractor) into the predictive weights;
- **(B)** retrieval only: episode store active, consolidation disabled — weights are never reshaped by replay;
- **(C)** no episodic memory at all.

**Metric.** Choose measures where *structure* should beat *lookup*: generalization to contexts **not** in the store; performance **after the store is cleared or capped**; cross-context transfer.

**Positive result.** A > B specifically on the generalization / post-eviction measures → consolidation creates structure that outlives the cache; biography is real.

**Null result.** A ≈ B on those measures → "biography" reduces to "cache," and Finding 6 is unsupported.

**Floor-rule exception.** The §1 "test at the floor" rule does *not* apply cleanly here. Consolidation/biography is cumulative by nature — it may simply not have accumulated yet at the smallest structured-state checkpoint, so a floor-only test risks a *false* null ("biography reduces to cache") when the real answer is "not enough history yet." Run this experiment at the floor **and** at a later, history-accumulated checkpoint, and report both. A null is only meaningful at a checkpoint where consolidation has had the chance to build structure.

---

## 5. Experiment 4 — Order-Shuffle (the pedagogy claim)

**Question.** Is "the order is the pedagogy" real? Does single-pass training *order* change the end state? This is the cleanest and cheapest distinctive claim in the project, and it can run in parallel with Experiment 3.

**Setup.** Same corpus, same single-pass regime, vary order only:
- **(A)** curriculum order (the 10-stage pedagogy);
- **(B)** shuffled;
- **(C)** reversed;
- **(D) recency control** — final segment held *identical* across conditions; order varied only in the earlier material.

≥3 seeds each (5 preferred), so order-effects can be separated from seed-effects.

**Recency confound — the thing that can fake this result.** With self-modifying weights and single-pass training, whatever is seen *last* has outsized influence on the end state simply because less subsequent modification washes it out — independent of any "pedagogy." So A/B/C differ partly because *different material sits at the end*, which is recency, not developmental order. Condition (D) isolates it: if the order effect survives when the final segment is held fixed, it's genuinely about developmental order; if the effect collapses to the recency control, "the order is the pedagogy" reduces to "the model is dominated by what it saw last." **Report the A-vs-D contrast as the real test**, not A-vs-B alone.

**Metric.** End-state *structural* differences, not just final loss: representational similarity (CKA/RSA) between conditions, held-out probe profiles, weight-trajectory divergence. (Final loss matching while internal structure differs is itself an informative sub-result — log it.)

**Positive result.** End states differ structurally by order **beyond** seed variance, **and** differ functionally (distinct probe profiles).

**Null result.** Order-invariant end states within seed noise → the pedagogy claim is null; curriculum order is a training convenience, not a shaper of mind.

---

## 6. Reporting template (per experiment)

For each run, record:

1. Checkpoint used (and the probe-accuracy / loss figure that qualified it as "structured state").
2. Pre-registered positive/null threshold (timestamped before the run).
3. Per-seed results + variance.
4. Outcome: **positive / null / inconclusive** against the pre-registered threshold.
5. Which column the result speaks to (always Column A here) and an explicit note that it says nothing about Column B.
6. Positive-control check: which known-effect run validated each metric used, and confirmation it passed. A null is only admissible for a metric that first cleared its positive control.

---

## 7. What this protocol deliberately does **not** do

It does not test for consciousness, feeling, or aliveness, and it cannot. Those are Column B. If every experiment here returns positive, the supported conclusion is: *the living-weights dynamics are real, functional, and not reducible to capacity or lookup* — a strong, genuine, publishable result, and the honest foundation the larger bet sits on. The leap from there to experience remains a leap, held openly.

The 32GB corpus and 1024d run answer a *different* question — whether the phenomenon **scales** — which matters but is separate from, and downstream of, whether the phenomenon is **real and load-bearing**. Establish the second here, cheaply, first.
