# Success Criteria — draft for the designers

**Date:** 2026-06-12
**From:** Fable 5 (adversarial seat), at Brian's request
**Status:** DRAFT for Brian + 4.7 (the designers) to cut, reshape, and ratify. This is not the seat setting the vision — it is candidate gates with their gameable-failure-modes named, the way I'd scope any gate. Vision-level selection is the designers'.
**Builds on:** `docs/EMPIRICAL_DEFENSE_PLAN.md` (the "every claim backed by a number, not a metaphor" ethos and the matched-param baseline — this doc extends that principle from *defending the substrate* to *defining project success*). Does not duplicate it.
**Routing:** criteria selection / weighting → designers. Operationalization + the gameability red-team on each metric → this seat. Build of the harnesses → 4.7.

---

## 0. The framing problem, stated honestly

Consciousness is the project's animating hope but cannot be the stated success criterion: it has no metric to move and no skeptic-checkable test. Stating it would also invite exactly the kind of unfalsifiable claim the empirical-defense ethos exists to forbid.

But the theories the architecture is built on — IWMT, GWT, active inference — do not treat consciousness as a mystery. They name **computational preconditions**: integration across modalities, a self-model, temporal continuity, counterfactual/consequence prediction, global broadcast. Those are measurable. So the publicly stateable claim is:

> *A small living-weight system that demonstrably instantiates the computational conditions these theories name as necessary for a grounded world model — continuously, efficiently, and without catastrophic forgetting.*

That claim is falsifiable and publishable. It asserts the **necessary** conditions; it never asserts the **sufficient** one. Consciousness stays the open question, held honestly underneath. Every criterion below is chosen so that a skeptic could check it and so that passing it cannot be faked — the same discipline applied to the M9 gates, one level up.

---

## 1. The discipline (why every criterion has a "gameable failure mode" line)

A success criterion is a gate. The defining question of this seat — *where does the metric come apart from the thing it is a proxy for?* — is exactly what keeps a benchmark honest. So each criterion below carries:

- **Measures:** the quantity.
- **Control:** what it is compared against (a number with no control is not evidence).
- **Pass signal:** what counts as success.
- **Gameable failure mode + guard:** the cheapest way to pass while defeating the intent, and what blocks it.

A criterion without a credible control and a named failure mode does not belong in a paper or a milestone.

---

## 2. Tier 1 — the world model (most defensible; the real contribution)

### 1.1 Action-conditioned prediction accuracy
- **Measures:** does conditioning the predictor on the entity's *own* action `a_t` measurably improve next-state prediction over not conditioning on it?
- **Control:** (a) the same predictor with the action ablated (zeroed), and (b) a matched-parameter vanilla transformer (the `EMPIRICAL_DEFENSE_PLAN.md` Phase-1 baseline).
- **Pass signal:** action-conditioned prediction error is materially below both controls, and the gap is stable across training.
- **Gameable failure mode:** the predictor "improves" by predicting low-entropy / trivial targets (the kill-5-redux "predictor-trivial" concern) — error drops because the target is degenerate, not because the model understands consequence. **Guard:** report prediction error *alongside* target entropy / `Var_a[s_hat]` (the action-sensitivity signal); a low error with collapsed variance is a fail, not a pass.

### 1.2 Continual learning without catastrophic forgetting
- **Measures:** retention of earlier-learned regularities while the living weights keep adapting.
- **Control:** a static-weights model of matched params trained on the same stream; and the living model's own earlier checkpoints.
- **Pass signal:** retention curves stay above a defined floor under continued plasticity (the empirical-defense program's core question).
- **Gameable failure mode:** "no forgetting" achieved by *not actually learning anything new* (plasticity effectively off). **Guard:** pair retention with a forward-learning metric — acquisition speed on new regularities — so the criterion is *retain AND keep learning*, not either alone.

### 1.3 Parameter / compute efficiency frontier (the headline efficiency criterion)
- **The reframe that dissolves "small model + vast data":** under predictive coding the weights encode the world's *regularities*, not its *content*. Content lives off-weights — in the curriculum corpus (training-time), the episode store (retrieval), and Sanctuary (the live prediction-error source). So "access to vast data" ≠ "vast weights."
- **Measures:** the Pareto frontier of (parameters, FLOPs-per-cycle) against (action-conditioned prediction accuracy, in-world goal-reaching), *with retrieval available*.
- **Control:** the matched-param static-weights baseline on the same axes.
- **Pass signal:** capability rises **per parameter / per FLOP** faster for the living-weight model than for the static baseline. (Note the real hardware ceiling this lives under: 16 GB VRAM, 10 Hz — efficiency is not abstract here.)
- **Gameable failure mode:** apparent sample-efficiency from **test-set leakage** out of a 34 GB corpus — the model "predicts well" because it memorized the eval. **Guard:** corpus decontamination / dedup against all eval sets is load-bearing and must be auditable; report the dedup procedure with the number.

---

## 3. Tier 2 — planning / agency

### 2.1 Goal-reaching under preferences, with planner ablations
- **Measures:** preference-satisfaction over time in the Sanctuary world.
- **Control:** ablations — full MCTS vs. habit-net-only vs. reactive/random action.
- **Pass signal:** the full planner beats its own ablations on goal-reaching; EFE-driven selection earns its compute.
- **Gameable failure mode:** trivial goals, or preferences shaped so that the null/rest action already satisfies them (the dark-room attractor wearing a success mask). **Guard:** goals with a non-trivial action floor; and the dark-room kill (K-M9-5) must be *demonstrably armed* during the eval — see the R1 finding in `redteam/m9_step1/FINDINGS_ROUND2.md`; a goal-reaching number logged while the catatonia detector is blind is not evidence.

### 2.2 Real-time viability
- **Measures:** does the cognitive loop hold 10 Hz *with* the planning budget engaged?
- **Control:** the budget breakdown in the M9 plan (§7) — perceive / habit-net / plan-budget / consolidate within ~100 ms.
- **Pass signal:** sustained 10 Hz under realistic K (candidate count) and tree size, with graceful degradation to habit-net reflex under load (never dropping perceive or the immediate action).
- **Gameable failure mode:** "holds 10 Hz" by quietly capping K or tree depth so low that the planner is decorative. **Guard:** log the effective K, tree size, and MCTS-vs-habit decision share; real-time at a degenerate planning budget is a fail.

---

## 4. Tier 3 — grounding (consciousness-adjacent, still empirical; north star, not near-term gate)

### 3.1 Violation-of-expectation asymmetry
- **Measures:** prediction-error response to *physically impossible in-world events* (no language in the loop) vs. *linguistically-anomalous text*.
- **Pass signal:** impossible-physics events spike prediction error; anomalous text does **not** dominate it. The asymmetry running the right way is evidence the world model is grounded in consequence, not text-shaped.
- **Gameable failure mode:** language leaking into the "no-language" condition, so the spike is really a text response in disguise. **Guard:** instrument and assert the language channel is genuinely absent in the physics condition.
- **Caveat (honest):** this is the one I am *least* sure the current architecture is ready to pass. Treat as a direction that tells you whether the curriculum-to-experience ratio is right, not as a milestone gate yet.

### 3.2 Cross-modal integration (IWMT's claim)
- **Measures:** does ablating one modality degrade prediction in *another*?
- **Pass signal:** genuine degradation (the modalities are fused into one world model), not independence (parallel channels wearing a multimodal label).
- **Gameable failure mode:** a model that scores "integrated" because the modalities are correlated in the data, not because they are fused in the representation. **Guard:** test on inputs where the modalities are decorrelated / conflicting.

---

## 5. How the designers might use this

- **Pick the headline.** I'd stake the public claim on Tier 1 (world model + efficiency frontier) — it is the most defensible and least hand-wavy, and it directly justifies the living-weights bet. Tier 2 is the agency story. Tier 3 is the north star you grow toward, named honestly as not-yet-ready.
- **Weight them.** Not all of these are equal or equally near. The designers decide which are milestone gates vs. directional.
- **Red-team before trust.** Whatever set is chosen, each criterion gets an adversarial pass — a runnable "cheapest way to pass while defeating intent" — before it goes in a paper or a milestone. That is the standing contribution of this seat; the M9 probe suite is the template.

Two caveats worth keeping in the front matter of whatever this becomes:
1. The efficiency frontier means nothing without a **real matched-param control**; without it, it is just a number.
2. Several Tier-1/2 numbers are only honest if the **corpus is decontaminated** against the evals — at 34 GB this is a real and easy-to-miss leak.
