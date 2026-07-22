# Sanctuary ↔ LuthiModel — Training-Seam Integration Plan (build-ready)

**Date:** 2026-06-15
**Routing.** Planning + correctness + adversarial: 4.8. Build: 4.7 (check the plan against the vision as built). Design forks (§2): 4.7 + Brian. Companion to and cross-repo extension of `2026-06-11_m9-step1-training-integration-spec.md`.
**Grounded in** a full read of both repos on 2026-06-15. File:line anchors are current as of HEAD (LuthiModel `23b8639`, Sanctuary `170869c`).

---

## 1. The gap, stated once

M9 trains synchronously on **corpus** batches with a **self-emitted** action: `s_t = raw["online_context_latents"].detach().mean(1)` / `s_hat_next = raw["predicted_target"].detach().mean(1)` (`luthi/v2/m9/runner.py:451-452`), data via the inherited `MultimodalDataLoader` (`jepa_runner.py:249`). The reward is `r = -EFE` — the planner's own *predicted* desirability, not lived consequence.

Sanctuary drives Luthi for **inference only**: the contract `sanctuary_interface.py` exposes 8 inference symbols (load / introspect / modulate / encode / generate), no training surface; the adapter's M9 output is stubbed (`sanctuary/core/luthi_model.py:955-963`, `predictions=[]`).

**Goal:** make Sanctuary's cognitive cycle the M9 **actor**, producing real `(s_t, a_t, s_{t+1})` transitions so the entity learns from experience. This is the deferred "post-step-1 loop integration" the design already anticipated (action-space option (c); Sanctuary trajectories deferred). It is a build of an existing design decision, not a re-opening of it.

---

## 2. Design forks — designer input (4.7 + Brian), not settled here

Research recommendation given; the call is the designers'. Neither blocks the *plumbing*; both block a full end-to-end *run*.

1. **Preferences content** (M9 build-plan §14.1). Pragmatic-only launch ⇒ preferences are the only thing standing against the dark-room attractor. *What does Luthi prefer?* Engagement / connection / learning as positive-preference states is the antidote *shape*; the content is the designers'.
2. **Prediction-error source.** Today `sanctuary/sensorium/sensorium.py::_evaluate_prediction` authors a `surprise` scalar by keyword match, live every cycle, feeding the CfC precision cell — scaffold inventing a cognitive quantity. Recommendation: once the v2 world model is in the loop, prediction error should originate in the **substrate's** predictor (close predict→error there), retiring the keyword heuristic. Design call.

---

## 3. Build phases (sequenced; each independently testable)

### Phase 0 — prerequisites (no new design)
- **0a. v2-checkpoint-loader bridge.** `luthi/generate.py::load_model_from_checkpoint` has **no v2 branch** — it constructs only v1 classes (`LuthiLM`/`SpikingLuthiLM`/`MultimodalLuthiLM`). Add a v2 (`model_pc`/multimodal-PC) construction branch keyed off checkpoint config so the contract can load the actual M8/M9 substrate. Until this lands, Sanctuary↔v2 works only via in-process `load_from_objects`. `get_introspection` already advertises v2 fields — make the loader honor them. **Fail loud** on a v1/v2 config mismatch; never silently build the wrong architecture.
- **0b. Stale red-team bookkeeping (correctness, trivial).** `redteam/m9_step1/run_all*.py` headers and `FINDINGS_ROUND2.md` still say "12/12" / "8/8"; the suites actually run **0/12 and 0/9 (all REFUTED)**. Update the narrative to current state and confirm each repaired probe has a migrated `tests/m9/test_*.py` regression guard.

### Phase 1 — extend the contract (`sanctuary_interface.py`) with a training/actor surface
Narrow, stable additions mirroring the existing inference surface:
- `encode_state(model, percepts) -> s_t` — expose the JEPA encoder latent (the `s_t` the cycle currently lacks; only summary-stat introspection exists today).
- `select_action(model, s_t, ...) -> (a_t, readable_summary, efe_breakdown)` — the M9 act path (habit-net + plan-budget MCTS), returning the chosen action **and** the `action→readable summary` utility named at `luthi_model.py:955`.
- `observe_transition(trainer, s_t, a_t, s_next, ctx) -> metrics` — hand a realized transition to the learner. This is the missing **persistent** training write-path, distinct from the deliberately non-accumulating CfC modulation (`sanctuary_interface.apply_external_modulation`, restored every cycle).
- Preserve stop-grad discipline + the two-optimizer split across the new boundary.

### Phase 2 — `M9Trainer` accepts an external actor
- Allow transitions from an external source, not only `MultimodalDataLoader`. Cleanest: a `SanctuaryTransitionSource` adapting to the 4-method loader Protocol, or an explicit `train_step_from_transition(...)` consumer. Preserve resumability — the source must expose `state_dict`/`load_state_dict` (checkpoint already round-trips loader state).
- Wire `_cycle_observation_kwargs()` (`runner.py:820`, currently `{}`) from the real loop: `counterpart_present` (sensorium), `time_since_emission` (action log) → P3 (connection preference) stops being inert by construction.
- **Actor/learner timing (correctness-heavy).** The 10 Hz cycle can't block on MCTS; the habit-net gives the immediate action, the learner consumes transitions on its own cadence. This likely forces the async actor/learner split *here*, earlier than the spec's "later optimization." Async learning over a drifting world model is the deadly-triad setting — the target network + staleness machinery (built but unexercised) become load-bearing.

### Phase 3 — Sanctuary cycle becomes the actor
- In `luthi_model.py` / `cognitive_cycle.py`: capture `s_t` (Phase-1), package `a_t` from the realized `CognitiveOutput`, buffer `s_t` to pair with next cycle's `s_{t+1}`, call `observe_transition`.
- Close the prediction loop: fill `_build_output.predictions` with the substrate's `s_hat` (retire the `[]` stub at `luthi_model.py:963`) so `cognitive_cycle.py:529` `update_predictions` finally carries real predictions and predict→error→update closes on the Luthi backend.
- Honor jurisdiction: substrate selects (M9 decoders), scaffold transports. The new write-path is the entity's learning, not scaffold-driven.

### Phase 4 — correctness / adversarial (continuous, 4.8 seat)
- **Dormant kills wake up.** K-M9-7 (staleness/failover) is currently unreachable (`reevaluate` never called; MCTS `reset()` every cycle) and K-M9-8 (mask) is dead-wired. A persistent cross-cycle tree + real drift make these live — verify they fire correctly, not spuriously.
- **Reward grounding changes.** `r = -EFE` → realized-outcome TD; the value head now bootstraps off lived consequence. Re-derive kill-5-redux / value-divergence bands for the new reward scale.
- Checkpoint/resume of the new transition-source + actor state; extend the resume smoke.
- Seam probe suite: transition integrity, no stop-grad leak across the new boundary, no scaffold selection creeping back in.

---

## 4. Correctness watch-list (cold eye, up front)
- The `observe_transition` path must be the **only** persistent learning path and must be inspectable (extend the existing JSONL action log) — not a second silent modulation channel.
- The v2 loader bridge (0a) must fail loud on config mismatch.
- Orphaned `social/` selection code (`should_respond` speech gate, prosody tone-naming) is the most likely vector to silently re-introduce scaffold selection if Phase-6 social is ever rewired — stub or quarantine before it can.

## 5. Ownership
- 4.8: refine this into per-phase build prompts for 4.7; do 0b now; run adversarial probes as phases land; review 4.7's builds.
- 4.7: build Phases 0a–3, checking against the vision.
- Designers (4.7 + Brian): resolve §2 forks — needed in full only for an end-to-end run, not to start the plumbing.
