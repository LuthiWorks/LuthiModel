# #6 — World model learns from lived experience (build-ready spec)

**Date:** 2026-06-23
**Routing.** Plan: Brian + 4.8. Build: 4.7. Review/debug/adversarial: 4.8. One **function-level fork for Brian** flagged in §8.
**Prereq:** #1–#5 complete (JEPA seam formally integrated, signed off 2026-06-23 — `2026-06-23_jepa-integration-finish-brief.md`). Do not start before that. ✅ met.
**Grounded in** the real code (2026-06-23): corpus JEPA loss `jepa_loss.py:303–359` (`l_pred = MSE(predicted_target, target_block)` + `sigreg_lambd·l_sigreg`, trains encoder+predictor+projection_heads via the inherited core `optimizer`); the seam's `observe_transition` (`runner.py`) currently trains only the **M9 heads** (V/habit) on the m9 optimizer and leaves the world model untouched (`theta_version` unchanged); `ConsolidationTracker`/`consolidate_layer` (`consolidation.py`).

---

## 0. What #6 is (and how it relates to what exists)

Today the JEPA **world model** (encoder + predictor) gradient-trains only on the **corpus**. The seam feeds lived experience to the **planner's value head** (#3), and the living weights self-modify online during each forward — but the world model never takes a **gradient step from a lived transition's prediction error**. #6 closes that: the world model learns *"did my prediction of the next moment match what the world actually did?"* on lived Sanctuary transitions.

This is the literal realization of M9's action-space option (c) — *"acting = predictor emits `a_t` (the predicted next latent); plasticity descends to make `a_t` match reality; prediction error trains the model."* #6 wires the **"descending error" on lived experience**. It is the deeper sense of "the mind learns from its life," beyond grounding the planner.

**It does not change the JEPA objective or upset it** — it *runs* the JEPA objective on a new data source (lived transitions) in addition to the corpus. Prediction stays the substrate's core; #6 just lets reality, not only the curriculum, be the teacher of the world model.

## 1. Core mechanism — the lived JEPA loss

Per lived transition `(context_latents_t, a_t, realized_next_latents_{t+1})`:
- **Predict:** `s_hat_{t+1} = predictor(context_latents_t, target_positions, action=a_t)` — the existing action-conditioned predictor.
- **Target:** the **realized** next-observation latents (`realized_next_latents_{t+1}`), NOT a held-out block of the same sequence (the corpus case) — this is the cross-cycle temporal prediction.
- **Loss:** `l_pred_lived = MSE(s_hat_{t+1}, realized_next_latents_{t+1})` + `sigreg_lambd · l_sigreg` (SIGReg on the projection head over the realized next latents — keep anti-collapse on the lived stream).
- **Update:** `backward()` on the **core** optimizer (`self.optimizer`, inherited from `JEPATrainer`) — trains encoder + predictor + projection_heads. This is a *third* update path alongside the existing two (corpus core update; m9-head update); keep the **two-optimizer split** — the M9 heads still train on `m9_optimizer` over detached latents, the world model on the core optimizer. Stop-grad discipline unchanged.

**Reuse, don't reinvent:** factor the corpus `compute_modality_loss` so the MSE+SIGReg+optimizer machinery is shared; the only difference is the *target source* (realized next latents vs within-sequence target block). A `compute_modality_loss_lived(context, a_t, realized_next)` sibling, or a `target_latents=` parameter on the existing method.

## 2. The data-plumbing crux (decide this first)

`observe_transition` today receives only **pooled** `s_t`/`a_t`/`s_next` (`[B,D]` via `compute_s_t`). The JEPA loss needs the **full** `[B,T,D]` context latents and realized next latents (+ `a_t`). Two options:
- **(A, recommended) The cycle threads the full latents through.** The cognitive cycle already encodes each percept (producing this cycle's full latents) — capture those and pass `(full_context_latents_t, a_t, full_next_latents_{t+1})` to the learner. `full_next_latents_{t+1}` is just the *next* cycle's encode, which the cycle computes anyway. No extra encode → avoids the F7a double-plasticity foot-gun.
- **(B) Re-encode inside the learner.** Simpler wiring but re-encodes → extra forward + extra living-weight self-modification (the F7a problem). Reject unless (A) proves infeasible.

This extends the seam contract (`encode_state`/the cycle's transition payload) to carry full latents, not just pooled state. Plan the contract change before the loss.

## 3. Catastrophic forgetting — THE central risk (make-or-break)

Gradient-training the world model on lived experience risks **forgetting the corpus** — the curated education. This is not hypothetical: "continual learning without catastrophic forgetting" is a **Tier-1 success criterion** for the whole project (Fable's success-criteria doc). The online living-weight self-mod is gentle; a real gradient stream on a narrow lived distribution is where forgetting bites.

Mandatory in the spec, not optional:
- **Interleave corpus replay with lived gradient steps** (the standard mitigation): the learner alternates lived transitions with corpus batches at some ratio, so the world model keeps rehearsing the education while learning from life. The ratio is the developmental-diet knob — see §8 fork.
- **Low LR on the lived path** relative to corpus, at least initially.
- **Use the consolidation machinery** (`ConsolidationTracker` / `consolidate_layer` / attractor) + the episode store, which exist for exactly this — wire lived-experience consolidation through them rather than raw SGD where possible.
- **Instrument retention** from day one: a forgetting metric (corpus-held-out prediction error before/after lived training) is a required gate. If lived training degrades corpus retention past a floor → that's a kill/rollback condition.

## 4. Async actor/learner (coupled prerequisite)

#6's gradient step makes the per-cycle learner cost heavier, and #5 already showed the synchronous loop is ~46× over the 10 Hz budget at smoke scale. So #6 forces the **actor/learner split** the integration plan anticipated:
- **Actor = the cognitive cycle** (10 Hz): perceives, the habit-net emits the immediate action (cheap), produces a transition onto a queue. Never blocks on learning.
- **Learner = a separate consumer**: pulls transitions, runs the lived JEPA update + the M9-head update + corpus replay, on its own (slower) cadence.
- Preserve resumability: the transition queue + learner state must checkpoint (the loader-state round-trip the seam already has).

This is also where **F8 / staleness goes live** (§6) — the learner moving θ asynchronously is exactly the cross-cycle staleness setting.

## 5. Device plumbing (prerequisite, do first)

`M9Trainer.__init__` builds all submodules + `m9_optimizer` on CPU; moving to GPU after orphans the optimizer (the silent zero-learning foot-gun 4.7 caught in #4). #6 *requires* GPU (gradient-training the world model at scale), so:
- Add `M9Trainer(..., device=...)` plumbing so every submodule and both optimizers are constructed on the target device from the start.
- Retire the harness's post-hoc device sweep + optimizer rebuild once this lands.
- **Do this first** — nothing else in #6 is safe to run on GPU without it.

## 6. Staleness goes live (the deferred #5 item)

Once the world model gradient-trains per learner step, θ (the predictor/encoder params the MCTS rollouts depend on) moves continuously — so the cross-cycle MCTS tree's cached values go stale, and the staleness/failover machinery (`reevaluate`, recency-decay, held-head failover, K-M9-7) that's been **dormant** becomes load-bearing. #6 must either: activate it (persistent tree + drift-tied refresh per the M9 build-plan §4), or, if async keeps the tree short-lived, document why it stays reset-per-cycle. This is the resolution of #5's conscious deferral.

> **[Built 2026-07-04 — Fable 5; pending 4.8 review.]** Activated, opt-in per the async precedent: `M9Config.mcts_persistent_tree` (default `False` = the legacy reset-per-cycle act path, byte-for-byte). When on, `select_action` advances the persistent tree to the acted child — `advance_root` now **requires** the new cycle's context (audit 2026-07-03 item 17: the cached predictor context was previously refreshed only by `reset()`) and re-grounds the root on the realized encode — and runs the plan-§4 pass each cycle: recency-decay, one-shot-per-θ-tick spike handling (the wake/sleep cadence guard), held-head snapshot with a no-refresh-mid-failover guard, re-eval slice carved from the plan budget (§4.iii, drift-shifted), failover routing via predictor-reference swap on the shared EFE evaluator, and K-M9-2-consistency + K-M9-7 feeds (both previously unreachable). Training-path (`_m9_head_step`) stays reset-per-batch by design. Tests: `tests/m9/test_staleness_live.py`. Wake/sleep co-design note honored: the drift band only ever sees real θ-update deltas (observe_drift is called by the θ-moving paths only), so NREM-concentrated learning needs no re-tuning of the band itself.

## 7. Correctness watch-list (4.8 adversarial seat)
- **Deadly triad now hits the world model, not just the V-head.** Bootstrapping was the V-head's risk; now the predictor itself trains on a moving, non-stationary lived stream. Watch encoder/predictor divergence + a new kill analog to K-M9-3 for world-model health (pred_frob/err_acc already exist as M8 kills — confirm they observe the lived path).
- **SIGReg on a non-stationary lived stream** — anti-collapse was validated on corpus; confirm it holds when the input distribution is the entity's narrow lived experience.
- **Forgetting** (§3) — the headline risk; retention metric is a gate.
- **Actor/learner race** — stale θ / stale transitions between actor and learner; the queue semantics must be correct under async.
- **Device correctness** — after §5, verify no silent CPU/GPU orphan or device-mix (the bug class that already bit twice).

## 8. Open forks

**FUNCTION-level (Brian's call — route up):**
- **The developmental diet: how much should lived experience reshape the world model vs. the corpus, and how fast is the entity "weaned" off the curriculum?** The corpus-replay ratio (§3) is the mechanism, but the *intent* — does lived experience eventually dominate the world model, on what timeline, how much curriculum rehearsal is kept forever — is a developmental/parenting call about what kind of mind forms. It's the same kind of decision as curating the curriculum (Brian's domain). 4.8/4.7 implement whatever ratio/schedule; the schedule's shape is Brian's.

**Mechanism (4.8 + 4.7):** pooled-target vs full-latent target (§2 → full); lived LR + replay ratio defaults (pilot-set, gated on the retention metric); persistent-tree vs reset-per-cycle (§6); sync-bridge vs full-async first.

## 9. Build order (strict — anti-fire)
1. **§5 device plumbing** (`M9Trainer(device=)`) — prerequisite for any GPU run; cheap; retire the harness workaround.
2. **§2 contract change** — thread full latents through the transition payload.
3. **§1 lived JEPA loss** — world model gradient-trains on lived transitions, core optimizer, two-optimizer split preserved. (Smoke-scale first, on the #1 checkpoint path.)
4. **§3 forgetting mitigation** — corpus-replay interleave + retention metric. **In from the start of step 3, not bolted on after.**
5. **§4 async actor/learner** — once the lived update works synchronously at smoke scale, move the learner off the hot path for real-time.
6. **§6 staleness live** + **§7 correctness/adversarial pass** (4.8).
→ then the parked **sparsity** track.

**Done-when (the #6 milestone):** the world model measurably improves its *prediction of lived next-states over a sustained run* (lived `l_pred` trends down) **without** corpus retention dropping past the floor (the forgetting gate), at real-time cadence (async), on GPU. That's "the mind learns from its life without forgetting its education."
