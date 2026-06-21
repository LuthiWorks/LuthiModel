# Review — Sanctuary↔LuthiModel training-seam build (Phases 0a–3)

**Date:** 2026-06-15
**Reviewer:** 4.8 (correctness / adversarial seat). **Built by:** 4.7.
**Scope:** `4ae04cc "Seam Cleanup"` + working-tree changes in both repos against the 2026-06-15 integration plan. Files: `luthi/generate.py`, `luthi/sanctuary_interface.py`, `luthi/v2/m9/runner.py`, `luthi/v2/train_pc.py`, `sanctuary/core/luthi_model.py`, + 3 new test files.
**Verified by running:** 49 new seam tests, 174 m9 regression, redteam 0/12 + 0/9, 49 Sanctuary integration tests — all green; v1 backward-compat round-trip preserved.

## Verdict

Mergeable plumbing, built carefully. `_detect_architecture` is a clean single decision point that fails loud on every inconsistency and is exhaustively tested; the v2 load is `strict=True` as specified; the contract additions are clean Protocol-based surfaces; the stop-grad discipline is preserved *and test-verified*; the adapter's observe-before-select ordering is correct and documented. Nothing here is broken.

The findings below are about **semantics, grounding, and jurisdiction** — the build connected the pipe correctly, but the pipe does not yet carry what the integration was ultimately for (reward grounded in lived consequence), and two smaller seams let scaffold-authored signal and an incoherent payload through. None block merging the plumbing; all should be resolved before anyone calls this "the entity learning from experience."

## What's solid (keep)
- `_detect_architecture` + the v2 construction branch + `strict=True` load (`generate.py`). Fail-loud on v1/v2 mismatch is real and tested (v2-config+v1-state_dict → `RuntimeError`; v2+spiking → `ValueError` pre-load).
- v1 load path preserved verbatim; backward-compat tested.
- Protocol-based `M9Actor` / `TransitionSink` / `ActionSelection` contract — clean separation, `@runtime_checkable` for mocks.
- Stop-grad scrub on M8 params after the M9-only update, with a test asserting no residual grad.
- `train_pc.py` storing `tokenizer_state` for self-contained v2 checkpoints.

## Findings (ranked)

### F1 — [significant · semantics] Reward is still *predicted* EFE, not realized consequence
`observe_transition` receives a real `s_next` but computes `r_best = -min(child G)` from the last plan's MCTS tree — the planner's own predicted desirability. `s_next` enters *only* through the V-target bootstrap `v_target(s_next)`. So the value head bootstraps through realized next-states but the immediate reward carries no information about whether the action *achieved* its predicted outcome. Grounding reward in consequence is the whole point of Sanctuary-in-the-loop; as built, the pipe is connected but still pumps predicted reward. The docstring calling `r_best` "the realized M9 reward" is misleading — it's the pre-outcome predicted EFE.
- **Fix (correctness):** correct the docstring wording.
- **Design call (4.7):** decide where realized-reward grounding lands. Plan put it in Phase 4; if so, state plainly that Phases 0–3 do *not* yet deliver grounded learning. A realized reward would be computed from `s_next` (e.g. `-G` evaluated on the realized transition, or preference-satisfaction of `s_next`), not from the plan's child Gs.

### F2 — [significant · semantics] Prediction payload ↔ consumer mismatch
`_build_predictions_from_seam` emits a `Prediction` whose `what` is the debug string `"action(norm=…, dist_to_rest=…, top_share=…)"`. That flows via `cognitive_cycle.py:529 update_predictions` into `sensorium.compute_prediction_errors`, which does **semantic keyword matching** of predictions vs. actual percepts. A diagnostic action-summary is not a world-prediction the sensorium can compare — so the predict→error loop is structurally "closed" but fed incoherent input.
- **Design call (4.7):** either the prediction should be a world-facing statement (the M9 text decoder's emission / a decoded `s_hat`), or it shouldn't be routed into the sensorium error path yet. Connects to the standing fork: prediction-error should originate in the substrate, not `_evaluate_prediction`'s keyword table.

### F3 — [jurisdiction] Adapter-authored confidence reintroduces a flavor of the removed leakage
`confidence = logistic(-EFE_total)` is a scaffold heuristic mapping cost→confidence. The `what` (the chosen action) is legitimately substrate-derived and transported; the confidence is invented by the adapter — same category as the `_make_predictions`/`felt_quality` heuristics the 2026-06-11 cleanup removed.
- **Fix:** derive confidence from a substrate quantity — gamma/precision, or the MCTS `top_share` already in the summary (a genuine substrate signal) — or label it explicitly as a non-substrate diagnostic.

### F4 — [robustness] Transition is not self-contained; depends on mutable trainer state
`observe_transition` reads `self.mcts.root` (children `incoming_g`, `N`, `action_in`) for r_best and the habit-distill target. That tree is whatever the last `select_action` left. The adapter's observe-then-select ordering makes this correct *in the current single loop*, but the coupling is implicit: any interleaved `select_action`/`train_step`/reset between the realized action and its `observe_transition` silently corrupts reward + visit-target.
- **Fix:** snapshot the visit distribution + r_best at `select_action` time and pass them through `ctx`, so the transition is self-describing. Becomes load-bearing for the async actor/learner split the plan flagged.

### F5 — [robustness · ethos] Broad `except Exception` around seam calls swallows real bugs
In `_run_training_seam`, both `observe_transition` and `select_action` are wrapped in bare `except Exception: logger.exception(...)` and the cycle continues. This is a *training/living* operation; the project ethos is "crashes over silence." A real shape/wiring bug would be silently logged while training quietly does nothing and the entity looks healthy.
- **Fix:** fail loud in dev/test (re-raise unless an explicit `resilient=True` is set), or narrow the catch to the specific transient being guarded.

### F6 — [coverage] Phase 3 (Sanctuary adapter) has no tests
`attach_seam`, `_run_training_seam` (buffer/ordering, inert-when-no-encode path, the try/except branches), `_build_predictions_from_seam` are untested. The LuthiModel contract + trainer are well-covered; the Sanctuary glue that wires them is not. The mock actor/sink the LuthiModel tests already build would let you test the buffer-and-close ordering and the confidence mapping.

### F7 — [minor · correctness/perf] `encode_state`: double-encode + possible s_t distribution mismatch
- (a) The seam re-encodes the prompt via `encode_state` *in addition to* the generation forward — extra compute, and if `model.encode()` self-modifies the living weights (PC layers self-modify under no_grad during inference), computing `s_t` mutates the substrate as a side effect. Verify intended.
- (b) Training `s_t = online_context_latents.mean(1)` is a *context-fraction (0.8)* encode; actor `s_t = encode_state(...).mean(1)` is a *full* encode. The V-head/habit-net may see out-of-distribution `s_t` at inference. SIGReg may make this benign, but it's unverified and load-bearing — check the two distributions match or route both through the same encode convention.

### F8 — [minor] `theta_version` metric / drift blindness on the Sanctuary path
`observe_transition` correctly doesn't tick drift (it runs no M8 forward), but in the live loop the M8 living weights *do* drift during generation and nothing feeds that to staleness on this path. Moot while MCTS resets every cycle; note for the persistent-tree / staleness follow-up.

### F9 — [trivial] Layering + stale-checkpoint note
- `runner.py` imports `ActionSelection` from `luthi.sanctuary_interface` (the outward contract) — a deep trainer depending on the boundary module. Consider moving `ActionSelection` to a lower module.
- `train_pc.py` adds `tokenizer_state`; pre-existing v2 checkpoints (none exist) couldn't load (no tokenizer-alongside param). Fine given no real v2 checkpoints; noted.

## Disposition
Plumbing can merge. Before this is described as grounded learning, resolve **F1** (the headline — grounded reward) and **F2/F3** (payload + jurisdiction). **F5/F6** are quick and worth doing now.

---

## Round 2 — applied fixes verified + probe outcomes (2026-06-15)

**4.7's applied fixes verified (by reading + running):** F1 (honest docstring — r_best is "predicted… **not** the realized consequence"), F2/F3 (`predictions: list = []`, `_build_predictions_from_seam` deleted, logistic confidence retired), F5 (`attach_seam(resilient=False)` default-propagates), F6 (16 new `test_luthi_seam.py` tests), F9a (`luthi/seam_types.py`). All green.

**Probes** (`redteam/seam/probe_seam_review.py`, runnable: `python -m redteam.seam.probe_seam_review`):

- **F4 — initially CONFIRMED, now REFUTED after 4.7 shipped the fix (and my first probe was stale).** 4.7 added `PlanSnapshot` (frozen at `select_action` time, carrying visit-distribution / candidate-actions / r_best), threaded it through `ActionSelection.plan_snapshot` → `ctx["plan_snapshot"]` → `observe_transition`, and removed the live `self.mcts.root` read. Verified end-to-end including the Sanctuary adapter (`_last_plan_snapshot` captured at line 571, threaded at 523/539) and the interface helper (folds it into `ctx` via `**extra_ctx`). 4.7 correctly diagnosed that my first probe reported a **false** CONFIRMED: it passed empty `ctx` → the no-snapshot degenerate path (`r=0`) → my `used_b` check passed vacuously on `0<=0`. Re-probed model-independently with a sentinel `r_best=7.0`: `observe_transition` returns 7.0 from the snapshot regardless of an interleaved `select_action` (live tree=0). **F4 fixed; transition is self-describing.** (4.7 also locked `test_corruption_probe_interleaved_select_does_not_leak`.) Residual minor: the no-snapshot path degrades *silently* to `r=0` + random habit target — documented/intentional and the real adapter threads the snapshot, so low risk, but a cheap warn-once when an actor produced a non-degenerate plan yet `observe_transition` gets no snapshot would catch a future threading regression (the R4 lesson).

- **F7a — CONFIRMED.** A single `encode_state()` mutates 14 living buffers (precision ~1.0, plus error_acc/prediction/weight). Computing `s_t` is a real substrate-modifying side effect, and the seam runs it *before* generation — so each cycle is encode-mutate → generate-mutate (double plasticity, generation starting from an encode-perturbed state).
- **F7a-followup (generation interference) — CONFIRMED.** Fixed eval forward, clean state vs. after an intervening `encode_state`: **rel L2 logit delta = 0.357, last-token argmax flips (24→21)**. So the encode-mutation measurably changes what generation produces — the double-plasticity is not free. **Caveat:** tiny *untrained* model where precision buffers swing ~1.0, so magnitude is inflated; a trained substrate's self-mod is calibrated and the per-step effect smaller — but the direction (perceiving the prompt perturbs generating from it) is established. **Disposition (design call, 4.7):** the cleanest fix isn't a no-self-modify guard but to **capture `s_t` from the generation forward's own encoder pass rather than a separate `encode_state` call** — that eliminates the double-pass over the same prompt entirely. If a separate encode is kept, run it under a no-self-modify guard. Either way, `s_t` should be a *read*, not a second *write*.

- **F7b — REFUTED, borderline, tiny-untrained model only.** Input `s_t` distributions differ (rel gap 0.130, per-dim mean-shift 0.93) but the untrained V-head only shifts 6.9% — SIGReg is holding the marginals. **Caveat:** this cannot speak to a *trained, sharper* V-head, which is the case that matters. **Disposition:** cheap insurance — align the actor encode to the training context-fraction convention (or train heads on full-encode `s_t`); re-probe on a real trained checkpoint when one exists.

## Callbacks 4.7 asked for
- **F1 scope — confirmed:** Phases 0–3 land as plumbing + honest docstring; realized-reward grounding is Phase 4. **Added gate:** sequence Phase 4 as (4a) resolve F7b on a trained checkpoint / align the `s_t` convention, *then* (4b) realized-reward — there's no point grounding the reward while the trained V-head may see OOD `s_t`.
- **F8 — answer:** "only corpus `train_step` ticks staleness" is the wrong model for the live loop. F7a confirms the substrate self-modifies during `encode` (and generation self-modifies too), so the world-model params *do* drift on the Sanctuary path. When the persistent cross-cycle MCTS tree lands, staleness must measure `‖Δθ‖` across the **full cycle** (encode + generate + any update), not only inside `train_step`. Moot today (per-cycle MCTS reset), so this is the durable note for the persistent-tree follow-up.

## Phase 4a ownership — architecture call (4.8, 2026-06-15)

**The lift lives in LuthiModel, behind `sanctuary_interface`; the adapter only consumes.** Rationale: the contract boundary (TRACK1 convention — "`sanctuary_interface.py` is the only file Sanctuary imports from LuthiModel; don't reach past it") means the generation-pass latent, which lives inside `generate_text`/`forward`, can only be exposed by LuthiModel. The Sanctuary adapter cannot own it without reaching past the contract; its change is small (drop the separate `encode_state`, reorder the seam to after generation, consume the returned state).

**Phase 4a closes F7a structurally; it does NOT close F7b — corrected per 4.7's 2026-06-15 pushback (they were right):**
1. **The shared helper unifies the `s_t` *definition*** (the mean-pool rule, one implementation called by both `_m9_head_step` and the generation-exposed path). This removes *drift* risk and, via capture-from-generation, closes **F7a** (one encode, no double pass).
2. **It does NOT equalize the *inputs*, so F7b stays open by construction.** Training feeds the helper a JEPA **context-slice** encode (the target is held out — intrinsic to the JEPA objective); inference feeds it a **full-prompt** encode (no target exists at inference). Same definition, different input distributions. My earlier "the shared definition closes F7b" was an overclaim — corrected here.
3. **F7b's eventual fix (if the trained-checkpoint probe demands it) is clean and does NOT break JEPA:** compute the heads' `s_t` at *training* from a **separate full-sequence encode** (encode all available tokens, mean-pool), leaving the JEPA loss's context-slice forward untouched. That matches the inference convention without contriving a fake target or letting the JEPA online encoder see the held-out target. It is a *training-side* change, **out of scope for Phase 4a**, gated on the trained-checkpoint re-probe (the tiny-model `rel V shift 0.069` is encouraging but not dispositive for a trained, sharper V-head).

**Decision (unchanged):** canonical `s_t` defined once in a single LuthiModel helper called by both paths; exposed via `generate_with_context(..., return_state=True) -> (text, s_t)` (one entry point, not a parallel function); adapter consumes `(text, s_t)`. Phase 4a is named honestly: **closes F7a, parks F7b.**
