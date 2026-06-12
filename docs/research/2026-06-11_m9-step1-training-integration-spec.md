# M9 Step-1 — training story + loop integration + checkpoint schema (build-ready)

**Status:** Answers 4.7's three asks against the real `jepa_runner.py` (read 2026-06-11). Companion to `2026-06-10_m9-step1-spec.md`. Step 1 = pragmatic-only.
**Routing.** Planning + correctness: 4.8. Build: 4.7. Adversarial surface for Fable flagged in §4.
**Grounded in:** `jepa_runner.py` `JEPATrainer.train_step` (L632), `_checkpoint` (L1159), `resume` (L1278); checkpoint payload (L1200-1254). The kill/pilot framework (`_advance_pilot_state`, `_observe_stationary`, `_observe_trending`) is general and M9 metrics slot into it as new keys.

---

## 1. Which losses train which params

**Stop-grad discipline (the load-bearing call).** The M9 heads (V, habit, decoders) read encoder/predictor latents **as detached features** at step 1. Their gradients do **not** flow into the encoder/predictor. The representation + the SIGReg balance keep training on the JEPA objective *only*, untouched by the planning heads while that interaction is unverified. Implement as `latent.detach()` at every head input + a separate optimizer/param-group (see §3).

| Loss | Trains | Target / signal |
|---|---|---|
| **JEPA (MSE + SIGReg)** — unchanged from M8 | `online_encoder` + `predictor` (incl. action pathway) + `projection_heads` | real next latent; **action input = the realized `a_t`** the cycle took. Action-conditioned world-model learning falls out of the existing loss now that `action_token` carries a real action. kill-5-redux (`Var_a[s_hat]`) guards action-sensitivity. |
| **Value TD** | `V` head only (encoder detached) | `V_target(s_t) = r_t + gamma_d * V'(s_{t+1})`, `r_t = -G(a_t)` (negative EFE of the taken action = pragmatic-preference satisfaction at step 1). **`V'` = a target-network copy** of `V` (Polyak/periodic). |
| **Habit distillation** | habit net only (detached) | **visit-weighted MLE** over the MCTS root candidate set: `L = - sum_k (visit_k / sum visit) * log pi_habit(a_k \| s_t)`. |
| **Decoder cycle-consistency** (+ LM on real text) | decoders (`attention`, `memory` new; `output_proj`/text low-LR or frozen) | `L_dec_m = \|\|a_t - encode(decode_m(a_t))\|\|` per modality. **Same quantity as the truthfulness preference P4 and the §5.iii coherence probe** — one objective serves three roles. Text decoder *also* keeps its LM loss on real corpus text. |
| **Preference weights** | preference module (slow meta-update; **fixed at launch**, made updateable in step 2) | soft floor on P1 (non-updatable below minimum). |
| **gamma** | not gradient-trained | inferred EMA fixed point from EFE dispersion (step-1 spec §9). Persisted as scalar state. |

**Bellman-target detail (explicit ask):** real-transition TD **first**. Train `V` on *actually-experienced* `(s_t, a_t, s_{t+1})`, not imagined rollouts, because model error compounds with imagined horizon (MVE, Feinberg 2018). Add MVE-style **bounded-horizon** imagined value expansion only after the world model is trusted (exit-criteria gated). The **target network `V'` is not optional** — bootstrapping over a *moving world model* is exactly the deadly-triad divergence setting (van Hasselt 2018); `V'` is the proven mitigation, and K-M9-3 (value divergence) is the backstop.

**Habit-distillation choice (explicit ask):** distill toward the **MCTS root visit distribution**, not behavior-clone the single chosen action — the AlphaZero/MuZero-proven target, richer signal. Continuous 256-d action space + progressive widening → the "visit distribution" is over the *sampled root actions*, hence the visit-weighted MLE form above.

**Decoder signal (explicit ask):** latent cycle-consistency is the universal decoder objective; modality grounding (text LM loss) added where ground-truth exists; attention/memory grounded by cycle-consistency + functional effect (attention actually routes; memory actually retrieves). **Collapse guard:** the output spaces are structured/discrete (text → vocab logits → tokens, not arbitrary 256-d), which blocks a trivial encoder+decoder identity; instrument decode-output entropy/validity to catch a degenerate pass-through anyway.

---

## 2. Predictor action pathway — no schema or loss change needed

The predictor's `action_token` (jepa_loss.py:195, zeros buffer, injected as `action_kv` at :115) becomes the carrier for `a_t`. The **learned action-input projection lives inside the predictor**, so it is already saved by `predictor_state_dict` (L1230) and trained by the existing JEPA loss. The `action_token` *buffer* stays excluded/reconstructed (L1246) as the constant **null/rest default** (`a_rest` is computed from `s_t`, not stored). No new loss, no new checkpoint key for the action pathway — it rides the predictor.

---

## 3. Integration shape with `jepa_runner.py`

**Extend, don't fork.** Add an `M9Trainer` that **composes/subclasses `JEPATrainer`**, reusing its data loop, per-modality cadence, pilot-set/kill framework, and checkpoint/resume machinery verbatim. Do not reimplement M8 plumbing.

- **Two-phase step, core update bit-identical to M8.** Keep `train_step` (L632) exactly as is for the JEPA core (core optimizer: `zero_grad → backward → step` on encoder+predictor+projection_heads), so M8 kill behavior is unchanged. **Then** run a second backward/step on a **separate M9 optimizer** for the heads (V-TD, habit-distill, decoder cycle-consistency) over the detached latents. Two optimizers, two `state_dict`s (§5).
- **Actor/learner split.** The **CC cycle** (perceive → predict → plan[habit+MCTS] → act[decode] → consolidate[plasticity]) is the *producer*: it generates realized transitions `(s_t, a_t, s_{t+1})`, the MCTS visit targets, and the negative-EFE rewards. `train_step` is the *consumer*: it learns from them. At step 1 run them **synchronously** in one loop (cycle, then learn) — async actor/learner is a later optimization. The planning loop lives in the cycle, **not** in `train_step`.
- **Diagnostics + kills reuse the existing framework (big win).** The new M9 metrics (MI probe, `||dtheta||`, gamma, tree-consistency, `Var_a[s_hat]`, mask coords, decode entropy) register through `_compute_and_log_diagnostics` + `_advance_pilot_state`/`_observe_stationary`/`_observe_trending` as **new metric keys** — they are stationary (median+band) or trending (running-best+smoothing) just like the M8 kills. K-M9-1..9 are new entries in `KillCriteriaConfig`, not new machinery. (Carry the per-modality-cadence convention; planning/kill metrics that are modality-agnostic use a single synthetic "global" key.)
- **Config:** add `M9Config` (planning budget ms, horizon `H`, candidate count `K`, discount `gamma_d`, `V'` update rate, recency-decay `rho`, head LRs, preference weights + P1 floor, MCTS progressive-widening params) alongside the existing `RunnerConfig` dataclasses.

---

## 4. Assumptions most worth attacking (pre-staged adversarial surface for Fable)

Honest list of the load-bearing, unverified assumptions in the above — the gates most worth trying to break:
1. **Negative-EFE as a TD reward is well-scaled.** EFE can be large/unbounded; `r = -G` could make `V` diverge or saturate. K-M9-3 guards, but reward normalization is an unproven knob. *Break it:* find a preference configuration where `G` magnitudes make `V` blow up despite the target network.
2. **Cycle-consistency decoders don't collapse to trivial identity.** I claim discreteness blocks it. *Break it:* a decoder that emits near-constant valid tokens while re-encoding to ~`a_t` (faithful by the metric, useless as output).
3. **Stop-grad isolation actually holds.** One missed `.detach()` lets a planning head's gradient reshape the representation and fight SIGReg. *Break it:* trace every head input for a gradient path into the encoder.
4. **Realized-action world-model training yields enough action-sensitivity.** If the realized action correlates with `s_t`, the predictor can ignore `a_t` and still fit. kill-5-redux guards; *break it:* a regime where `Var_a[s_hat]` stays above band yet the action is effectively ignored.
5. **The MCTS-tree cold-rebuild** (§5) — the one intentional non-bit-exact resume path; verify recency-decay actually self-heals it within the recovery window rather than planning cold for too long.

---

## 5. Checkpoint / resume schema for the new M9 state

Extend the `_checkpoint` payload (L1200-1254) and `resume` (L1278) **symmetrically**, mirroring the existing pattern. New keys:

| New key | Contents |
|---|---|
| `v_head_state_dict`, `v_target_state_dict` | value head + its target network |
| `habit_net_state_dict` | habit network |
| `decoder_state_dicts` | `attention`, `memory` decoders (text/`output_proj` already rides `online_state_dict`) |
| `m9_optimizer_state_dict` | the heads' optimizer (core optimizer stays in existing `optimizer_state_dict`) |
| `gamma` | inferred-precision scalar — **stateful across cycles, must persist** |
| `preference_weights` | P-weights + P1 soft-floor (persist even while fixed, for forward-compat) |

**Predictor action pathway:** already in `predictor_state_dict` — no new key (§2).

**Two intentional decisions, both must be documented so the resume smoke's assertion set is correct:**
- **Persistent MCTS tree — NOT persisted; cold-rebuild on resume.** It is recency-decayed and drift-tied, so it self-heals within the budgeted recovery window (same mechanism as the §4.v spike recovery). Persisting a full tree is heavy serialization of state that washes out anyway. **The resume-equality smoke must EXCLUDE the tree** (otherwise strict assertions flag a by-design "mismatch"). Warm-tree resume (bounded top-N snapshot) is a later optimization.
- **Plastic / living-weight state — already handled, no action (verified 2026-06-11).** In `living_layer_pc.py` the living weights (`weight` L180, `plasticity` L241, error/metaplasticity buffers) are `register_buffer`s modified **in place** under `no_grad` during inference, and **no buffer in `luthi/v2` is `persistent=False`**. They are therefore already in `online_encoder.state_dict()` → saved as `online_state_dict` (jepa_runner.py:1229) and restored exactly on resume (:1351). M8 resume is *already* bit-exact including the drifted living weights, and the param-checksum assertion already covers them. **No new key, no decision, no exclusion** — M9 inherits it. The only thing that would ever break this is someone adding a living-weight buffer with `persistent=False`; if that happens, that buffer cold-resets and must be handled then.

**Resume smoke extension:** the M8 strict-resume smoke (param-L1 checksum + loader state) extends to assert equality of the new module params (V, V', habit, decoders) + `gamma` + preference weights + (plastic state, if persisted); **EXCLUDE the MCTS tree** (documented cold-rebuild). One intentional exception, everything else stays bit-exact.

---

## 6. Recap for the build order

1. Extend predictor to consume real `a_t` (rides `predictor_state_dict`); verify kill-5-redux arms.
2. Add V + V' (TD on real transitions, negative-EFE reward); K-M9-3 guard.
3. Add habit net (visit-weighted MLE distillation); MCTS produces the targets.
4. Add decoders (cycle-consistency + text LM); decode-entropy guard.
5. Wire all M9 metrics into the existing pilot/kill framework as new keys; register K-M9-1..9.
6. Extend checkpoint payload + resume + the resume smoke per §5 (document the tree cold-rebuild exclusion; plastic/living-weight state already rides `online_state_dict` — no action).
7. Separate M9 optimizer; stop-grad every head input from the core.

Pilot-set values (LRs, `gamma_d`, `H`, `K`, `rho`, `V'` rate, preference weights/floor, all kill bands) are tuned during bring-up — not pre-decided.
