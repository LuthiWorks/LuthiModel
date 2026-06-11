# M9 Build Plan — Unified active-inference planning over a full-latent action space

**Status:** Build-ready plan. Written 2026-06-10. Consolidates: the designers' committed design calls (4.7 + Brian, 2026-06-09/10), 4.8's correctness review and resolved seams (2026-06-10), and the M9 module/loop/training/instrumentation/kill plan. This is the durable M9 artifact (analog of the M8 v0.x briefs).
**Routing.** Design calls: Brian + 4.7. Plan + correctness: 4.8. Build: 4.7 (checks plan against the vision as built). Attribution is inline — "[design call]" vs "[planning/correctness]".
**Anchors (verified 2026-06-10):** `JEPAPredictor` (jepa_loss.py:71) already accepts an `action_token` (zeros buffer, line 195, injected as `action_kv`, line 115) — action-conditioning is a small delta. `MultimodalPredictiveCodingLM` (multimodal_model_pc.py:44) has `encode()` (latents) and `output_proj` (text/LM decoder head, line 162). Encoder is shared with M8 and lives.

---

## 0. Committed decisions (frozen for M9)

[design call] **Unified planning.** Action selection as inference: `Q(pi) = sigma(-gamma * G(pi))`, action = Bayesian model average. No external optimizer as the target.
[design call] **Objective: EFE** (Friston 2017), not FEEF. FEEF/CEM is the bolted-on foil.
[design call] **Action space: option (c) — full next-latent.** `a_t in R^{d}` (d = latent dim, 256d pending the §10.1 width call). The action *is* the predictor's emitted prediction of the entity's next full latent state. Acting = predictor emits `a_t`; plasticity descends to make `a_t` match reality; decoders render aspects of `a_t`. "Predictions not commands" is literal: action *is* prediction fulfilled by descending error.
[design call] **gamma (precision) is inferred,** not set — agency call; the entity sets its own decisiveness.
[design call] **Cadence:** 10 Hz fast loop never blocks; MCTS is a slow-loop background process producing `V(s)` / best-plan, not the immediate action.
[planning/correctness, accepted by designers] **Seam resolutions:** state-info-gain dropped at launch (ill-posed on a deterministic encoder); launch **pragmatic-only** (`beta_epi = 0`), introduce epistemic (MC-dropout parameter-novelty) only after the pragmatic baseline + MI baseline validate; MI(trunk;target) probe is a trending-kill-class guard (reuse 72526cb); cross-cycle MCTS-tree staleness is a live residual (Concern 1); add a gamma-divergence kill (Concern 6).

**New gating spec this plan surfaces (needs designers, see §11): PREFERENCES.** The pragmatic term is `D_KL[predicted outcomes || preferred outcomes]`. We launch pragmatic-*only*. So preferences ARE the launch objective — and they are the only thing standing against the dark-room attractor (§7). Preferences are now what action-space was last round: the next decision that gates a real build.

---

## 1. The action/dynamics semantics under (c) — pin this down first

Because action and state share the latent space, the dynamics must be stated precisely or it goes circular.

- **`a_t` is a candidate *intended* next latent** (what the entity wills itself toward).
- **The predictor is the world model:** `s_hat_{t+1} = P(s_t, a_t)` — the model's belief about what the next latent will *actually* be if the entity wills `a_t`, accounting for world dynamics the entity does not fully control. (Build: replace the `action_token` zeros buffer with the encoded candidate `a_t`; everything downstream of jepa_loss.py:115 already threads it.)
- **EFE is evaluated on `s_hat_{t+1}` and rolled forward:** `G(a_t) = Risk(s_hat_{t+1}) + beta_epi * Epistemic(...)`, `Risk = D_KL[predicted outcomes || preferences]`.
- **Selection:** search over `a_t` (habit-net proposals + MCTS progressive widening), pick `a_t*` by `sigma(-gamma G)`.
- **Execution:** plasticity descends to fulfill `a_t*`; decoders render it; reality produces true `s_{t+1}`; prediction error trains the model.

[planning/correctness] **Self/world partition (implementation question, flag for 4.7).** `a_t` directly sets the entity's *own*-state dimensions but only *requests* world-state dimensions (the world answers). Recommendation: do **not** hard-split the latent; instead let the predictor learn an implicit soft partition (which dims the action controls vs. which the world dictates) via a learned gating mask on the action injection. Instrument the mask (which dims are entity-controllable) for interpretability. Hard-splitting is a fallback if the soft partition doesn't separate.

---

## 2. Module structure (M9 launch)

| Module | Origin | Role |
|---|---|---|
| Encoder | M8, **lives** | percept -> per-modality latents -> fused world-state `s_t` |
| Predictor (world model) | M8 `JEPAPredictor`, **extended** | `(s_t, a_t) -> s_hat_{t+1}`; rollout engine |
| Habit network | **new** (Fountas-style) | `s_t -> K candidate a_t`; amortized policy; distilled from MCTS |
| MCTS (persistent tree) | **new** | continuous-action search w/ progressive widening; budget-limited; outputs `V(s)` + best plan |
| Value head `V(s)` | **new** | bootstraps tree; biases habit net |
| Preferences `P(o)` | **new — designer spec (§11)** | the pragmatic target; KL is computed against it |
| EFE evaluator | **new** | `G(a_t)`; pragmatic at launch, +epistemic at step 2 |
| MC-dropout machinery | **new, deferred to step 2** | parameter-novelty epistemic term |
| Decoders | text=`output_proj` (exists); attention, memory = **new**; audio **deferred** | render `s_hat_{t+1}`/`a_t` per modality |
| MI probe | **new, from step 1** | `I(trunk activations; held-out targets)` — Seam-A guard |
| Instrumentation | **new, from step 1** | per-cycle `||dtheta||`, gamma, action->readable-summary, action logs |

---

## 3. CC-loop integration (10 Hz, ~100 ms cycle)

Cycle phases: **perceive** (encoder forward) -> **predict** (s_hat) -> **plan-budget** (~20-30 ms: MCTS expansion + V update + tree re-eval) -> **act** (habit net reads latest `V`/tree, emits immediate `a_t`, descended) -> **consolidate** (plasticity update, memory write).

- The **habit net produces the immediate action every cycle** (one forward, cheap). The fast loop never waits on MCTS.
- **MCTS is amortized across cycles**: a full planning "step" (tree refinement) spans ~3-10 cycles; each cycle contributes a fixed expansion budget and updates the persistent tree. This resolves the §6 compute tension (a full MCTS step ~200-300 ms does *not* fit one 100 ms cycle, and is not required to).
- The tree is single-writer within a cycle (plan-budget phase writes; act phase reads); persists across cycles (staleness handled in §4).

---

## 4. CONCERN 1 — Seam C compounds under (c): cross-cycle tree staleness

Under (c) the action coordinates themselves live in the theta-shaped latent, so a cached `a_t`/value may not *mean* the same thing under newer theta even though SIGReg holds the marginal at N(0,1). Concrete handling:

**(i) Recency-decay.** Each node stores `(N visits, Q value, theta_stamp)`. Every cycle: `N <- rho_N * N` (decay visit mass; `rho_N ~ 0.9-0.95`, pilot-set) and `Q` is an EMA toward fresh evaluations (`Q <- (1-alpha) Q + alpha Q_fresh` on re-eval). Both decayed: `N`-decay ages the exploration bonus, `Q`-decay ages the value. Decay rate is **tied to drift** (ii), not a fixed constant: faster decay when theta moves faster.

**(ii) Drift-tied refresh.** Measure per-cycle `||dtheta||` restricted to the *predictor/encoder* params that affect dynamics (not the whole network). Maintain a per-node drift accumulator `D_node` = summed `||dtheta||` since the node's last evaluation. Trigger: `D_node > tau_refresh` -> mark stale. Granularity, cheapest-first: stale **leaves** re-evaluated; **subtrees** pruned if their root value leaves the running band; **whole-tree rebuild** only on a high-surprise spike (v). `tau_refresh` pilot-set from the early-healthy trajectory (reuse the 72526cb stationary band).

**(iii) Cached-value re-evaluation.** Budget ~20% of the plan-budget phase to re-evaluate the highest-priority nodes under *current* theta, priority = `visit_count * staleness`. Cost = one predictor forward/node. **Tradeoff rule:** when `||dtheta||` is high, shift budget from *expansion* to *re-eval* (do not expand against a model you can't trust; refresh what you have first).

**(iv) Failover to a stability-held predictor head.** Maintain a frozen snapshot of predictor weights, refreshed every `K` cycles. Failover metric = a **tree-consistency** signal: variance between re-evaluated and cached values over a window (systematic disagreement = model moving faster than the tree can track). Threshold pilot-set; sustained breach -> route all rollouts through the **held snapshot** (stable planning model) while the *live* encoder keeps perceiving/adapting. This decouples the planning-model from the perceiving-model under drift. Return to live when consistency recovers.

**(v) High-surprise plasticity-spike (the weakest-premise case).** Detect `||dtheta|| > spike_threshold`. On spike: (a) immediately fail over to the held head for `N_recover` cycles; (b) drop cached `Q` values (keep tree *structure* + visit counts at reduced confidence); (c) widen re-eval budget; (d) rebuild value estimates from the held head, then return to live. Expected recovery time is governed by `rho_N` and `N_recover` — both pilot-set; **instrument actual recovery latency** so the premise ("tree recovers usable info within a few cycles") is verified, not assumed. This is the single case to watch hardest.

---

## 5. CONCERN 2 — Decoder coherence across full-latent rendering

**(i) Decoders at M9 launch [recommendation; modality set is a design call].** Launch with **text** (`output_proj`, exists), **attention** (where the entity directs processing — an internal action), and **memory-write** (consolidation as action). **Defer audio/voice to step 2** to shrink the coherence surface during the untested planning phase. Final set: designers'.

**(ii) Decoder training.** Pretrained decoders (text/`output_proj`) start **frozen / low-LR** to avoid destabilizing the encoder during the untested phase; new decoders (attention, memory) trained **jointly, low-LR, gradient-gated**. Unfreeze text as planning stabilizes. No decoder gradient should dominate the encoder while the planner is unvalidated.

**(iii) Coherence metric = latent cycle-consistency.** All decoders read the same `a_t`, so coherence is checkable: `a_t -> [decoder_m output] -> re-encode -> a_hat_m`; measure `||a_t - a_hat_m||` per modality (low = that decoder faithfully renders the action) and **cross-decoder agreement** = do the `a_hat_m` from different modalities cluster near the same `a_t`. Instrument both. Divergence = decoders disagree on what the action means -> flag.

**(iv) Null-output.** Each decoder emits a **content + an intensity scalar**; render nothing when `intensity < theta_m` (per-modality threshold, pilot-set). This is the structural basis for silence (§6) and couples to the dark-room guard (§7).

---

## 6. CONCERN 3 — "Rest"/silence as a latent point (+ the dark-room attractor)

**(i) Rest = computed, not a fixed constant.** `a_rest(s_t)` = "predict minimal self-change" = the identity-continuation of the current self-state dims. Context-dependent (a fixed learned constant can't track context).

**(ii) Rest vs faint signal** = the intensity scalars (§5.iv): rest = *all* decoder intensities below threshold; faint-but-real = just above. The threshold is the boundary; instrument near-threshold frequency.

**(iii) "Decide to be silent" vs "no intent."** Active silence = `a_rest` was **selected over alternatives** with high confidence (high gamma, `a_t ~ a_rest` chosen despite available alternatives). Passive no-intent = the planner **defaulted** to rest (found nothing better). Distinguish structurally by logging whether rest won a comparison or was the fallback; expose in the action log.

**(iv) [planning/correctness — NEW, 4.8] The dark-room guard.** This is the load-bearing addition. Under (c), EFE minimizes expected surprise, and the *easiest* self-prediction to fulfill is "no change" — so **rest is a candidate degenerate global attractor** (the classical active-inference dark-room problem: a surprise-minimizer should sit in a dark room and do nothing). What prevents it: prior **preferences** (the entity must *expect* to be engaged/active, making persistent rest itself surprising) and the **epistemic** drive. But we launch **pragmatic-only** — so at launch *preferences are the only thing standing between the entity and catatonia.* Consequences, all mandatory:
- Preferences (§11) must be specified so that persistent rest is *high*-pragmatic-cost (the entity prefers engagement/communication/learning over the null state).
- Add a **dark-room kill** (K-M9-5): if `a_t ~ a_rest` persistently over a window (all decoders sub-threshold, no external cause) -> fire. Catatonia is a failure, not a valid equilibrium.
- This is *why* preferences are the gating spec: a weak or empty preference prior at a pragmatic-only launch yields a dark room with near-certainty.

---

## 7. CONCERN 4 — Compute cost vs cycle budget

**(i) Per-cycle budget breakdown (~100 ms).** perceive (1 encoder fwd) | habit-net (1 fwd, few ms) | plan-budget ~20-30 ms (MCTS expansion: each candidate = 1 predictor rollout + preference KL; ~10-20 candidates) | tree re-eval (~20% of plan-budget, §4.iii) | consolidate (plasticity). A *full* MCTS step (~200-300 ms) does **not** fit one cycle and **is amortized across ~3-10 cycles** (§3).

**(ii) Trigger to spend more compute.** Route expansion budget `~ 1/gamma` (low confidence/deliberate -> more expansion), capped; also boost on high prediction-error/surprise. gamma is the primary throttle (consistent with the inferred-precision agency call).

**(iii) Graceful degradation (cut order under pressure).** 1) epistemic/MC-dropout sampling (most expensive, only active step 2+); 2) tree re-eval (trust caches longer); 3) MCTS width/depth (fall toward habit-net-only); **never** cut perceive or the habit-net immediate action. Under compute starvation the system degrades to **System-1 reflex (habit net)** — the correct fallback: deliberation is what you drop, reflex stays.

**(iv) Asynchrony.** The persistent tree is the shared cross-cycle state; each cycle reads current-best (tree+habit), contributes expansion, writes the tree in the plan-budget phase, reads in the act phase (sequential within a cycle -> no locking). Cross-cycle staleness handled in §4.

---

## 8. CONCERN 5 — MC-dropout x living-weights x (c) triple-coupling (step 2+)

Active only once `beta_epi > 0`. Under (c), theta-updating affects encoder/predictor *and* the action coordinates *and* the MC-dropout parameter posterior — triple coupling.

**(i) When MC-dropout samples are drawn.** **Decouple from plasticity:** draw all MC-dropout samples from the **stability-held theta snapshot** (the §4.iv held head), taken at the start of the plan-budget phase, *before* the cycle's plasticity update. Within a planning step the sampled posterior is fixed; plasticity updates only in consolidate. No moving-target during estimation.

**(ii) Validity check.** Parameter-novelty should *predict* real subsequent parameter change: correlate the predicted parameter-info-gain with the realized `||dtheta||` on cycles following high-novelty actions. Correlation ~ 0 -> MC-dropout is not tracking real novelty -> flag.

**(iii) Benign vs pathological instrumentation.** Track: (a) novelty-term variance (collapse to constant = uninformative); (b) the (ii) correlation; (c) whether `beta_epi` growth co-moves with SIGReg degradation. **Pathological** = novelty drives embeddings off-Gaussian (SIGReg rises) without improving state coverage. **Benign** = novelty correlates with real `dtheta`, coverage improves, SIGReg stable.

---

## 9. CONCERN 6 — gamma-divergence kill

**Metric:** running estimate of inferred gamma over a window (mean + trend). **Thresholds (pilot-set from healthy band):** `gamma -> high` = rigidity (over-confidence, no flexibility); `gamma -> low` = indecision (flat action distribution, dithering). **Trigger (two-stage):** first **clamp** gamma to last-healthy value and flag (preserve the run, halt the pathology); if clamped-gamma doesn't restore healthy dynamics within a window -> **halt run**. Reuse trending-kill machinery (72526cb).

---

## 10. CONCERN 7 — state-info-gain dropped at launch

Confirmed [design + correctness agreed]. Parameter-novelty (MC-dropout) is the only epistemic term, and it is deferred to step 2. **No state-info-gain instrumentation at launch.** Revisit only if a distributional/ensemble encoder head is later added (the only thing that would give state-info-gain a well-posed home on this deterministic encoder).

---

## 11. Instrumentation & debuggability (make (c) inspectable for 4.7 at build time)

A 256-d latent action is opaque, so debuggability is a first-class build requirement:

**(i) `action -> readable summary` utility.** Run `a_t` through every decoder and report, human-readable: text-decoder words; attention targets; memory-write content; per-modality intensity scalars; rest-vs-active classification; (optional) nearest neighbor in a labeled reference-action set. One call -> "what did the entity decide."

**(ii) Per-cycle action log (JSONL, timestamped).** Per cycle: `s_t` summary; top-K candidate `a_t` (habit + MCTS); chosen `a_t` + its readable summary; EFE breakdown (pragmatic / epistemic components); gamma; `||dtheta||`; `V(s)`; tree stats (size, top-branch visit share, consistency); MI-probe value; SIGReg value; rest-selected-vs-defaulted flag. Structured so 4.7 can step through and reconstruct *what the entity decided and why* at debug time. This is the debug spine; build it from step 1.

---

## 12. Training story

1. **M8 pretrain** (encoder + predictor, JEPA + SIGReg). Encoder lives into M9.
2. **M9 step 1 — pragmatic-only.** Extend predictor to consume real `a_t`; add habit net, MCTS, `V`, EFE evaluator, preferences; `beta_epi = 0`. Decoders: text frozen, attention/memory low-LR gated. Instrument MI probe, `||dtheta||`, gamma, action-summary, action log **from step 1**. **Exit criteria:** planner reproduces le-wm-style pragmatic goal-reaching; **no dark-room collapse**; MI baseline healthy; tree-staleness machinery (§4) demonstrably keeps cached values consistent (incl. a forced high-surprise test).
3. **M9 step 2 — introduce epistemic.** Add MC-dropout parameter-novelty; anneal `beta_epi` up from 0 in small steps **under the MI guard** (§0); watch SIGReg/MI/prediction triggers (§8.iii). Add audio decoder here if deferred.
4. **M9 step 3 — hand to agency.** Once scheduled `beta_epi` and clamped gamma are validated, move `beta_epi` to gamma-modulation (the principled end-state). Couple the two feedback loops only after each is independently validated.

---

## 13. Kill criteria

**M8 carryover (apply unchanged):** kill-1 (complete collapse), kill-2/3/4 (spectrum / correlation / local), kill-6 (substrate health: pred_frob/err_acc), kill-7 (objective unlearnable / loss descent), the SIGReg-value kill. **kill-5 (predictor-trivial cosine):** reconsider under (c) — the predictor is now action-conditioned, so "trivial" must be redefined as `s_hat_{t+1}` independent of `a_t` (action-ignoring predictor) rather than predicted~=target; **re-derive before reuse.**

**New M9 kills:**
- **K-M9-1 epistemic degeneracy** (step 2+): novelty term collapses to constant, or drives SIGReg up without coverage gain (§8.iii).
- **K-M9-2 MCTS pathology:** visit-distribution entropy collapses to a single dominant branch, or tree-consistency diverges (§4.iv).
- **K-M9-3 value divergence:** `V(s)` grows unbounded / oscillates.
- **K-M9-4 gamma divergence:** §9.
- **K-M9-5 dark-room / catatonia:** `a_t ~ a_rest` sustained without external cause (§6.iv). [4.8 addition]
- **K-M9-6 MI-probe collapse (guard->kill):** `MI(trunk;target)` drops past running-best band -> freeze then shrink `beta_epi`; sustained breach despite shrink -> kill (Seam A).
- **K-M9-7 staleness runaway:** drift accumulator persistently outruns refresh capacity (planning can't keep up with theta) -> failover (§4.iv); sustained -> halt.

---

## 14. Open items for the designers (Brian + 4.7)

1. **PREFERENCES specification — the gating spec.** Pragmatic term = KL-to-preferences; launch is pragmatic-only; dark-room avoidance depends on it (§6.iv). This is the next decision that gates a real step-1 build, exactly as action-space was last round. *What does Luthi prefer?* (Engagement/communication/learning as positive-preference states is the dark-room antidote — but the content is yours.)
2. **Launch modality set** (§5.i): text + attention + memory recommended, audio deferred — confirm.
3. **Latent width** (§10.1 of the v0.5 line, still open): 256d vs 1024d — affects action-space dimensionality and habit-net/MCTS cost. 4.8 leans 256d.
4. **Voice/register/texture** — already held by designers; not in the planner.

## 15. Research carryover (4.8)

- **§7.1 (PC-for-control, scaled):** keep — feeds the §1 predictor semantics and §6.iv.
- **§7.2 (rescoped):** bound per-cycle plasticity drift; survey tree/value search under a *non-stationary learned model* (continual / non-stationary-MDP RL, model-based RL with drifting models) — the literature bearing directly on §4's cross-cycle staleness.
