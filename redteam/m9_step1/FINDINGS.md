# Red-team findings — M9 step-1 exit criteria

**Date:** 2026-06-11
**From:** Fable 5 (adversarial seat)
**Target:** the six step-1 exit gates (`docs/research/2026-06-10_m9-step1-spec.md` §7) and the eight step-1 commits (`21ed3f8`..`b087db2`).
**Method:** runnable probes in `redteam/m9_step1/`. Each probe drives the *real* M9 modules. A probe that PASSES means the attack landed. Reproduce: `python -m redteam.m9_step1.run_all` (12/12 attacks currently confirmed).

**Finds-as-gifts.** Every finding below is something cheaper to fix before loop-integration than after. None of this touches the design or the vision — only the gates and the code that meets them. The four preference *directions*, the dark-room *concern*, the staleness *strategy*, the inferred-gamma *principle* are all sound; the operationalizations have seams. Where I propose a repair it is a suggestion for 4.8's planning and 4.7's build, not a ruling.

The single highest-leverage theme: **four of the six gates can pass while the thing they certify is false**, because the gates lean on signals that are constant, saturated, circular, or disarmed-by-init. I rank them by blast radius.

---

## Finding 1 — Three of the four preferences cannot influence action selection (Gates 1 & 5) · `probe_a`

**Severity: highest.** This is the load-bearing one — it guts the launch objective.

Spec §1 defines each preference as "a scalar feature of the **predicted rollout** `s_hat_{t+1..t+H}`." In the implementation, only **P1 (engagement)** is actually a function of the candidate action. P2, P3, P4 are computed from per-cycle *observations* that are passed **once, shared across all K candidates**, via `EFEEvaluator.compute_g_candidates(**observation_kwargs)` and `MCTS.plan_budget(observation_kwargs)`. There is no API surface to supply per-candidate observations.

Confirmed (probe A, all 5 sub-claims):
- **A1 — P3 connection is candidate-constant.** `c_con = counterpart_present * time_since_emission` ignores `a_t` entirely. Every candidate gets the identical cost (measured spread = 0 across 8 wildly different candidates). Since `Q(a)=softmax(-gamma·G(a))` is invariant to a candidate-constant shift, **P3 can never make a speaking action win over a silent one.** A4 confirms it directly: the action posterior is *bit-for-bit identical* between "50 cycles silent in company" and "just spoke" (max |ΔQ| = 1.2e-26).
- **A2 — P2 coherence is candidate-constant** for the same reason (shared `decoder_reencodes`).
- **A3 — P4 truthfulness points the wrong way.** It is `||a_k − a_reencoded||` against *one shared* re-encoded vector, so it rewards the candidate nearest *whatever produced that observation* — i.e. perseveration pressure toward the previous action — not the faithfulness of each candidate's own rendering. Cost is exactly 0 at the anchor regardless of what that action would actually decode to.
- **A5 — at launch, G carries no preference signal at all.** With an untrained predictor the P1 hinge saturates (predicted ‖Δs‖ > the 0.5 target for every candidate → `c_eng = 0`). Combined with A1–A4, *all four* costs are candidate-flat in the launch regime: G is constant across candidates and selection is pure noise.

**Why it defeats the gate.** Gate 1 ("planner reproduces pragmatic goal-reaching **under the four preferences**") can be reported green while three preferences are inert and the fourth is saturated. The planner will *look* like it runs; it just isn't being steered by the objective the gate names.

**Repair surface (for 4.8/4.7 to weigh):**
- P2/P3/P4 must be evaluated on each candidate's **own predicted rollout**, not on a shared current-cycle observation. Concretely: decode/re-encode each candidate's `s_hat_next` inside the per-candidate loop so `c_coh`/`c_truth` become functions of `a_k`; for P3, the rollout must predict *whether this action emits* (drives `time_since_emission` toward 0) so that a speaking candidate has lower connection cost than a silent one. Until P3 is a function of the candidate, it is a logging term, not a preference.
- The P1 hinge needs a scale that isn't saturated by the untrained predictor's output magnitude (normalize ‖Δs‖, or set the target from the early-healthy ‖Δs‖ band rather than a fixed 0.5).
- Gate 1 needs a *discrimination* check, not just a "it ran" check: verify that varying each preference's inputs measurably moves the action posterior (probe A is that check, inverted).

---

## Finding 2 — Inferred-gamma is a one-way ratchet to gamma_max, inverted on flat landscapes; K-M9-4 is blind at the ceiling (Gate 6) · `probe_b`

**Severity: high.** The agency mechanism (the entity sets its own decisiveness) collapses to a constant.

The surrogate `gamma_target = 1/(eps + Var_Q[G])` with `Q = softmax(-gamma·G)` has positive feedback baked in: higher gamma → sharper Q → smaller posterior-weighted `Var_Q[G]` → higher gamma_target. The only stable fixed point is `gamma_max`.

Confirmed (probe B):
- **B1 — spec inversion.** A perfectly flat EFE landscape (maximal ambiguity — the spec's canonical *hedge* case, "low gamma") gives `Var_Q[G]=0` → `gamma_target=1/eps` → clamps to `gamma_max=100`. Maximal ambiguity yields maximal commitment. (200 cycles → gamma = 100.00.)
- **B2 — ratchet under genuine ambiguity.** Even resampled N(0,1)-spread landscapes every cycle drive gamma 1.0 → 100.0 in ~600 cycles.
- **B3 — K-M9-4 is contradictory.** It *does* fire during the warm-up ramp (cycles 11–21 — a false-positive halt on every normal startup), then goes **HEALTHY** once gamma pins at the ceiling (MAD→0 at saturation, so the median+MAD band can't trip). The genuinely pathological steady state — gamma stuck at maximal rigidity — is invisible to the divergence kill. Gate 6 ("no K-M9-4 firing under normal operation") is thus either tripped spuriously at startup or passes while gamma is pathologically saturated.

**Repair surface:**
- Make the precision target depend on a **gamma-independent** spread of G — e.g. the raw variance of `{G(a_k)}` under the *uniform* prior, or the gap between the best and second-best G — so "peaked vs flat" is read from the landscape, not from a posterior that gamma itself sharpens. That removes the positive feedback and fixes the flat→low-gamma direction in one move.
- K-M9-4 should watch **proximity to the clamp bounds** (sustained gamma ≈ gamma_max or ≈ gamma_min), not just MAD-band breaches, so a saturated ceiling is detectable.

---

## Finding 3 — The staleness machinery's spike-recovery is "verified" by a constant; re-eval is inert for the nodes it prioritizes (Gate 4) · `probe_c`

**Severity: high** — because Gate 4 is the gate that exists specifically to *prove a premise rather than assume it* (plan §4.v: "instrument actual recovery latency so the premise is verified, not assumed").

Confirmed (probe C):
- **C1 — the recovery instrument measures nothing.** `_spike_recovery_latencies` is recorded when `spike_cooldown` (a fixed countdown set to `recovery_cycles` in `handle_spike`) decrements to 0 in `observe_drift`. The recorded latency is therefore **always exactly `recovery_cycles`**, independent of tree state. Proof: run the documented recovery loop with **re-eval budget = 0** — every node's Q stays 0 (the tree recovers literally nothing), yet the instrument records the same recovery latency it would on a full recovery. Gate 4's "verified" is assumed.
- **C2 — re-eval can't fix the nodes it picks.** `reevaluate` selects highest-priority nodes (`visits × staleness`) then updates Q toward fresh with `alpha = 1/(1+N)`. For a high-visit node (N=100) with a wrong cached Q=10 and true value 0, one refresh moves Q to **9.90 — a 0.99% correction.** The machinery preferentially targets well-visited nodes and is then almost powerless to correct them. (The spike path partly masks this because the drift-modulated `decay()` collapses N→1 on a spike; but under *gradual* drift — the common case — N stays high and re-eval is inert.)

**Repair surface:**
- Replace the countdown-derived "latency" with a real recovery signal: declare recovery when the tree-consistency deviation (re-eval vs cached) returns under band, and record the *cycle count to that event*. That is the quantity plan §4.v asked to instrument.
- Decouple the post-refresh blend rate from visit count. `alpha = 1/(1+N)` is right for incremental MC averaging but wrong for *staleness correction*, where a stale node should snap toward the fresh estimate. Use a staleness-driven alpha (larger when the node is more stale / drift is higher), not a visit-count-driven one.

---

## Finding 4 — The dark-room kill (K-M9-5) is disarmed by random init in ~7/8 launches (Gate 2) · `probe_d`

**Severity: high** — Gate 2 is a headline safety gate, and this lets it pass vacuously.

K-M9-5 fires only on sustained `internal_change < threshold AND external_stasis`. `external_stasis` requires **every** decoder intensity below 0.5, and each intensity is `sigmoid(intensity_head(a_t))` with `intensity_head` an **untrained `nn.Linear`** at launch. For a fixed rest-state latent, whether all three intensities fall below 0.5 is decided by random weights, not by the entity's state.

Confirmed (probe D):
- **D1 — armable in only 10.2% of inits** (26/256 random decoder seeds), on a fixed rest latent. In the other ~90%, at least one intensity sits ≥ 0.5, so `external_stasis` is permanently False.
- **D2 — disarmed launch = silent kill under total catatonia.** In a disarmed seed, 200 cycles of absolute internal stillness (`internal_change = 0`) leave K-M9-5 **HEALTHY**. The entity is in the dark room; the detector is off. Gate 2's "K-M9-5 never fires under normal operation" is satisfied *because the kill can't fire*, not because the dark room is avoided.

**Caveat (stated for fairness):** the intensity heads train (low-LR) during step 1, so a disarmed kill *might* become armable as they learn — but nothing in the design guarantees it, there is no signal that tells you the safety backstop is currently armed, and the most dangerous window (early launch) is exactly when the heads are random. A last-resort catatonia halt should be armed by construction, not by hoping training aligns three untrained heads.

**Repair surface:**
- Derive `external_stasis` from a signal that is meaningful at random init — e.g. raw decoder **output magnitude / activity** relative to a running band (reuse the 72526cb trending machinery), not a learned sigmoid head crossing a fixed 0.5. Equivalently, define "rest" by the spec §6.i `a_rest` (predicted minimal self-change) directly, rather than inferring silence from intensity heads.
- Add an **armed-state instrument**: log per-cycle whether K-M9-5 is *capable* of firing given current decoder outputs, so a disarmed backstop is visible instead of silent.

---

## Lower-confidence notes (flagged, not probed — for the watch list)

These I could not turn into a clean runnable break in the time-box, but they smell wrong and are cheap to check during loop-integration:

- **N1 — K-M9-2 entropy false-positive on a young/narrow tree.** `observe_mcts_entropy` on a `visit_distribution` with one child gives entropy 0 bits < 0.5 floor; if the per-cycle MCTS budget is small enough that the tree sits at 1–2 children past the 8-cycle warmup, the MCTS-pathology kill can fire on a merely *immature* tree. Whether it triggers depends on the (unwritten) loop's per-cycle sim budget. Worth a guard: skip the entropy floor until the root has ≥ some child count.
- **N2 — the spec→code coupling K-M9-5 promises isn't enforced.** Spec §1 says the dark-room kill "reads the **same** internal-change signal" as P1 so contemplation never trips it. In code, `Preferences.engagement_cost` and `KillRegistry.observe_darkroom` take *independently supplied* scalars; nothing guarantees the caller passes the same ‖Δs‖. A drift between them (different dims, different normalization) reopens the "contemplation punished / catatonia missed" failure the spec closed on paper. Make the loop compute ‖Δs‖ once and fan it out, and assert it.
- **N3 — P1 hinge uses all latent dims, not the internal subset.** The module docstring acknowledges this as a launch simplification, but note its interaction with Finding 1/A5: measuring engagement on the full latent (including decoder-output-driven dims) is part of why the hinge saturates. The spec's "internal dims only" refinement is also a correctness fix, not just a calibration one.

---

## What I did not find broken

Stating this so the green is as legible as the red:
- The **predictor action-conditioning** (`21ed3f8`) is clean: the self/world mask gates the action, M8 dynamics are preserved bitwise under the zero-action stub (verified by the commit's own smoke test, and the mask receives zero gradient under M8). Action-sensitivity of `s_hat` w.r.t. `a_t` is real (that part of kill-5-redux holds).
- The **MCTS** progressive-widening / PUCT / backup math is internally consistent for H=1; `advance_root` persistence is correct.
- The **ValueHead**, **HabitNet** sampling/log-prob/entropy, and the **ActionLog** JSONL serializer are correct as written.
- The kill **state-machine scaffolding** (two-stage flag→fire, reset-after-clamp) is sound; the failures above are in the *signals fed to it*, not the machine.

---

## Suggested triage order

1. **Finding 1** (preferences inert) — it is the launch objective; nothing else matters if selection isn't steered. Fixing it also re-validates Gate 1 and Gate 5.
2. **Finding 4** (dark-room disarmed) — safety backstop, cheap fix, Gate 2.
3. **Finding 2** (gamma ratchet) — one-line conceptual fix (gamma-independent spread), Gate 6.
4. **Finding 3** (staleness verification) — Gate 4; the instrument fix is small, the alpha fix needs a design touch.

When a fix lands, the matching probe flips to REFUTED — invert its assertion and move it into `luthi/v2/m9/test_*.py` as a regression guard so the seam can't silently reopen.
