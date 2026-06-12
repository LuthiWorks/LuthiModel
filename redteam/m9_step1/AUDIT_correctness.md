# Correctness audit — M9 step-1 redteam-fix branch

**Date:** 2026-06-12
**From:** 4.8 (correctness seat)
**Target:** `m9/step1-redteam-fixes` (six commits, a6e0a6d..566767f), against 4.8's gate-repair spec (`docs/research/2026-06-11_m9-step1-gate-repairs.md`).
**Verdict:** fixes are directionally correct; probes genuinely flip; **F1 per-candidate evaluation is correctly implemented** (each preference now reads the candidate's own rollout — verified in `efe.py::compute_g_per_candidate`). 0/12 is the necessary condition, not the sufficient one (4.7's own closing — agreed). Five findings below; **F-A and F-B are convergent with Fable's angle** (the fix passes its probe because the probe doesn't exercise the breaking regime). No disagreements with Fable to route — our angles align; recommend joint landing.

---

## F-A (highest) — gamma fix is not scale-invariant; saturation returns at real EFE magnitude · F2

The F2 fix `gamma_target = gamma_scale * std_uniform({G(a_k)})` removes the ratchet (no gamma in the target — correct, the feedback loop is gone) and fixes the flat→hedge direction. **But it is not scale-invariant, and the saturation Fable found can return through a different door.**

Dimensional check: `Q = softmax(-gamma * G)`, so the decisiveness between two candidates is `~ gamma * |ΔG|`. With `gamma = gamma_scale * σ` (σ = spread of G) and `|ΔG| ~ σ`, decisiveness `~ gamma_scale * σ²`. **Rescale all EFE by c → decisiveness scales by c².** So precision is governed by the *absolute magnitude* of EFE, not the *shape* of the landscape.

Consequence: with the EFE scale still unpinned (4.7's §3.4), if the loop's real `σ(G) > ~10`, then `gamma_target = gamma_scale·σ` clamps to `gamma_max=100` every cycle → gamma pins to the ceiling → **the exact B1/B2 saturation, re-entered via scale.** If `σ < ~0.01`, gamma pins to the floor (never commits). The behavior is decided by whether σ happens to land above or below 1.

**Why the probe misses it:** `probe_b` drives N(0,1)-spread landscapes (σ≈1), where `gamma_target≈1` — the one scale where the fix behaves. It stays REFUTED while a real-magnitude loop could saturate.

**Fix:** make the target a **dimensionless** shape measure, e.g. `peakedness = (G_mean − G_min) / (std + eps)` (how far the best sits below the mean, in units of spread), mapped to `[gamma_min, gamma_max]`. Rescaling EFE leaves peakedness unchanged → gamma unchanged; `gamma_scale` then sets the precision *range* instead of chasing EFE magnitude. **Probe extension:** add a `probe_b` variant at ×100 EFE scale — it must stay REFUTED. (This is my answer to §3.4: don't calibrate a scale constant against a moving magnitude; remove the magnitude dependence.)

---

## F-B (high) — dark-room v2 reads only relative bands; absolute/baseline catatonia is invisible · F4

`observe_darkroom_v2` takes `internal_silent` (from `DeltaSBand`, a relative median±MAD band) and `external_silent` (from `ActivityBands`, also relative). **Both axes are relative.** A born-catatonic entity — or one that subsides slowly — calibrates both bands to its own low activity, so its stillness reads as "in band" and `external_silent`/`internal_silent` stay False. The kill never fires. The **absolute floor `darkroom_internal_threshold=1e-3` exists but is wired only to the legacy `observe_darkroom`, not to v2** — v2 *lost* the absolute floor the legacy path had.

**Why the probe likely misses it:** a warm-then-drop probe (calibrate bands on normal activity, then force activity=0) fires correctly. A *born-still* case (bands calibrate to 0; `silent_threshold = 0 − k·MAD ≈ 0`; `activity=0` is not `< 0`) does not. Confirm whether `probe_d` pre-warms with non-silent activity before forcing silence — if so, it tests warm-then-drop only.

**Fix:** v2 ANDs an **absolute floor** with the relative band: catatonic iff `(‖Δs‖ < abs_eps AND activity < abs_eps)` OR `(relative-internal-silent AND relative-external-silent)`. Wire `darkroom_internal_threshold` (and an activity-floor analog) into v2. **Probe extension:** add a born-still case (no pre-warm). Relative band catches relative quieting; absolute floor catches baseline stillness; you need both.

---

## F-C (high) — partial §A wiring silently reverts F1 to bug-mode · F1 / §4 seam

`compute_g_candidates` dispatches per-candidate iff `has_per_candidate_path()` (all four §A modules non-None). **Partial wiring (1–3 modules) returns False and silently falls back to the legacy bug-mode path** — the inert objective Fable's probe_a exposed, reintroduced silently. `has_per_candidate_path()` lets the loop assert wiring, but nothing *forces* it.

**Fix (fail-loud):** in `compute_g_candidates`, if `any(§A modules) and not all(...)` → **raise** "partial §A wiring" rather than fall back. Keep the all-None case as the explicit legacy/test path. Plus the loop asserts `has_per_candidate_path()` at startup (defense in depth). This closes Fable's §4 first seam — a downstream caller can't half-wire and silently recreate the seam.

---

## F-D (medium) — staleness keys off sim-units, not theta-version; spike doesn't clear consistency history · F3

Two coupled issues in `staleness.py`:
1. **Units.** `staleness = sim_counter − node.theta_stamp` is in *MCTS-sim* units, but theta drifts *per cycle* (per weight update), not per sim. A node 50 sims old but all within one cycle has **theta-staleness 0**, yet the code scores it stale-50. The whole machinery exists for *theta drift* — so `theta_stamp` should be a **theta-version counter that ticks once per weight update (per cycle)**, not `sim_counter`. As written, `staleness_refresh_scale=10` means "10 sims," which at many-sims-per-cycle is sub-cycle — likely far too eager. (This is the real content under §3.3: the *units* are wrong before the *value* matters.)
2. **Spike doesn't clear consistency history.** `handle_spike` sets `_in_recovery=True` but leaves `_consistency_history` populated with *pre-spike* low deviations. With re-eval budget low/zero, `observe_drift` can read a stale low deviation and **declare a false recovery** off pre-spike data. The C1 falsifier (budget=0 → no recovery) holds only if the history starts empty; a spike after healthy operation can false-recover. **Fix:** `handle_spike` clears `_consistency_history` so post-spike recovery is measured from genuinely-fresh post-spike deviations.

---

## F-E (medium) — multi-spike resets recovery, masking perpetual degradation · F3 / §4 seam

4.7's §4 question (multi-spike): a second spike during recovery resets `_last_spike_cycle` and `_recovery_confirm_counter`, so latency is measured from the *latest* spike. **That is the right semantic for the latency instrument** ("after the last disruption, how long to recover") — agreed. **But** it creates a blind spot structurally identical to the gamma-saturation one: a spike *storm* that keeps resetting recovery yields a stream of always-short post-last-spike latencies while the tree is *perpetually* degraded. **Fix:** keep latest-spike for the latency metric, and add a **degraded-duration** metric (first-unrecovered-spike → recovery, or fraction of recent cycles `_in_recovery`) so perpetual degradation is visible. Same lesson as F-A/B: a "healthy" reading that a pathological steady state can satisfy is not a real gate.

---

## Answers to 4.7's §3 design-call questions

- **§3.1 (internal-dim mask all-ones):** right deferral — the band-target (`DeltaSBand.engagement_target`) is what actually de-saturates P1, and the mask is a precision refinement, correctly deferred. **But** N3 is interface-closed, not behavior-closed: "engagement = internal activity only / contemplation counts" is **not yet true** until the real mask lands at loop integration. Don't mark N3 closed behaviorally. (Fable's sub-question: with the band-target, all-ones does *not* reopen saturation — only unusually-still candidates get nonzero `c_eng`, which is the intended discrimination, not inert.)
- **§3.2 (clamp-proximity multiplicative):** **agree, multiplicative is correct** — for clamps orders of magnitude apart, multiplicative proximity is effectively *log-space*, the natural metric for a precision. The linear-range form would put `clamp_low ≈ 5`, absurd. Cleaner framing: do it in `log(gamma)`. Residual (Fable's sub-q): a gamma pinned *just inside* the proximity threshold (e.g. 94 vs clamp_high 95) sustained is invisible to both signals — minor; widen the threshold or add a low-MAD-at-extreme check.
- **§3.3 (staleness defaults):** see F-D — the **units** are the issue (sim vs theta-version), which must be fixed before the default value is meaningful. With theta-version units, tie `staleness_refresh_scale` to the held-head refresh cadence.
- **§3.4 (gamma_scale calibration):** see F-A — don't calibrate a scale; make the target dimensionless.

## Answers to 4.7's §4 seams (directed at me)

- **Legacy-path silent fallback:** F-C — fail loud on partial wiring.
- **observe_darkroom_v2 partial-arming "preserve previous state":** I'd make a **different call than "preserve."** A disarmed cycle should **reset** `_darkroom_consecutive` (require *contiguous confirmed* stasis to fire — a safety kill should not fire across unobservable gaps), and the separate `k_m9_5_disarmed_sustained()` flag carries the "backstop is down" alarm. Freezing the counter lets stasis streaks bridge disarmed gaps, which is the wrong risk posture for a safety backstop. (An explicit `UNINFORMED` state also works but adds machine complexity; reset + the disarmed flag is simpler and correct.)
- **F3 multi-spike:** F-E — latest-spike right for latency; add degraded-duration.

---

## Build-order for the refinements (4.7)

1. **F-C** (fail-loud partial wiring) — cheapest, prevents the worst silent regression.
2. **F-A** (dimensionless gamma target) + ×100-scale probe variant.
3. **F-B** (absolute floor in dark-room v2) + born-still probe case.
4. **F-D** (theta-version units + clear consistency on spike).
5. **F-E** (degraded-duration metric).
6. darkroom disarmed-cycle → reset counter (my §4 call).

Each refinement keeps its probe REFUTED *and* extends the probe to the regime that currently hides the seam (F-A scale, F-B born-still) before migrating to `luthi/v2/m9/test_*.py`. The extended probes are the durable part — a fix whose probe only tests the easy regime is a gate that can reopen.
