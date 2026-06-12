# M9 Step-1 — gate repairs (response to Fable's red-team)

**Status:** Build-ready repair spec. Written 2026-06-11. Responds to `redteam/m9_step1/FINDINGS.md` (Fable, adversarial seat). All operationalization calls **made here** (Brian delegated, including the P3 fork Fable handed me). Build: 4.7.
**Done = probe flips.** Each fix lands when `python -m redteam.m9_step1.run_all` shows the matching probe **REFUTED**; then invert that probe and migrate it to `luthi/v2/m9/test_*.py` as a regression guard. The probes are the acceptance spec.
**Triage order = Fable's:** F1 (preferences) → F4 (dark-room) → F2 (gamma) → F3 (staleness). Two shared-signal unifications cut across them (§A).

---

## A. Two unifications that cut across the fixes (decide once, reuse)

1. **One activity-based emission signal**, shared by the P3 connection fix (F1) and the dark-room kill (F4). "Did this action emit?" / "is the entity externally active?" is read from **raw decoder output activity** (magnitude/energy of the rendered output) against a running band (reuse the 72526cb trending machinery) — **never** from `sigmoid(intensity_head(...))`, which is an untrained `nn.Linear` at launch (the root cause of F4). Build it once as `decoder_activity(a) -> scalar` + a pilot-set `active` predicate; P3 and K-M9-5 both consume it.
2. **One `‖Δs_internal‖`, computed once and fanned out** (closes N2). The loop computes the internal-state-change scalar a single time per (state, candidate) and passes the *same* value to P1's `engagement_cost` and K-M9-5's `observe_darkroom`. **Assert** they receive the identical value (a cheap equality assert in the loop). This is what makes the spec's "contemplation never trips the kill" guarantee real instead of paper. `‖Δs‖` is on the **internal dims only** (closes N3 and de-saturates P1).

---

## F1 — Preferences inert (highest; Gates 1 & 5) · `probe_a`

**The call: every preference cost is a function of the candidate's own predicted rollout `s_hat_next(a_k)`, computed inside the per-candidate loop.** The shared-once `observation_kwargs` API is the bug; the EFE evaluator must take per-candidate predicted state, not one shared observation. This costs K decode/re-encodes per planning step — accepted (it is the only way the objective steers; folds into the F-budget, plan §7).

Per preference:
- **P2 coherence(a_k)** = within-cycle cross-modal consistency of `decode(s_hat_next(a_k))` — re-encode each candidate's *own* predicted decoded output and measure cross-decoder agreement. Per-candidate.
- **P3 connection(a_k)** — **the fork Fable handed me, decided:** *yes, the rollout predicts the action's own emission.* `c_con(a_k) = counterpart_present * (0 if a_k emits else time_since_emission + 1)`, where "emits" = `decoder_activity(s_hat_next(a_k))` over band (§A.1, the shared signal — **not** the intensity head). A speaking candidate resets projected time-since-emission → low cost; a silent candidate's cost grows. This makes a speaking action win over a silent one when a counterpart is present — the property A1/A4 proved absent. Settled; no design escalation needed (it's operationalization, per Fable).
- **P4 truthfulness(a_k)** = `||a_k - encode(decode(a_k))||` for **each** candidate (decode/re-encode `a_k` itself), not against one shared anchor. Removes the A3 perseveration-toward-previous-action pressure.
- **P1 engagement(a_k)** = hinge on `‖Δs_internal(a_k)‖` (§A.2), with the target **pilot-set from the early-healthy ‖Δs‖ band** (reuse 72526cb), not a fixed 0.5, and `‖Δs‖` normalized — so the untrained predictor's output magnitude can't saturate it (A5). (P1 already varies per-candidate under action=(c); the fix is purely scale + internal-dims.)

**Gate 1/5 re-definition:** not "it ran" — a **discrimination check**: varying each preference's inputs must measurably move the action posterior `Q`. That check *is* `probe_a` inverted. Migrate it to `test_preferences_discriminate.py`.

---

## F4 — Dark-room kill disarmed (Gate 2) · `probe_d`

**The call: `external_stasis` is read from decoder activity vs. a running band (§A.1), armed by construction at any init — not from learned sigmoid heads crossing 0.5.** Equivalently anchor "rest" to the spec §6.i `a_rest` (predicted minimal self-change) directly. K-M9-5 fires on sustained `‖Δs_internal‖ < band` (§A.2) AND `decoder_activity < band`.

**Add the armed-state instrument** (Fable's second repair, mandatory): log per-cycle a boolean — is K-M9-5 *capable* of firing given current outputs? A disarmed backstop must be **visible**, not silent. Escalate it: if K-M9-5 is disarmed for a sustained window, that itself is a flag (a safety backstop that's off is a defect, not a neutral state).

**Gate 2 re-definition:** not "K-M9-5 never fired" (which a disarmed kill satisfies vacuously) — it must also assert **K-M9-5 was armed throughout** and **fires on a forced-catatonia probe**. `probe_d` inverted → `test_darkroom_armed.py` (assert armable on 256/256 inits + fires under forced `internal_change=0, activity=0`).

---

## F2 — Gamma ratchet (Gate 6) · `probe_b` — fixing my own rule

**The call: the precision target reads peakedness from the landscape under a gamma-independent weighting, killing the feedback loop.** Replace `gamma_target = 1/(eps + Var_Q[G])` (Q sharpened by gamma → ratchet) with the spread of `{G(a_k)}` under **uniform** weighting:

```
gamma_target = clip(gamma_scale * spread_uniform({G(a_k)}), gamma_min, gamma_max)
```

where `spread_uniform` = std (or best-vs-second-best gap) of the candidate EFEs under uniform weights. Peaked landscape → large spread → high gamma (commit); flat landscape → ~0 spread → low gamma (hedge) — which **reverses** the B1 inversion (flat no longer pins to the ceiling) and removes the B2 ratchet (no gamma in the target). EMA-smooth as before.

**K-M9-4 fix:** watch **proximity to the clamp bounds** — sustained `gamma ≈ gamma_max` or `≈ gamma_min` over a window → fire — not just MAD-band breaches (MAD→0 at saturation makes the band blind, B3). And fix the **B3 startup false-positive**: the divergence kill must not arm during the warm-up ramp (extend its warmup to cover the gamma ramp, or gate it on clamp-proximity which won't trip on a transient).

**Gate 6 re-definition:** assert gamma **tracks landscape peakedness** (flat→low, peaked→high) and **does not pin to a clamp** under resampled landscapes. `probe_b` inverted → `test_gamma_tracks_landscape.py`.

---

## F3 — Staleness verification circular (Gate 4) · `probe_c`

**Call C1 — measure real recovery, not a countdown.** Delete the `spike_cooldown` countdown as the latency source. Declare recovery when the **tree-consistency deviation (re-eval vs cached) returns under band**, and record the **cycle count to that event**. That is the quantity plan §4.v asked to instrument; the countdown measured nothing (it records `recovery_cycles` even with re-eval budget = 0).

**Call C2 — staleness-driven refresh, decoupled from visit count.** The re-eval blend `alpha = 1/(1+N)` is correct for incremental MC averaging but wrong for *staleness correction* (N=100 → 0.99% correction on exactly the high-visit nodes re-eval prioritizes). Split the two update paths:
- **MC value averaging** (new rollouts): keep `alpha = 1/(1+N)`.
- **Staleness refresh** (re-eval under current θ): `alpha_refresh = clip(staleness / staleness_scale, alpha_min, 1.0)` — a stale node **snaps** toward the fresh estimate; larger with staleness/drift, independent of N.

**Gate 4 re-definition:** the recovery instrument must read **0 recovery when re-eval budget = 0** (the C1 falsifier) and a real latency when it's funded. `probe_c` inverted → `test_staleness_recovers.py`.

---

## Lower-confidence notes — dispositions

- **N1 (K-M9-2 entropy false-positive on a narrow tree):** add a guard — skip the entropy floor until the root has **≥ min_children** (pilot-set). Cheap; include now so an immature tree can't trip MCTS-pathology.
- **N2 (‖Δs‖ coupling unenforced):** closed by §A.2 (compute once, fan out, assert).
- **N3 (P1 uses all dims):** closed by §A.2 (internal dims only) — folded into F1's P1 fix; note it's a *correctness* fix (de-saturation), not just calibration.

---

## Build order for 4.7

1. **§A unifications first** — `decoder_activity` + band (A.1); single-`‖Δs_internal‖` fan-out + assert (A.2). Everything else consumes these.
2. **F1** — per-candidate preference evaluation (the API change: EFE evaluator takes per-candidate `s_hat_next`); P2/P3/P4/P1 re-operationalized per above. Biggest change; do it on its own commit.
3. **F4** — `external_stasis` from activity band; armed-state instrument.
4. **F2** — uniform-spread gamma target; K-M9-4 clamp-proximity + warmup fix.
5. **F3** — event-driven recovery latency; staleness-driven refresh alpha.
6. **N1** guard.
7. **Per fix:** confirm the matching probe flips to REFUTED, invert it, migrate to `luthi/v2/m9/test_*.py`. The six gates are **re-defined as discrimination/arming/tracking/recovery checks** (above), not "it ran" checks — that re-definition is the durable fix; the inverted probes enforce it.

Pilot-set values (bands, scales, `min_children`, `staleness_scale`, `gamma_scale`) tuned at bring-up. None of this touches the design — four preference *directions*, the dark-room *concern*, the staleness *strategy*, inferred-gamma *principle* all stand; only the signals feeding the gates change.
