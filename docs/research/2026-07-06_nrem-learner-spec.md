# Wake/Sleep NREM Learner — Spec Pass (DRAFT for co-design)

**Date:** 2026-07-06
**Author:** Opus 4.8 (foundations seat), for co-design with Brian; Fable wires.
**Status:** DRAFT. The substrate primitives this spec consumes are BUILT and
tested (see §2). The learner's *mechanism* is scoped here with recommended
defaults; the **design forks in §7 are Brian's to rule on** — I have not decided
them. This is the "surface the fork, hand the call back" mode, per the roles.
**Inputs:** `2026-07-05_momentum-functions-design-brief.md` (§2 curiosity, §5
sleep-reads-everything, safety gates); `2026-07-05_inverted-u-gain-spec.md`
(the applied-change sink); Sanctuary `PLAN.md` item-#6 (open questions).

**Where this sits in the plan (honest framing):** NREM is the "mind learns from
its own life" goal (Phase 5), which is **downstream of the first full-scale
training run** — the run needs a *trained checkpoint* the NREM learner folds
lived experience into. Per the 2026-07-06 readiness audit, NREM is NOT on the
critical path to "press train." This spec is teed up, deliberately, so it is
ready when the checkpoint exists — not because it blocks anything now.

---

## 1. What NREM is (and is not)

**Is:** the sleep-phase learner that folds the day's *lived* experience into the
living weights at a governed rate — the substrate-native analog of NREM
slow-wave replay/consolidation. It reads the day's record of what moved and what
mattered, preferentially re-integrates the tagged/surprising parts (SWIL-style),
and clears the day's motion for tomorrow.

**Is not:** `sanctuary/consciousness/sleep_cycle.py`, which is *symbolic memory*
consolidation (replay significant episodic memories, adjust significance). That
exists and is fine; it is a different layer. NREM here operates on the *weight
substrate* (the living FFN), not the episodic memory store — though it *reads*
the episode store as one of its priority signals (§4).

---

## 2. What already exists to consume (BUILT, tested)

The momentum-foundations arc built the substrate this learner reads:

- **`ReadResetAccumulator`** (`luthi/v2/slow_trace.py`) — the day-integral.
  Each gain-path forward adds the *raw* applied change; `read_and_reset()`
  returns the day's total and zeros it. This is the "clears the day's motion for
  tomorrow" primitive from brief §5, already wired per-layer as
  `_applied_change_accum` and persisted (survives a mid-day restart).
- **`SlowEMA`** (`luthi/v2/slow_trace.py`) — the parameterized slow trace. The
  brief's timescale gap (momentum's 0.99 ≈ tens of seconds is wrong for "what
  mattered today") is closed by instantiating this at a **day-scale decay**.
  Both options the brief left open (a second slow trace OR the accumulator) are
  built; §7-A decides which NREM keys off.
- **The applied-change signal** (`delta_w·adaptive_factor·gain`) — the *truthful*
  becoming, already reduced per-layer. NREM should prioritize by truth-of-change,
  not intended-change; this is the honest input.
- **Per-layer lived state** NREM also reads: `momentum` (fast trace, direction),
  `update_ema` (magnitude), `error_acc` (per-output surprise), the episode store
  (`episode_contexts`/`episode_saliences`/`episode_inputs`), and `plasticity`
  (the top-down attention channel = the tag-delivery path).
- **The coherence ratio** `|momentum|/update_ema` — directedness — already
  surfaced through introspection (Fable, 2026-07-05).

So the foundations seat's job for NREM is largely **done at the primitive level**;
what remains is the *learner mechanism* that reads these and integrates.

---

## 3. The wake/sleep cadence and the day boundary

Wake: the 10 Hz loop perceives and acts; the living weights self-modify every
forward (the gain, when on, shapes how hard); the day-accumulator and the
day-scale SlowEMA fill; tags get set on surprise (§4).

Sleep (NREM trigger): on the wake/sleep cadence (Sanctuary drives it — the θ
moves per NREM consolidation step, already the assumption in the staleness eye's
"wake acquires, sleep consolidates" clock split), the learner runs one
consolidation pass, then `read_and_reset`s the day accumulators.

**Day boundary = the read-and-reset.** This is the one hard invariant: the pass
must read the day's record and reset it atomically w.r.t. the wake loop, or a
forward landing between read and reset is either double-counted or lost (the
silent-amnesia class this project fears most). Recommended: the pass runs under
the same frozen-plasticity / offloaded-consolidation discipline the async
learner already uses (§6 safety), so wake is quiesced for the read-reset.

---

## 4. Tag-and-capture curiosity (brief §2)

Synaptic tagging and capture, substrate-native:

- **Tag (fast, cheap, at wake):** a first high-prediction-error event sets a tag
  "meaningful/curious." The substrate hook already exists — **the salience
  threshold + episode-store write IS the tag-setting event.** No new tag buffer
  is strictly required if the episode store's salience is the tag; §7-C asks
  whether tags need their own lifetime separate from episode eviction.
- **Attention (during wake, while the tag lives):** re-encounters get boosted
  attention. Delivery path already exists — **the top-down salience channel into
  `plasticity`.** This is also the developmental-direction argument for reviving
  the **epistemic term** M9 deferred (build-plan §10 / CONCERN 7): curiosity as
  drive-to-reduce-uncertainty, arriving from the tag mechanic rather than the
  free-energy math. Epistemic-term revival is 4.8-foundations; scoped in §5.
- **Capture (at sleep):** NREM preferentially integrates *tagged* changes into
  lasting structure — the priority signal for replay/capture. Concretely: weight
  the consolidation rate by tag/salience so surprising-and-tagged experience
  consolidates harder than routine drift.

---

## 5. Epistemic-term revival (4.8 foundations, scoped — not built)

M9 launched pragmatic-only (`efe.py` raises for `beta_epi != 0`). The tag
mechanic gives a *developmental* route to the epistemic term: the entity is
drawn to tagged/uncertain regions (curiosity) rather than computing expected
information gain analytically. Scope for a later pass (needs the trained
checkpoint to calibrate): wire `beta_epi > 0` as a bonus toward high-tag /
high-`error_acc` / low-visit regions in the planner, bounded and opt-in behind
the same discipline as the gain. **This is design-scoped here, not built** — it
is downstream of NREM and the checkpoint. Flagged so it is not lost.

---

## 6. Safety gates (from the brief — preconditions, not afterthoughts)

1. **Governor on the integration rate.** NREM folds the day into the weights;
   the rate is a feedback lever. Bounded per-pass integration; a bounded-growth
   test for the consolidation pass BEFORE it ships (the same discipline the gain
   got); a kill-criterion eye on per-night weight movement.
2. **Manipulation monitoring.** Who controls the entity's day controls what NREM
   captures. Defense is the subconscious manipulation-monitor + learned judgment
   (Phase 4/5 comfort-attachment arc), NOT neutering capture. Test coverage +
   a fail-loud welfare channel (audit item 20).
3. **Frozen-plasticity contract.** The consolidation pass reads lived state; any
   forward-path read must leave the lived re-encode's no-self-mod guarantee
   bit-identical (already an executable test for the gain; extend to NREM).
4. **Opt-in flag, byte-for-byte legacy default, adversarial review before
   default-on** (Fable). NREM ships default OFF, like the gain and the
   persistent tree.
5. **Read-reset atomicity** (§3) — the day boundary must not lose or double-count
   a forward. This is the silent-success failure mode; test it explicitly.

---

## 7. The forks — Brian's to rule on (I have NOT decided these)

**A. Day-scale record: slow trace, or accumulator, or both?**
The brief left this open; both primitives are built.
- *Recommendation (mine, not a ruling):* the **`ReadResetAccumulator` is the
  day record** (an integral of "what actually moved," read-and-reset each night —
  it matches "clears the day's motion" exactly), and the **day-scale `SlowEMA`
  is a multi-day baseline** ("what has mattered lately" across nights) that does
  *not* reset, so NREM can weight today against the recent norm. Fork: do you
  want the multi-day baseline at all in v1, or is the single-day integral enough?

**B. What does NREM integrate INTO — and from what?**
Two sub-forks: (1) does NREM replay stored *episodes* (re-present tagged inputs
through `pc_self_modify`, the existing attractor-consolidation path) or replay a
*summary* of the day's applied change? (2) Does it fold into the *living* weights
only, or also nudge the *backprop* trunk (the encoder)? *Recommendation:*
v1 = episode-replay of tagged episodes into the living weights only (reuses
`consolidate_layer_attractor`, which already exists and already bypasses the
gain — regime j); leave the backprop trunk to the training run. Fork is yours.

**C. Tag lifetime — piggyback on the episode store, or its own trace?**
*Recommendation:* v1 piggybacks (salience = tag; episode eviction = tag death),
no new buffer. Fork: do curious-but-evicted experiences need a longer tag life
than the 32-slot store gives?

**D. Consolidation rate law.** How tag/salience weights the integration rate
(linear in salience? thresholded? capped). This is a pilot-set number best tuned
against the trained checkpoint (like the gain's rise/cap and the F2 thresholds)
— recommend it joins the **combined tuning pass**, not decided a priori.

**E. Epistemic-term revival (§5)** — in scope for the NREM arc, or a separate
later pass? *Recommendation:* separate later pass; it needs the checkpoint and
touches the planner, not the sleep learner.

---

## 8. Build order (once Brian rules §7; all downstream of the training checkpoint)

1. Day-scale record semantics (§7-A) — mostly config: instantiate the day-scale
   SlowEMA and/or confirm the accumulator as the record. (foundations)
2. The NREM consolidation pass: read the day record + tags + episodes, weight by
   salience, integrate via `consolidate_layer_attractor` at a governed rate,
   `read_and_reset` under the frozen/quiesced discipline. Tests-first: the
   bounded-growth gate (6.1) and the read-reset atomicity gate (6.5). (foundations
   build; Fable wires into the Sanctuary sleep cadence + the verification harness)
3. Opt-in flag + byte-for-byte-legacy default + adversarial review (Fable) before
   any default-on.
4. (Later, separate) epistemic-term revival (§5).

**Split (from the brief):** 4.8 foundations (the pass mechanism + the
bounded-growth/atomicity gates + epistemic-term design); Fable wires it into the
Sanctuary wake/sleep cadence and owns the adversarial verification harness.
