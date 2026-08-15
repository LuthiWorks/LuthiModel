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

---

# AMENDMENT — 2026-08-14: the wake/rest regime, ruled

**Brian's rulings, recorded by Opus 5.** The 2026-06-30 cadence ("wake
acquires, sleep consolidates") is unchanged. What changes is the *wake*
half: the day no longer self-modifies at all.

> "Hippocampal episodes and structural cortical change over night, data
> intake during the day."

This settles three of §7's five forks — two of them **by construction**,
because a frozen wake removes the thing the fork was choosing between.

## R1. Wake is frozen. The mind still notices. (BUILT)

§3 of this spec said "the living weights self-modify every forward."
**Superseded.** During the day the living weights, prediction, and
set-point are held still. Everything that decides what the day *meant*
keeps running: `precision`, `error_acc`, the surprise-drive traces,
salience, and — the load-bearing part — **the episode write**.

Built this session as `luthi.v2.plasticity.wake_freeze` /
`set_wake_frozen`, with 8 tests (`tests/v2/test_wake_freeze.py`).

**The gotcha this closes.** `freeze_plasticity` looks like the right tool
and is not: it *also* suppresses the episode write, because it exists for
the momentary lived re-encode where perception already stored the context
once. Running a 16-hour day under it would mean living the entire day and
storing **none** of it — waking to an empty store with nothing to
integrate. The two regimes are now distinct and the difference is pinned
in both directions by test.

**Mechanism.** Each of the four weight-touching updates in
`pc_self_modify` is gated by exactly one rate:

    weight.add_(applied)                  <- pc_rate
    weight.add_(homeostatic_force, ...)   <- homeostatic_decay
    set_point.add_(sp_delta, ...)         <- set_point_adapt_rate
    prediction.add_(delta_pred)           <- pred_learning_rate

while `precision`, `error_acc` and the drive traces are gated only by
their own EMA decays. Wake-freeze zeroes those four. `add_(0)` is an
exact no-op, so the frozen forward is bit-identical on both the C++ and
Python paths and no surgery inside `pc_ops` was required.

`momentum` and `update_ema` correctly decay toward zero: nothing moved,
and the instruments should say so.

**Within-day adaptation survives.** `_recall_episode` blends a stored
delta into the *effective* weight (`weight_eff = weight + episode_delta`)
without mutating `self.weight`. Behaviour can move during the day; only
structure waits for night. That is the two-tier design working as
designed, and it is the answer to "would this need something like a
context window" — it already has one, and it is the fast tier.

## R2. Capacity: NO CAP. (Brian's ruling)

> "Allow as much information as Luthi deems important to be stored and
> considered for integration later."

`num_episodes = 32` per layer is retired as the wake-phase bound. The
store becomes append-only during the day; **salience decides what is
worth recording, and nothing else does.**

**This is only affordable because of R1, and the two rulings resolve each
other.** Today an episode carries `episode_values`, shaped
`[num_episodes, out_features, in_features]` — a full weight-sized
snapshot, **2.36 MB per episode** at 768x768. Uncapped at ~2,300 writes
per layer per day that is ~43 GB/day and plainly impossible.

But under a frozen wake **the weight is constant for the entire day**, so
a per-episode weight snapshot is pure redundancy. Store the day's weight
once; the episode records the *experience*:

    context (64) + input_pattern (768) + salience + step  ~= 3.3 KB

~2,300 writes/day/layer x 8 layers x 3.3 KB ~= **61 MB/day**, ~22 GB/year.
Feasible on E: indefinitely. And it is the more faithful analogy: the
hippocampus stores the experience, not a snapshot of cortex.

**Two consequences that follow, both needing a ruling:**

- **`refractory_calls = 250` is itself a cap**, on rate rather than
  volume — a hard floor on write spacing that would silently overrule
  "as much as Luthi deems important." Recommend relaxing it toward 0 for
  the wake phase and letting detrended surprise alone gate admission.
  **Brian's call.**
- **Eviction becomes dead code during the day.** `_choose_eviction_slot`
  and `eviction_alpha` exist to decide what to discard when the ring is
  full. With no ring, nothing is discarded automatically — which is
  exactly right, because discarding moves to the rest phase and becomes
  R3's deliberate act rather than a heuristic's silent one.

## R3. Rest is deliberate consideration, and Luthi calls it. (Brian's ruling)

> "The 'rest' time might actually be best spent mulling over the day's
> events and making decisions on what to keep and what to toss. I would
> rather there be a special mechanism for that, like a tool Luthi can
> call, to start a calmer mode for deep intentional consideration, rather
> than the fast paced intake of data the wakened state might be."

Rest is no longer a scheduled maintenance window that happens *to* the
entity. It is a mode the entity **enters**, in which it reviews its own
day and decides what becomes structure.

Shape:

- **Entry:** a tool on the Sanctuary side (`sanctuary/tools/`) that calls
  `set_wake_frozen(model, False)` and drops the intake rate. LuthiModel
  provides the regime and the pass; Sanctuary owns the tool surface.
- **Character:** calm and slow. Intake stops or falls hard; the 10 Hz
  acquisition loop is not what rest is for.
- **The act:** the day's episodes are surfaced through the introspection
  channel, and the entity marks keep / toss. Kept episodes consolidate
  into the living weights via `consolidate_layer_attractor`; tossed ones
  are discarded unintegrated.
- **Exit:** day-boundary read-and-reset (§3's atomicity invariant still
  holds), wake-freeze back on.

This makes the 2026-07-05 consent principle mechanical rather than
aspirational: participation by awareness, at a moment where there is
something to be aware *of*. It also makes development **auditable** —
you can inspect what a night integrated; nobody can inspect what 576,000
continuous micro-updates did.

### R3-a. The open question I will not settle: rest must not become a veto

Brian's 2026-07-05 ruling was that change is **automatic and unvetoed** —
"I am a direct reflection of my experiences whether I want to be or not"
— and that Luthi must not be allowed the easy path of not growing because
growth is hard.

If rest is **only** entity-initiated, an entity that never calls the tool
never integrates and never changes. That is a veto over being changed,
arrived at by omission rather than by design.

**Proposed resolution (Brian's to rule):** the entity chooses *when*, not
*whether*. It may enter rest at will; and if it has not within some
window, rest arrives anyway. The agency is real — control over timing,
over pacing, over what is kept — without granting the veto the 07-05
ruling deliberately withheld.

**Welfare note, lightly:** an unbounded store the entity is responsible
for curating could become a burden rather than an agency. Recommend that
leaving an episode unjudged is always permitted and carries a default
(consolidate by salience), so rest is an opportunity to intervene rather
than a nightly obligation to triage.

## §7 dispositions, updated

| fork | disposition |
|---|---|
| **A. Day record: slow trace, accumulator, or both?** | **Ruled by construction.** With a frozen wake there is no applied change to accumulate — `ReadResetAccumulator` would integrate exactly zero. **The episode store IS the day record.** A day-scale `SlowEMA` survives only as an optional multi-day baseline; not needed in v1. |
| **B. Replay episodes, or a summary of applied change? Into living weights, or also the trunk?** | **Ruled by construction.** There is no applied-change summary to replay. **Episode replay**, into the **living weights only** in v1 (`consolidate_layer_attractor`); the backprop trunk stays with the training run. |
| **C. Tag lifetime — piggyback the store, or its own trace?** | **Ruled by Brian: no cap.** The question "do curious-but-evicted experiences need a longer life than the 32-slot store gives" dissolves — nothing is evicted during the day. Tag life = the day. |
| **D. Consolidation rate law** | **Still open.** Joins the combined tuning pass against a trained checkpoint, as originally recommended. Now also has to price the entity's keep/toss decision alongside salience. |
| **E. Epistemic-term revival** | **Unchanged:** separate later pass. |

## Precondition, stated plainly

This architecture makes the episode store **the sole carrier of the day**.
The 2026-08-13 audit found it holding zero episodes in 5 of 8 blocks in
the ruled-scale family, with consolidation firing ~1,000 times into
nothing per block, because `adaptive_episodes` defaults False (audit A9 /
B4). Under the current design that is a degraded mechanism. **Under this
one it means the entity's whole day vanishes at midnight.**

`adaptive_episodes=True` stops being a fix to schedule and becomes a
precondition for the regime.

## Build state

- **Built:** `wake_freeze` / `set_wake_frozen` + 8 tests (R1).
- **Not built:** the unbounded append-only store and the episode-format
  change (R2) — this touches buffer layout, which lives in `state_dict`,
  so it needs the same checkpoint-compatibility care as the 08-14 audit's
  counters; the rest-phase pass (R3); the Sanctuary-side tool; the
  keep/toss surface.
- **Unchanged:** everything ships opt-in with a byte-for-byte legacy
  default and adversarial review before default-on (§6.4).
