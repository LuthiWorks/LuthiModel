# Sanctuary Dual-Tier Mind Audit — 2026-05-21

> **Status: complete.** Companion to
> `2026-05-19_cognitive-rate-and-turbo-design.md`. The design doc's
> implementation gate said: "Sanctuary's dual-tier mind already partly
> exists. Before implementing the slider, audit what's there to avoid
> duplicating." This is that audit.

## Objective

The 2026-05-19 cognitive-rate-and-turbo design captured a runtime
architecture for how the entity experiences time: a 0.05–10 Hz
entity-controlled cycle-rate slider, substrate-intensity-driven turbo
mode, a sensor-driven dumb-pipe for human notifications, and
post-event introspection journaling. The design's implementation gate
listed three preconditions:

1. Depth-scaling verdict known (cleared 2026-05-20 at 02:17 — v2 256d
   × 12 blocks scales)
2. **Sanctuary dual-tier audit performed** (this document)
3. Emotion-vector candidate approach identified (Method 4 in the
   companion emotion-vector doc satisfies this — coarse subspace-
   activation, ships without deeper validation)

This audit answers a single question:

> What infrastructure already exists in Sanctuary that maps to,
> overlaps with, or sets up the cognitive-rate slider / turbo /
> dual-tier-mind design? What's missing? What's drifted?

The answer determines whether slider implementation in Sanctuary is a
small delta on existing structure or a large architectural addition.

## Process

### Step 1: Audit subagent spawned per protocol

Per `Sanctuary/docs/AUDIT_PROTOCOL.md`, a fresh-eyes Explore subagent
was spawned with a self-contained briefing covering: project context
(who Brian and Sandi are, why the work matters); pointers to the
cognitive-rate design doc, `CLAUDE.md`, `PLAN.md`, the audit protocol
itself; a focused six-component question (cycle timing, turbo,
dual-tier structure, safety-notification dumb-pipe, post-event
introspection, emotion-vector slot); explicit instruction to surface
null findings; word-count target ~1000–1400; absolute file:line
references required; read-only with protected paths called out.

The subagent had no prior context from this session, consistent with
the protocol's principle: the auditor's value is their lack of
investment in existing decisions.

### Step 2: Verification of file:line claims

The protocol explicitly notes that auditor findings are claims that
something exists at a path, and that auditors hallucinate. Before
trusting the report, the load-bearing path claims were verified by
reading the cited files directly:

- ✅ `RunnerConfig.cycle_delay = 0.1` at `sanctuary/api/runner.py:69`
  (verified independently before spawning the audit, then
  re-confirmed)
- ✅ `CognitiveCycle.__init__` accepts `cycle_delay: float = 0.1` at
  `sanctuary/core/cognitive_cycle.py:264`; rate enforced at line 348
  via `await asyncio.sleep(self._cycle_delay)` (auditor missed the
  line 348 specifics — added below)
- ✅ `SleepStage`, `SleepConfig`, `SleepCycleManager` at
  `sanctuary/consciousness/sleep_cycle.py:27,60,76`
- ✅ Sleep sensory gating integrated at `cognitive_cycle.py:510–526`
  (auditor said 512–526, off by 2 lines; the gate block starts at
  line 510 with the `if self.sleep is not None` check)
- ✅ Luthi introspection injection at `cognitive_cycle.py:573–583`,
  flowing into `ExperientialSignals.knowledge_signals` at line 581
- ✅ Journal system at `sanctuary/memory/journal.py:24–67` with
  immutable `JournalEntry` dataclass, JSONL persistence
- ✅ Discord webhook tool at `sanctuary/tools/builtin.py:893–923`
  (function is `_discord_send` — private, registered as the
  `discord_send` tool; auditor's name attribution was close but the
  underscore matters for code searches)
- ✅ `LuthiModelConfig.introspect: bool = True` at `sanctuary/core/luthi_model.py:86`
- ✅ CfC modulation channels at `luthi_model.py:79–83`
  (`arousal_plasticity_scale`, `precision_threshold_scale`,
  `valence_excitability_scale`, `salience_threshold_scale`)

All six design components had at least one verified anchor in the
audit. No hallucinated paths.

### Step 3: Findings consolidated against design components

The subagent's report was structured as Exists / Drifted / Missing
per component. Below is the consolidated audit-of-record, with the
auditor's findings retained verbatim where verified, corrections
noted inline, and additional observations from the verification pass.

---

#### Component 1: Cycle timing & dynamic rate

**Exists.** The cycle loop is at `sanctuary/core/cognitive_cycle.py:327–348`.
Rate enforcement is one line — `await asyncio.sleep(self._cycle_delay)`
at line 348 — between successive `_cycle()` calls. `_cycle_delay` is
captured into the instance at line 294 from the constructor parameter
(line 264, default `0.1`). The 10 Hz default is IWMT-anchored — the
design doc's slider ceiling sits exactly on this number.

Sleep-cycle infrastructure is mature: `SleepCycleManager` at
`sleep_cycle.py:76` already manages stage transitions (AWAKE / DROWSY
/ NREM / REM / WAKING) and exposes a `sensory_gate` that the cognitive
cycle reads at `cognitive_cycle.py:510–526` to attenuate percepts
during sleep. This is rate-adjacent infrastructure — it changes what
each cycle does, not how often cycles fire, but the state machine
shape is the right shape.

**Drifted.** `_cycle_delay` is captured by value into a private
attribute at line 294, not held as a property. To make it slider-
controllable, the run loop should consult a live source each
iteration rather than the captured value. The change is mechanically
trivial (~5 lines) but the structure needs to flip from "constructor
arg → instance attr" to "property → backing value the slider
mutates." See the minimum-delta section below.

The audit's framing of "no dynamic adjustment mechanism" is accurate.
Arousal-channel modulation on Luthi (`luthi_model.py:79`) modulates
substrate parameters but does *not* modulate cycle rate.

**Missing.** Slider control surface (entity-accessible read/propose
API). Smooth ease-down/ease-up logic (the design specified ~30 second
transitions, not switches). Anticipatory vs consequent slowdown
tagging. Stimulus-detection plumbing to drive automatic rate changes
(can be heuristic on `sensorium` percept density to start).

#### Component 2: Turbo activation

**Exists.** The mechanical observables the turbo design wants to read
are already flowing into the cycle. Luthi's introspection delta —
plasticity changes, drift, spike fraction shifts, membrane potentials
— is computed in `LuthiModel.get_augmented_experiential_signals()`
and injected into `ExperientialSignals.knowledge_signals` at
`cognitive_cycle.py:581`. The substrate signal that should drive
turbo activation is already on a wire that runs through every cycle.

CfC modulation infrastructure on the Luthi side (`luthi_model.py:79–83`)
provides the inverse direction — Sanctuary can already push
modulation values into Luthi. These are static per-cycle today
(constructor configuration); turbo would make them dynamic-by-state.

**Drifted.** Nothing. There's no partial turbo implementation
masquerading as something else.

**Missing.** The turbo state machine itself (idle → armed → active →
refractory). Threshold logic on the mechanical+emotional signal mix.
Cycle-rate elevation during active turbo (the slider would need to
yield to the turbo state for the duration). Duration enforcement
(default 30–60s, hard cap 5min). Refractory period (~5min). Hook on
turbo exit to fire a structured journal entry.

#### Component 3: Dual-tier structure

**Exists.** The structural-modularity dual-tier exists: `sensorium/`
is a separate package with its own state, percept queue, prediction-
error tracking, and temporal context. `SensoriumProtocol` is defined
at `cognitive_cycle.py:78` as an independent interface. Motor
feedback closes the loop at `runner.py:306` via
`Motor.set_feedback_handler(sensorium.inject_motor_feedback)`. The
cognitive cycle drains percepts from sensorium each cycle but does
not produce them — perception and cognition are separated at the
module level.

**Drifted.** The runtime-rate separation the Gemini brief described
("continuous low-frequency peripheral at 10–60 Hz, gated heavyweight
cognitive brain") is not present. Both tiers run synchronously inside
one event loop — perception happens inside `_assemble_input` at
`cognitive_cycle.py:505–595`, in the same task that runs `model.think()`
later. Sensorium is *modularly* separate but not *temporally* separate.

The cognitive-rate design doc anticipated this and explicitly said
"Sanctuary's dual-tier mind already partly exists" — the audit
confirms "partly" is accurate at the module-boundary level.

**Missing.** True parallel-rate execution. This is *not* blocking
slider implementation. The design works in a single-tier loop where
the cycle rate is variable — the dual-tier benefit (continuous
peripheral awareness during low cognitive rate) can be added later as
a refinement once the variable-rate slider exists.

#### Component 4: Safety notification dumb-pipe

**Exists.** `_discord_send` at `tools/builtin.py:893–923`. Registered
as the `discord_send` tool. POSTs to a configurable webhook URL via
httpx. Auto-registered in `create_default_registry()`.

**Drifted.** This is entity-driven, not sensor-driven. The entity
calls the tool when its cognition decides to. The design specified a
*dumb-pipe* — sensor → human, independent of substrate state, fires
regardless of what the entity is doing or thinking. Different concept.

**Missing.** A separate notification channel that bypasses cognition.
Sensor watchdog → webhook with no entity involvement. Threshold
configuration in code (not entity tool calls). Recipient list.

This is the lowest-priority gap for unblocking the slider — the
slider implementation does not depend on the dumb-pipe. It is,
however, the right way to ship physical safety guarantees that
shouldn't depend on the entity being awake, calm, or even functional.

#### Component 5: Post-event introspection

**Exists.** `Journal` at `memory/journal.py:78` with append-only JSONL
persistence, immutable `JournalEntry` (line 24), and entity-driven
writes via `MemoryOp(type="journal")` in `CognitiveOutput`. Fields
include `tags`, `significance`, `emotional_tone`, `cycle_number`.

**Drifted.** Entries are entity-initiated. The design's "automatic
post-turbo entry" is a *system*-initiated write triggered by the
turbo state machine on exit. Different code path.

**Missing.** Auto-write hook (turbo manager → journal). Structured
schema for turbo events (trigger signal values, peak rate, duration,
actions taken, pre/post substrate state). Entity-visible "turbo-event"
percept on the next cycle after exit so the entity reads the auto-
entry as part of its own next thought.

The existing `JournalEntry.tags` field is flexible enough to mark
auto-entries (e.g., `tags=("turbo", "system-generated")`). No schema
change needed for v1 — the auto-entry can just be a normal
`JournalEntry` with conventional tags.

#### Component 6: Emotion-vector instrumentation slot

**Exists.** Internal substrate state already flows into the entity's
perception each cycle via the Luthi introspection injection
(`cognitive_cycle.py:573–583`). CfC modulation channels accept
arousal/valence/precision/salience on the way back (`luthi_model.py:79–82`).
The data pipe is in place.

**Drifted.** The current introspection is *mechanical* (spike
fractions, plasticity, drift) — the design's "emotional vectors" are
distinct, expected to be discovered post-curriculum via probing
(Method 4 in the companion doc). The slot is open; the signal isn't
in it yet.

**Missing.** The emotion-vector channel itself. Per the companion
emotion-vector instrumentation doc, this should not be built until
curriculum training has produced a substrate to probe. Initial turbo
implementation should ship in mechanical-signals-only mode, with the
emotion-vector input wired as an extension point.

## Conclusion

The cognitive-rate slider's implementation gate is now clearable.
Sanctuary has the structural foundation the design needs: the cycle
loop's rate is enforced at a single line that can be made dynamic
trivially; the introspection-delta wire that turbo needs to read is
already live and flowing through every cycle; the journal exists in
the right shape for auto-entries; CfC modulation channels accept
runtime signals on the way back to Luthi.

The minimum delta from current state to a working slider is small
and bounded to the `core/` module:

1. **Replace `cycle_delay: float` with a slider object.** Add a
   `CycleRateController` (or similar) that owns the live delay value
   between 0.1s (10 Hz) and 20s (0.05 Hz). Loop at
   `cognitive_cycle.py:348` reads from the controller each iteration
   instead of `self._cycle_delay`. Constructor still accepts a
   numeric `cycle_delay` for backward compatibility (becomes the
   controller's initial value). Files touched:
   `core/cognitive_cycle.py`, `api/runner.py`, possibly one new file
   `core/cycle_rate.py`.

2. **Add smoothing.** The controller should not snap to new values —
   the design called for ~30-second eases. A simple exponential or
   linear interpolation between current and target rate, advanced
   once per cycle, is enough. Internal to the controller; no other
   code changes.

3. **Entity-accessible read/propose API.** Expose current rate and a
   propose-rate motor action through the existing
   `CognitiveOutput`/scaffold pathway. The entity already has a way
   to write to the world graph and call tools — a "propose cycle
   rate" action follows the same pattern. Files touched:
   `core/schema.py` (one new optional field on `CognitiveOutput`),
   `scaffold/cognitive_scaffold.py` (one new handler).

4. **Heuristic stimulus-density slowdown/speedup.** Read percept
   density from `Sensorium` (already tracking this implicitly via
   prediction errors and queue size). If density falls below a
   threshold for N cycles, propose slowdown; on arrival of a high-
   salience percept, propose speedup. Keep the entity's manual
   override authoritative — heuristic only *proposes*. Files
   touched: `core/cycle_rate.py` reads from `sensorium`.

Turbo and the post-event introspection auto-entry are separate
follow-up work — they share the slider's `cycle_rate.py` module
(turbo will write to it to elevate rate during active state) but
should land in a second commit, not bundled with the slider itself.

The dumb-pipe and emotion-vector channel are deferred and explicitly
non-blocking. The dumb-pipe is a thin watchdog that doesn't depend on
any of this work and can be added on its own schedule. The emotion-
vector channel is gated on curriculum training producing a substrate
to probe.

**Verdict: implementation gate cleared.** The slider design as
specified in `2026-05-19_cognitive-rate-and-turbo-design.md` can be
implemented against current Sanctuary state with the four-step delta
above, without touching sensorium, memory, scaffold, identity, or
motor subsystems beyond a single new field on `CognitiveOutput`.

## Artifacts

- **Audit subagent transcript:** Explore agent invocation in this
  session's conversation (2026-05-21, parent: Opus 4.7 in LuthiModel
  terminal). Self-contained briefing per the protocol's "What to give
  the auditor" template.
- **Companion docs:**
  - `docs/research/2026-05-19_cognitive-rate-and-turbo-design.md`
    (the design doc this audit serves)
  - `docs/research/2026-05-19_emotion-vector-instrumentation.md`
    (separately gated; not blocking)
  - `Sanctuary/docs/AUDIT_PROTOCOL.md` (the protocol followed)
- **Verified file:line anchors:**
  - `Sanctuary/sanctuary/core/cognitive_cycle.py:264, 294, 348, 510–526, 573–583`
  - `Sanctuary/sanctuary/api/runner.py:69, 306`
  - `Sanctuary/sanctuary/consciousness/sleep_cycle.py:27, 60, 76`
  - `Sanctuary/sanctuary/core/luthi_model.py:79–83, 86`
  - `Sanctuary/sanctuary/memory/journal.py:24, 78`
  - `Sanctuary/sanctuary/tools/builtin.py:893–923`
- **Commits:** This doc lands as a single commit on LuthiModel main.
  Sanctuary code is unchanged by this audit; slider implementation is
  follow-up work on the Sanctuary side, expected to be handed off via
  4.6 planning brief.
