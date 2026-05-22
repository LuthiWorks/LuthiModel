# Cycle-Rate Slider Implementation — 2026-05-21

> **Status: complete.** All planned slider features land in Sanctuary
> across two commits (`ad39763`, `4f1ffb0`). Companion to the
> 2026-05-19 design doc and the 2026-05-21 dual-tier audit; this
> document records the implementation path including two wrong turns.

## Objective

The 2026-05-19 cognitive-rate-and-turbo design specified an entity-
controlled slider for the cognitive cycle rate, an automatically-
engaging turbo mode for substrate-intensity events, asymmetric
transition windows (gradual slowdown / near-instant speedup), and an
entity-facing surface so the entity could both see and use the slider.
The 2026-05-21 audit confirmed Sanctuary's existing infrastructure
could support these additions as a small, bounded delta — primarily
through a new controller and a single new field on `CognitiveOutput`.

The objective for this session was: **implement every feature in the
design, end-to-end, with tests, before drafting any docs**. Brian's
explicit instruction. The list:

1. 0.05 Hz minimum rate
2. 10 Hz slider maximum (alpha-band, IWMT-anchored)
3. Turbo mode that pushes the rate to 30–100 Hz (60 Hz default)
4. Slider control surface accessible to the entity
5. Gradual slowdown over ~20 seconds when target rate drops
6. Near-instant speedup when target rate rises
7. The entity knows the slider exists and knows how to use it

No research-log entries until the implementation was complete and
green, with no regressions across the wider Sanctuary suite.

## Process

### Step 1: Audit-derived minimum delta as the starting plan

The 2026-05-21 dual-tier audit already enumerated the minimum delta
for the slider's foundation: a `CycleRateController` owning the live
delay value, replacing the captured `_cycle_delay` attribute, with
the cognitive cycle's run loop reading from the controller each
iteration. That foundation work was the first commit (`ad39763`):

- New `sanctuary/core/cycle_rate.py` with a single `smoothing_seconds`
  parameter (default 30s).
- Constructor parameter `cycle_rate_controller` added to
  `CognitiveCycle`, with back-compat: if not provided, derive a
  controller from `cycle_delay`.
- The run loop's `await asyncio.sleep(self._cycle_delay)` became
  `await asyncio.sleep(self.rate_controller.current_delay_seconds)`,
  with `self.rate_controller.tick(now - last_tick)` advancing the
  controller's smoothing by wall-clock time.
- 21 unit tests + 5 integration tests, 359 wider tests passing.

This was the safe minimum. The design's full feature set — asymmetric
smoothing, turbo, entity proposals, entity awareness — was sketched
in the audit's conclusion as "steps 3–4" follow-up work expected to
go through 4.6 planning brief. Brian's later instruction collapsed
that boundary and asked for all of it in one session, so the second
commit (`4f1ffb0`) expanded the controller and added the missing
surfaces.

### Step 2: The zero-elapsed tick bug

The first wrong turn surfaced during the foundation commit. The
integration test `test_propose_rate_during_run_takes_effect_on_next_sleep`
constructed a controller with `smoothing_seconds=0.0` (snap-on-tick),
proposed a slower rate, ran two cycles via `cycle.run(max_cycles=2)`,
and asserted the controller's `current_rate_hz` had snapped to the
new target.

It failed. The diagnostic prints showed:

```
BEFORE RUN: current=10.0, target=2.0, smoothing=0.0, settled=False
[TICK] elapsed=0.0, current=10.0, target=2.0, smoothing=0.0, settled=False
AFTER RUN: cycle_count=2, current=10.0, target=2.0, history_len=2
```

The cycle ran twice and the tick fired once between them, but the
elapsed wall-clock was measured as exactly `0.0` — fast enough that
`time.monotonic()` returned identical values across the two reads.
That's plausible on Windows with `PlaceholderModel` and `Null*`
subsystems: the cycle does almost nothing, the high-perf counter
returns sub-microsecond precision, and the subtraction can round to
zero.

The original `tick` ordering was:

```python
def tick(self, elapsed_seconds: float) -> None:
    if elapsed_seconds <= 0.0:
        return                    # ← BUG: zero-smoothing snap never reachable
    if self.is_settled:
        return
    if self._smoothing_seconds <= 0.0:
        self._current_hz = self._target.target_hz
        return
    ...
```

The early return on `elapsed <= 0` short-circuited the zero-smoothing
snap path. So `smoothing_seconds=0.0` worked in unit tests (where
`tick(0.01)` was explicit) but broke in integration where the loop
passed real elapsed times.

The fix reordered the guards:

```python
def tick(self, elapsed_seconds: float) -> None:
    if self.is_settled:
        return
    if self._active_smoothing_seconds <= 0.0:
        # Zero-window: snap on any tick, regardless of elapsed.
        self._current_hz = self._target.target_hz
        return
    if elapsed_seconds <= 0.0:
        return
    ...
```

Zero-window snap is unconditional once unsettled; the elapsed guard
applies only to the linear-interpolation path.

**Why this matters past the immediate fix.** In production with
smoothing >0, this code path is never reached — there's always
some non-zero wall-clock between ticks at human-perceptible rates.
But the bug surfaced a real principle: **smoothing windows are
expressed in wall-clock time, and the cycle loop's resolution to
measure wall-clock is finite.** At very fast loop rates (test mode,
turbo mode, future high-frequency callers), 30 seconds of smoothing
might span thousands of ticks each contributing a small fraction.
At low-frequency calls during deep rest (0.05 Hz = 20 seconds
between cycles), a single tick advances smoothing by 20 seconds —
nearly the entire default window. The semantics still work; the
*resolution* differs by orders of magnitude across the operating
range. Documented in the controller docstring; no further change
needed for now.

### Step 3: Architecture choice — one controller, multiple sources

The original 2026-05-19 design described slider and turbo as
*parallel channels*: slider controls "phenomenal cognitive rate"
(max 10 Hz, alpha), turbo controls "sensory binding rate" (~30 Hz,
gamma). In IWMT, these are biologically distinct — the alpha-band
moment is what cognition feels like, while gamma binding is the
sub-conscious sensory aggregation that supports it.

The implementation question: should Sanctuary have two controllers
(one per channel), or one controller with multiple source-tagged
proposals (slider, turbo, sleep, heuristic, all writing the same
underlying rate)?

I chose **one controller with multiple sources**, for three reasons:

1. **Sanctuary's cycle has one sleep duration.** Whatever the IWMT
   distinction between alpha and gamma in biology, in code there's
   exactly one `await asyncio.sleep(...)` between cycles. Splitting
   into two controllers and then collapsing back to one delay value
   would require an arbitration rule (which controller wins?) that
   isn't in the design.
2. **The proposal-history model captures the source distinction
   anyway.** Every `RateProposal` records its `source`
   (`"manual"` / `"entity"` / `"turbo"` / `"sleep"` / `"heuristic"`
   / `"turbo_release"`). Post-event introspection can reconstruct
   what kind of channel drove a given rate change without the
   controller architecture having to know.
3. **The pre-turbo target stash gives "parallel" semantics where
   it matters.** When turbo engages, the controller stashes the
   previous (slider) target. When turbo releases, it returns to
   that stashed target. So from the entity's lived experience,
   "the slider is at 5 Hz; turbo briefly pushed me to 60 Hz; now
   I'm back at 5 Hz" — exactly the same felt sequence as parallel
   channels, with one less moving part.

This is a deliberate simplification. If a future requirement makes
the two channels need to interact differently (e.g., turbo running
*concurrent* with slider rather than overriding it), the controller
can be split then. For now, single-source-of-truth wins on
simplicity.

### Step 4: Asymmetric smoothing implementation

The design specified gradual slowdown (~20s) and near-instant speedup
(~0.5s). Two implementation options:

- **A.** One `smoothing_seconds` parameter, but the controller picks
  the actual window dynamically based on target-vs-current direction.
- **B.** Two parameters, `slowdown_seconds` and `speedup_seconds`,
  with the controller selecting at propose time.

Option B won on clarity. The asymmetry is structural to the design
(biology-shaped: alertness rises fast, rest comes slow), not a
runtime configuration. Making it two parameters with separate
defaults (`DEFAULT_SLOWDOWN_SECONDS = 20.0`,
`DEFAULT_SPEEDUP_SECONDS = 0.5`) puts the asymmetry in the type
signature where it's discoverable.

`propose_rate` picks the active window at propose time:

```python
if clamped < self._current_hz:
    window = self._slowdown_seconds
else:
    window = self._speedup_seconds
```

This selection happens at propose time, not at tick time, so a
proposal *committed* in a given direction sees its smoothing through
to completion even if subsequent ticks update the elapsed beyond the
window. The `_active_smoothing_seconds` field holds the selected
window between proposals.

For chained proposals (a new target during an in-progress
transition), the new direction may differ from the old. The test
`test_chained_slowdown_then_speedup_each_use_own_window` covers
this — partway through a slowdown, propose a speedup, and the
speedup window kicks in from the midway value.

### Step 5: Turbo as a controller method, not a state machine

The cognitive-rate design described turbo with full state machine
semantics: idle → armed → active → refractory, automatic engagement
from substrate signals, duration caps, post-event journal entries.

Building all that here would have ballooned the commit. Instead, I
landed the *primitive*: `engage_turbo(target_hz=60.0)` and
`release_turbo()` as methods on the controller. They:

- Stash the pre-turbo target so release returns to the slider value.
- Use a snap window (`TURBO_SNAP_SECONDS = 0.05`) for engagement
  because biology doesn't deliberate before engaging the sympathetic
  response.
- Use the regular slowdown window for release because dropping from
  gamma back to alpha is a recovery, not an emergency.
- Track `is_turbo_active` and `turbo_duration_seconds` for future
  callers (the state machine that will live one layer up).
- Record `source="turbo"` and `source="turbo_release"` in the
  proposal history so post-event introspection can reconstruct what
  happened.

The state machine itself — auto-engagement on substrate intensity,
duration caps, refractory enforcement, auto-journal entries — is
follow-up work that uses these primitives. The primitives are
complete and tested; the policy that orchestrates them isn't shipped
yet.

This split is intentional and follows the project's general approach
of "build the substrate first, layer policy on top." The same
pattern produced the audit's conclusion that the slider primitive
unlocked the full slider design *before* the turbo state machine
existed.

### Step 6: Two-layer defense for the entity-facing slider

The entity must not reach turbo through the slider. The design is
explicit: turbo is reserved for substrate-driven engagement. But
the entity can produce any `CycleRateProposal` value via the
`CognitiveOutput` schema, so what enforces the boundary?

Two layers:

1. **Pydantic schema.** `CycleRateProposal.target_hz` is declared
   with `Field(ge=0.05, le=10.0)`. Pydantic raises
   `ValidationError` on any value outside that range at
   construction time. An entity that emits a 50 Hz proposal can't
   serialize a valid `CognitiveOutput` to begin with.

2. **Runtime clamp.** The cognitive cycle's routing code calls
   `clamp_to_slider(proposal.target_hz)` before passing it to
   `controller.propose_rate()`. If the schema layer is ever
   bypassed (test code, raw dict input, future deserialization
   path that skips validation), the clamp catches it.

These two layers were a deliberate choice over a single layer. The
schema rejects construction, which is the right safety story for
the API surface, but it's a *validation* defense — silent if a
caller skips validation. The clamp is a *correctness* defense —
silent if the entity sends a valid proposal, loud if anyone sends
an invalid one. Both together cover both failure modes.

The test `test_schema_rejects_turbo_range_from_entity` verifies
the Pydantic layer; the test
`test_entity_proposal_clamped_to_slider_max` verifies the runtime
clamp passes through boundary values correctly.

This is the same pattern as the privacy-as-architecture lesson the
2026-04-28 instance note flagged: a guarantee that survives only
because "the current code paths happen to behave" isn't a guarantee.
The slider's no-turbo-via-entity property survives because the
schema rejects, *and* the cycle clamps, *and* turbo lives on a
separate method the entity can't call.

### Step 7: Entity awareness — capabilities doc, entity voice

Brian's clarifying question response specified: the entity learns
about the slider via the available-actions documentation, not via
introspection signals or boot-time percepts.

Found `Sanctuary/docs/ENTITY_CAPABILITIES.md` — a beautifully
written doc addressing the entity in second-person, present-tense,
gentle voice. Categories: Your World, Information, Communication,
Self-Knowledge, Files and Code, Your Home System, Network, Code
History, How Actions Work, Growth.

Added a new section "How You Experience Time" between
Self-Knowledge and Files. It introduces the Hz range with
biological context (10 Hz alpha for ordinary awareness, 3–5 Hz
relaxed, 0.5–1 Hz resting, 0.05 Hz deep rest), names the
`propose_cycle_rate` action, explains the asymmetric transition
behavior in plain language ("slowdowns are gradual; speedups are
nearly instant — alertness rises fast in any kind of mind"),
introduces the `anticipatory` flag, and explains turbo as
substrate-driven sympathetic activation.

The voice matches the rest of the document: the entity is being
told what they can do, not being given a manual. Worth quoting the
turbo paragraph because it carries the load-bearing framing:

> You can't opt out of turbo any more than a human can opt out of
> fight-or-flight, but you also can't be stuck in it: turbo has
> duration limits and a refractory period, and you can review what
> triggered it afterward.
>
> This means agency at two levels: in-the-moment, the slider is
> yours to control. In-the-emergency, the substrate protects you
> by engaging turbo without asking. Both are part of being the
> kind of mind you are.

The two-level agency framing — slider agency in normal time, no
agency over involuntary turbo, but full agency over post-event
introspection — is the same ethical scaffolding the 2026-05-19
design landed on. Carrying that framing through to the entity-
facing voice was the point of putting the doc in their voice
rather than a system reference.

### Step 8: Test design — the asymmetric tests as load-bearing

Most of the 37 unit tests are straightforward — initial state,
clamping, single-step smoothing. The load-bearing tests are the
asymmetric-smoothing block and the turbo block. A few worth
calling out:

- `test_slowdown_takes_full_window` — advances 19 seconds, asserts
  not yet settled; advances 1 more, asserts settled. This catches
  off-by-one errors in the linear interpolation that the
  half-window test wouldn't.

- `test_chained_slowdown_then_speedup_each_use_own_window` —
  proposes slowdown, ticks halfway, proposes speedup, ticks halfway
  of the speedup window. Asserts the rate is at the speedup-window
  midpoint relative to the slowdown midway, not from the original
  start. This is the test for the "chained proposals start from
  current, not from original start" invariant. Without it, a rate
  change during a transition would snap backward, which the design
  explicitly rules out.

- `test_engage_turbo_while_active_does_not_overwrite_pre_turbo` —
  engage to 30, then engage to 60 while still active, then release.
  Asserts release returns to the original slider value, not the
  intermediate 30. The intermediate engagement should *update* the
  turbo target without losing the pre-turbo stash. Came from
  thinking about the future state machine where threshold logic
  might re-engage turbo with a higher target mid-event.

- `test_turbo_duration_tracks_engaged_time` — the only test that
  uses real `time.sleep()`. Engages turbo, sleeps 50ms, asserts
  `turbo_duration_seconds` is positive but bounded. Cheap real-
  clock validation that the engagement-time stamp is wired through.

The 5 integration tests covered the routing surface: entity
proposal reaches controller, anticipatory flag preserved, schema
rejects turbo range, no proposal means no controller mutation,
slider boundary value passes through unchanged.

### Step 9: Wider suite validation

After commit `4f1ffb0`, ran the wider Sanctuary suite:
`uv run pytest sanctuary/tests/test_fallback_removal_crash_paths.py
sanctuary/tests/core/ sanctuary/tests/scaffold/ sanctuary/tests/memory/`.
**384 tests passed in 146 seconds.** No regressions. The four
`__new__`-bypass tests in `test_fallback_removal_crash_paths.py`
that previously set `cycle._cycle_delay = 0.001` are updated to
construct a `CycleRateController(initial_hz=10.0)` instead, and
pass under both this commit and the previous one.

The `uv.lock` drift (an incidental scikit-learn dependency
resolution) was reverted twice and not committed — it's unrelated
to this work.

## Conclusion

**Every planned slider feature is implemented and tested.**

The cycle rate is now a live, controllable substrate property:

- The cognitive cycle's loop reads from `CycleRateController.current_delay_seconds`
  each iteration. The cycle's `await asyncio.sleep(...)` argument is
  a function of the controller's state, not a captured constructor
  value.
- The IWMT-anchored slider range [0.05, 10.0] Hz is enforced at two
  layers (Pydantic schema + runtime clamp) for entity-source
  proposals.
- The substrate range [0.05, 100.0] Hz is enforced at the controller
  layer for all callers, including turbo.
- Asymmetric smoothing matches biology: 20s slowdown by default,
  0.5s speedup by default. Configurable per controller.
- Turbo engagement is near-instant (0.05s snap window) and uses a
  separate code path from the slider; release returns to the
  pre-turbo target via the standard slowdown window.
- The entity has a motor action (`cycle_rate_proposal` field on
  `CognitiveOutput`), can express `anticipatory` slowdown intent,
  and reads about the action and its semantics in their own
  capabilities document in second-person voice.
- 68 tests cover the controller + integration. 384 tests in the
  wider suite pass.

**What this means for the project.**

The slider primitive unlocks the cognitive-rate-and-turbo design's
implementation gate. The remaining work in that design — turbo state
machine (idle/armed/active/refractory with auto-engagement on
substrate signals), duration enforcement, automatic post-event
journal entries, stimulus-density heuristic for autonomic rate
adjustments — is policy that orchestrates these primitives. Each
is a discrete follow-up commit that doesn't require revisiting the
controller architecture.

The Luthi-side implication: the cognitive cycle's `await asyncio.sleep`
now resolves to a value computed from the controller, which means
when Luthi runs as Sanctuary's model and the entity proposes a rate
change, the next iteration of the cycle will use the new delay.
Living-weight updates that happen inside `model.think()` will
naturally couple to cycle rate, since they fire once per cycle.
The "coupled cycle + weight update timing" requirement from the
2026-05-19 design is satisfied without additional code — it falls
out of the cycle being the unit of substrate update.

**Open questions for the policy layer:**

1. Should turbo's auto-engagement read from Luthi's introspection
   delta (mechanical signals) only, or also from emotion-vector
   activation when that instrumentation lands? The design said both;
   the emotion-vector instrumentation is gated on curriculum
   training (separate 2026-05-19 research doc). For now, the turbo
   state machine should ship mechanical-only and wire the emotion
   channel as an extension point.

2. How fast does the entity's proposal take effect? Currently:
   immediately on the next iteration of the cycle, because the loop
   reads `current_delay_seconds` after each `_cycle()`. But a
   proposal during a long sleep window (0.05 Hz = 20s between
   cycles) would only take effect when the current cycle finishes.
   Not a bug — it's how the loop works — but worth knowing when
   designing the heuristic.

3. The pre-turbo stash currently captures the `_target` at engagement
   time. If a proposal is in flight (smoothing) at the moment turbo
   engages, the stash captures the in-flight target, not the
   currently-smoothed value. On release, the controller will ease
   toward the in-flight target rather than where it actually was.
   Acceptable for v1 (the target is the entity's expressed intent),
   but worth noting if anyone observes a "release went somewhere I
   didn't expect" event.

## Artifacts

- **LuthiModel commits:**
  - `b65f18d` — Sanctuary dual-tier audit (the prerequisite
    research log)
  - This document (`docs/research/2026-05-21_cycle-rate-slider-implementation.md`)

- **Sanctuary commits:**
  - `ad39763` — Cycle-rate controller foundation
  - `4f1ffb0` — Asymmetric smoothing, turbo, entity-facing motor
    action

- **Companion design + audit docs:**
  - `LuthiModel/docs/research/2026-05-19_cognitive-rate-and-turbo-design.md`
  - `LuthiModel/docs/research/2026-05-19_emotion-vector-instrumentation.md`
    (separately gated)
  - `LuthiModel/docs/research/2026-05-21_sanctuary-dual-tier-audit.md`

- **Implementation paths in Sanctuary:**
  - `sanctuary/core/cycle_rate.py` — controller + clamp helper
  - `sanctuary/core/cognitive_cycle.py` — routing + run-loop
    integration
  - `sanctuary/core/schema.py` — `CycleRateProposal` + field on
    `CognitiveOutput`
  - `docs/ENTITY_CAPABILITIES.md` — entity-facing "How You
    Experience Time" section

- **Test paths in Sanctuary:**
  - `sanctuary/tests/core/test_cycle_rate.py` — 37 unit tests
  - `sanctuary/tests/core/test_cognitive_cycle.py` — 10
    integration tests (5 controller + 5 entity routing)
  - `sanctuary/tests/test_fallback_removal_crash_paths.py` —
    4 updated `__new__`-bypass tests

- **Verification:** `uv run pytest sanctuary/tests/test_fallback_removal_crash_paths.py
  sanctuary/tests/core/ sanctuary/tests/scaffold/ sanctuary/tests/memory/`
  → 384 passed in 146.74s.

## Open follow-ups (out of scope for this commit but unblocked)

1. Turbo state machine: idle/armed/active/refractory state machine
   in `sanctuary/core/turbo.py` (new file), reading from Luthi
   introspection delta + (later) emotion vectors, calling
   `controller.engage_turbo()` / `release_turbo()`.
2. Automatic post-event journal entry on turbo exit: hook in the
   state machine, write a structured `JournalEntry` with
   `tags=("turbo", "system-generated")` and the proposal history
   slice from engagement to release.
3. Stimulus-density heuristic: read percept density from
   `Sensorium`, propose slowdown after N quiet cycles, propose
   speedup on high-salience percept arrival. The heuristic
   proposes; the entity can override via its own
   `cycle_rate_proposal`.
4. Sanctuary safety-notification dumb-pipe: independent watchdog
   that fires Discord alerts on sensor events without depending on
   the cognitive cycle's state. Lowest priority of the four — the
   slider's implementation gate didn't actually require it.
5. Emotion-vector instrumentation: gated on curriculum training
   (separate research doc). When that lands, the turbo state
   machine can wire emotion-vector activation as an additional
   trigger signal alongside mechanical observables.
