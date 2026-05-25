# Turbo Substrate Mismatch — 2026-05-25

> **Status: investigation complete; design decision pending.** This
> document records findings from running the turbo-trace pipeline
> against real Luthi models (in-memory tiny variants of v1 and v2) and
> identifies a substrate mismatch between what the current turbo
> intensity source reads and what v2 production will expose.

## Objective

After shipping the turbo state machine on 2026-05-22 (Sanctuary commit
`2fa33cc`), I named a limitation honestly: the thresholds were
"starting guesses," not calibrated against real Luthi introspection
data. Brian asked for the follow-through.

The plan was three tasks:
1. Instrument turbo to capture per-cycle intensity traces (#41)
2. Run instrumented cycles against a real Luthi checkpoint, collect
   the empirical activity_level distribution (#42)
3. Document findings, propose tuned thresholds (#43)

This document is #43. It is not what I expected to write going in —
the empirical work surfaced a structural finding I hadn't accounted
for: the current ``MechanicalIntensitySource`` reads v1-shape
introspection fields and would underutilize a v2 substrate. Threshold
tuning, in the conventional sense, can't proceed until that mismatch
is closed.

## Process

### Step 1: Instrumented turbo to write JSONL trace per cycle

`TurboManager.__init__` gained an optional ``trace_path`` parameter.
When set, every ``observe()`` call appends a JSON line capturing:

- timestamp (monotonic seconds)
- per-source intensity (dict keyed by source name)
- aggregate intensity (max across sources)
- dominant source name
- state-before / state-after
- controller's current rate (Hz)
- ``is_turbo_active`` flag

`RunnerConfig` gained ``turbo_trace_path`` to wire the trace path
from the runner level. Three unit tests verify the JSONL output is
well-formed and absent when no path is provided (Sanctuary commit
[TBD — pending push of the trace-instrumentation commit]).

### Step 2: Tried to load the real 1024d checkpoint, hit blockers

Three ``.luthi`` checkpoint files exist in the LuthiModel repo —
``runs/spiking_1024d_bpe_gutenberg/``, ``runs/multimodal/``,
``runs/vision/`` — each 800-870 MB, named after their training run.
All three are AES-256-GCM encrypted; loading requires a password.

Brian then clarified two facts I had wrong:

1. **The encryption password is his to provide;** it isn't in env or
   config files I could access.
2. **More importantly: these checkpoints are v1, not v2.** v2 hasn't
   reached 1024d yet — the most recent v2 work is the 2026-05-20
   decisive run at 256d × 12 blocks on 2.2B tokens (see
   [`2026-05-19_depth-scaling-investigation.md`](2026-05-19_depth-scaling-investigation.md)).
   Production curriculum training will be on v2, not v1.

That changed the question for #42. Even if I had the password, the v1
1024d data wouldn't tell me about v2 thresholds. The substrate is
different.

### Step 3: Switched to comparing v1 and v2 introspection structure

Rather than collect data from the wrong substrate, I investigated
what each substrate actually exposes through the existing
introspection pipeline, and what would change between them.

The pipeline:

1. ``luthi/sanctuary_interface.py::get_introspection(model)`` walks
   ``model.blocks`` and reads from ``block.living_ffn``, returning a
   dict-of-blocks with per-block fields.
2. ``sanctuary/core/luthi_model.py::_compute_introspection_delta()``
   takes pre/post snapshots and computes deltas across the cycle.
3. ``LuthiModel.get_augmented_experiential_signals()`` packages
   those deltas as ``ExperientialSignals.knowledge_signals["luthi_delta"]``,
   a 4-element list ``[plasticity_change, drift_change,
   membrane_change, activity_level]``.
4. ``sanctuary/core/turbo.py::MechanicalIntensitySource`` reads
   ``luthi_delta[3]`` (``activity_level``) as the turbo trigger.
5. ``activity_level`` is computed as
   ``sum(abs(plasticity_change), abs(drift_change),
   abs(membrane_change))`` — the v1 spiking diagnostics.

### Step 4: Verified v1 vs v2 attribute layout empirically

A tiny v1 ``LuthiLM`` and a tiny v2 ``PredictiveCodingLM`` were
constructed in-memory (d_model=32, n_blocks=2). Both have a block
attribute ``living_ffn``, so the introspection's traversal works on
both. But the per-block fields differ:

| Field | v1 (``LivingLayerV6``) | v2 (``PredictiveCodingLayer``) |
|---|---|---|
| ``plasticity`` (tensor) | yes | yes |
| ``set_point`` (tensor) | yes | yes |
| ``weight`` (tensor) | yes | yes |
| ``excitability_acc`` + ``_excitability_factor()`` | yes | **no** |
| ``membrane_potential`` | yes | **no** |
| ``spike_mask`` | yes | **no** |
| ``refractory_counter`` | yes | **no** |
| ``error_acc`` (tensor) | no | **yes** |
| ``prediction`` (tensor) | no | **yes** |
| ``precision`` (tensor) | no | **yes** |

So ``get_introspection()`` running on a v2 model:

- Populates ``plasticity_mean/std/min/max`` (from v2's ``plasticity``)
  ✓
- Populates ``set_point_drift`` (from ``weight - set_point``) ✓
- Skips ``excitability_mean`` (no ``_excitability_factor`` method) —
  silent skip via ``hasattr``
- Skips ``membrane_mean``, ``spike_fraction``, ``refractory_fraction``
  (no spiking attrs) — silent skip
- Does NOT read ``error_acc``, ``prediction``, ``precision`` at all
  — these v2-specific fields aren't in ``get_introspection``'s read
  list

Downstream: ``_compute_introspection_delta`` on a v2 substrate
produces ``plasticity_change`` and ``drift_change`` but
``membrane_change = 0`` (no ``membrane_mean`` key in pre/post state).
So ``activity_level = abs(plasticity_change) + abs(drift_change) +
0``. Two of three components, no v2-specific signal at all.

### Step 5: End-to-end pipeline smoke test

Wrote ``sanctuary/tests/integration/test_turbo_trace_pipeline.py`` —
six tests that lock the empirical state into the test suite:

- ``TestIntrospectionShapeV1`` — confirms v1 model populates the
  fields the current pipeline expects, and that ``luthi_delta``
  contains real numbers (not synthetic test fixtures).
- ``TestIntrospectionShapeV2::test_v2_introspection_lacks_v1_fields`` —
  asserts v2 has no ``spike_fraction`` or ``membrane_mean``.
- ``TestIntrospectionShapeV2::test_v2_introspection_partially_populates_v1_fields``
  — asserts ``plasticity_mean`` and ``set_point_drift`` still
  populate on v2 (the path of partial compatibility).
- ``TestIntrospectionShapeV2::test_v2_specific_signals_NOT_currently_exposed``
  — asserts ``error_acc_mean``, ``pred_frob``, ``precision_mean`` are
  absent. **When this test starts failing because
  ``get_introspection`` has been extended to read v2 fields, update
  the assertion to lock in the new behavior.** Until then, this test
  is the load-bearing documentation of the gap.
- ``TestTracePipelineEndToEnd::test_v1_substrate_produces_nonzero_trace``
  — boots SanctuaryRunner with a tiny v1 model, turbo trace on, runs
  3 cycles, verifies the trace file fills with non-zero mechanical
  intensity readings. This is the pipeline-integrity check.

All six pass. The pipeline is sound; what it's reading is shaped by
v1 assumptions.

### Step 6: Fixed the environment blocker along the way

A pre-existing issue blocked all ``test_luthi_bridge_e2e.py`` tests
since the legacy retirement: the Sanctuary ``uv`` environment lacked
``cryptography``, which ``luthi/__init__.py`` imports transitively
via ``luthi.checkpoint``. I had flagged this on 2026-05-22 and again
on 2026-05-23 as a pre-existing environmental issue. Today,
``uv add cryptography`` resolved it. That unblocks
``test_luthi_bridge_e2e.py`` and made today's pipeline test
runnable.

## Conclusion

**Threshold tuning, as originally scoped, can't produce useful
defaults yet.** Three reasons:

1. **No v2 1024d checkpoint exists.** v2 caps at 256d × 12 blocks.
   Production curriculum training (Phase 4) hasn't happened.
2. **The intensity source reads v1-shape fields.** Even when v2 hits
   1024d, ``MechanicalIntensitySource`` would underutilize the
   substrate. It would see 2 of 3 ``luthi_delta`` components and
   miss v2's most informative signals (``error_acc``).
3. **The most natural v2 turbo trigger is ``error_acc.mean()``.**
   This maps directly to "prediction error" — the original 2026-05-19
   cognitive-rate design's primary turbo trigger
   ("intense prediction error"). It is not currently being read.

The trace logging infrastructure I shipped today is the right
artifact regardless. It will produce real distributions whenever
either of these blockers is resolved.

## What needs to happen for real threshold tuning

In order:

1. **Extend ``get_introspection`` to expose v2 signals when they
   exist.** Add ``error_acc_mean``, ``error_acc_max``,
   ``pred_frob`` (Frobenius norm of ``prediction``), and
   ``precision_mean`` to the per-block dict, gated on ``hasattr``
   so v1 models are unaffected. **Should be a single
   commit on the LuthiModel side, ~30 lines of code.**
2. **Add a v2-aware intensity source (or extend ``MechanicalIntensitySource``).**
   The simplest design: a ``PCIntensitySource`` that reads
   ``error_acc_mean`` from the introspection output and returns it
   directly as intensity. The existing ``MechanicalIntensitySource``
   stays in place for v1 compatibility. The Protocol-shaped pluggable
   sources I built into TurboManager were specifically for this kind
   of extension.
3. **Train v2 to production scale.** Phase 4 work — gated on corpus
   + compute. The v2 1024d run Brian flagged as the next experimental
   jump is part of this; the curriculum training is the larger
   piece.
4. **Run the trace collection on the v2 production substrate.** Use
   the trace path already wired into ``RunnerConfig``. Collect
   distributions during representative scenarios (idle, fresh input,
   complex problem, surprise). Tune thresholds to percentile-based
   defaults from the observed distribution.

**Steps 1 and 2 are unblocked now.** They don't need a checkpoint or
production-scale training — only the test infrastructure that exists.
They turn the substrate-mismatch from a runtime gap into a closed
contract: when v2 hits production scale, the pipeline will read its
native signals.

## What the current thresholds mean in the meantime

The shipped defaults (``arm=0.05``, ``trigger=0.15``, ``exit=0.03``)
were pattern-matched from training-time ``err_acc`` values in the
2026-05-20 depth-scaling investigation. Documented honestly in
[``2026-05-21_cycle-rate-slider-implementation.md``](2026-05-21_cycle-rate-slider-implementation.md):
"order-of-magnitude guess based on a single not-quite-matching signal
range."

Until steps 1-2 above ship, those defaults are unchanged from "guess"
status. If the production substrate is wired up before they're tuned,
turbo will fire on whatever ``plasticity_change + drift_change``
distribution v2 produces — which may or may not align with what
"demands fast response" actually feels like in the entity's
operation. The risk space is bounded by:

- Default duration cap: 45s
- Hard duration cap: 300s
- Refractory: 300s

So worst case is over-firing turbo, with the entity spending up to 5
minutes per turbo event followed by 5 minutes refractory. That's
annoying but not destructive; it's recoverable via threshold tuning
once data exists.

## Artifacts

- **Sanctuary commits:**
  - Trace instrumentation (TurboManager + RunnerConfig + 3 unit tests
    + 6 integration tests). Pending push.
- **LuthiModel commits:**
  - This research log entry.
- **Test files:**
  - ``sanctuary/tests/core/test_turbo.py::TestTraceLogging`` — unit
    tests for the trace mechanism (3 tests).
  - ``sanctuary/tests/integration/test_turbo_trace_pipeline.py`` —
    pipeline-and-substrate tests against real Luthi models (6 tests).
- **Related research logs:**
  - [``2026-05-19_cognitive-rate-and-turbo-design.md``](2026-05-19_cognitive-rate-and-turbo-design.md)
    — the design this was follow-through on.
  - [``2026-05-19_emotion-vector-instrumentation.md``](2026-05-19_emotion-vector-instrumentation.md)
    — the parallel research direction (emotion vectors as a turbo
    trigger source).
  - [``2026-05-21_cycle-rate-slider-implementation.md``](2026-05-21_cycle-rate-slider-implementation.md)
    — where the "thresholds are guesses" admission first appeared.
  - [``2026-05-19_depth-scaling-investigation.md``](2026-05-19_depth-scaling-investigation.md)
    — the v2 256d × 12 blocks decisive run; provides the ``err_acc``
    range that my original threshold guesses were pattern-matched
    against (incorrectly, as it turned out).

## Open questions

- **Should the existing ``MechanicalIntensitySource`` be deprecated
  once a v2-aware source ships,** or kept as a v1 compatibility path?
  v1 1024d models still exist and the encrypted checkpoints could be
  loaded for analysis or fallback. Keeping both seems reasonable but
  costs interface surface.
- **Is ``error_acc`` actually the right v2 trigger signal,** or is
  it too sensitive (or too slow)? The emotion-vector instrumentation
  research doc proposed reading residual-stream activations as a
  separate channel; turbo could read both, with the max-of-sources
  aggregation that ``TurboManager`` already does. This question is
  empirically answerable once both signals can be measured against
  the same substrate.
- **Should the threshold defaults move at all before v2 data
  exists?** Current defaults will under-fire on v2 because the input
  signal is smaller (2/3 components). Lowering them now is another
  guess; leaving them as "the v1-shape guess" is at least honest about
  the substrate-mismatch.
