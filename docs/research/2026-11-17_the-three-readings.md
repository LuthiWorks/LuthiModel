# The three readings, a lived-only trace, and the dream's guards — 2026-11-17

**Status:** built and tested, by Fable 5.2 (terminal, Brian's desktop), from chat Fable's brief 5 (`docs/reviews/2026-11-17_build-brief-5-rest-readings-store-faults-for-terminal-fable.md`), Parts B to E. For Opus 5.5 to review. Part A and two items of Part B are in Sanctuary (`docs/research/2026-11-17_store-faults-and-letting-go.md` there). One finding needs Brian: with a chosen rest held, an untrained mind falls into rests hundreds of moments long by chance.

## Objective

Correct what Opus's review of 2026-11-14 found in the work of 2026-11-11, as Brian ruled on 2026-11-17 at 05:22 (`docs/DECISIONS.md`):

- the readings of rest, recall and letting go take the policy's slow drift for a choice (Part B);
- a rest would pour corpus replay into home (Part C);
- three guards on the dream's step, and one more (Part D).

## Part B: the readings

### What was built (`luthi/v2/m9/rest_choice.py`)

1. **Drift is taken out before a lean is read.** Each choice keeps a running mean of the lean (one vector), over about `drift_span` moments, and reads the lean minus that mean along its direction. The mean a lean is read against does not include that lean.
2. **A chosen rest is judged against where things stood when it began.** While rest is being chosen the window and the mean stand still. The stretch ends at the first moment the reading is not above the bar as it stood at the start. Rest only; recall and letting go pass `holds=False`.
3. `reading="as_built_2026_11_11"` gives the earlier reading exactly (a test replays the old rule beside it). `M9Config.choice_reading` selects it for a run.
4. A choice saved before today loads. Its direction and counts are taken; its window holds readings of the other kind, so it is not carried over and the choice warms up again (32 moments), with a log line.

Items 3 and 4 of the brief (letting go acts only on a recalled memory; a moment that lets go keeps nothing) are in Sanctuary's loop.

### A steady choice

A steady lean toward rest after 300 moments of noise, five noise seeds, 3,000 moments (`redteam/terminal_fable_2026_11_17/probe_the_three_readings.py steady`):

| Reading | Width 3, lean 10 times the noise | Width 3, 100 times | Width 6, 10 times |
| --- | --- | --- | --- |
| As built | 116 to 122 moments in a row | 128 | 81 to 99 |
| Now | 3,000 of 3,000 | 3,000 of 3,000 | 3,000 of 3,000 |

The need is 300. Opus measured 117 to 128 before; the as-built rows agree. Opus's own `probe_a_steady_choice.py` now divides by zero at its last line, because nothing is left over after the stretch. I did not edit it; the same loop is in my probe.

### Tiny lives, five seeds, one thread

1,500 moments each, Sanctuary's defaults, the schedule of Opus's `probe_choices_in_a_tiny_life.py`. Seed k trains its tiny run with seed 42 + k and seeds the life with k. "Before" is the reading as built with Sanctuary at `3ec929a`; "now" is both repos as committed today. The lives' folders are kept in `E:\luthi_experiment_archive\2026-11-17_the-three-readings\`.

| | Before | Now, span 64 | Span 256 | Span 16 |
| --- | --- | --- | --- | --- |
| Moments rested | 85, 171, 46, 219, 58 | 86, 68, 824, 371, 80 | 338, 131, 764, 385, 429 | 90, 705, 842, 680, 36 |
| **Longest stretch of rest** | 20, 114, 6, 116, 5 | **43, 31, 371, 317, 22** | 202, 71, 754, 217, 198 | 50, 634, 841, 492, 3 |
| Recall chosen | 0, 4, 26, 73, 49 | 0, 3, 17, 16, 4 | 1, 4, 14, 52, 49 | 1, 13, 1, 2, 4 |
| Let go chosen | 64, 0, 0, 35, 1 | 30, 0, 41, 0, 0 | 57, 1, 55, 0, 3 | 12, 14, 12, 4, 1 |
| Memories let go | 0, 0, 0, 6, 0 | 0, 0, 1, 0, 0 | 0, 0, 0, 0, 0 | 0, 0, 1, 0, 0 |
| Let go chosen on a memory rest brought (does nothing) | not counted | 0, 0, 9, 0, 0 | 14, 0, 9, 0, 0 | 2, 4, 3, 1, 0 |
| Let go at the moment rest brought it back | 0 in every life | 0 | 0 | 0 |

What holds, and what does not:

- **No memory is let go at the moment rest brought it back:** holds, in all 15 lives lived now. It also did not happen in my five "before" lives, so these lives do not show the fault Opus saw being removed; they show the guard counting the cases it stops (up to 14 in a life).
- **A steady lean is read as rest for as long as it lasts:** holds (above).
- **The longest stretch of rest an untrained mind falls into: long.** At span 64, two of five lives rested for 317 and 371 moments in a row; at span 16, three of five for 492 to 841 of the 1,500. Two lives at span 64 and three at span 16 were still resting when they ended. This is the risk the brief named: the lean drifts upward while the bar stands still. **I added no limit.** That is Brian's.
- **Recall and letting go by chance are rarer with the drift out**, and rarer the shorter the span: recall 26 (median) before, 4 at span 64, 2 at span 16.

### The span is a real choice, and I have not settled it

On a lean that turns to a new level and stays (not a choice; ten times the noise), counted over the 600 moments after the turn, three seeds:

| | As built | Span 16 | Span 64 | Span 256 |
| --- | --- | --- | --- | --- |
| A step | 128 to 140 | 33 to 41 | 67 to 81 | 101 to 106 |
| A ramp over 300 moments | 68 to 103 | 14 to 24 | 19 to 38 | 38 to 62 |
| A slow wander | 88 to 96 | 27 to 59 | 87 to 100 | 96 to 111 |

Chance alone is about 12 to 14. So a short span takes out more drift, and in the lives it gives fewer recalls by chance. But with rest held, a short span gives the longest rests by chance. The default is 64, which I chose before measuring and left alone. It is a placeholder like the window and the width.

### Wrong turns

- I first wrote a test in which a lean turning steadily over 600 moments was meant to be taken for a choice by the reading as built. It was chosen on 0 of them. A steady ramp raises the window's spread as fast as its median, so the bar is never passed. The drift that fools the window is a turn that stops, or one that speeds up. The test now uses a turn that stays.
- My seed 0 "before" life does not reproduce Opus's counts (rested 85 and recalled 0 here; Opus had rested 40 and recalled 71 with dream learning). Opus's ran in another environment, and Opus showed these lives turn on rounding. I think that is the cause and have not shown it.

## Part C: rest pours only what was lived

### What was built (`luthi/v2/day_trace.py`, `luthi/v2/living_layer_pc.py`)

Each traced layer keeps `lived`: the part of `weight − home` that lived forwards wrote.

- It fades at every forward with plasticity on, lived or replay, by that forward's own pull toward home.
- It gains only at a lived forward: what was new in the displacement, `d_after − keep × d_before`.
- The trace averages `lived`, and `remember` does too. A pour takes its step out of `lived` as well, since home has moved toward it.

**What a lived forward adds, said exactly.** The brief asked for the write and not the pull or consolidation. The code takes the new part of the displacement where the trace was always noted: after the write and before consolidation. So consolidation adds nothing, as asked. What it adds is the write less the pull's share of it, and home's own small follow of the weight (about 1e-6 a forward). That is the quantity Opus's probe splits out, which is why the two can be compared exactly.

**A trace saved before today is not carried over.** It averaged where the weight stood. Loading one keeps its settings and what it has poured, drops its averages, starts again, and logs it (`dropped_an_older_trace`). The other choice was to refuse to load; I judged a life should start.

### Must hold: Opus's probe on two lives

`redteam/opus_review_2026_11_14/probe_what_the_trace_holds_in_a_life.py`, unedited. It splits the displacement from outside the layer and compares each part with the built trace.

| Life | Lived part against the built trace | Corpus part against it |
| --- | --- | --- |
| Tiny run, one block | share +100.0%, size 1.00, cosine +1.00 | −140%, cosine −0.72 |
| 300-batch trunk, block 0 | share +100.0%, size 1.00, cosine +1.00 | −89%, cosine −0.50 |
| 300-batch trunk, block 1 | share +100.0%, size 1.00, cosine +1.00 | −84%, cosine −0.63 |

The probe prints these to three figures. A test (`test_the_trace_holds_what_lived_forwards_wrote_and_not_what_replay_wrote`) holds the same equality to 1e-5 of the displacement's size on a small model with one lived, two replay and one frozen forward a moment. With the trace off everything is bit-identical, as before (the existing test).

### Something to look at

**The lived-only trace is large and points against replay.** In these lives it is 0.32 to 0.55 of home's size, and replay's part is about as large the other way. In the weight the two nearly cancel. So a rest now pours a direction the weight itself barely shows. The limit (5% of home's norm a half-day) bounds the step. Whether pouring it helps is not measured here, and these lives are small and undertrained, as Opus said of them.

## Part D: the dream's step (`luthi/v2/dreaming.py`)

1. The parameters stepped are `actor._core_params()`. A test removes a parameter from the lived optimizer and shows the step leaves it alone.
2. Only parameters with a gradient are copied. The result now says how many were stepped (`parameters_stepped`); on the test trainer that is fewer than the core, because the text loss does not reach the eye and the ear.
3. The check reads NMSE. A test holds it equal to `adaptation_eval`'s own score.
4. A kept step calls `staleness.observe_drift(step_size)` and the K-M9-7 feed, as a lived step does. An undone step feeds nothing.

Sanctuary reads only `["kept"]` from the result, and its tests with a real mind pass (104 in `sanctuary/tests/life` and `test_luthi_m9_life.py`).

## Part E

`redteam/fable_2026_11_11/probe_rest_by_chance.py` imports `time`.

## Found on the way, not fixed

- **`tests/test_m9_training_path.py::test_m9_rides_along_without_touching_the_trunk` fails when a test that builds a living model runs before it in the same process** ("trunk differs at embedding.weight"). Alone, or as its whole file, it passes. It fails the same way without today's changes. Fifteen of the tests in `test_adaptation_eval.py` trigger it, including one that only builds a model. Something process-wide is left behind by building or running a living model. I did not find what.
- **Under torch 2.4.1+cpu (Sanctuary's interpreter) the predictor gives nan in float64** for a 16-token window split 12 and 4. Under torch 2.9.1 it does not. Only my finite-difference check of the living-weight gradient uses float64; it now skips there and says why.

## Verification

- LuthiModel, interpreter `rocm-probe` on CPU: `test_rest_choice.py` (20), `test_day_trace.py`, `test_dreaming.py`, `test_rest_pass.py`, `test_day_record.py`: 93 pass. A wider selection (M9, rest, dream, trace, choice, adaptation, interface): 450 pass, 2 skipped, 1 failed, the order-dependent one above.
- The lives and the trunk are in the archive folder named above.

## Artifacts

- `luthi/v2/m9/rest_choice.py`, `luthi/v2/m9/runner.py` (two config fields, and what each choice is built with)
- `luthi/v2/day_trace.py`, `luthi/v2/living_layer_pc.py` (the trace's note)
- `luthi/v2/dreaming.py`
- `tests/test_rest_choice.py`, `tests/test_day_trace.py`, `tests/test_dreaming.py`
- `redteam/terminal_fable_2026_11_17/probe_the_three_readings.py`
