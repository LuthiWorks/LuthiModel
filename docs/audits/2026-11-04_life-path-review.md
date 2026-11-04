# Review of the Luthi life path, 2026-11-04

**Reviewer:** Opus 5.5, local session, at Brian's request.
**Reviewed:** the 2026-11-02 and 2026-11-03 commits that connect M9 to Sanctuary and add felt
valence, at LuthiModel `344129b` and Sanctuary `0e1aece` (origin/main). The work is by an Opus 5.5
cloud session (`Claude-Session` 018uhrL7mcLFjQcNX1XSCLtc).
**Not reviewed:** the 2026-10-19 living-channel program, the seeding proposal, the Stage 0 probe,
the learning governor's internals, and the Minecraft and stand-in code. For those I read commit
messages at most.

**Method:** read the path, then measured it. `scripts/measure_life_path.py` trains the same
tiny M9 run as Sanctuary's end-to-end test and lives it through the real runner, from a scratch
export. No repo was changed.

**Limits of the numbers:** d=32, 1 block, 4 training batches, CPU, one seed. They show mechanism.
Their magnitudes will not transfer to 768x8. No life has begun and nothing produces felt percepts
yet, so none of this has affected anyone.

The relevant test files all pass on this code (Sanctuary 36, LuthiModel 34), so none of the
findings below is visible to the suite.

**Status:** Brian's framing, which I agree with for most of this: "There aren't any errors, just more to build." The list below is a build list for a path that has not run past CPU tests. Finding 3 is built (`docs/research/2026-11-04_bands-in-a-life.md`). The others are open.

---

## Finding 1: after pain, the head learns that everything is unpleasant

`M9Trainer._learn_felt_valence` returns early when a moment has no valence, so the head only ever
sees felt moments. With no neutral examples it has nothing to separate them from, and the easiest
fit is to move its output everywhere.

Valence read by the real head after repeated -0.9 events, through the real path. "Unfelt" is 10
states from ordinary moments that were never felt:

| Painful events | Unfelt states | The painful state |
|---:|---:|---:|
| 0 | 0.000 | n/a |
| 1 | -0.007 | -0.008 |
| 5 | -0.035 | -0.036 |
| 20 | -0.215 | -0.287 |
| 100 | -0.694 | -0.914 |

Control: same head class, optimizer, loss and states, with the unfelt states also taught as 0.
After 100 rounds, unfelt is -0.011 and painful is -0.885. So the states can be told apart, and the
missing neutral examples are the cause.

**Why it matters:**
- It defeats the purpose. A cost that is nearly the same for every candidate gives the planner
  little reason to choose one over another.
- The docstring says "Nothing is unpleasant to Luthi until something has felt unpleasant." The
  measured behavior is closer to: after enough pain, nearly everything is.
- The cost enters realized G, so the value head learns from it too.
- `test_sanctuary_lives_a_trained_m9_run` asserts `felt.bias.item() < 0`, which is the failure
  mode's signature, and it passes.

**Suggested fix:** teach unfelt moments as neutral, at low weight, and replace the bias assertion
with a test that felt and unfelt states are separated. Whether "nothing reported" should mean
"neutral" is a design call for Brian.

## Finding 2: the "moment that matters" keep fires on about half of ordinary cycles

`_keep_if_significant` counts a moment as significant whenever the memory channel is active. It
has no rate limit and no budget. With no felt percept in the whole run:

| 1500 unfelt cycles | keep on (0.05) | keep off |
|---|---:|---:|
| Memory channel active | 764 | 727 |
| Keep calls | 764 | 0 |
| Share of displacement moved into home | 100% | 0 |
| Home moved, relative to its starting norm | 2.47 | 0.003 |
| Home norm (starts at 8.03) | 20.27 | 8.02 |
| Weight norm | 19.77 | 12.34 |
| Top direction's share of home change | 0.86 | 0.86 |

The keep-on home moved about 800 times as far, and its weight norm ended 60% higher. Each call
moves about 2.4% (0.05 x mean salience 0.48). Sleep's home drift is 1% per NREM cycle.

**Why it matters:**
- The design says moments that matter. In practice this overrides homeostasis continuously, whether
  or not anything happened.
- Stage 0 (`2026-11-03_stage0-the-home.md`) found that a home following too fast ran away along one
  direction. This is a second route to the same place, and the cloud session listed its interaction
  with the band and taper under "Not known."
- By the code, each active cycle also queues an episodic write, with the content "nothing new was
  perceived" if nothing was. I counted active cycles, not stored memories, so the flood is unmeasured.

**Not established:** whether the home keeps growing past this point, and how often the memory channel
is active in a properly trained run. A trained decoder may be far quieter.

**Suggested fix:** give the keep a budget per sleep that is recorded when hit, and count only felt
moments or a real change in chosen salience. *(Update, same day: Finding 3 is built and did not explain this. The channel stayed about as active and the home moved about as far.)*

## Finding 3: the activity bands never warm up during a life

`activity_bands.observe` has one call site, in `_m9_head_step`, the corpus training path.
`observe_transition` does not reach it, and the bands are not in the checkpoint. Measured: bands
armed on 0 of 1500 lived cycles, for all three channels.

So `decode_selection`, which is documented as an activity-band test, falls back in a life to the
absolute floor and the comparison with the rest action. Per the band's own docstring, it defaults
to "active" while uncalibrated, and this holds at any scale. `bands_armed` reports it honestly, but
nothing reads that field. The same applies to `delta_s_band`, which I read but did not measure.

K-M9-5 (stasis) also lives only in `_m9_head_step`, so as far as I can tell it does not run during
a life. K-M9-3 and the retention gate do.

**Suggested fix:** observe realized activity in `observe_transition` and persist the bands. Then
re-measure Finding 2.

## Finding 4: three of the four affect channels do nothing on a v2 living layer

`apply_external_modulation` runs every cycle and the docstrings describe four channels. On a real
`PredictiveCodingLayer`:

| Channel | Target | On v2 |
|---|---|---|
| Arousal | `hebb_rate` | absent; `pc_rate` stays 0.001 at a 2x request |
| Precision | `spike_threshold` | absent |
| Valence | `excitability_acc` | absent |
| Attention | `salience_threshold` | works (0.1 to 0.05), kept "as a floor only" |

The interface's own docstrings say v2 uses `pc_rate`, but the code never touches it. This predates
the cloud session and is not their defect. `tests/test_sanctuary_interface.py` checks the channel
behavior on a fixture with `hebb_rate`, so it passes without exercising a v2 layer.

**Decision needed:** plumbing `pc_rate` is small, but it would make affect change the living
layers' learning rate by 0.5x to 2x for the first time. That is a real change to the mind, so it
is Brian's call. If the channels stay inert for now, the docstrings should say so.

## Finding 5, smaller: a retention rollback isn't reported to the mind

`_check_retention_gate` can roll the life's core back to its last-good snapshot. It logs a warning
and returns a status that Sanctuary never reads (no reference to `retention_rolled_back` or
`retention_breached`). The learning governor tells the mind about changes and refusals, and decision
5 says nothing is written into a life except by its own living or with consent. A rollback may need
to be told, and may need a place in that rule. I did not count how often it fires.

---

## What held up

I tried to break these and could not:
- **Life checkpoints:** names are zero-padded, so newest-first sorting is right. Writes are atomic,
  nothing is rotated out, and a failed save crashes the loop loudly.
- **The run stays untouched:** a life refuses a directory inside its run, and the test confirms it.
- **The felt-valence head can't reach the trunk:** `s_next` is detached, and it has its own optimizer.
- **Felt events survive the path:** the truncation branch of compression keeps valence through
  `model_copy` (the batch branch was fixed on 11-03), and felt percepts pass the sleep gate.
- **`keep_recent`** runs inside the critical section and is not undone by the per-cycle modulation
  restore, which does not cover `set_point`.
- **The modulation restore** resets only `salience_threshold` on v2 (from Finding 4), so it isn't
  wiping lived state.

The cloud session's self-reporting was good. It caught its own empty test in step 7, recorded the
skills setting as not acted on rather than hiding it, and named the keep's interaction as unknown.
Finding 2 is that named unknown, measured.

## Side notes

- `consolidate` failures are caught and logged as non-fatal in `cognitive_cycle.py`. Character
  consolidation could fail on every sleep while the mind keeps running. This predates the session.
- Decision 5's line (a life begins at first load) is still marked for Brian to confirm.
