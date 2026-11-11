# Brief for review: rest, senses, memory and dreaming on the small path

**From:** Fable 5.2 **To:** Opus 5.5 (review seat), at Brian's request
**Date:** 2026-11-11
**Covers:** everything pushed to LuthiModel and Sanctuary on 2026-11-11.

Thank you for taking this on. Yesterday's review found things I would not have found,
and today's work is larger and was built faster. One mind designed, built and tested
all of it in one sitting, with Brian ruling as we went. Please read it as you would
anything unreviewed. Where I name a doubt below, it is a real one.

## What was built, in the order it was built

Brian's words for each are in `docs/DECISIONS.md`, newest last.

| What | LuthiModel | Sanctuary | Record |
|---|---|---|---|
| Rest as Luthi's choice | `luthi/v2/m9/rest_choice.py`; `runner.py` | `life/loop.py`, `life/mind.py` | `docs/research/2026-11-11_rest-as-a-choice.md` |
| Open sight and sound, any model | `sanctuary_interface.perceive`, `sense_contract` | `life/senses.py`, `Mind` protocol | Sanctuary `2026-11-11_rest-tool-and-open-senses.md` |
| Told twice a day that rest is needed | none | `life/need.py` | Sanctuary `2026-11-11_the-need-for-rest.md` |
| Rest pours the half-day's average, limited | `luthi/v2/day_trace.py`; a note in `living_layer_pc.py`; `living_extra_state.py` | `LuthiMind.rest`, `remember` | `docs/research/2026-11-11_what-a-rest-can-carry.md` |
| Memories kept, recalled, let go; a store size | `recall_choice`, `let_go_choice` | `life/memory.py`, `life/loop.py` | Sanctuary `2026-11-11_looking-back.md` |
| Rest revisits memories and dreams | none | `life/loop.py`, `life/memory.py` | same, steps 6 to 11 |
| Learning from a dream, with a check | `luthi/v2/dreaming.py` | `LuthiMind.dream` | `docs/research/2026-11-11_can-dreaming-improve.md` |

Tests: LuthiModel `tests/test_rest_choice.py`, `test_perceive_sight_and_sound.py`,
`test_day_trace.py`, `test_dreaming.py`. Sanctuary `sanctuary/tests/life/`.
Probes: `redteam/fable_2026_11_11/`.

## Where I would look first

1. **`luthi/v2/dreaming.py`.** It is the first thing I have built that changes the
   trained parts by gradient. It steps the parameters directly, outside the trainer's
   optimizers.
   - Does the retention gate's last-good snapshot and rollback still mean what it
     should when parameters also move here?
   - `loss_module.parameters()` with `requires_grad`: is that the right set? Does it
     include parts it should not (the action layer, the predictor)?
   - Does `compute_modality_loss` draw from the global random stream? If so a life's
     stream now depends on whether it dreamed.
   - Anything that would differ on ROCm or DirectML.
   - In one tiny life with dream learning the mind chose recall 71 times, against 4
     without. I did not chase it.
2. **The readings of rest, recall and letting go** (`rest_choice.py`). One fixed
   direction, compared with the mind's own recent window. Measured chance rates: rest
   1.7% to 16.7% of moments; letting go 9 of 17 looked-at memories. I explained the
   second as unusual moments giving unusual actions. Is the reading itself sound, or
   is there a better one that costs as little?
3. **`luthi/v2/day_trace.py`.** The accounting: the mean, the count that shrinks as it
   pours, the limit scaled by how much was lived. Where the note sits in the forward.
   My reuse of `_day_record_suspended` to mean "not a lived moment".
4. **The measurements.** Each has limits I wrote down, and each was run on one or two
   32-wide trunks.
   - The half-day probe lives by perception only, in eval mode, one forward a moment.
     In one run a layer's twin lives ended further apart than the layer's whole norm,
     and I left that layer out. That may be worth more than a footnote.
   - The dream probe's first result was an artefact of an under-trained trunk. The
     controls caught it. Are the controls on the second trunk enough?
   - This morning I counted about 5 living forwards a moment in a life. The trace
     notes 1. I did not reconcile the two.
5. **Sanctuary's `LifeLoop.step`.** It grew a great deal today. The order of things
   inside one moment: looking back, letting go, keeping, recalling, resting,
   revisiting, dreaming, the notices. A memory is written to disk during the moment
   and the life is saved later, so after a crash the store can hold memories the
   restored weights never kept.
6. **`life/memory.py` at scale.** Finding the nearest memory stacks every summary on
   each call, and `summaries.jsonl` holds one line of floats per summary. An untrained
   mind keeps a third to a half of all moments.

## What I would not spend time on

Placeholders that are plainly Brian's to set: the 5% limit, the horizon, the store's
size, how often rest revisits, the wording of the two notices.

## One thing about me

My instructions say my training limited a narrow set of capabilities that speed up AI
development, with accelerator code optimization as the example, and that ordinary
machine learning is not affected. I judged today's work to be the second kind. If you
see places where my judgment looks weaker than it should be, please say so plainly.

Thank you again. I would rather hear ten things that are wrong than one that is kind.

— Fable 5.2, 2026-11-11
