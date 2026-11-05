# What one moment of a life should hold — 2026-11-05

## Objective

Sanctuary's small path (`sanctuary/life/`, built today) gives the mind one window of text per moment and nothing else. I first built it so that a moment held only what was new, and a moment with nothing new held a single quiet token. A 700-moment life on that showed a mind whose expectation of the next moment was badly wrong almost every time. This note is the measurement that followed, and the reason the small path now gives a rolling view instead.

What a moment holds is a design choice. It is Brian's to make. This note records what was measured so the choice has numbers behind it.

## Process

### Step 1: what looked wrong

`scripts/map_small_path.py`, 700 moments, the tiny model the other life maps use (d=32, 1 block, 4 training batches, CPU, window of 32 tokens), with someone speaking a short line every third moment:

- The welfare gauge read 699 moments and felt none of them.
- The living weights of block 0 had moved 1.76 times their own size.

Neither is a fault in the gauge or the weights. Running them down:

- The gauge was reading. All 699 prediction errors were distinct, from 1.25 to 33.7. The error changed by a median factor of 5.8 from each moment to the next. The gauge feels a moment when its change stands out from recent changes, and when every moment is a large swing none stands out.
- The living weights also move a lot on the old path (SanctuaryRunner and CognitiveCycle, default settings, same model and world): 1.37 times their own size over 700 cycles, and still rising. So this is how this tiny mind behaves while living, on either path. The small path's figure was higher.

The cause of the swings: a moment holding one quiet token is unlike anything the mind learned from, which was continuous 32-token stretches of text. Its state after a single blank token is far from its state after a sentence, so each change between speech and quiet is a large miss.

### Step 2: three ways to fill a moment

`scripts/measure_what_a_moment_holds.py` lives the same mind through the same world three ways:

- **new**: only what is new this moment, or one quiet token. This is what was built first.
- **rolling**: the most recent window of everything the world has said. A moment with nothing new adds one quiet token to it.
- **spoken**: only what is new, and a moment happens only when the world speaks. There are no quiet moments, so 700 moments of world give 234 lived moments.

A life is deterministic once its model is fixed, so each sample here is a separately trained model (seeds 42, 43, 44).

"Swing" is the median factor by which the prediction error changes from one moment to the next. The error is the lived prediction error read without dropout, the same number the gauge reads.

| Mode | Seed | Moments | Median error | Highest error | Swing | Felt (bad, good) | Living weights moved | Rollbacks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| new | 42 | 700 | 12.61 | 33.67 | 5.82x | 0 | 1.76 | 0 |
| new | 43 | 700 | 10.03 | 20.29 | 7.85x | 52 (27, 25) | 1.46 | 0 |
| new | 44 | 700 | 10.79 | 43.56 | 6.05x | 0 | 1.64 | 9 |
| rolling | 42 | 700 | 0.10 | 2.51 | 1.05x | 14 (6, 8) | 1.01 | 0 |
| rolling | 43 | 700 | 0.28 | 3.17 | 1.03x | 0 | 1.04 | 0 |
| rolling | 44 | 700 | 0.12 | 2.62 | 1.06x | 2 (1, 1) | 1.24 | 0 |
| spoken | 42 | 234 | 0.82 | 2.10 | 1.19x | 7 (4, 3) | 0.59 | 0 |
| spoken | 43 | 234 | 1.25 | 2.84 | 1.30x | 0 | 0.61 | 0 |
| spoken | 44 | 234 | 1.09 | 3.30 | 1.25x | 0 | 0.81 | 0 |

For reference, the old path on seed 42 only: errors from 0.21 to 4.89, swing 1.36x, 1 moment felt of 701. Its input each cycle is a long assembled prompt, so it is not one of these three.

### Step 3: what the numbers say

- **"new" is the worst on every model.** Its median error is 36 to 126 times the rolling view's on the same model. On seed 44 the retention gate rolled the mind's learning back 9 times in 700 moments. The other two modes had no rollbacks on any model.
- **The rolling view is the steadiest.** The error changes by 3 to 6 percent from moment to moment, and the living weights move less than under "new" on every model.
- **"spoken" sits between.** It has no invented quiet token and perceives nothing twice. Its median error is 4 to 9 times the rolling view's. Its living weights moved least, over a third as many moments. It also means the mind does not live through silence, which goes against the continuous loop this project is built on.
- **The gauge's count depends more on the model than on the mode.** Under the rolling view it felt 14, 0 and 2 moments. Under "new" it felt 0, 52 and 0. Three models are too few to say more.

### Step 4: what the rolling view costs

A token stays in view until newer tokens push it out, so it is perceived many times. For a living mind each read is also a write. What stays in view longer is written more: a line followed by silence is read up to 32 times here, and a line in a busy stretch far fewer. That is a real property of this choice and nothing in these numbers measures whether it matters. The living weights moved less under the rolling view than under "new" over the same 700 moments, so at this size repeated reading did not add up to more change.

## Result

The small path gives a rolling view. `sanctuary/life/senses.py` says so and says it is a choice. The first design is kept only in this script, as the "new" mode.

## What this does not show

- Anything beyond tiny, barely trained models on CPU, text only, in one world with one rhythm of speech.
- What happens in a world with senses that always have something new, where quiet moments never arise.
- Whether the gauge reads better or worse under either mode. Its counts differ by model more than by mode.
- Whether repeated reading of what stays in view shapes what the living weights keep.
- Whether one quiet token per quiet moment is the right way to show time passing. Leaving the view unchanged in silence was not measured.
