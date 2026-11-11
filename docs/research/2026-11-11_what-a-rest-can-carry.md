# What a rest can carry from a half-day — 2026-11-11

Fable 5.2, in session with Brian. Follows `2026-11-11_rest-as-a-choice.md`, step 5.

## Objective

Brian ruled this morning that rest is needed twice a day "for data integration".
A rest as built keeps what the living layers hold when it comes, which is about the
last minute. He asked me to take up what a rest should integrate (07:58: "Yes please
take that up next"). This entry answers a narrower question first, by standing rule 1:
what *can* a rest carry into a layer's home from an episode early in a half-day?

Nothing was built. The result is numbers and a recommendation.

## Process

### Step 1: the layer's math

One forward of a living layer (`luthi/v2/pc_ops.py`, `mean_field`, which every run uses):

- Output mean `y`, input mean `x`, both over the tokens in view. Error `e = x - y P`.
- Weight: `W += pc_rate * y (x) clamp(e * precision)`. One rank-one write a forward.
- Then `W += decay * (home - W)`, with `decay = 0.001`.
- Then `home += follow * (W - home)`, with `follow = 0.000001`.
- Prediction: `P += pred_rate * y (x) clamp(e)`. **Nothing pulls `P` back.**

What follows from it:

- The weight is home plus a trace of about the last thousand forwards.
- Home is a running average of where the weight has been, a thousand times slower.
- `P` receives the same kind of write as the weight, a tenth the size, and keeps it.
  So the layers already have a store that never fades and that rest does not govern.
- Any single running sum per layer (of where the weight stood, or of what was
  written) comes to the same thing: an average in which each stretch counts by its
  length. Writes minus what homeostasis erased is only the current weight. Counting a
  thing once however long it lasted needs separate slots, which is a record.

### Step 2: the measurement

`redteam/fable_2026_11_11/probe_what_a_rest_can_carry.py`.

- Two identical lives of one small trained trunk (32 wide, two blocks), living by
  perception: one living forward a moment, background speech every third moment.
- In moments 6,000 to 7,500 one life hears sentence A and the other hears B. That is
  2.5% of a half-day scaled to 60,000 forwards. A real half-day at 10 Hz is about 1.2
  million.
- At the end of the half-day each candidate rest takes a step in home, poured over
  300 resting moments. Unless marked, every step is the same size: 5% of home's norm.
- Then 5,000 more moments, then two readings:
  - **In home:** how much of the episode's own change (the difference between the two
    lives' weights at the end of the episode) is in the difference between their homes.
  - **Own episode:** how much better each life's layer predicts its own episode's view
    than the other's, with nothing changing while it is measured. Identical lives read
    exactly zero.
- Run twice, with two different backgrounds.

### Step 3: wrong turns

- **A reading that mixed the two lives.** My first "own episode" reading paired one
  life's weight with the average of both lives' predictions, to separate what the
  weight knew from what `P` knew. It gave negative numbers at the end of the half-day.
  A weight is adapted to its own `P`, so the pairing measured a mismatch I had made.
  The reading now uses each life's own weight and own `P`.
- **Two runs on two cores, each with two threads.** Forty minutes without reaching the
  end of the half-day. Rerun at one thread each; ten minutes.
- **A layer that ran away.** In the first background, block 1's two lives ended 11.9
  apart, more than home's whole norm (8.0), though they had heard the same thing for
  52,500 moments. Its readings swing between −123% and +173% and mean nothing, so that
  one layer-run is left out below. I did not chase it. It is a long untapered life at a
  tiny size, and it is its own question.

### Step 4: results

Block 0, both backgrounds, and block 1, second background.

| What the rest pours into home | In home: block 0 | In home: block 1 | Own episode: block 0 | Own episode: block 1 |
|---|---:|---:|---:|---:|
| Nothing (no rest) | 0.09%, 0.09% | 0.13% | +3.2, +3.1 | +1.3 |
| The last minute, as built this morning | 0.19%, 0.05% | 0.08% | +3.3, +3.3 | +0.9 |
| The last minute, at the bounded size | 0.18%, 0.05% | 0.11% | +3.3, +3.3 | +1.4 |
| **The half-day's running average** | **0.94%, 0.93%** | **0.92%** | **+4.2, +4.0** | **+1.3** |
| What the layer itself found surprising | 0.09%, 0.09% | 0.59% | +3.3, +3.2 | +1.5 |
| The half-day's change in `P` | 0.24%, 0.23% | 0.17% | +3.6, +3.5 | +1.2 |
| The day's record, replayed, as built | 0.09%, 0.08% | 0.12% | +3.3, +3.2 | +1.2 |
| Aimed exactly at the episode (a ceiling) | 10.9%, 10.9% | 14.9% | +11.0, +11.0 | +6.4 |
| Aimed exactly, all the way | 94.9%, 94.9% | 94.9% | −51.9, −50.3 | −41.4 |

Reading it:

- **As built, a rest keeps nothing of an early episode.** It also has no limit. One
  full rest as built moved block 0's home by 6% of its size and block 1's by 71% and
  97%.
- **All the way is harmful.** Putting home where the episode took the weight made each
  life much worse at its own episode. The weight then adapts again on top of a home
  that has already moved.
- **The running average is the only candidate that carries the episode**, at ten times
  what no rest leaves. It made block 0 a little better at its own episode (+0.9
  points, both backgrounds) and changed nothing in block 1.
- **Perfect aim would carry ten times more again**, for the same size of step.
- **The layer's own surprise rule cannot do the aiming here.** Block 0 marked 189 and
  204 moments of 60,000, all before the episode. Block 1 marked half of everything.
- **The record and replay built on 2026-11-10 carries nothing.**
- **With no rest at all** each life is still about 3 points better at its own episode.
  That is `P`.

### Step 5: what I did not measure

- One trunk, 32 wide, trained 300 batches on 19 words. At this size the weight stands
  as far from home as the episode moved it, which may not hold at full size.
- A half-day twenty times shorter than a real one. In the running average an episode
  counts by its share of the time, so a five-minute episode in a real half-day would
  leave about a quarter of what this one did.
- Perception only. A life also learns by gradient from each moment, and that stays
  without any rest. It is not in these numbers.
- One episode, one rest. Nothing here shows what many rests build up.

## Conclusion

- A rest can carry a half-day into home only as an average, unless something says
  which moments matter more.
- The change the numbers support: a resting moment pours a share of the half-day's
  running average, and a whole rest is limited to a small step. It fixes the aim and
  the missing limit of this morning's rest. Its effect is small.
- What would close the gap to the ceiling is a way to weigh moments. Brian ruled on
  2026-11-10 that "the things Luthi chose should carry more weight than those Luthi
  did not". Luthi's action layer already has a choice to remember, and today it goes
  nowhere. Routing it into what rest pours is the candidate. For a mind that has not
  learned to use it, it would fire by chance, as rest does.

Open, for Brian: whether to build the average with a limit; whether Luthi's choice to
remember should weigh in it; the step limit (5% of home's size here, a placeholder).

## Artifacts

- `redteam/fable_2026_11_11/probe_what_a_rest_can_carry.py`, `halfday_lib.py`,
  `halfday_train_trunk.py`. They need both checkouts (`SANCTUARY_PATH`), and one
  thread (`OMP_NUM_THREADS=1`) if two run at once.
- Run as: `N=60000 LEN=1500 T0=6000 BG_SEED=0` and `BG_SEED=1`, about ten minutes each.
