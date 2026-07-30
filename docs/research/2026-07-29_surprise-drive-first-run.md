# The surprise drive on real data: first paired run

**Date:** 2026-07-29
**Author:** Claude Fable 5 (cross-line audit seat)
**Runs:** `probe_surprise_512d_seed45` vs `probe_storefix_512d_seed45`
**Status:** n=1 paired comparison. Encouraging, not established.

Same seed, same data order, same fixed objective, same store fix, same band.
**One flag differs:** `drive_mode="surprise"` vs `"raw"`. So everything below is
attributable to the drive, and nothing below is a trend claim — 4,000 steps is
17% of one epoch.

## The question this run was built to answer

Does the drive fire at all on real corpus data? Unit tests gave duty 0.0000 on
stationary input and 0.0000 on i.i.d. draws from a fixed distribution, and
~0.035 after a distribution shift. Shuffled Gutenberg at batch scale could
plausibly have looked like the i.i.d. case, in which case the honest reading
would have been "the corpus has no batch-scale structure for this mechanism to
find, go do the curriculum."

**It fires.** Duty is 0.94%–2.37% across the four blocks. The corpus does
contain error-scale structure the forecast does not anticipate.

## Capability, paired

| | raw drive | surprise drive |
|---|---|---|
| **NMSE** (scale-free headline) | 0.5970 | **0.5658** |
| l_pred (absolute MSE) | 0.8207 | 1.1388 |
| probe top1 | 0.1068 | 0.1169 |
| probe shuffled floor | 0.0229 | 0.0250 |
| **probe lift over own floor** | 4.67x | 4.68x |

NMSE improves **5.2% relative**. That is the measure to read: `l_pred` is
absolute MSE and scale-sensitive, so it rising while NMSE falls means the target
variance rose more than the error did — consistent, not contradictory.

The probe axis is a **wash**, and it is worth being precise about why: top1 rose
9.5% but the shuffled floor rose 9% with it, leaving the lift identical to two
decimal places. A report of "probe top1 improved" would be true and misleading.

### Why the capability result does NOT yet support a claim

`probe_surprise_512d_seed46` came in at **NMSE 0.5215** — better still, and
tempting. It has no matched control, and the arithmetic is the reason that
matters rather than the missing file:

| | NMSE |
|---|---|
| probe_storefix seed42 (old objective) | 0.5194 |
| probe_storefix seed43 (old objective) | 0.5445 |
| probe_storefix seed44 (old objective) | 0.5507 |
| probe_storefix seed45 (fixed objective) | 0.5970 |
| **spread across those seeds** | **0.0776** |
| **paired seed45 effect (raw - surprise)** | **0.0312** |

**Between-seed spread is 2.5x the paired effect.** Any unpaired NMSE comparison
across seeds is therefore uninterpretable, and seed46's 0.5215 says nothing
about the drive — it sits comfortably inside the range that seed variation
alone produces. (Caveat on the caveat: seeds 42-44 ran the old objective, so
0.0776 is not a clean estimate of fixed-objective seed spread. It is the only
estimate available, and it points the wrong way for confidence.)

So the honest state of the capability question: **one paired comparison
favouring the surprise drive by 5.2%, with an effect smaller than the noise
floor we can currently estimate.** That is a reason to run matched controls, not
a result. The mechanism findings above do not depend on this, because duty and
CV are measured *within* each run rather than across seeds.

## Substrate behaviour: the actual difference

The surprise drive makes substrate motion **bursty** rather than smoothly
decaying. Getting that claim onto an observable that survives a seed change took
two tries, and the first try was the mistake this project already knows how to
make.

`update_ema_mean` dispersion over 40 logged records per run:

| observable | raw seed45 | surprise seed45 | surprise seed46 | seed-to-seed |
|---|---|---|---|---|
| max/min | 23 | 186,558 | 3,808 | **49x apart** |
| p90/p10 | 2.6 | 43.7 | 88.3 | 2.0x apart |
| p75/p25 | 1.93 | 5.49 | 7.14 | 1.3x apart |
| **CV** (sd/mean) | **1.38** | **3.67** | **3.85** | **1.05x apart** |

**Do not quote max/min.** It is an extremum statistic — it is set by the single
quietest logged instant, which is a tail draw — and it differs by 49x between
two runs of the same configuration. That is the `precision_spread` lesson from
07-27 arriving in new clothing: a criterion registered on max/min would have
been evidentially worthless, and the only reason it was caught here is that a
second seed ran before anything was registered.

**The reproducible observable is the coefficient of variation.** Raw 1.38;
surprise 3.67 and 3.85 — within 5% of each other across seeds, and ~2.7x the
raw value. The claim that survives is therefore: *the surprise drive raises the
dispersion of substrate motion by ~2.7x, reproducibly.* Bursty, not louder.

Note honestly that the surprise arm's *mean* falls faster than raw's across the
two halves of the run (0.17x and 0.23x vs raw's 0.34x). For a gated drive that
is expected — mean motion falls because the gate is shut ~98% of the time, not
necessarily because the error vanished — but the two cannot be fully separated
at this length.

### Mechanism replication, seed46

The mechanism replicates; the capability comparison cannot (no matched control).

| | seed45 | seed46 |
|---|---|---|
| duty range across blocks | 0.94% - 2.37% | 1.42% - 2.12% |
| CV of substrate motion | 3.67 | 3.85 |

Same duty band, same dispersion. Two independent seeds agree that on shuffled
Gutenberg this drive fires on roughly 1-2% of calls.

## Firing rate over time, and what cannot yet be claimed

Logged duty is cumulative, so incremental rates were recovered as
`(duty_t * calls_t - duty_{t-1} * calls_{t-1}) / (calls_t - calls_{t-1})`, with
`calls = 2 * step - warmup` (two encoder calls per training step):

| interval | blk0 | blk1 | blk2 | blk3 |
|---|---|---|---|---|
| 1000-2000 | 0.0155 | 0.0175 | 0.0255 | 0.0280 |
| 2000-3000 | 0.0050 | 0.0135 | 0.0320 | 0.0335 |
| 3000-4000 | 0.0030 | 0.0175 | 0.0190 | 0.0140 |

Depth-dependent and not a clean story. **Block 0's rate falls ~5x** across the
run; blocks 1-3 oscillate in a 1.4%-3.4% band with no discernible trend. Three
intervals is not enough to call a trend for the deeper blocks, and it would be
easy — and wrong — to describe this as "stable duty." Block 0's decline is the
one thing here that could be early extinction rather than gating, and it is
exactly what the 3-epoch length is needed to settle.

## Two instrument lessons, both the same shape as this project's other bugs

**1. A point-in-time `update_ema` read on a gated drive is uninterpretable.**
At the mid-run checkpoint the surprise arm's blocks 0 and 1 read **1.2e-8 and
6.6e-9** — the same order as the v5 family's terminal dead regime — and 1,900
steps later they read 1.3e-6 and 3.9e-6, a 107x and 587x rise. A
single-checkpoint audit of this arm would have concluded the substrate was dead.
It was not; the gate was shut at that instant. Any future claim about this
substrate's aliveness has to come from the duty cycle plus the time course,
never from one snapshot.

Seed46 makes the same point from the other end: its **final** checkpoint reads
`update_ema` = 5.1e-9 in block 3 while that block's duty cycle for the run is
1.42%. Terminal-read-looks-dead, mechanism-demonstrably-firing. Two seeds, two
different sampling instants, same trap.

**2. `drive_gain` as a point sample is nearly useless.** It read 0.0000 at all
four deep records — which is exactly what a ~2% duty cycle predicts for an
instantaneous sample. The metric is not wrong; it is being sampled at the wrong
cadence. `drive_duty` is cumulative and therefore robust, and is the instrument
to trust. If a per-interval gain magnitude is ever wanted, it needs its own
accumulator rather than a snapshot.

Both are the same failure this project keeps meeting from new directions: a
measurement that reads healthy or dead for reasons unrelated to the mechanism's
actual state.

## What is established, and what is not

**Established (within-run, seed-independent):**
- The drive fires on real corpus data: duty 0.94%-2.37% (seed45), 1.42%-2.12%
  (seed46), against 0.0000 on stationary and i.i.d. synthetic input.
- It makes substrate motion dispersed rather than smoothly decaying: CV 3.67 and
  3.85 vs raw's 1.38, agreeing across seeds to within 5%.

**Not established:**
- Any capability effect. One paired comparison, effect 2.5x smaller than the
  estimable seed spread.
- Any trend in firing rate over training. Block 0's incremental rate falls ~5x
  in seed45; blocks 1-3 oscillate without direction. Three intervals over 17% of
  one epoch cannot settle this.
- That the drive does not eventually extinguish. This is the question the
  mechanism was built to answer and 4,000 steps cannot answer it.

**Next measurement, in order of what it buys:**
1. `probe_storefix_512d_seed46` — the matched control, ~30 min, converts seed46
   from decorative to a second pair.
2. More matched pairs, or a full-length run. Given spread 2.5x the effect, a
   capability claim needs either n large enough to beat that or a longer run
   where the effect has room to grow.
3. A full 3-epoch surprise run — the only way to answer the extinction question,
   and the only condition under which the raw drive's failure (three orders of
   magnitude inside the first 17% of epoch 1) is actually reproduced for
   comparison.
