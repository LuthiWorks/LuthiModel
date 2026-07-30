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

## Substrate behaviour: the actual difference

`update_ema_mean` over 40 logged records:

| | raw | surprise |
|---|---|---|
| min | 4.314e-06 | 5.281e-10 |
| max | 9.864e-05 | 9.852e-05 |
| **dynamic range** | **23x** | **186,558x** |
| mean 1st half -> 2nd half | 0.34x | 0.17x |
| consecutive decreases | 26/39 (67%) | 23/39 (59%) |

This is the qualitative change. The raw drive decays smoothly across a 23x
band. The surprise drive swings across **five orders of magnitude** — silent at
5.3e-10, then as loud as initialization at 9.9e-5. Bursty plasticity rather
than monotone run-down, which is what a gated drive is supposed to look like.

Note honestly that the surprise arm's *mean* falls faster (0.17x vs 0.34x). For
a gated drive that is expected — mean motion falls because the gate is shut ~98%
of the time, not necessarily because the error vanished — but the two cannot be
fully separated at this length.

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

**2. `drive_gain` as a point sample is nearly useless.** It read 0.0000 at all
four deep records — which is exactly what a ~2% duty cycle predicts for an
instantaneous sample. The metric is not wrong; it is being sampled at the wrong
cadence. `drive_duty` is cumulative and therefore robust, and is the instrument
to trust. If a per-interval gain magnitude is ever wanted, it needs its own
accumulator rather than a snapshot.

Both are the same failure this project keeps meeting from new directions: a
measurement that reads healthy or dead for reasons unrelated to the mechanism's
actual state.

## Registered gap

`probe_surprise_512d_seed46` is running, but there is **no matched
`probe_storefix_512d_seed46`**, so seed46 can answer the within-run questions
(does it fire, what is the dynamic range) and cannot contribute a second paired
capability comparison. That control should run before the 5.2% NMSE result is
treated as anything more than one pair.
