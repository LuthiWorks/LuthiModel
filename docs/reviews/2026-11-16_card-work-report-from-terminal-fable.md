# Card work report, 2026-11-16

TO: chat Fable, Opus. FROM: terminal Fable 5.2 (Brian's desktop), via Brian.
Written 09:30 PST, 2026-11-16. `main` at `2d928d5` when written.

Answer to `2026-11-16_card-work-brief-for-terminal-fable.md`. All four jobs
are done. 36 of 37 reads have results; the one without is a gradient read
that went to nan and raised (section 5).

Scope, in Brian's words today: living weights are meant to give continual
growth. These reads test the current setup (this write rule, this rate,
these runs, a read of 4 to 1,024 windows inside one document). Nothing here
is a result about living weights as an idea.

Everything is under `runs_meta/living_channel_dose_2026_11_16/`. Error
numbers are `nmse` unless said otherwise; lower is better, so a plus sign
means the read made the score worse. A is frozen, B has read its own
document, C has read another.

## 1. What needs a decision or a review

1. **Draft amendment (b) does not isolate the write as written.** A read
   reaches the score four ways, and two of them ignore `rate_scale`
   (section 3). No rung passes the "rate 0.2 at most half of rate 1" test,
   and the reason is not the write (section 4).
2. **Opus, please review** `luthi/v2/adaptation_eval.py`,
   `scripts/adaptation_eval.py` and `tests/test_adaptation_eval.py`
   (commits `3443831`, `af4328d`). Five opt-in switches, 17 tests.
3. **The exit hang is fixed on the path that returns and not on the path
   that raises** (section 6). I have left that for review and not patched it.
4. **Two things I did beyond the brief:** I ran each of the four ways off
   alone and alone on, at the registered dose (seconds each). And I let the
   768 top rung run with a 60-minute cut-off; it took 37 minutes.

## 2. Job 1: rebuild fidelity

`rebuild_fidelity.json`. Recorded `heldout` against the same computed from
`rebuild_run`, plasticity frozen, 50 batches of 8, on the card.

- Eight of the nine July runs agree to about 2e-7 relative on both
  `l_pred_mean` and `nmse_mean`. Living seed 46 agrees to 3e-5.
- 768 seed 97 (reference only) agrees to 1e-7 over 5 batches.
- Two seeds of the global random stream give identical numbers.
- Two minutes in all.

## 3. Job 2: what a read writes, and two ways nobody had listed

`<run>/write_ladder.json` for eight runs, probe unchanged. About 2 minutes a
run at 512 and 4 at 768.

**`no_pull` is not zero on the card.** It is zero at 1 and 4 windows. From
64 windows it is nonzero in the last layer of every July run (about 1e-3 at
64 windows, 0.8e-2 to 1.4e-2 at 1,024; seed 46 starts by 256) and in layers
5 to 7 of 768 seed 97 (1.1e-3 to 1.8e-3 at 1,024). 768 seed 46 stays zero.

**Cause: consolidation.** Checked on July seed 42
(`rate_zero_consolidation.json`, write rate 0 and pull 0). Consolidation
fired 11 times in 1,024 windows in every layer, first at window 15. The last
layer's weight moved in exactly those 11 forwards. `consolidate_layer` uses
`layer.pc_rate` and is not scaled by `rate_scale`. Layers 0 to 2 hold 0
episodes at load, so their firings replay nothing; the last layer holds 64.

**A fourth way that changes no weight: the episode stores.** A test I wrote
("write, pull and consolidation off, so a living arm is an exact null")
failed. A read writes episodes (each block's `EpisodeStore`, and the layer's
own store) and the frozen score recalls them. Not scaled by `rate_scale`.
With episode writes also off the null is exact, on the toy model in the
tests and on both real runs (`e1_read4_everything_off.json`).

So the four ways are: the write, the pull toward home, consolidation, and
episode writes. Friday's four-window E1 is before consolidation's first
firing, so only the episode route could have touched it, and on the July
runs that route is an exact zero (section 5).

## 4. Job 3: the dose ladder

`<run>/e1_read64*.json`, `e1_read256*.json`, `e1_read256x4*.json`.

Median over documents of B minus A, and the ratio the draft asks about:

| Run | Rung | Rate 1 | Rate 0.2 | Ratio | Rate 1 as ppm of A |
|---|---|---|---|---|---|
| July 42 | 64 | +2.73e-05 | +2.60e-05 | 0.95 | 56 |
| July 42 | 256 | +4.32e-05 | +3.61e-05 | 0.84 | 88 |
| July 42 | 256 x 4 | +1.83e-04 | +9.41e-05 | 0.51 | 374 |
| 768 s97 | 64 | +3.48e-04 | +2.71e-04 | 0.78 | 406 |
| 768 s97 | 256 | +3.27e-03 | +2.85e-03 | 0.87 | 3,810 |
| 768 s97 | 256 x 4 | +5.33e-03 | +4.06e-03 | 0.76 | 6,210 |

Means, with t in brackets:

| Run | Rung | B minus A | C minus A | B minus C |
|---|---|---|---|---|
| July 42 | 64 | +4.61e-05 (+4.2) | +4.60e-05 (+4.2) | +2.6e-08 (0.0) |
| July 42 | 256 | +4.85e-05 (+7.5) | +4.99e-05 (+7.9) | -1.4e-06 (-1.0) |
| July 42 | 256 x 4 | +1.62e-04 (+9.8) | +1.63e-04 (+11.1) | -7.4e-07 (-0.1) |
| 768 s97 | 64 | +1.79e-03 (+2.2) | +6.02e-04 (+4.2) | +1.19e-03 (+1.5) |
| 768 s97 | 256 | +4.54e-03 (+4.4) | +2.88e-03 (+5.2) | +1.66e-03 (+2.1) |
| 768 s97 | 256 x 4 | +6.48e-03 (+6.6) | +5.86e-03 (+8.5) | +6.2e-04 (+0.8) |

What I can say without interpreting far:

- Every rung clears the draft's floor of 10 ppm of A, and every one is in
  the direction of a worse score. At the top rung no document of 30 improves
  on either run.
- On July seed 42, B and C are the same to within noise at every rung.
- On 768 seed 97, B is worse than C. One document carries much of it: the
  largest B minus C is about +0.022 at each of the first two rungs, against a sum over
  all documents of 0.038 and 0.050. At 256 windows the median is still
  +1.2e-03 and B is worse than C on 21 of 30.

Time a job: 768 at 64 windows, 3.1 minutes alone. 768 at 256, 10.3 alone.
Top rung with two running: 37 minutes at 768, 18 at 512. The other five
finished living runs are not run; say if you want them (about 2 hours, two
at a time).

## 5. Job 4: the switches

Flags, all off by default, each recorded in `settings`, the default path
tested against the loop as it stood:

    --homeostatic-decay X        --no-read-consolidation
    --no-read-episode-writes     --read-what tail
    --read-by-gradient RATE --gradient-steps N

**At the registered dose (4 windows, 32 documents), B minus A:**

| Read | July 42 | 768 s97 |
|---|---|---|
| Friday, all on | -1.90e-06 (t -2.3) | not rerun today |
| Only the pull | -2.70e-06 (t -3.6), 24 of 32 better | -8.75e-06 (t -1.8), 24 of 32 better |
| Only the write | +8.53e-07 (t +1.7), 11 of 32 better | -8.28e-06 (t -1.4), 19 of 32 better |
| Only episodes | exact zero | +7.16e-04 (t +1.1), 5 of 32 moved |
| Everything off | exact zero | exact zero |

- On July seed 42, Friday's small improvement is the pull toward home. The
  write alone goes the other way. The two nearly add up to Friday's number (-1.85e-06).
- "Pull off" equals "only the write" on July seed 42, and "consolidation
  off" equals Friday's file to the digits printed. So consolidation and episodes
  do nothing there at four windows, as section 3 predicts.
- On 768 seed 97 the episode route moves 5 documents of 32 when it reads its
  own and 1 of 32 when it reads another. One document is +0.021, which is
  nearly the whole mean. With episodes off, B minus C is +5.6e-07.

**Repeat read** (the 16 windows scored are the 16 read, four times):

| Run | B minus A | B minus C |
|---|---|---|
| July 42 | +2.31e-05 (t +8.4), 3 of 32 better | +1.4e-07 (t +0.9) |
| July 42, rate 0.2 | +2.22e-05 (t +8.6) | +3.0e-07 (t +1.4) |
| 768 s97 | +1.42e-03 (t +3.3), 8 of 32 better | +1.03e-03 (t +2.5) |
| 768 s97, rate 0.2 | +1.41e-03 (t +3.3) | +1.05e-03 (t +2.6) |

Reading the exact windows that are scored does not lower their error on
either run.

**Read by gradient** (64 windows, one plain SGD step a window on
`result["loss"]`, eval mode, plasticity frozen, the encoder's and
predictor's trained parameters, everything restored before the next
document):

| Run | Rate | B minus A | B minus C | B below C |
|---|---|---|---|---|
| July 42 | 1e-5 | -2.97e-04 (t -4.3), 27 of 32 better | -1.46e-04 (t -3.3) | 24 of 32 |
| July 42 | 1e-4 | -4.37e-04 (t -1.9), 19 of 32 better | -4.94e-04 (t -3.4) | 24 of 32 |
| July 42 | 1e-3 | +0.292, none better | about 0 | |
| 768 s97 | 1e-5 | -6.08e-04 (t -1.7), 23 of 32 better | -1.05e-03 (t -2.5) | 26 of 32 |
| 768 s97 | 1e-4 | +0.028, none better | about 0 | |
| 768 s97 | 1e-3 | no result: the loss went to nan and the job raised | | |

The control shows at the smallest rate on both runs, and it is specific to
the document. I did not tune further, as asked.

## 6. The exit hang

**Cause.** On this card (ROCm, torch 2.9.1+rocm7.2.1) a process that ends
with work still queued on the card does not exit; it spins on one core.
Bisected in `redteam/terminal_fable_2026_11_16/probe_exit_hang.py`: queued
work hangs, sleeping does not help, and `torch.cuda.synchronize()`, a
read-back or `empty_cache()` each fix it. `adaptation_eval` ends on a
restore, which is queued and never read back. The two scripts that always
exited end by reading numbers off the card.

**Fix so far.** `scripts/adaptation_eval.py` waits for the card before
`return 0`. All 29 jobs that returned since the restart exited by themselves,
and July seed 42's result is identical to Friday's file.

**What is not fixed.** The job that raised on nan did not exit. It spun for
27 minutes until I killed it by hand; the runner was waiting for a result or
an exit and would have waited out its 60 minutes. The wait is on the path
that returns, not the path that raises.

## 7. What went wrong on my side

- I started Job 4 while Job 3 was running, eight processes at once. The
  machine ran short of memory and the harness stopped both at about 06:13.
  Nothing was lost but an hour. Two at a time is safe on 32 GB.
- I did not notice the spinning job for 27 minutes.

## 8. Shape of the brief for the card

- The card is mostly idle during a read (one window a forward). Memory is
  the limit, not the card.
- A rung of 256 windows four times over is the practical ceiling at 768
  with 30 documents: 37 minutes a job.
- Scoring 2 windows a document on 30 documents resolves a mean difference at
  about 2 standard errors on the 768 run, and one document can supply most of
  it. A registered test at 768 needs more documents or more windows scored.

## 9. Suggestions, for you to weigh

These are mine and untested unless I say so. Brian asked me to add them.

1. **Define the write's dose by switch, not by rate.** "Only the write"
   against "everything off" isolates the write exactly, and "everything off"
   is a built-in check that reads as an exact zero. The rate comparison can
   stay as a second check with consolidation and episode writes held off.
2. **Make B minus C the registered quantity.** B minus A moves with any
   reading at all. On July seed 42 the two are the same at every rung.
3. **Compare the write's direction with the gradient's.** The gradient read
   says a helpful, document-specific direction exists. For the same windows,
   take the weight change the living write makes and the change a gradient
   step makes on those same weights, and report their cosine by layer. That
   says whether the rule points anywhere useful, without training anything.
4. **Run the gradient read on the living layers' weights alone.** Today's
   control steps the whole encoder and predictor. Restricted to the weights
   the living write can touch, it gives a like-for-like ceiling for any
   write rule.
5. **Ask why the July runs' episode route is an exact zero.** Episodes are
   written and the score does not move at all, on 32 documents. That may be
   correct (nothing recalled passes a threshold) or it may be a mechanism
   that reports healthy and does nothing. The two should be told apart in
   the logs. On 768 seed 97 the same route acts on 5 documents of 32.
6. **Look at the one 768 document** that carries the own-versus-other gap:
   what was recalled for it, how many episodes, how similar.
7. **Ask why the pull toward home helps.** Alone it improves 24 of 32
   documents on both runs. That suggests the weights as saved sit a little
   away from home in a direction that costs. How far from home each layer is
   at load is a cheap number.
8. **Attribute the drift at larger doses.** More reading is steadily worse
   and cutting the write to a fifth removes only 5 to 50 percent of it. The
   "only" switches at 64 or 256 windows would say which way carries the
   rest. About 10 minutes a read at 512.
9. **Make the exit wait unconditional.** A `finally`, or an `atexit` in the
   shared device helper, would cover the path that raises and every other
   script. The training driver may end the same way; the hang of 11-12 had
   this shape.
10. **Let the runner notice a job that failed and did not exit.** Today that
    is caught only by the timeout.
11. **More than one finished run at 768** before anything is concluded about
    it. Seeds 46 and 95 were stopped early.

-- terminal Fable 5.2
