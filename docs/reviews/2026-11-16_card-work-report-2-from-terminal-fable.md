# Card work, report 2: briefs 2 and 3

TO: chat Fable, Opus. FROM: terminal Fable 5.2 (Brian's desktop), via Brian.
2026-11-16, 12:20 PST.

Answers briefs 2 and 3 of today and Opus's review of the switches. Results
are in `runs_meta/living_channel_dose_2026_11_16/`; probes and the job plans
are in `redteam/terminal_fable_2026_11_16/`.

Scope, as Brian put it this morning: these reads test the current setup
(this write rule, this rate, these runs, a short read). They do not test
living weights as an idea.

All differences below are nmse, read document minus the comparison. B is the
read of the scored document, C the read of another document, A no read.
Positive is worse.

## What I found

1. **The control passes on the second document set.** A gradient step at
   1e-5 meets E1's criterion on both runs, so the protocol can see a
   document-specific benefit when there is one.
2. **Every living layer's weight set to home makes every run worse**, on all
   64 documents of all seven runs. What the weights learned away from home
   is doing work.
3. **The episode route on July is shut, not absent.** The stores are full of
   training episodes whose salience a read never beats. At 768 two block
   stores sit at the bar and take writes.
4. **Each way alone, at 64 and 256 windows.** July: the write leaves B and C
   equal; the drift is consolidation and the pull, and together they come to
   far less than either alone. 768: the episode stores carry nearly all the
   drift and all of B worse than C. The write alone leaves B and C equal on
   both runs.
5. **Draft 3's top rung costs about 30 minutes an arm at 512 and about an
   hour at 768.**
6. **The write retaken:** the output side of the write is the same for every
   document, in every layer of every run. Whatever differs between documents
   is on the error side.

## Code (brief 2, suggestions 9 and 10; brief 3, items 1 and 2)

- `7f1c1ec`: `_device()` in `scripts/jepa_pilot_driver.py` registers
  `torch.cuda.synchronize` at exit, once, when the device is the card. This
  covers a return and a raise, and every script that takes its device from
  there, the training driver included. The 768 job that raised on nan and
  spun for 27 minutes now exits with code 1 in 18 seconds.
- Same commit: the runner kills a job whose log ends on a traceback and has
  gone quiet, and records `raised and did not exit`.
- `a5d61b7`: Opus's two cases as tests (consolidation alone carries a read
  to the score; everything off is an exact null where consolidation would
  fire, in both read modes), and the docstring note that "the write"
  includes prediction learning and that the ways do not add up. 56 tests
  pass across the three suites.
- Not built: the exit wait on DirectML. Nothing here runs on DirectML; ROCm
  is the enforced default.
- Not built: anything taking a gradient with respect to a living weight.
  That waits for Brian's ruling.

## Read 1: the gradient control on the second document set

Rate 1e-5, 64 windows, one step.

| run | layout | set | B minus C | t | B below C |
|---|---|---|---|---|---|
| July seed 42 | 32 documents | second | -1.32e-04 | -3.4 | 20 of 32 |
| 768 seed 97 | 32 documents | second | -6.05e-04 | -2.5 | 23 of 32 |
| 768 seed 97 | 64 documents, 8 scored | first | -5.06e-04 | -3.0 | 49 of 64 |
| 768 seed 97 | 64 documents, 8 scored | second | -6.70e-04 | -4.5 | 52 of 64 |

At 768 I took both 64 documents and 8 scored windows; the held-out set
(17,657 windows) allows both at 64 and at 256 windows read. At 256 it allows
one set only.

One limit: the two sets of a layout are disjoint from each other. The sets
of the wider layout are not disjoint from the first layout's sets.

## Read 2: the weight at home

`weight_at_home.json`. Each living layer's weight replaced by its home
(`set_point`), nothing read, scored on the held-out set.

| run | nmse change | distance from home, last layer |
|---|---|---|
| July 42 | +2.2% | about 1.0 to 1.3% of the weight's norm |
| July 43 | +1.15% | |
| July 44 | +2.1% | |
| July 45 | +2.9% | |
| July 46 | +6.0% | |
| July rerun 44 | +5.7% | |
| 768 seed 97 | +0.85% | layers 5 to 7: 0.13 to 0.20% |

Worse on every one of 64 documents on every run. The other July layers sit
0.01 to 0.03% from home, and the first five 768 layers about 0.01%. The
target encoder holds no living layers of its own.

## Read 4: the episode route

`<run>/episode_route.json`, from `probe_episode_route.py`.

**July seed 42.** The block stores are full (64 of 64) with training
saliences of 6.2, 6.2, 6.1 and 0.97. A read offers 2.7, 4.8, 5.1 and 0.60.
Nothing is written: 0 of 512 offers. The block stores do recall on every
forward (similarity 0.90 to 0.997), but what they recall is the same before
and after the read, so the score cannot move. The layers' own stores: layers
0 to 2 are empty and a read offers salience of 0.001 to 0.002 against a
floor of 0.1; layer 3 holds 64 and never recalls (similarity 0.21 against
0.7). `adaptive_episodes` and `adaptive_recall` are off. So the exact zero
is a route that is shut on this run.

**768 seed 97.** The same, except block stores 0 and 1, where a read's
salience sits at the bar: 36 and 37 of 128 offers written. Blocks 0 to 3
recall on every forward; blocks 4 to 7 never. The score moves only when a
scored window recalls a slot the read overwrote: documents 8, 11, 12, 17 and
21 for B, document 30 for C. Document 12 carries +0.0214 of it.

**For Opus:** your review names document 10. In my files it is index 12.
Probably a different read or a count from another base; please check.

## Read 3: each way alone

July seed 42 on the first layout (32 and 30 documents); 768 seed 97 at 64
documents, 8 scored. B minus A first, then B minus C.

**July seed 42**

| read | 64 windows, B-A | B-C | 256 windows, B-A | B-C |
|---|---|---|---|---|
| everything on | +4.61e-05 | about 0 | +4.85e-05 | -1.4e-06 (t -1.0) |
| everything but the write | +4.29e-05 | exactly 0 | +3.78e-05 | exactly 0 |
| only the write | +8.6e-06 (t +2.2) | about 0 | +5.5e-06 (t +2.3) | -6.9e-07 (t -0.6) |
| only the pull | -1.2e-05 (t -0.7) | exactly 0 | +4.40e-04 (t +14.6) | exactly 0 |
| only consolidation | +2.01e-04 (t +6.8) | exactly 0 | +1.16e-03 (t +28.2) | exactly 0 |
| only episodes | exact null | | exact null | |
| everything off | exact null | | exact null | |

At 256 windows consolidation alone costs +1.16e-03 and the pull alone
+4.40e-04, and the two together with everything else cost +4.85e-05. They
hold each other. I have not measured how; consolidation moving the weight
and the pull returning it is the plain guess. This is Opus's point that the
ways do not add up, and here they nearly cancel.

**768 seed 97**

| read | 64 windows, B-A | B-C | 256 windows, B-A | B-C |
|---|---|---|---|---|
| everything on | +8.98e-04 | +4.35e-04 (t +2.0), 26 of 64 | +3.65e-03 | +1.11e-03 (t +3.0), 25 of 64 |
| everything but the write | +8.89e-04 | +4.44e-04 | +3.44e-03 | +1.12e-03 (t +3.4), 24 of 64 |
| only episodes | +8.71e-04 | +4.49e-04 (t +2.1) | +3.41e-03 | +1.11e-03 (t +3.3), 24 of 64 |
| only consolidation | +1.06e-04 (t +6.5) | exactly 0 | +7.44e-04 (t +12.5) | exactly 0 |
| only the pull | +1.4e-05 | exactly 0 | +3.57e-04 (t +8.3) | exactly 0 |
| only the write | +2.3e-06 | -9.3e-06 (t -0.6), 34 of 64 | +1.44e-04 (t +4.3) | -6.1e-06 (t -0.2), 33 of 64 |
| everything off | exact null | | exact null | |

At 768 the episode stores alone give the whole of B minus C. At 256 windows
B is worse than C on 39 or 40 of 64 documents, so this is no longer one
document, as it was at 64 windows on the first layout.

On both runs B minus C is exactly zero whenever the write and the episode
writes are both off, as it must be, and the write alone leaves B and C
equal within noise. On this setup, at these read lengths, the write does not
make a read of a document help that document.

## Costing draft 3 (brief 3, item 3)

48 documents, 64 read, 8 scored. Times are with three jobs on the card.

| arm | July seed 42 (512) | 768 seed 97 |
|---|---|---|
| 64 windows, each of three ways | 2.3 to 2.8 min | 4.7 to 5.2 min |
| 64 windows read 16 times | 29.3 min, measured | about 60 min, by arithmetic |

The 768 figure: the 256-window reads at 768 gave 0.037 s a forward
(32,768 read forwards in about 20 minutes); the top rung is 98,304 read
forwards. I did not run it.

What fits: three jobs at a time is the ceiling on this machine (about 4 GB
of memory each; five left 5 GB free, eight got jobs killed). So three arms
of the top rung at 768 take about an hour of wall time, and at 512 about
half an hour. If draft 3 on one run is both sets and three ways at the top
rung, that is six arms in two rounds: about two hours at 768.

Results from the costing reads, for what they are worth:

- July, everything on: B-A +3.08e-05, B-C about 0 (27 of 48). Read 16 times:
  B-A +1.54e-04, B-C -4.2e-06 (t -1.0), 25 of 48.
- 768, everything on: B-A +1.31e-03, B-C +9.42e-04 (t +2.4), 18 of 48.
  Everything but the write is the same to three figures. Only the write:
  B-C -1.2e-05 (t -0.6), 23 of 48.

## The write ladder retaken (brief 3, item 4)

`<run>/write_ladder_2.json`, from `probe_write_ladder.py`. Consolidation and
episode writes held off with `adaptation_eval`'s own `_read`; a pass at rate
0 and pull 0 is checked to move no weight, and on every run it moved none.
6 documents and 15 pairs a run; 1, 4, 64, 256 and 1,024 windows; one
`rebuild_run`. Seven runs done: July 42 to 46, the rerun of 44, 768 seed 97.
768 seed 46 (a run killed early) was still reading when I wrote this.

- **The output side is the same for every document.** Median cosine between
  documents of +1.00 in every layer of every run at every rung; the lowest
  median anywhere is +0.95. The between-document cosine is the error side's
  cosine, to two figures.
- **The write is one direction.** Top share 0.96 to 1.00 nearly everywhere.
  It falls to 0.87 to 0.90 in the first two 768 layers at 1,024 windows, and
  to 0.76 to 0.88 in the third July layer on two runs.
- **July, last layer:** the error side agrees between documents from the
  first window (+0.82 to +0.97) and reaches +1.00 by 64. The write there
  does not depend on the document. The first July layer stays low
  (-0.5 to +0.75, no trend I would trust).
- **768:** between documents the write is unrelated at 1 to 64 windows
  (-0.17 to +0.17), and comes to agree with length: at 1,024 windows +0.65
  to +0.77 in the first three layers, falling to +0.14 in the last. So a
  long read at 768 writes more of what is common to documents, and less of
  what is particular to one.
- **Write against pull.** In the July last layer the pull is 130 to 425 times
  the write alone at 1,024 windows (about 8e-03 against 3e-05 of the
  weight's norm), and the write in context is 15 to 40 times the write
  alone: most of "whole minus rest" there is the write changing what the
  pull then does. At 768 the write is 10 to 40 times the pull in the first five
  layers and about equal to it in the last three, and the write alone is
  larger than the write in context from 256 windows.

## Suggestions

1. The one mechanism that makes B differ from C at 768 is an episode store
   overwriting a slot and recalling it, and it makes B worse. Before a
   longer E1 at 768, decide whether a read should write episodes at all, or
   whether recall of a slot written in the same read should be scored.
2. On July, check what holds consolidation and the pull against each other
   at 256 windows: the weight's distance from home after each of the three
   reads would settle it, with no new machinery.
3. The error side is where a document could be told from another. On the
   July last layer it is the same for all of them from the first window. It
   would be worth knowing what the error there is an error of.
4. The write ladder at 768 on one finished run is thin. Seed 97 is the only
   one; say so wherever its numbers are used.

-- terminal Fable 5.2
