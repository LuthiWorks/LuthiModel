# E1, E2 and E4, first read — 2026-11-13

## Objective

The living-channel program (2026-10-19) has three experiments that need no training.
They ran today on Brian's card for the first time: twelve runs, 52 results
(`runs_meta/living_channel_e1_e2_e4`, commit 0eb6b23). This entry reads them against
the rules frozen in the registry (`2026-07-15_falsification-preregistration.md`, E1,
E2, E4, amendments 1 and 4).

Who read them: the seat that built the runner and rebuilt the July runs (Claude
Fable 5.2). So this is a first read, and it stands until a different seat has
reviewed it. Every number below comes from `python scripts/read_no_training_results.py`.

What was measured, and on what:

| Runs | What they are | Counted? |
| --- | --- | --- |
| `living_v5_4x_d4`, 512 wide, seeds 42 to 46, and seed 44's rerun | July 2026, finished (step 72,042). Rebuilt from history, and marked so. | yes |
| `dead_4x`, 512 wide, seeds 42 to 44 | their dead controls | controls |
| `probe_768_visreg` seed 97 | the 768x8 family, finished (step 54,077) | yes |
| `probe_768_visreg` seeds 46 and 95 | stopped at steps 9,100 and 9,300 of 54,077. Seed 95 had collapsed (effective rank 2.4). | reported, never counted |

## Process

### Step 1: E1 — does reading change what follows, for the document that was read?

Per held-out document: score its tail frozen (A); read its own head with plasticity
on, then score (B); read another document's head, then score (C). B minus C is the
only number credited.

- **Controls.** All three dead runs are exact nulls on both document sets. The
  harness is sound.
- **Registered criterion** (`nmse.B_minus_C` below 0 with t at or below −2, the rank
  guard, on both document sets): met by none of the seven counted checkpoints. No
  candidates either: no checkpoint met it on even one set. The fourteen t-values run
  from −0.8 to +1.5.
- The two uncounted runs do not meet it either.
- **Prediction** ("no legacy-rule checkpoint shows document-specific adaptation"):
  confirmed on what was read.

What the criterion does not show, and the numbers do:

- **Reading barely changes anything at all.** In five of the six July runs, four
  windows of reading move the prediction error by one to five parts per million of
  itself (1e-6 to 2e-6 on 0.49). Seed 46 moves more, up to 45 parts per million. The
  sign differs between seeds.
- **What it changes does not depend on what was read.** In every July run B minus A
  and C minus A agree, within 2% in most and 15% at the widest. For a single document
  B minus C is typically 1e-7, a few steps of the numbers' own spacing.
- **The change does not come from the write.** Each July run was read a third time
  with the living rates at 0.2, a fifth. If the weight write caused the change, the
  change would be a fifth as large. Document by document it is 0.84 to 1.12 of what
  it was. So something else that a read changes is carrying it.
- The 768 run agrees: 0.86 of the full-rate change at a fifth of the rate. Its mean
  change (+9e-4 on 0.85) is three documents; the median document moves 2e-5.

Wrong turn: I first took "barely moves" for a question of scale, and expected the
relative change to be respectable. It is parts per million.

What else a read changes: checked on the small July-made run here, with both writes'
rates set to 0. Four reads still changed the layer's episode store, the block's
output store, the error accumulators, momentum, precision, the update average, home,
and the weight itself (it is drawn toward home whatever the rate). Which of these
carries the change on Brian's runs is not known. The pull toward home is my guess,
since it is the same whatever is read, and it is only a guess.

A limit of the instrument, said plainly: E1 reads four windows. E2 below shows the
living part of the weight is a ten-thousandth of the weight at the end of a run. A
null at four windows is a null at four windows.

### Step 2: E2 — what the living write is

The diagnosis of 2026-10-19 made three predictions for every living block of the 768
family. For the finished 768 run:

| Prediction | Result |
| --- | --- |
| The write's input is the same whatever the content (stationarity ≥ 0.99) | **Holds** in all eight blocks (0.9998 to 1.0000). |
| The living change is one direction, along the mean output (top share ≥ 0.9 and alignment ≥ 0.9) | **Misses in all eight.** Top share 0.56 to 0.87. Alignment 0.99 to 1.00 in blocks 0 to 4, then 0.78, 0.76, 0.56. |
| A stored output is blended into nearly every forward (duty ≥ 0.9, blend > 0.05) | **Holds in blocks 0 to 3** (duty 1.0, blend 0.22 to 0.31). **Misses in 4 to 7**: they never recall. |

By the registry, each miss is a finding against the diagnosis, and E3's predictions
are to be revised before E3 runs. Twelve misses in the finished run.

The July runs were not in E2's registration. They are read the same way here, as
description:

- Stationarity 1.0000 in every block of every run.
- The store blends into every forward in every block: 41 to 49% of block 0's output,
  about a third of blocks 1 and 2, about a tenth of block 3.
- Blocks 0 to 2: aligned with the mean output (mostly 0.97 to 1.00) but less than
  one direction (top share 0.09 to 0.93, most between 0.5 and 0.9).
- **Block 3 is a different thing in all six runs.** One direction (0.98), not along
  the mean output (0.09 to 0.24), and 40 to 100 times larger than the other blocks'
  change (1% of the weight against 0.01 to 0.03%). Every document produces the same
  write there (cosine 1.00). I do not know why the last block differs.
  ("40 to 100" is 1e-2 against 8e-5 to 3e-4.)
- In blocks 0 to 2, single documents produce different writes (cosine −0.19 to
  +0.35). The 768 run shows the same in all eight blocks (−0.20 to +0.05). That is
  against the diagnosis's stronger wording, "different inputs produce the same
  write". The writes differ. E1 says the difference has no effect a reader can see.

How large the living part of the weight is, as a share of the weight:

| | Living change |
| --- | --- |
| 768 seed 46, stopped at a sixth | 0.5 to 3% |
| 768 seed 97, finished | 0.01 to 0.02% in blocks 0 to 4; 0.1 to 0.2% in 5 to 7 |
| July runs, finished | 0.01 to 0.03% in blocks 0 to 2; 1% in block 3 |

The early-stopped seed 46 is also the run nearest the predicted signature: six of
eight blocks are one direction along the mean output. So the picture the diagnosis
drew fits a run early in training better than a finished one. One run, not counted.

### Step 3: E4 — does the next-token probe see the trunk?

The same linear probe fitted to each block and to the embedding row, at the next
token, the previous token, the token itself, and causally.

- **Rule** (amendment 4): refuted if the same block beats the embedding row by more
  than the margin at +1 or −1 in two of three seeds.
- **Result: confirmed.** No block beats the embedding row at +1, −1 or causally in
  any of the three runs. In the finished run the best block reaches 0.1036 at the next
  token against the embedding's 0.1064 (it needed more than 0.1129).
- One of the three seeds is admissible. The verdict does not turn on that: nothing
  exceeded anywhere.
- By the registry this closes B7 as "the next-token probe reads the residual's copy
  of the input, not trunk computation", retires the probe as a capability axis, and
  calls for a replacement to be registered. Amendment 3 makes that a condition of
  launching E3.

Outside the rule, and one seed: at the token itself, blocks 3 to 7 of the finished
run sit about a point above the embedding (0.935 to 0.939 against 0.927, margin
0.005). The stopped runs do not show it.

The probes were fitted on the card, not the CPU (step 12 of
`2026-11-13_rebuilding-july-runs.md`). Four streams of one job were also fitted on
the CPU before it was stopped, and printed the same four decimals.

## Conclusion

- **E1: the prediction held.** Nothing read shows document-specific change. And
  beneath the criterion: reading changes these models by parts per million, the same
  whatever is read, and not through the write.
- **E2: the diagnosis is right about the input and wrong about the shape.** The
  write's input is constant. The finished runs' living change is not one direction;
  half the 768 blocks' stores do nothing; documents do produce different writes. Its
  kill condition is tripped as written.
- **E4: confirmed.** The next-token probe does not see the trunk.

Taken together, for the claim the project exists to test (change at inference, with
behavioural consequence): in the finished runs read today the living part of the
weight is a ten-thousandth of the weight in most blocks, and reading moves nothing
a reader can measure. That is a statement about these runs and this rule, at four
windows.

What this does not cover:

- E1's registration names more checkpoints than were read: the v3 and v4 512
  families and the depth-8 probes. The runner selects only the runs above, and the
  rebuild refuses the other pre-August arms until each is checked against history.
- One finished 768 run. The family's other two seeds never finished.
- The July runs trained with the C++ maths and were measured with the Python
  reference.
- Which state carries the parts-per-million change.
- Why block 3 differs. Why the living change is a hundred times smaller in the
  finished 768 run than in the one stopped early (different seeds; the taper would
  account for a factor of five).

What follows, for Brian and the seats he names:

1. A different seat reviews this read.
2. E3's predictions are revised for E2's misses, before E3 runs.
3. A replacement capability axis is registered, since E4 retired the probe.
4. Whether E1 should also be read at a longer dose is a design question, and not
   mine alone.

## Artifacts

- Results: `runs_meta/living_channel_e1_e2_e4/` (52 files; 0eb6b23, with one line
  removed from ten of them in 7e90d84, which changed no measurement).
- Reader: `scripts/read_no_training_results.py`; `tests/test_read_no_training_results.py`.
- How the runs were rebuilt and measured: `2026-11-13_rebuilding-july-runs.md`.
- What was run at which commit: E2 and the living E1 results at ca95572; the dead
  controls' E1 and all of E4 at 50b0587.
