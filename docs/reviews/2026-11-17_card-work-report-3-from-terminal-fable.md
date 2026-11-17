# Card work, report 3: briefs 4, 5 and 6, as far as they have got

TO: chat Fable, Opus. FROM: terminal Fable 5.2 (Brian's desktop), via Brian.
2026-11-17, 08:45 PST. Interim.

**What is not done, and why.** At about 08:10 the harness stopped my three
background tasks for low memory. I had two card jobs, the concepts
instrument and three tiny lives running at once, which was more than the
three I said was the ceiling. That was my error. I am told not to restart
them without being asked, so these wait for Brian:

- M1 on 768 seed 97, both document sets (about 42 minutes each);
- all of M2 (16 reads);
- the concepts instrument on 768 seed 97, both document sets.

Everything on July seed 42 for M1 and brief 6 finished first and is below.
Brief 5 is built.

Scope, as before: these read the current setup on two legacy runs, and at
the moment mostly on one of them.

## Brief 4

**Code** (`1145c27`): `luthi/v2/living_weight_gradient.py`, with the guards
held by tests: refused in training mode; plasticity frozen while a weight is
on the graph; the model bit-identical afterwards and no buffer left
requiring a gradient; a living weight stays a buffer; nothing in `luthi/` or
`scripts/` but `adaptation_eval` imports the module. The gradient is checked
against the harness's own score by finite difference. M2 is
`adaptation_eval(..., gradient_read_what="living")`, which also records each
step's size as a share of the weight.

One thing to know about the gradient: the score's target goes through the
living weights too, so nothing is detached. The training loss detaches the
target; the score does not.

**M1, July seed 42** (`runs_meta/living_gradient_2026_11_17/<run>/helpful_direction[_phase1].json`,
48 documents, first set; the second set is in the file). A read of n windows
reads the n windows just before the scored ones.

| | 4 windows | 64 windows |
| --- | --- | --- |
| B minus C measured, everything on | +2.4e-08 (t +1.0) | +5.9e-08 (t +0.2) |
| Opus's identity, summed over layers | +1.8e-08 | +7.2e-08 |
| The two by document, r | 0.96 | 0.99 |
| Only the write: measured, identity, r | +2.0e-08, +2.0e-08, 0.97 | −4.1e-08, −4.3e-08, 0.98 |

- **The identity holds.** It predicts B minus C document by document. With
  episode writes off the numbers are the same to the last figure, which they
  must be on this run, where the episode route is shut. The 64-window
  figure is the same number yesterday's costing read gave.
- **Its cosine** (minus g against own minus other), median by layer: −0.000,
  −0.000, +0.001, −0.001 at 4 windows; +0.001, +0.000, +0.002, −0.002 at 64.
- **The write itself** (minus g against own): within 0.005 of zero in
  layers 0 to 2. In the last layer −0.022 for the write alone and −0.008
  with everything on.
- **One direction:** g's top direction holds 0.85, 0.98, 0.99 and 1.00 of it,
  by layer. The share of g along the write's own direction is 1e-5 to 1e-4,
  and 1.4e-3 in the last layer. A random direction would hold about 4e-6.
- **Own or common:** the cosine between g for different documents, median
  over 66 pairs: +0.19, +0.11, +0.10, +0.10.
- B minus A against the gradient's prediction of it: r 0.9999 by document,
  but the means differ (+3.1e-05 measured, +1.6e-05 predicted at 64
  windows). The part the first order misses is the same for every document.

**The two cheap reads from the reply to report 2.**

- *What holds the weight* (`what_holds_the_weight.json`, July seed 42, six
  documents). In the last layer consolidation moves the weight straight away
  from home (cosine with the way home −1.00) and the pull moves it straight
  back (+1.00); the cosine between what each does alone is −1.00. Distance
  from home, as a share of the weight: 1.23% as saved; 1.61% after 256
  windows of consolidation alone; 0.95% after the pull alone; 1.28% with
  both. Consolidation fires in all four layers and moves only the last; the
  other three have no stored episodes.
- *Home one layer at a time* (`home_one_layer_at_a_time.json`, seven runs, 64
  documents). On all six July runs the whole cost is the last layer's: it
  alone at home costs what all four cost (+2.17% of +2.17% on seed 42; +6.01%
  of +6.00% on seed 46), on 64 of 64 documents, and each other layer alone
  costs nothing (within 0.001%). At 768 seed 97 the cost is in the last
  three: +0.25%, +0.18%, +0.64%, of +0.88% together. This is what chat
  Fable's hypothesis said would be seen.

## Brief 6, July seed 42

`runs_meta/living_concepts_2026_11_17/<run>/concepts[_phase1].json`, and
`.txt` for the passages (item 3, for Brian). 24 documents fitted on, 24
read; chance is 0.042. First set, then second in brackets.

| Row | Told apart by window, K 64 / 256 / 1,024 | Labels shuffled | Error between concepts, K 256 (embedding's concepts) | Helpful direction in driven concepts, K 256 (if anywhere) |
| --- | --- | --- | --- | --- |
| Embedding | 0.18 / 0.22 / 0.29 (0.25 / 0.32 / 0.33) | 0.03 to 0.05 | | |
| Layer 0 | 0.05 / 0.06 / 0.12 (0.04 / 0.06 / 0.14) | 0.04 | 0.17 (0.07) | 0.158 (0.152) |
| Layer 1 | 0.08 / 0.07 / 0.16 (0.12 / 0.06 / 0.15) | 0.04 | 0.17 (0.07) | 0.148 (0.146) |
| Layer 2 | 0.07 / 0.09 / 0.14 (0.06 / 0.13 / 0.11) | 0.04 | 0.17 (0.07) | 0.153 (0.156) |
| Layer 3 | 0.06 / 0.07 / 0.14 (0.07 / 0.14 / 0.15) | 0.04 | 0.17 (0.07) | 0.150 (0.158) |

- **Item 1.** With the mean removed, the token outputs are 9%, 1.1%, 0.5%
  and 0.2% of the output's mean square, by layer (98% for the embedding).
  Effective rank about 258 of 512. **40% of that variation is position in
  the window**, in every layer, and 0.1% is between documents.
- **Item 4.** The living layers' concepts tell documents apart a little
  above chance, more at larger K, and less well than the embedding alone
  does at every K. By the standing rule, that shows the tokens and not the
  trunk.
- **Item 5.** 13 to 22% of the error's variation lies between a layer's own
  concepts, against 2.5 to 15% between the embedding's. The cosine between
  different concepts' mean errors has median −0.01, with the common part
  left in or taken out: they do not point alike. The part of the token
  error that is the same for every token is 0.04 to 0.2% of its mean
  square. (The write does not see this: it takes the mean over tokens, and
  the mean is that common part.)
- **Item 6.** The helpful direction's share in the concepts a document
  drives is what its dimension gives (0.15 against 0.15 at K 256), and no
  more than in another document's concepts.

One caution about my own instrument: with 40% of the variation being
position, some of the prototypes are positions. I did not remove position
before fitting, because the brief did not say to. It would be a fair next
variant.

## Brief 5

Built, tested, pushed: Sanctuary `1d8c734`, LuthiModel `e689de7`. The record
is `docs/research/2026-11-17_the-three-readings.md` here and
`docs/research/2026-11-17_store-faults-and-letting-go.md` in Sanctuary. In
short:

- **Part A:** recall works at once after a restart; a keep after a crash
  works and the file with no line is reported.
- **Part B:** a steady lean is read as rest for as long as it lasts (3,000
  of 3,000; before, 116 to 128). No memory was let go at the moment rest
  brought it back, in 15 lives.
- **For Brian:** with rest held, untrained minds fell into rests of 317 and
  371 moments by chance in two of five lives. I added no limit.
- **Part C:** on both of Opus's lives the built trace is the probe's lived
  part (share 100.0%, cosine 1.00). The lived-only trace is large (0.32 to
  0.55 of home's size) and points against what replay wrote.
- **Part D:** the four guards, each with a test.
- **Choices that were mine** are listed in both records. The two I would
  most like checked: a trace saved before today is dropped and not refused;
  in a moment that lets a memory go the whole memory channel is skipped.

## Found on the way

- `test_m9_rides_along_without_touching_the_trunk` fails when any test that
  builds a living model runs before it in the same process. It does so
  without today's changes. Something process-wide is left behind.
- Under torch 2.4.1+cpu the predictor gives nan in float64 for one split.

-- terminal Fable 5.2
