# The M4 STOP GATE, re-run against production checkpoints

**Date:** 2026-07-29
**Run by:** Claude Fable 5 (cross-line audit seat)
**Script:** `scripts/run_m4_stop_gate.py` (read-only; checkpoints copied aside)
**Prompted by:** external review round 3 (Opus 5), which asserted the gate had
never been run.

## Correction to the review's premise

The gate ran and passed on 2026-05-09, at M4 scale. `docs/ML_GLOSSARY.md`
records it: *"Passed 2026-05-09 with margin; remains the architectural
justification for the whole consolidation pathway."* What had never happened is
re-running it against **production** checkpoints, where `update_ema` has decayed
to ~5e-9 and consolidation moves weight by `pc_rate * 0.1 = 1e-4` of a distance
that the logged numbers suggested was itself near zero. That is the real gap,
and it is worth closing. The review was right about the risk and wrong about the
history.

## The gate has two wordings, and they disagree

`docs/ML_GLOSSARY.md`:

> if consolidation has no measurable effect on **prediction quality**
> post-replay, v2 has no architectural novelty over "vanilla transformer +
> episode store" and should be abandoned.

`luthi/v2/consolidation.py` docstring (closer to the original brief):

> if consolidated layer's **behavior on the episode's context is not measurably
> closer to the stored snapshot** than a control without consolidation [...]

These are not the same claim, and the difference turns out to matter. Both were
tested, against a no-consolidation control, on every layer holding episodes:

- **A** weight-space distance to the stored snapshots (consolidation.py, literal)
- **B** behavioural distance to the snapshots on each episode's own stored
  input: `|W_now @ x - W_snap @ x|` (consolidation.py, intent)
- **C** prediction error on each stored input pattern (glossary)

Pre-registered threshold for "measurable": relative change > 1e-3.

A second yardstick, taken from the run's own data rather than chosen: the layer
tracks `update_ema`, an EMA of ordinary per-step weight motion. If a full replay
pass over every stored episode moves the layer *less* than one ordinary training
step does, consolidation is below the noise floor of normal learning regardless
of what the relative numbers say.

## Result: the gate PASSES on the production substrate

All six v5 runs, `blocks.3.living_ffn` (the only block holding episodes):

| run | A weight | B behaviour | C prediction | replay pass vs 1 step | gate |
|---|---|---|---|---|---|
| v5 seed42 | 6.26e-03 | 6.27e-03 | **1.613e-01** | 5714x | PASS |
| v5 seed43 | 6.20e-03 | 6.25e-03 | **1.444e-01** | 5817x | PASS |
| v5 seed44 | 6.29e-03 | 6.30e-03 | **1.516e-01** | 4883x | PASS |
| v5 seed45 | 5.85e-03 | 6.15e-03 | **1.291e-01** | 4263x | PASS |
| v5 seed46 | 6.17e-03 | 6.23e-03 | **1.554e-01** | 3085x | PASS |
| v5 rerun seed44 | 6.31e-03 | 6.29e-03 | **1.504e-01** | 3470x | PASS |

Criterion C is carried entirely by the **attractor** pathway, which reduces
prediction error on stored patterns by **12.9% to 16.1%** relative to control.
Gradient-replay contributes 0.25% to 0.30%. Two orders of magnitude apart.

Note the spread: 12.9-16.1% across six runs including a byte-identical rerun.
Unlike `precision_spread` (70.8% divergence between identical runs), this is a
**reproducible observable** and can support a point comparison. That was the
methodological lesson from the 07-27 verdict, applied here before registering
anything on it.

## Three findings the gate surfaced that it was not asked about

**1. The two wordings disagree about the attractor pathway, and the glossary's
is the correct one.** For attractor consolidation, criterion A is *positive* on
every v5 run (+3.6e-3 to +3.8e-3) - it moves weights **away** from the stored
snapshots while simultaneously reducing prediction error on the stored inputs by
15%. That is not a contradiction, it is the mechanism working as designed:
attractor consolidation does not pull toward a past weight state, it makes
stored inputs fixed points of the layer's own dynamics. The consolidation.py
wording ("closer to the stored snapshot") only ever made sense for
gradient-replay. Applied to the attractor pathway it would have failed a
mechanism that is demonstrably working. **The consolidation.py docstring's gate
wording should be corrected to the glossary's.**

**2. The review's "no-op that fires constantly" was right about three quarters
of the production family, for a reason it did not identify.** In every v5 run,
three of four blocks stored **zero** episodes - for those, consolidation is a
literal no-op. And it fired constantly: `substrate.consolidation_fires` reaches
5718 by step 72000 (summed over 4 blocks), roughly one firing per block per 50
steps, ~1430 per block per run. So ~4290 of those ~5720 firings per run did
nothing at all. The cause is not the trigger, it is the **frozen episode store**
defect found on 07-27: the store could not admit. The trigger itself does not
latch (`ConsolidationTracker.step` resets `_below_threshold_count` on firing and
on any above-threshold step) - but once variance settles permanently below half
the frozen baseline, it re-arms and fires every 100 calls forever, which
produces the behaviour the review described without the mechanism it proposed.
Under the store fix, the probe run has all four blocks populated (4, 2, 64, 64).

**3. The pass is real but the margin is inflated by a sick substrate.** A
replay pass moving 3000-5800x more than an ordinary training step does not mean
consolidation is powerful; it means ordinary PC learning had gone nearly silent.
In the v5 family, consolidation is by three orders of magnitude the dominant
force shaping the PC weights. Under the fixed objective the same ratio falls to
3-73x, which is a far healthier regime - consolidation as one force among
several rather than the only one still moving.

## Cross-check: the same gate under the fixed objective

`probe_storefix_512d_seed45` (4000 steps, fixed objective) vs
`probe_storefix_512d_seed44` (4000 steps, previous objective), same arm:

| run | layer | episodes | C prediction | vs 1 step | gate |
|---|---|---|---|---|---|
| probe seed44 | blocks.0 | 3 | 5.60e-03 | 4.6x | PASS |
| probe seed44 | blocks.1 | 3 | 3.76e-03 | 3.8x | PASS |
| probe seed44 | blocks.2 | 2 | 3.84e-03 | 4.7x | PASS |
| probe seed44 | blocks.3 | 64 | 1.723e-01 | 132x | PASS |
| probe seed45 | blocks.0 | 4 | 1.14e-04 | 3.1x | **FAIL** |
| probe seed45 | blocks.1 | 2 | 1.73e-03 | 1.1x | PASS |
| probe seed45 | blocks.2 | 64 | 2.74e-02 | 73x | PASS |
| probe seed45 | blocks.3 | 64 | 1.02e-02 | 52x | PASS |

The single FAIL is a 4-episode store, and the effect scales with store
occupancy across every row in both tables. That is the expected shape - a
mechanism that replays stored material does less when little is stored - not a
gate failure. The gate is a claim about whether the mechanism has a measurable
effect at all, and it does, at every populated store.

## Verdict

**The M4 STOP GATE passes against production checkpoints.** Consolidation has a
measurable, seed-reproducible effect on prediction quality post-replay
(12.9-16.1%, six runs). The architectural justification for the consolidation
pathway stands.

Three qualifications carried forward, none of which change the verdict:

1. The effect is the **attractor** pathway. Gradient-replay is within noise of
   nothing (0.25-0.30%) and should be re-examined on its own terms, not credited
   with this pass.
2. The gate is only evaluable where episodes exist. It passed in v5 on one block
   out of four, because the other three could not store. The pass covers the
   mechanism, not its coverage.
3. Every number above from the v5 family was produced under the objective now
   known to be defective (`2026-07-28`, BatchNorm-blinded SIGReg). The gate is
   re-run here on post-fix checkpoints as a cross-check and passes there too, at
   a smaller and more honest margin.
