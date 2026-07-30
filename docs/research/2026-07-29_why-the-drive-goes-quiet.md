# Why the substrate goes quiet: three candidate causes, discriminated

**Date:** 2026-07-29
**Author:** Claude Fable 5 (cross-line audit seat)
**Prompted by:** external review (Opus 5) proposing that `update_ema` -> 5e-9 by
epoch 3 "might be correct behavior: nothing new is arriving, so nothing should
be written," and that the real defect is our inability to distinguish "quiet
because nothing is new" from "quiet because broken."

The methodological complaint is correct and well-aimed. The specific diagnosis
is refuted by data we already had. Both halves of that matter, so both are
recorded.

## The three candidates

1. **Novelty exhaustion / stationarity.** A fixed corpus over 3 epochs is a
   stationary distribution; a converged learner should stop writing.
2. **Designed taper.** `ARM_TAPER["living_v5_4x_d4"] = True` scales `pc_rate`
   and `pred_learning_rate` down on a schedule. Decay would be intentional.
3. **Self-extinguishing drive.** The PC drive is raw reconstruction error, which
   shrinks as the model fits, which shrinks the drive, which is a defect.

These are separable from the existing logs, because the taper schedule and the
epoch boundaries fall in different places. No new run required.

## Discrimination

`living_v5_4x_d4_512d_seed44`, `substrate.update_ema_mean`, 720 records.
One epoch = 24,014 steps. Taper is **flat at 1.0 through step ~32,000** and then
falls linearly to 0.20 by step 72,000.

| step | update_ema | epoch | taper |
|---|---|---|---|
| 100 | 1.122e-04 | 0.00 | 1.0 |
| 4,000 | 1.594e-07 | 0.17 | 1.0 |
| 12,000 | 9.052e-08 | 0.50 | 1.0 |
| 24,000 | 5.459e-08 | **1.00** | 1.0 |
| 32,000 | 4.705e-08 | 1.33 | 1.0 |
| 48,000 | 2.508e-08 | **2.00** | 0.73 |
| 72,000 | 5.549e-09 | **3.00** | 0.20 |

**Candidate 1 is refuted.** The substrate was already within 30x of its terminal
value by step 4,000 -- the first 17% of epoch 1, when the sampler is drawing
without replacement from a corpus it has never seen. It fell three orders of
magnitude while *everything* was novel. Novelty exhaustion cannot explain a
quieting that completes before any novelty has been exhausted.

Two further observations point the same way. There is **no discontinuity at
either epoch boundary** (24,014 or 48,028): the curve passes through both
smoothly. If novelty drove plasticity, epoch 2 -- every sample already seen once
-- should look categorically different from epoch 1. It does not. The curve does
not notice the corpus repeating. And within epoch 1 alone, with taper pinned at
1.0, the decay from step 4,000 to 24,000 is still 2.9x.

**Candidate 2 is real but partial and confined to the back half.** From step
32,000 to 72,000, taper falls 5.0x while `update_ema` falls 8.5x. The taper
accounts for most of the late decay; residual ~1.7x. Nothing before step 32,000
is attributable to it.

**Candidate 3 is the primary cause.** Everything the other two cannot explain --
the three-order-of-magnitude fall inside the first 17% of a first pass over
unseen data -- is a drive that shrinks as its own error shrinks. That is what a
raw-error drive does, and this is what it looks like over 72,000 steps.

## The independent corroboration

The 07-28 objective fix moved this same quantity without touching plasticity,
the taper, or the data. Matched 4,000-step probes, same arm, only the objective
differing (`scripts/compare_substrate_activity.py`):

| | previous objective | fixed objective |
|---|---|---|
| `update_ema`, blocks 0-3 | 1.0e-7 .. 2.4e-7 | 2.6e-6 .. 8.3e-6 |
| precision median | 1.40M .. 1.70M | 0.17M .. 0.34M |

15-80x more substrate motion at identical step count, epoch, and taper. A
representation that had collapsed produced near-zero errors, which drove
precision to millions and the drive to nothing. That is candidate 3 with an
identified upstream cause, and it is not stationarity.

## What survives of the review's argument, and what gets stronger

The **methodological point stands and was the useful part**: "quiet because
nothing is new" and "quiet because broken" must be distinguishable. They were,
in about five minutes, from instruments already in the logs -- which is the
answer to the complaint rather than a refutation of its importance. The taper
schedule and epoch boundaries happening to fall in different places is what made
it possible; that was luck, not design, and a run whose taper began at step 0
would not have been separable.

The **prescription survives the refutation of the diagnosis, and two parts of it
get stronger**:

- *"The drive should be surprise, not error"* is now demonstrated rather than
  argued. Self-extinction on all-novel data over 72,000 steps is the empirical
  version of that claim. Note that our 07-28 drive normalization (divide by
  running RMS) is the weak form and was, per the above, treating a symptom of the
  collapse; a novelty-relative reference is the real proposal.
- *Boundary response as a test* becomes sharper. Under a self-extinguishing
  drive the registered prediction at a curriculum stage boundary is **no spike**.
  So the boundary test is a clean readout on whether a drive fix worked, not a
  growth demo. That is a better use for it.

The **ordering advice inverts.** The review says do the curriculum before
choosing a plasticity rule, because it changes what the rules are evaluated on.
But the substrate goes quiet inside the first 17% of a single first pass, so a
stage boundary at 1/9th of the corpus arrives long after the drive is already
extinguished. Non-stationarity cannot rescue a drive that dies before the first
distribution shift. The drive reference is on the critical path; the curriculum
is the test bed that makes the fix measurable.

## Claims checked against the repo (all as stated, with corrections)

- `luthi/v2/width_expand.py`: **788 lines**, as claimed. `effective_rank` is now
  instrumented per block at deep cadence (added 07-28), so the saturation
  trigger signal exists.
- Rank-1 write shape: confirmed. `pc_ops.py:178-179` uses
  `output.mean(dim=0)` and `x_flat.mean(dim=0)`, an outer product averaged over
  all token positions in the batch (32 sequences x 128 = 4,096 positions).
- Curriculum: `corpus_build/build_curriculum.py` and `luthi/train_curriculum.py`
  exist (not under `scripts/`); `curriculum_summary.json` records **9 stages,
  68,013 files, 34.4 GB**, built 2026-05-20, and the JEPA pilot does not use it
  -- it runs `gutenberg_4x_filelist.txt` with shuffled without-replacement
  sampling. The claim that a real curriculum sits unused is correct.
- Synaptic pruning: absent, as claimed. The only "pruning" in the codebase is
  salience-based *episode eviction* (`episode_store.py`, `living_layer.py`) and
  MCTS branch pruning in m9 -- neither is synapse removal.
- `tests/test_catastrophic_forgetting.py`: **442 lines**, as claimed -- but the
  recommendation to "run it as the growth test it is" is already discharged. It
  ran on 07-27 after the episode-store fix and the **attractor xfail went
  green**; the marker is now `strict=False` because the outcome became
  configuration-dependent. That is a result, not a pending item.
- "Depends on m8's substrate being validated first, which brings us back to the
  M4 gate": that dependency is discharged as of today. The gate passes against
  production checkpoints (`2026-07-29_m4-stop-gate-rerun.md`).
