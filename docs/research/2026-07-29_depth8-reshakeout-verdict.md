# Depth-8 re-shakeout with clipping: VERDICT — FAIL (stable but dead)

**Date:** 2026-07-29, ~22:15
**Run:** `probe_surprise_d8_512d_seed96`, 3000 steps, 0.57 h, `grad_clip_norm=1000`
**Compare:** `probe_surprise_d8_512d_seed97` (unclipped, diverged)
**Criteria:** `docs/research/2026-07-29_depth8-collapse-shakeout-criteria.md`

**Do not start the 18-hour run.** Clipping fixed the divergence and revealed a
second, independent problem underneath it.

## Scoring against the registered criteria

| # | condition | value at 3000 | result |
|---|---|---|---|
| 1 | `std_p5` >= 0.85 | 1.1746 | pass |
| 2 | `cos_pred` <= 0.75 | 0.5070 | pass |
| 3 | `L_sigreg` <= 300 | 388.9 | **FAIL** |
| 4 | direction 2000->3000 | std rising, cos falling | pass |
| 5 | `cos_pred` >= 0.40 | 0.5070 | pass |
| 6 | `L_pred` <= 4.0 | 4.93 | **FAIL** |

FAIL on two of six, both marginal. On the training metrics alone this reads as a
near miss, and I was about to describe it that way.

## The held-out evaluation says it is not a near miss

| | depth 4 (seeds 45/46) | depth 8 unclipped | depth 8 clipped |
|---|---|---|---|
| heldout NMSE | 0.52 - 0.60 | 5.675 | **1.505** |
| heldout l_pred | 0.82 - 1.14 | 9576.9 | 7.489 |
| probe top1 | 0.107 - 0.121 | 0.069 | 0.0132 |
| probe shuffled floor | 0.023 - 0.025 | — | 0.0128 |
| **probe lift over floor** | **4.67x - 4.80x** | — | **1.03x** |

**NMSE 1.505 is worse than predicting the mean** (NMSE 1.0 is the
predict-the-mean line). **Probe lift 1.03x means the representation carries
essentially no retrievable information above chance**, against 4.67-4.80x at
depth 4.

Clipping worked at what it was for: NMSE 5.675 -> 1.505, `l_pred` 9577 -> 7.5, no
divergence, no runaway. It produced a **stable model that learns nothing.**

## The fourth defect in my criteria, and the instructive one

The three earlier criteria errors were mechanical: two one-sided bounds on
two-sided quantities (`cos_pred`, then `std_p5`), and one threshold set on the
raw predictor cosine when the mean-centered version — which this project built
yesterday precisely because the raw one is confounded by offset — was available.

This one is structural. **Every condition I registered measures stability, and
the question was whether the model learns.** A stable-but-dead model passes four
of six. The two measurements that actually decided it, held-out NMSE and probe
lift over the shuffled floor, are not in my criteria at all — and the probe lift
is the single most informative number in the table.

Any future gate of this kind must include at least one capability condition. A
proposal, for the record: **probe lift over the run's own shuffled floor >= 2.0x**,
which depth 4 clears at 4.67x and this run misses at 1.03x.

## Diagnosis: two independent problems, not one

**1. The clip value is too aggressive.** 13 of 30 log points sat exactly at the
ceiling — **43% of sampled steps clipped**, median grad_norm 829 against a clip
of 1000. When I chose 1000 I wrote that it was "deliberately aggressive for a
stability probe: over-damping shows up as poor learning, which is a diagnosable
failure, where divergence is not." That is now diagnosed, and it is the predicted
failure mode rather than a surprise. A clip bounding only the excursions that
preceded divergence (2158, 2941, 5555, 8645) would sit nearer 3000-5000.

**2. Offset dominance at depth, which clipping barely touched.**

| | offset dominance (median) | centered cosine (median) |
|---|---|---|
| depth 4 | 0.143 - 0.150 | 0.62 - 0.68 |
| depth 8 unclipped | 0.719 | 0.329 |
| depth 8 clipped | **0.561** | 0.484 |

Depth 8 remains **~4x more offset-dominated than depth 4**. Clipping improved it
(0.719 -> 0.561) without coming near depth 4's level. This is the same failure
mode the 2026-07-28 objective fix addressed at depth 4 — a representation
dominated by one batch-constant direction — and it evidently is not solved at 8
blocks. It is independent of the gradient problem: bounding step size does not
change what direction the representation prefers.

### Hypothesis for (2), untested

SIGReg targets zero-mean isotropic N(0, I), so an offset is exactly what it
should penalize, and it is visibly fighting (`L_sigreg` in the hundreds). But it
is applied to the output of a per-modality `nn.Linear` projection head
(`jepa_loss.py:260`) — **and that Linear has a bias.** A bias can absorb the
batch-mean offset, presenting SIGReg with centered latents while the trunk
retains the offset.

That is structurally the same defect as the BatchNorm removed on 2026-07-28: a
learnable layer standing between SIGReg and the quantity it exists to constrain.
At depth 4 the offset is small (0.14) and it does not matter; at depth 8 (0.56)
it may.

Cheap to test — `sigreg_projection="none"` already exists and runs SIGReg on
trunk latents directly. **Not measured. Stated as a hypothesis, not a finding.**

## The guards behaved correctly

The divergence guards did **not** fire, and should not have: NMSE 1.505 is below
the 2.0 threshold and the loss never sustained 10x its frozen baseline. This run
was bad, not diverging. That distinction is the guards working as designed —
they are a floor against self-destruction, not a quality gate. The quality gate
is the criteria doc, and per the section above it needs a capability condition.

Worth noting for the record: had the unclipped seed97 run been executed under
these guards, the loss guard would have tripped around step 2600 and the run
would have reported `killed:divergence` instead of `completed`.

## Recommended next steps, one variable at a time

1. **Raise the clip to ~3000-5000** and re-shake. Bounds the pre-divergence
   excursions without clipping 43% of steps. Cheapest test, and it isolates
   whether over-damping alone explains the dead representation.
2. **Test `sigreg_projection="none"`** at depth 8. Directly tests the
   projection-bias hypothesis. If offset dominance drops toward 0.15, that is
   the second problem identified and it is a one-line fix.
3. Only then consider `mu_pc_exponent`. It has still never been tested at 8
   blocks, but it governs the PC substrate rather than either problem above.

Do not vary two at once. The whole reason this is diagnosable at 90 minutes of
compute rather than 18 hours is that each run changed one thing.
