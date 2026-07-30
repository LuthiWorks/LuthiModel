# SIGReg projection-bias hypothesis: REFUTED

**Date:** 2026-07-30, ~01:00
**Run:** `probe_surprise_d8_noproj_512d_seed95`, 3000 steps, 0.57 h
**Control:** `probe_surprise_d8_512d_seed96`
**Prediction registered before the run:**
`docs/research/2026-07-30_sigreg-projection-hypothesis.md`

## Verdict

**REFUTED.** The projection head's bias is not what preserves the offset at
depth 8.

| primary metric | control (linear) | test (none) | registered bound |
|---|---|---|---|
| trunk offset dominance, median | 0.5606 | **0.4762** | confirm <= 0.35 |
| trunk offset dominance, mean | 0.6083 | **0.5206** | refute >= 0.50 |

The median lands at 0.4762, inside the 0.35-0.50 band that was declared in
advance as **refuted for decision purposes**; the mean refutes outright. Depth-4
reference is 0.143-0.150, so the test run remains ~3.3x more offset-dominated
than depth 4 despite SIGReg looking straight at the trunk.

Pre-declaring the ambiguous band as refuted was load-bearing. 0.4762 against a
0.5606 control is a 15% relative reduction, and without the advance rule that is
exactly the number I would have been tempted to write up as "a real improvement,
trending the right way, worth pursuing." It is a real effect. It is not the
predicted one, and it is not a fix.

## What the refutation establishes

The measurement is valid in a way worth stating: `offset_dominance_target` is
computed on **trunk latents, pre-projection** (`_light_collapse_metrics` receives
`target_latents`, the full-sequence encoder output). So both runs measure the
same quantity — the trunk's own geometry — and differ only in what the objective
is allowed to see. That is what makes this a clean test.

With `sigreg_projection="none"`, SIGReg is applied directly to trunk latents.
There is no learnable layer between the regularizer and the geometry. And the
offset barely moves.

**So the offset is not concealed from SIGReg. It is resistant to SIGReg.** That
is a more awkward finding than the hypothesis would have been, and it relocates
the problem: something in the trunk generates a batch-constant direction faster
than the regularizer at the output can remove it. Eight blocks of residual
accumulation and LayerNorm gains compounding a mean is the obvious candidate,
and it fits the dose-response — four blocks gives 0.14, eight gives 0.56.

A regularizer at the output can only penalize an offset after it exists. If the
trunk manufactures it at every block, the penalty is fighting eight sources with
one lever.

## Cost, and a correction to my interim reasoning

Removing the projection head made prediction dramatically worse:
centered cosine **0.4778 -> 0.0651**, a 7.3x collapse.

Mid-run I wrote that SIGReg in the test arm was "working nearly twice as hard,"
comparing `sigreg` 1736 to the control's 965. **That comparison was invalid and
is withdrawn.** The control's SIGReg is computed on projected latents and the
test's on trunk latents — different inputs, not a common scale. The
offset-dominance comparison stands precisely because it does not have that
problem.

## Capability, as pre-registered: not readable

| | NMSE | probe lift |
|---|---|---|
| depth 4 | 0.52 - 0.60 | 4.67x - 4.80x |
| d8 linear (control) | 1.505 | 1.03x |
| d8 none (test) | 1.854 | 1.23x |

Both arms carry the clip of 1000, which is independently known to kill
capability. As registered in advance, **these numbers cannot be used to judge
the hypothesis either way** and are recorded only for completeness. The test's
marginally higher lift (1.23x vs 1.03x) is not evidence of anything; both are
near the floor and both are confounded.

## Where this leaves depth 8

Two problems were separated on 2026-07-29. One is now partly diagnosed and one
is now better localized:

1. **Gradient magnitude / clip value.** Clipping stopped the divergence; the
   clip of 1000 is too aggressive (43% of steps clipped) and kills learning.
   **Untested at a looser value.** This is the outstanding cheap experiment.
2. **Offset dominance at depth.** Not the projection head. Localized to the
   trunk itself. The next candidates, in order:
   - **Trunk LayerNorm gains across 8 blocks** — already instrumented via
     `norm_gain_summary()` (added 2026-07-28) and not yet examined at depth.
     Cheapest next look, and it requires no run at all: the data is in the
     existing logs as `trunk_norm_gain_median` / `trunk_norm_gain_min`.
   - **Per-block SIGReg**, rather than output-only — apply the constraint where
     the offset is generated instead of once at the end.
   - `mu_pc_exponent`, still untested at 8 blocks.

**Recommendation: look at `trunk_norm_gain_median` in the runs already on disk
before spending another 45 minutes.** The instrument exists, the data exists,
and if the gains are compounding across depth it will show without a new run.

---

# Follow-up, same session: four candidates eliminated, no cause yet

Everything below cost zero training compute — existing logs and checkpoints.

## The depth-8 collapse is worse than "offset dominance" conveyed

`scripts/measure_offset_by_block.py`, and an independent input-sensitivity check:

| | within-batch pairwise cosine | interpretation |
|---|---|---|
| depth 4 | **0.023** | different inputs -> near-orthogonal representations |
| depth 8 | **0.970** | different inputs -> nearly the SAME representation |

The depth-8 encoder has largely stopped responding to its input. Two different
random token batches produce representations 97% aligned. The depth-4 control,
run through the identical procedure, gives 0.023 — which is what validates the
method rather than the result.

Per-block offset dominance (same checkpoints):

| block | depth 4 | depth 8 |
|---|---|---|
| 0 | 0.157 | **0.989** |
| 1 | 0.166 | 0.9997 |
| 2 | 0.176 | 0.9998 |
| 3 | 0.164 | 0.9999 |
| 4-7 | — | 0.9995 - 0.9999 |

**The collapse is already total at block 0.** It is not accumulated across
depth, which kills the "eight blocks compounding a mean" story from the section
above — including the LayerNorm-gain hypothesis it rested on.

**Caveat, unresolved:** this measures 0.99 where the training-time instrument
logged 0.56 for the same run. Differences: eval vs train mode, final checkpoint
vs run median, random tokens vs real text. The within-batch cosine of 0.970 is
independent of that discrepancy and is measured identically at both depths, so
the *qualitative* finding is solid; the exact number is not. Reconciling the two
instruments is outstanding work.

## Four candidates eliminated

1. **Projection-head bias** — refuted by the registered experiment above.
2. **Trunk LayerNorm gains** — flat at 0.9935-1.0000 median (min 0.90) at both
   depths. From existing logs, no run.
3. **Positional embedding dominating token embedding** — pos/tok norm ratio is
   **1.01 at both depths**. Killed in one command.
4. **muPC initialization scale** — block-0 `q_proj` std is 0.0322 at depth 4 and
   0.0325 at depth 8. muPC initializes with std proportional to
   `1/L^exponent`, so depth 8 starts ~16% smaller — and training has washed
   that out entirely. Init is not the difference at step 3000.

## What survives, and it is thin

`residual_scale = 1.0 / (n_blocks_total ** mu_pc_exponent)` —
**0.7071 at 4 blocks, 0.5946 at 8** (`hybrid_block_pc.py:109`). This is the one
structural difference between the two configurations that training *cannot*
wash out, because it is a fixed multiplier rather than a learned parameter.

Whether a 16% attenuation causes total input-insensitivity at block 0 is not
obvious and I am not going to claim it at 01:20 on the strength of it being the
last candidate standing. Being the only survivor of an elimination round is not
evidence; it is a reason to test it.

**Cheapest decisive test:** run depth 8 with `mu_pc_enabled=False`. That removes
both the residual scaling and the depth-dependent init in one change, so a clean
result implicates muPC as a whole; if the collapse persists, muPC is exonerated
and the cause is elsewhere entirely. One variable, 45 minutes.

Note this is the third time `mu_pc_exponent` has come up and the second time I
demoted it. It was demoted on both occasions for defensible reasons — it governs
PC substrate rates rather than the backprop gradient that diverged, and it
governs init which has now been shown to wash out. The residual scale is a third
thing that neither of those arguments covered.
