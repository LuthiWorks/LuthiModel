# The SIGReg projection-bias hypothesis: prediction registered before the run

**Date:** 2026-07-30, ~00:10
**Run:** `probe_surprise_d8_noproj_512d_seed95` (stage 15), 3000 steps, ~45 min
**Control:** `probe_surprise_d8_512d_seed96` (stage 14, already run)
**One variable:** `sigreg_projection` "linear" -> "none". Nothing else differs.

## The hypothesis

Depth 8 is ~4x more offset-dominated than depth 4 (median 0.561 vs 0.143-0.150)
— the representation is dominated by a single batch-constant direction. SIGReg
targets zero-mean isotropic N(0, I), so an offset is precisely what it should
penalize, and it is visibly straining (`L_sigreg` in the hundreds). So why does
the offset survive?

**Proposed:** SIGReg is applied to the output of a per-modality `nn.Linear`
projection head (`jepa_loss.py:260`), **and that Linear has a bias.** A bias is
free to absorb the batch-mean offset, presenting SIGReg with centered latents
while the trunk retains the offset. SIGReg would then be satisfied on the
projected view and blind to the trunk's actual geometry.

If true, this is structurally **the same defect as the BatchNorm removed on
2026-07-28**: a learnable layer standing between SIGReg and the quantity it
exists to constrain. The difference is only in degree — BatchNorm subtracted
the mean *and* divided by the std, unconditionally; a Linear bias can subtract
a mean only if doing so reduces the loss, which it does.

Why it would have gone unnoticed until now: at depth 4 the offset is 0.14 and
there is little for a bias to absorb. At depth 8 it is 0.56.

`sigreg_projection="none"` sets the head to `nn.Identity()`, so SIGReg sees
trunk latents directly. The option already exists (added 2026-07-28); this run
uses it.

## The readout — and what CANNOT be read

**Primary, and the only thing this run decides:**
`offset_dominance_target`, median over the run's light-cadence records.

| reference | median |
|---|---|
| depth 4 (seeds 45/46) | 0.143 - 0.150 |
| depth 8, linear projection, clipped (seed96) | **0.561** |
| depth 8, linear projection, unclipped (seed97) | 0.719 |

**Registered prediction.** If the bias is absorbing the offset, removing the
projection exposes it to SIGReg and the offset should fall substantially:

- **CONFIRMED:** median `offset_dominance_target` <= **0.35**
  (roughly halfway from 0.561 to depth 4's 0.15; a real effect, not noise)
- **REFUTED:** median >= **0.50** (essentially unchanged from 0.561)
- **AMBIGUOUS:** 0.35 - 0.50 — treat as refuted for decision purposes, and say
  so rather than reaching for the favourable reading.

**Secondary, supporting only:** `predictor_cosine_centered_mean` should rise
from 0.484 toward depth 4's 0.62-0.68. Not decisive on its own.

**CANNOT be read from this run: capability.** Held-out NMSE and probe lift are
uninterpretable here, because the clip of 1000 is carried over unchanged and is
independently known to kill capability (43% of steps clipped, probe lift 1.03x
at depth 4's 4.67x). A dead capability number in this run neither confirms nor
refutes the hypothesis. Stating this in advance so that a dead probe lift is not
later read as evidence against a projection fix that may have worked.

The clip is carried over *on purpose*: changing it too would leave two variables
moving and make the result unattributable. Fixing the clip is the next run, not
this one.

## Why the prediction is registered before the run

Four separate defects were found in the criteria I wrote for the two depth-8
shakeouts, in the space of one evening:

1. `cos_pred <= 0.75` — one-sided bound on a two-sided quantity; a broken
   cosine of -0.08 would have scored a pass (caught mid-run, blind to outcome).
2. `std_p5 >= 0.85` — the identical error, not generalized from (1) in the same
   edit; a scale explosion to 16.05 scored a pass.
3. The threshold was set on the *raw* predictor cosine when the mean-centered
   version — built the previous day precisely because the raw one is confounded
   by offset — was already available and logged.
4. Structural: every registered condition measured *stability* when the question
   was whether the model *learns*. A stable-but-dead model passed four of six,
   and the two numbers that actually decided it were not in the criteria at all.

The pattern in all four is the same: the criterion was written to match the
failure I was picturing, and reality failed differently. Hence a single primary
metric here, with numeric bounds on both the confirm and refute sides, and an
explicit list of what this run cannot answer.
