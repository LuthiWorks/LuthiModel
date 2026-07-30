# Is muPC the cause of the depth-8 collapse? Prediction registered before the run

**Date:** 2026-07-30, ~01:35
**Run:** `probe_surprise_d8_nomupc_512d_seed94` (stage 16), 3000 steps, ~45 min
**Control:** `probe_surprise_d8_512d_seed96` (stage 14, already run)
**One variable:** `mu_pc_enabled` True -> False. Nothing else differs.

## What is being tested

Depth 8 has stopped responding to its input: within-batch pairwise cosine 0.970
against depth 4's 0.023, with per-block offset dominance already 0.989 at block
**0**. Four candidate causes have been eliminated by measurement (projection-head
bias, trunk LayerNorm gains, positional-vs-token embedding scale, muPC init
scale).

One structural difference survives, and it is the only one training cannot wash
out because it is a fixed multiplier rather than a learned parameter:

    residual_scale = 1.0 / (n_blocks_total ** mu_pc_exponent)
    depth 4: 1/4^0.25 = 0.7071
    depth 8: 1/8^0.25 = 0.5946      (hybrid_block_pc.py:109)

Disabling muPC sets `residual_scale = 1.0` and skips the depth-scaled init.

**Being the last candidate standing is not evidence.** It is a reason to run the
test, and the test is cheap.

## Primary readout — a better metric than last time

The previous two gates were scored on offset dominance, which has an unresolved
train-vs-eval discrepancy (my post-hoc measurement reads 0.99 where the training
instrument logged 0.56 for the same run). **Within-batch pairwise cosine does not
have that problem**: it is measured identically at both depths, by one procedure,
and it directly asks the question that matters — does the encoder distinguish its
inputs?

| reference | within-batch pairwise cosine |
|---|---|
| depth 4 (healthy) | **0.023** |
| depth 8, muPC on (control) | **0.970** |

**Registered prediction.** Measured on the final checkpoint with the same
procedure used for both numbers above:

- **CONFIRMED (muPC is the cause):** <= **0.30**
- **REFUTED:** >= **0.70**
- **AMBIGUOUS:** 0.30 - 0.70 — treat as refuted for decision purposes.

**Secondary, supporting only:** per-block offset dominance at block 0 (0.989 in
the control, 0.157 at depth 4). Reported but not decisive on its own, given the
instrument discrepancy noted above.

## What CANNOT be read from this run

**Capability.** The clip of 1000 is carried over unchanged and is independently
known to kill capability (43% of steps clipped, probe lift 1.03x against depth
4's 4.67x). Held-out NMSE and probe lift are recorded for completeness and are
not evidence either way. Stated in advance so a dead probe lift is not later
read as evidence against a fix that worked.

## A confound to report alongside, not to hide

Removing the residual attenuation raises every block's contribution to the
residual stream by ~1.68x (0.5946 -> 1.0). That will change gradient magnitudes,
which changes **how often the clip of 1000 binds**. The clip is held fixed for
one-variable discipline, but its *effect* is therefore not held fixed.

So: `grad_norm` distribution and clip engagement rate will be reported with the
result. If clip engagement changes substantially, the run is still a valid test
of "does disabling muPC fix the collapse" but a poor test of "why" — and that
distinction goes in the verdict rather than being discovered later.

## Standing discipline note

Five gates have now been registered in ~30 hours. Four defects were found in the
first two (two one-sided bounds on two-sided quantities; a threshold set on the
raw cosine when the mean-centered version was already logged; and one structural
— every condition measured stability when the question was whether the model
learns). The third and fourth gates adopted: one primary metric, numeric bounds
on both the confirm and refute sides, the ambiguous band pre-assigned to refute,
and an explicit list of what the run cannot answer. That structure caught a 15%
effect that would otherwise have been written up as progress. It is kept here.
