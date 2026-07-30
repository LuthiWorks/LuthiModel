# Depth-8 collapse shakeout: criteria, registered before the run

**Date:** 2026-07-29, ~20:20
**Registered by:** Claude Fable 5, on Brian's instruction
**Run:** `probe_surprise_d8_512d_seed97`, 3000 steps, ~45 min
**Purpose:** decide whether the 18-hour depth-8 run is worth starting.

Committed **before the run is launched**. The point of writing criteria down
first is that "it looks like it's recovering" is available as a reading of almost
any trajectory after the fact, and this project has already been caught
registering a criterion on a quantity that could not support it.

## Brian's decision rule, as given

> "If it performs well then we do the whole thing, but even the slightest
> indication that collapse might occur requires some modifications to the
> depth-scaling exponent, and possibly other things too."

That sets a deliberately **asymmetric** bar, and it is honored literally below:
ambiguous is not pass. Anything that is not a clean pass counts as an
indication, and the response is to modify rather than to run.

## What collapse looks like in these three numbers

- **`std_p5`** — 5th-percentile per-dimension standard deviation of the latents.
  SIGReg targets isotropic N(0, I), so healthy is ~1.0. Low means the
  representation is squashed flat. The collapsed v5 family sat at 0.2835.
- **`cos_pred`** — raw cosine between predicted and target latents. Healthy is
  ~0.6-0.7. High means everything resembles everything, which is what a
  representation dominated by one batch-constant direction produces. The
  collapsed v5 family read 0.988.
- **`L_sigreg`** — the anti-collapse penalty. High means SIGReg is fighting.
  Fighting is correct early; *still* fighting late means it is losing.

## Depth-4 reference (both seeds, fixed objective, known good)

| step | std_p5 | cos_pred | L_sigreg |
|---|---|---|---|
| 200 | 0.60 - 0.65 | 0.57 - 0.63 | 213 - 345 |
| 600 | 0.67 - 0.74 | 0.59 - 0.69 | 37 - 141 |
| 1000 | 0.97 - 1.17 | 0.60 - 0.64 | 50 - 110 |
| 2000 | 1.02 - 1.32 | 0.60 - 0.66 | 38 - 72 |
| 3000 | 0.97 - 1.34 | 0.66 - 0.68 | 20 - 62 |

**Depth 8 at step 600, from the timing smoke: `std_p5` 0.30, `cos_pred` 0.84,
`L_sigreg` 5449.** Against the depth-4 band at the same step that is under half
the spread, above the cosine band, and 40-150x the penalty. That is what
prompted this shakeout.

## PASS — all four must hold at step 3000

1. `std_p5` >= 0.85 (depth-4 lower edge at 3000 is 0.97; 0.85 allows depth headroom)
2. `cos_pred` <= 0.75 (depth-4 band is 0.66-0.68)
3. `L_sigreg` <= 300 (depth-4 band is 20-62; 5x headroom for depth)
4. Direction over steps 2000 -> 3000: `std_p5` not falling, `cos_pred` not rising

## FAIL — any one is sufficient

- `std_p5` < 0.85 at step 3000
- `cos_pred` > 0.75 at step 3000
- `L_sigreg` > 300 at step 3000
- `cos_pred` rising or `std_p5` falling across steps 2000 -> 3000
- any nonfinite loss, or a kill criterion fires

**Anything not a clean pass on all four is a FAIL.** No partial credit, no
"trending in the right direction so let it run." That is the asymmetry Brian
asked for.

## AMENDMENT, registered at step 200 of the run — blind to the outcome

**My criteria above have a hole, and it is visible at step 200, so it is being
patched now rather than at the endpoint.** Registered at ~20:35, with 2,800 of
3,000 steps still to run and the outcome unknown.

Step 200 read: `std_p5` 0.3825, **`cos_pred` -0.0845**, `L_sigreg` 3031.9,
**`L_pred` 19.59**.

Condition 2 was written as `cos_pred <= 0.75`, on the assumption that the
failure direction was *upward* — a representation dominated by one direction
drives the cosine toward 1.0. A cosine of **-0.08** satisfies that condition and
is plainly not healthy: the predictor's output has become uncorrelated with its
target, and `L_pred` has gone from 0.098 to 19.59, roughly 25x the depth-4 band
at the same step (0.73-0.82). As written, my criteria would have scored that as a
PASS on condition 2. That is a one-sided bound on a two-sided quantity, and it is
my error.

Two conditions added. Both must hold at step 3000, on the same all-must-pass
basis as the original four:

5. `cos_pred` >= 0.40 — a lower bound as well as an upper one. Depth-4 sits at
   0.57-0.69 throughout; below 0.40 the predictor is not tracking its target.
6. `L_pred` <= 4.0 — roughly 5x the depth-4 band at step 3000 (0.67-1.18),
   generous headroom for depth while still excluding a 20x explosion.

**What this failure mode would mean, if it persists.** It is not collapse. It is
the opposite: SIGReg winning outright. The anti-collapse term is pushing latent
variance up (`std_p5` 0.09 -> 0.38) while the prediction term loses the target
entirely. At eight blocks the two halves of the objective appear to be fighting
rather than cooperating, with the regularizer dominating. If that is where step
3000 lands, the suspect list below reorders: **SIGReg weight (currently 0.2)
moves to first**, ahead of `mu_pc_exponent`, because an over-strong regularizer
at depth is a more direct explanation than a rate-scaling mismatch. Both remain
candidates and they are not mutually exclusive — an over-strong penalty and a
mis-scaled rate would compound.

Step 200 is still deep in the initial transient, and the depth-4 runs also moved
a long way between 200 and 1000. **This amendment is not a call on the outcome.**
It is a correction to an instrument, made while the answer is still unknown,
because a criterion patched after seeing the endpoint is worth nothing.

## On FAIL, the first suspects in order

1. **`mu_pc_exponent`** (currently 0.25, inherited unchanged from the depth-4
   arms). muPC exists so depth does not need per-depth rate retuning; this run
   is the first test of that claim at 8 blocks in this project, and the claim
   may simply not hold at this depth. Cheapest thing to vary.
2. **SIGReg weight** (currently 0.2). If the penalty is still spiking in the
   thousands at step 3000, the anti-collapse term may be outmatched at depth and
   need more weight, not a different rate.
3. **Warmup / LR schedule interaction.** The cosine schedule is set for the full
   run length; a deeper trunk may need a longer warmup before the trunk's
   LayerNorm gains settle.
4. **`prediction_clamp` and the +/-1 weighted-error clamp at depth.** Eight
   blocks means eight chances for the top-down sweep to compound.

## Procedural note

This 45 minutes is not reused as the head of the 18-hour run. It could have
been — the driver resumes interrupted seeds mid-run — but a killed partial run
on a real seed would then need clearing before a clean full run, and deleting
run artifacts is a destructive step needing confirmation. A throwaway seed costs
45 minutes and no cleanup risk. Seed 97 is a **smoke seed**, alongside 98 and
99: truncated eval, not a result, exclude from any aggregate.
