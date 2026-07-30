# Depth-8 shakeout: VERDICT — FAIL, with a diagnosed cause

**Date:** 2026-07-29, ~21:15
**Run:** `probe_surprise_d8_512d_seed97`, 3000 steps, 0.59 h
**Criteria:** `docs/research/2026-07-29_depth8-collapse-shakeout-criteria.md`
(registered before launch, amended at step 200 blind to the outcome)

**The 18-hour run must not start at these settings.** The shakeout cost 45
minutes and found a hard divergence at step ~2250.

## Scoring against the registered criteria

| # | condition | value at 3000 | result |
|---|---|---|---|
| 1 | `std_p5` >= 0.85 | 16.049 | passes *(see below)* |
| 2 | `cos_pred` <= 0.75 | 0.8611 | **FAIL** |
| 3 | `L_sigreg` <= 300 | 5572.9 | **FAIL** |
| 4 | direction 2000->3000: `cos_pred` not rising | 0.247 -> 0.861 | **FAIL** |
| 5 | `cos_pred` >= 0.40 | 0.8611 | passes |
| 6 | `L_pred` <= 4.0 | 9417.4 | **FAIL** |

**FAIL on four of six.** Unambiguous.

### My second one-sided-bound error, disclosed

Condition 1 reads `std_p5 >= 0.85` and **scores a pass at 16.049**, which is a
representation whose scale has exploded 16x past its target. This is the *same
error* I caught and patched for `cos_pred` at step 200 — a one-sided bound on a
two-sided quantity — and I failed to generalize the fix to `std_p5` in the same
edit, despite the target being ~1.0 and stated as such three lines above it in
the criteria doc.

It did not change the verdict, and the reason is structural rather than lucky:
the criteria were **conjunctive** (all must pass), so one defective condition
could not manufacture a pass. That is an argument for keeping the all-must-pass
form on future gates rather than scoring on a weighted total.

For any future use, condition 1 should read `0.85 <= std_p5 <= 1.5`.

## What happened

| step | L_pred | L_sigreg | std_p5 | cos_pred | grad_norm |
|---|---|---|---|---|---|
| 1800 | 16.5 | 1912 | 0.953 | 0.321 | 1107 |
| 2100 | 4.3 | 526 | 0.821 | 0.800 | 640 |
| 2200 | 28.2 | 5011 | 0.588 | 0.151 | 2158 |
| 2300 | **411.2** | 6492 | 0.862 | **0.967** | 2941 |
| 2500 | 879.2 | 5761 | 1.536 | 0.985 | — |
| 2800 | 10438.0 | 5959 | 0.607 | 0.990 | — |
| 3000 | 9417.4 | 5573 | **16.049** | 0.861 | 5555 |

Two phases:

1. **Steps 100-2200: sustained, non-damping oscillation.** `L_pred` swung across
   0.10-19.6, `cos_pred` across -0.08 to 0.80. Amplitude did not decay — sd over
   steps >1000 was 3.88 for `L_pred` against 5.42 before, and `std_p5`'s sd
   *grew* (0.192 -> 0.229). A startup transient damps; this did not.
2. **Step ~2250 onward: runaway.** `cos_pred` pinned at 0.985-0.990 and stayed
   there while `L_pred` grew four orders of magnitude. Note that 0.988 is the
   *exact* signature of the collapsed v5 family. Depth 8 reached the same
   one-direction collapse the 07-28 objective fix was meant to prevent, by the
   opposite route: not quiet shrinkage, but alignment plus scale explosion
   together.

For scale: no NaN or inf ever appeared. It diverged smoothly.

## Diagnosis: gradient magnitude at depth, and no clipping

| | depth 4 | depth 8 |
|---|---|---|
| grad_norm median | 28.4 | **1065** |
| grad_norm max | 374 | **8645** |
| learning rate | 3e-4 | 3e-4 (unchanged) |

**Gradients are ~37x larger at 8 blocks and the learning rate was not changed.**
The effective step size is therefore roughly 37x too large, which is exactly the
signature observed: oscillation from the start, then runaway once grad_norm
crossed ~2000 around step 2250.

**There is no gradient clipping anywhere in the JEPA runner.** `grep` for
`clip_grad`, `grad_clip`, `clip_norm`, `max_norm` across `jepa_runner.py` and
`jepa_pilot_driver.py` returns nothing. `grad_norm` is *logged* every 100 steps
and has never been *acted on* — the instrument existed and nothing consumed it.
At depth 4 that was harmless because gradients were small. At depth 8 it is the
failure.

Note also that the LR barely moved (2.999e-4 -> 2.897e-4): the cosine schedule
is sized for 24,014 steps, so a 3000-step run sits at essentially peak LR
throughout. The full 72,042-step run would *also* pass through peak LR in its
first epoch, so this is a live risk for the real run and not an artifact of
truncation.

### Retracting my step-200 hypothesis

At step 200 I proposed that the failure was "SIGReg winning outright" — the
regularizer overwhelming the prediction term — and moved SIGReg weight to the
top of the suspect list. **That was wrong.** The endgame shows SIGReg at 5573,
straining and losing, while the representation collapsed to one direction
anyway. An over-strong regularizer does not permit `cos_pred` = 0.99. The
hypothesis fit two data points and did not survive twelve.

## Recommended fixes, reordered by what the evidence supports

1. **Add gradient clipping.** It does not currently exist. This is the most
   direct fix and it is a plain omission rather than a tuning question.
2. **Lower the learning rate at depth**, or make it depth-aware. 3e-4 was set at
   4 blocks against gradients 37x smaller. Clipping may make this unnecessary;
   both together risk under-training, so vary one at a time.
3. **`mu_pc_exponent` (0.25).** Still untested at 8 blocks, but demoted: it
   scales the *PC substrate* rates, not the backprop gradient that diverged here.
   Worth checking after the gradient problem is fixed, not before.
4. **SIGReg weight (0.2).** Demoted to last on the evidence above.

## Secondary finding: no probe run today had any kill protection

`kill_criteria.warmup_batches = 5000`. Every probe run today was 3000-4000
steps. **No kill criterion could fire in any of them**, and this diverged run
reported `outcome: completed`, `admissible: True`, with heldout NMSE **5.675**
against a healthy ~0.57 and `l_pred_mean` 9576.9.

That is a silent success of exactly the kind this project keeps finding: a run
that destroyed itself, reported completion, and would have been picked up by any
aggregate as a valid data point. The four `probe_surprise` / `probe_storefix`
runs at 4000 steps were also entirely unprotected — they happened to be healthy,
and we had no safety net and did not know it.

Recommended: an absolute divergence guard that does **not** wait for
`warmup_batches` — e.g. abort if heldout NMSE exceeds 2.0, or if loss exceeds
some multiple of its own running median. The existing criteria are all relative
and warmup-gated, which is the right design for detecting subtle collapse and
the wrong one for catching a run that has already blown up.
