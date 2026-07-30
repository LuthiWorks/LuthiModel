# muPC is the cause of the depth-8 collapse: CONFIRMED

**Date:** 2026-07-30, ~02:30
**Run:** `probe_surprise_d8_nomupc_512d_seed94`, 3000 steps, 0.58 h
**Control:** `probe_surprise_d8_512d_seed96` (muPC on)
**Prediction registered before the run:** `2026-07-30_mupc-hypothesis.md`

## Verdict

**CONFIRMED**, on the registered primary metric and by a wide margin.

| within-batch pairwise cosine | value |
|---|---|
| registered CONFIRMED bound | <= 0.30 |
| registered REFUTED bound | >= 0.70 |
| depth 4 (healthy reference) | 0.0231 |
| depth 8, muPC **on** (control) | 0.9704 |
| **depth 8, muPC off (test)** | **0.0111** |

The depth-8 encoder responds to its inputs again — and by this measure slightly
better than depth 4 does.

## Every other axis agrees

**Per-block offset dominance** — the collapse was total at block 0 with muPC on:

| block | depth 4 | d8 muPC on | d8 muPC off |
|---|---|---|---|
| 0 | 0.157 | 0.989 | **0.165** |
| 1 | 0.166 | 0.9997 | 0.194 |
| 2 | 0.176 | 0.9998 | 0.186 |
| 3 | 0.164 | 0.9999 | 0.167 |
| 4 | — | 0.9999 | 0.158 |
| 5 | — | 0.9998 | 0.141 |
| 6 | — | 0.9995 | 0.134 |
| 7 | — | 0.9969 | **0.119** |

Flat and healthy at every block, matching depth 4 and drifting slightly
*downward* with depth rather than upward.

**Gradient magnitude.** This is the finding that reframes the whole evening:

| | grad_norm median | clip engagement |
|---|---|---|
| depth 4 | 28.4 | n/a (no clip) |
| depth 8, muPC on | 828.8 | 43% of steps |
| **depth 8, muPC off** | **39.0** | **3% of steps** |

The "gradients are ~37x larger at depth 8" result from 2026-07-29 — which
prompted adding gradient clipping in the first place — **was muPC's doing.**
Without muPC, depth-8 gradients (39.0) are comparable to depth-4's (28.4). The
divergence, the huge gradients, the offset dominance and the input-insensitivity
all trace to one cause.

**SIGReg**, the objective's own distance-from-isotropic measure, fell from ~965
to 10-26 and held there — inside depth 4's healthy 50-110 band and below it.

## Capability: readable after all, and good

Registered in advance as **not readable**, because the carried-over clip of 1000
was known to kill capability in the control (43% of steps clipped, probe lift
1.03x).

**That confound did not materialize**, and the reason is itself part of the
result: with muPC off, gradients fell 21x, so the fixed clip now binds on only
**3%** of steps. It is effectively inactive in this run.

| | NMSE | probe lift over own floor |
|---|---|---|
| depth 4 | 0.52 - 0.60 | 4.67x - 4.80x |
| depth 8, muPC on | 1.505 | 1.03x |
| **depth 8, muPC off** | **0.5569** | **4.19x** |

NMSE lands **inside depth 4's band** and probe lift is 4.19x against depth 4's
4.67-4.80x. This is a working model, where the muPC arm was worse than
predicting the mean.

Stated precisely: the *test* run's capability is essentially unconfounded (clip
binds 3%). The *control's* is not (43%), so test-vs-control capability comparison
remains confounded and the meaningful comparison is test-vs-depth-4. Depth 4 ran
with no clip at all, so even that comparison is not perfectly matched — but a 3%
engagement rate is close enough for the numbers to mean something.

## What this does NOT establish

**Which half of muPC.** Disabling it removes the residual scaling
(`residual_scale` 0.5946 -> 1.0) *and* the depth-scaled init, together and by
design. Init was previously shown to wash out by step 3000 (block-0 `q_proj` std
0.0322 at depth 4 vs 0.0325 at depth 8), which makes the residual scale the
likely half — but that is an inference, not a measurement. Separating them needs
`mu_pc_exponent=0.0`, which keeps the init scaling and removes the attenuation.

**Why a 1.68x attenuation does this.** I flagged before the run that a 16%
residual difference causing total input-insensitivity was not obviously
sufficient. It evidently is sufficient, and the mechanism is not explained by
this run. Worth understanding before muPC is used at any depth.

**Whether muPC is harmful at depth 4.** Depth 4 runs muPC at
`residual_scale` 0.7071 and is healthy (offset 0.15), so this is a
depth-*interaction*, not a blanket defect. Whether depth 4 would also improve
without muPC is untested and is a cheap, separate question.

## Consequences

1. **The 18-hour depth-8 run is now plausible** — first time tonight. It should
   run with `mu_pc_enabled=False`, and the gradient clip is probably
   unnecessary (3% engagement) though harmless. Recommend keeping it as a safety
   net rather than removing a guard on the strength of one run.
2. **muPC's registration should be revisited.** It was adopted for depth scaling
   and it is the thing that broke depth. Any future use needs the mechanism
   understood first.
3. **The gradient-clipping work from 2026-07-29 was treating a symptom.** It was
   still worth adding — the runner had no clipping at all and `grad_norm` had
   been logged and ignored since EMIT_BATCH_1 — but it was not the cause, and
   the verdict that named it "the diagnosis" was wrong about that. The
   divergence guards stand on their own merits.

## Method note

Five gates registered in ~30 hours; the first two contained four defects between
them. This one used the structure adopted after those: one primary metric chosen
for being measurable identically across arms, numeric bounds on both sides, the
ambiguous band pre-assigned to refute, and an explicit list of what the run could
not answer. The metric was deliberately switched away from offset dominance
because that instrument has an unresolved train-vs-eval discrepancy — scoring a
decision on a number I could not reconcile would have repeated the evening's
mistakes in a new place.

The pre-registered "capability is not readable" turned out to be unnecessary
caution. That is the correct direction for a registration to be wrong in.
