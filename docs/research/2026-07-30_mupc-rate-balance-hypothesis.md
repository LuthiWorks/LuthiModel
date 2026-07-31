# muPC rate balancing: mechanism, fix, and prediction registered before the run

**Date:** 2026-07-30, ~03:30
**Run:** `probe_surprise_d8_balanced_512d_seed92` (stage 17), 3000 steps, ~45 min
**Control:** `probe_surprise_d8_512d_seed96` (stage 14, collapsed)
**One variable:** `mu_pc_balance_rates` False -> True. Verified: exactly one
model-config difference.

## The problem this is trying to solve

Neither existing setting reaches production depth (36 blocks):

| | depth-8 trained outcome | activation growth at 36 blocks |
|---|---|---|
| muPC on (exponent 0.25) | **collapse** (within-batch cos 0.970) | 1.14 (controlled) |
| muPC off | healthy (0.011) | **3.92** (uncontrolled, still climbing) |

muPC's attenuation is doing real work. Disabling it trades a depth-8 collapse
for unbounded activation growth at production depth. Neither is a fix.

## Mechanism

`residual_scale` multiplies the block's output into the residual stream:

    x = x + residual_scale * attn_out
    x = x + residual_scale * ffn_out

By the chain rule, every **backprop-trained** parameter in the block receives a
gradient scaled by `residual_scale`. The **living FFN does not**: it
self-modifies locally inside the forward pass, from its own input and output,
*before* that multiplication is applied. Its update magnitude is set by
`pc_rate` and its input scale — and its input is `norm2(x)`, a LayerNorm.

**Measured, and this is the load-bearing fact:** PC-layer input RMS is
**1.0000 at depths 4, 8, 12 and 36**. LayerNorm pins it exactly; the residual
stream's magnitude never reaches the PC layer.

So muPC attenuates one half of a two-speed system. The PC substrate keeps
rewriting itself at full rate on a unit-scale input, while both its influence on
the output *and* the backprop path that could correct it are damped. The deeper
the trunk, the smaller `residual_scale`, and the worse the imbalance.

muPC (Innocenti et al. 2025) is derived for a network that is PC *throughout*,
where everything scales together. This trunk is a hybrid.

### Correcting the 2026-05-16 design doc

That doc identified the same interaction — "μPC's signal attenuation interacts
poorly with PC's living-weights property at depth" — and gave this mechanism:

> "At L=12 the residual is 2.4× weaker than at L=2, so the PC layer's input is
> dampened, its pred_error is dampened, and its self-modification is dampened."

**That is not what happens, and the measurement above refutes it.** `norm2`
stands between the residual stream and the PC layer, so the input is unit-scale
at every depth. The response at the time was to lower the exponent 0.5 -> 0.25,
which mitigates by weakening the attenuation rather than by fixing the
asymmetry — and the problem duly reappeared at depth 8.

The doc's *observation* was correct (NFF attenuates with depth): PC's influence
on the output really is attenuated by `residual_scale` and by muPC's init
scaling. Both are true at once — PC's influence falls with depth while its
internal churn does not.

## The fix

`mu_pc_balance_rates=True` scales `pc_rate` and `pred_learning_rate` by
`residual_scale`, so both halves are attenuated together and the two-speed
balance holds at any depth. muPC's depth-scale control is kept; the imbalance it
creates is removed.

Default False — bit-identical to every prior run.

## Registered prediction

Primary metric, same as the muPC gate, measured on the final checkpoint by the
same procedure: **within-batch pairwise cosine**.

| reference | value |
|---|---|
| depth 4 (healthy) | 0.023 |
| depth 8, muPC on, unbalanced (control) | 0.970 |
| depth 8, muPC off | 0.011 |

- **CONFIRMED:** <= **0.30**. The imbalance was the mechanism; muPC is usable at
  depth with balanced rates, and production depth is reachable.
- **REFUTED:** >= **0.70**. Balancing does not prevent the collapse, so the
  mechanism is wrong and attenuation harms by some other route.
- **AMBIGUOUS:** 0.30 - 0.70 — treated as refuted for decision purposes.

**Secondary, supporting:** per-block offset dominance (control is 0.989 at block
0; depth 4 is 0.157), and `L_sigreg` (control ~965; depth 4's healthy band is
50-110).

**Not readable:** capability. The clip of 1000 is carried over for one-variable
discipline. If gradients fall as they did in the muPC-off arm, the clip may
again turn out to be near-inactive and capability may become readable after all
— that will be reported, not assumed.

## What a CONFIRMED result would and would not establish

Would: the imbalance is the mechanism, and there is a configuration with both
depth-scale control and no collapse **at depth 8**.

Would **not**: that production depth works. That claim needs the depth ladder
re-run with balancing on, and at minimum a training shakeout at 12 blocks — the
glossary's "current decisive run" depth — before anything is claimed about 36.
Generalizing from the depth I happened to test is the exact error that produced
this document; the correction cost 4 minutes of probe time and would have cost
18 hours of run time.

---

# Addendum: power=+1 REFUTED, and the opposite direction registered

**Time:** ~04:00, with stage 17 still running. Registered before stage 18 runs.

## Stage 17 (power=+1, attenuate) is failing

Matched window, first 10 light records:

| | offset dominance | centered cosine |
|---|---|---|
| muPC off | 0.2063 | 0.5990 |
| power 0 (unbalanced control) | 0.5657 | 0.5087 |
| **power +1 (attenuate)** | **0.8277** | **0.1121** |

The fix made the thing it was fixing worse, on both measures.

## The mechanism I proposed is refuted by the ordering

My claim was that the **ratio** of PC rate to backprop attenuation drives the
collapse. Three points kill it:

| residual_scale | PC factor | PC/backprop ratio | offset dominance |
|---|---|---|---|
| 1.0 | 1.0 | 1.0 | 0.21 |
| 0.595 | 1.0 | 1.68 | 0.57 |
| 0.595 | 0.595 | 1.0 | **0.83** |

Rows 1 and 3 have the **same ratio** and land at 0.21 versus 0.83. The ratio is
not what drives it.

What does order the three is **total attenuation**: damping either path worsens
the offset, damping both is worst.

## The account that fits, and its prediction

A block computes `x = x_0 + s * sum(f)`, and the embedding `x_0` is **not**
scaled by `s`. If `x_0` carries a batch-constant component — the positional
embedding is identical across sequences by construction — then shrinking `s`
raises that constant's share of the representation. Less signal from the blocks,
same constant underneath.

**Withdrawn:** I earlier "killed" the positional-embedding hypothesis by showing
the pos/token norm ratio is 1.01 at both depths. That compares the size of the
embedding *tables* and says nothing about whether the positional component is
the batch-constant direction. It was not a valid test and I presented it as one.

**Prediction for power=-1 (amplify PC rates by 1/s):** more learned,
input-dependent structure per unit of attenuation should push offset dominance
back down.

- **CONFIRMED:** median offset dominance <= **0.40** on the matched window
  (below the 0.5657 unbalanced control, moving toward muPC-off's 0.21).
- **REFUTED:** >= **0.55** (no better than the unbalanced control).
- **AMBIGUOUS:** 0.40 - 0.55 — treated as refuted.

Secondary: within-batch pairwise cosine on the final checkpoint, same bounds as
before (confirm <= 0.30, refute >= 0.70).

## Honest note on what this experiment is

This is a **directional probe**, not a mechanism test. Brian's instruction was
to take the adjustment and go the same distance the other way, which is the
right move when a signed intervention produces a signed result: it establishes
whether the axis matters and which way it runs, without committing to a story
about why.

If power=-1 improves things, that supports "more block signal helps" and points
at the constant-component account — it does not confirm it. If it also makes
things worse, then both directions hurt, the axis is not monotonic, and the
whole rate-scaling family is the wrong lever.

---

# VERDICT, stage 17 (power=+1): REFUTED

**Time:** ~04:30. Scored on the registered primary metric.

| | within-batch pairwise cosine |
|---|---|
| depth 4 (healthy) | 0.0231 |
| d8 muPC off | 0.0111 |
| d8 power 0 (unbalanced control) | 0.9704 |
| **d8 power +1 (attenuate)** | **0.9807** |
| registered CONFIRMED | <= 0.30 |
| registered REFUTED | >= 0.70 |

**REFUTED at 0.9807**, and marginally worse than the control it was meant to
improve on. The two-speed-imbalance mechanism is wrong: matching the PC rate to
the backprop attenuation does not prevent the collapse, it deepens it slightly.

## Note on the interim readings

This run's intermediate windows swung twice and I reported both swings. At ~1000
steps the test looked clearly worse than control (offset dominance 0.83 vs 0.57);
at ~2000 the most recent five records reversed it (0.50 vs 0.77); the endpoint
landed at refuted anyway. None of the interim reads predicted the verdict, and
two of them pointed in opposite directions.

The registered metric at the final checkpoint was the only reading that
mattered, which is the entire reason it was registered. Recorded here because it
is the same failure mode as reading a point-in-time `update_ema` on a gated
drive -- a trap this project documented on 2026-07-29 and which I then walked
into conversationally the next night.

## What survives

`residual_scale` remains the only structural difference that separates a healthy
depth-8 trunk from a collapsed one, and PC-rate scaling in EITHER direction
fails to fix it (power=+1 refuted here; power=-1 in stage 18). The three-point
ordering that motivated the opposite-direction test still stands as data:

| residual_scale | PC factor | offset dominance |
|---|---|---|
| 1.0 | 1.0 | 0.21 |
| 0.595 | 1.0 | 0.57 |
| 0.595 | 0.595 | 0.83 |

But the endpoint metric is what decides, and by that measure power=+1 and
power=0 are indistinguishable (0.9807 vs 0.9704) while muPC-off is a different
regime entirely (0.0111). That pattern says the PC rate is not the axis at all
-- `residual_scale` is.

---

# VERDICT, stage 18 (power=-1, amplify): REFUTED — but not null

**Time:** ~05:20. Scored on the registered primary metric.

| | within-batch cosine | NMSE | probe lift |
|---|---|---|---|
| depth 4 healthy | 0.0231 | 0.5215 | 4.80x |
| d8 muPC off | 0.0111 | 0.5569 | 4.19x |
| d8 power 0 (control) | 0.9704 | 1.5054 | 1.03x |
| d8 power +1 (attenuate) | 0.9807 | 6.6805 | 2.89x |
| **d8 power -1 (amplify)** | **0.6384** | 1.7070 | 1.35x |

Registered: CONFIRMED <= 0.30, REFUTED >= 0.70, ambiguous 0.30-0.70 assigned in
advance to refuted. **0.6384 is refuted by that rule.**

But it is not the null result the interim predicted. 0.638 against 0.970 is a
substantial move, in the direction Brian proposed, and it says rate scaling is
**not** a dead axis — it is an axis with the wrong magnitude, or one that helps
without being sufficient.

## The interim was wrong for the third time, and I acted on it

At 18 of 30 light records, power=-1's offset dominance matched the control to
three decimals (0.5716 vs 0.5719). On that basis I told Brian the endpoint would
"almost certainly read ~0.97 like the others" and that the run's marginal value
was small. It read 0.638.

Tally for these depth-8 runs: stage 17's interims swung twice and contradicted
each other, then the endpoint contradicted both; stage 18's interim predicted a
null and the endpoint was the most informative result of the pair. **Three for
three.** Interim windows on these runs have zero demonstrated predictive value
and I have now been misled by them at every opportunity.

The registered-endpoint discipline is the only thing that has worked, and on
this occasion it survived my own argument against it.

## What this changes

Rate scaling was about to be written off. It should not be. The ordering on the
endpoint metric is:

    power +1 (attenuate)  0.9807
    power  0 (none)       0.9704
    power -1 (amplify)    0.6384

Monotonic in the right direction, with amplification producing by far the
largest single-knob improvement seen at depth 8 short of disabling muPC. That
suggests testing further amplification (power=-2, factor 1/s^2 = 2.83x at depth
8) before concluding the axis is exhausted.

Capability does not follow the same ordering: power=-1 sits at NMSE 1.707 and
lift 1.35x, barely better than the control's 1.505 / 1.03x and nowhere near
muPC-off's 0.557 / 4.19x. Note power=+1's lift of 2.89x is an artifact worth
ignoring — its NMSE is 6.68, a broken model, and lift over a degenerate floor is
not a capability signal.

So: geometry improves substantially with amplification, capability does not yet
follow. Those may reconnect at stronger amplification, or the geometry
improvement may be cosmetic. Untested either way.

## Standing conclusion on the depth question

Unchanged and worth restating: **muPC off is still the only configuration that
produces a working depth-8 model** (within-batch cosine 0.0111, NMSE 0.5569,
lift 4.19x — inside depth 4's band on every measure), and it is still the
configuration whose activation growth compounds to 3.92x at production depth.
The bind is not resolved. Nothing here licenses starting the 18-hour run.

---

# Stage 19 (power=-2): prediction registered before the run

**Time:** ~05:35. Brian's instruction: double the previous adjustment.
**Run:** `probe_surprise_d8_amp2_512d_seed90`, 3000 steps, ~45 min
**One variable vs stage 14:** `mu_pc_rate_power` -2.0. PC-rate multiplier
1.682x -> **2.828x** at depth 8.

## Why this is worth running

The endpoint ordering across the three tested powers is monotonic:

| power | PC-rate multiplier | within-batch cosine |
|---|---|---|
| +1 | 0.595x | 0.9807 |
| 0 | 1.0x | 0.9704 |
| -1 | 1.682x | **0.6384** |

Amplification produced the largest single-knob improvement seen at depth 8 short
of disabling muPC entirely. The axis is live; the open question is whether it is
still paying at 2.83x or has saturated.

## Registered prediction

Primary metric, unchanged: within-batch pairwise cosine on the final checkpoint.

- **CONFIRMED (axis still paying):** <= **0.45** — a clear further improvement on
  power=-1's 0.6384.
- **REFUTED (axis exhausted):** >= **0.60** — no better than power=-1.
- **AMBIGUOUS:** 0.45 - 0.60 — treated as refuted, per standing discipline.

Additional threshold, recorded separately: **<= 0.30 would mean depth 8 is
usable with muPC ON**, which is the outcome that would resolve the
depth-scale-control-versus-collapse bind. That is the prize; 0.45 is merely
"keep going".

## Capability is a registered WARNING, not a pass/fail

Power=-1 improved geometry (0.970 -> 0.638) while capability did NOT follow:
NMSE 1.707 vs the control's 1.505, against muPC-off's 0.557. Geometry and
capability came apart.

**If stage 19 improves the cosine further while NMSE degrades again, that is
evidence the geometry gain is cosmetic** -- the representation becoming less
uniformly-aligned without becoming more useful. Recorded in advance so a good
cosine with a bad NMSE is not reported as progress.

The clip of 1000 is carried over unchanged for one-variable discipline, so
capability remains partly confounded; the clip engagement rate will be reported.

## Interim readings will not be reported

Three for three, these runs' interim windows have pointed the wrong way --
twice contradicting each other within one run, once predicting a null that the
endpoint contradicted. No interim conclusions from this run. The endpoint metric
is the result.

---

# VERDICT, stage 19 (power=-2): CONFIRMED

**Time:** ~06:20. Scored on the registered primary metric.

Registered: CONFIRMED <= 0.45, REFUTED >= 0.60, ambiguous assigned to refuted.
**Result 0.4127 — CONFIRMED.**

| power | multiplier | within-cos | NMSE | lift | grad median | clip engaged |
|---|---|---|---|---|---|---|
| depth 4 | - | 0.0231 | 0.5215 | 4.80x | 28 | 0% |
| muPC off | - | 0.0111 | 0.5569 | 4.19x | 39 | 3% |
| +1 | 0.595x | 0.9807 | 6.6805 | - | - | - |
| 0 | 1.0x | 0.9704 | 1.5054 | 1.03x | 829 | 43% |
| -1 | 1.682x | 0.6384 | 1.7070 | 1.35x | 2427 | 73% |
| **-2** | **2.828x** | **0.4127** | **1.1742** | **2.04x** | **3591** | **93%** |

Monotonic on the primary metric across four settings. And the registered
capability WARNING did not fire: NMSE improved to 1.1742 (best of any muPC-on
arm) and probe lift to 2.04x. Geometry and capability moved together this time,
where at power=-1 they came apart.

## The confound that stops this ladder here

Clip engagement: 43% -> 73% -> **93%**. Gradient median: 829 -> 2427 -> **3591**,
against a fixed clip of 1000.

Amplifying the PC rates inflates gradients ~4.3x across the ladder, so this arm
is no longer a clean test of PC-rate amplification -- it is a test of
amplification PLUS a clip binding on nearly every step, at a clip value chosen
for a different regime (2026-07-29, when the depth-8 gradient median was 829).

**A power=-3 run would be almost entirely clip-dominated.** Running it would
produce a number attributable to the clip, reported as a PC-rate result. That is
the same confounding error this project has spent two days unwinding, so the
ladder stops here until the clip is addressed.

## Why the gradients grow, stated as a hypothesis and not a finding

Higher `pc_rate` means the living FFN's weights change more per forward pass.
The trunk's backprop gradient is computed against a substrate that is moving
faster underneath it, which plausibly raises gradient magnitude. Not measured;
recorded so it can be tested rather than assumed.

## What is and is not established

**Established:** PC-rate amplification monotonically improves depth-8 geometry
across four settings, and at 2.83x it also improves capability. The axis Brian
identified is real and was very nearly written off after stage 17.

**Not established:** that it reaches a working model. Power=-2 is at NMSE 1.174
and lift 2.04x against muPC-off's 0.557 and 4.19x and depth 4's 0.522 and 4.80x.
It is the best muPC-ON configuration found and it is still roughly half as good
as simply disabling muPC. The scale-control-versus-collapse bind is NOT resolved.

**Next, in order:**
1. Re-test the clip at this regime -- raise it to ~5000 (above the 3591 median)
   or disable it, at power=-2, one variable. Until that runs, everything at
   power<=-2 is confounded.
2. Only then consider power=-3.
3. The block-0 localization remains unstarted and remains the measurement most
   likely to explain WHY any of this works.

---

# Stage 20 (power=-4, clip 20000): registered before the run

**Time:** ~06:40. Brian's instruction, chasing a hunch.
**Run:** `probe_surprise_d8_amp4_512d_seed89`, 3000 steps, ~45 min.

## power=-4 is a landmark, not another rung

With `mu_pc_exponent = 0.25`, `residual_scale ** -4 == n_blocks` **exactly**, at
every depth:

| depth | residual_scale | PC multiplier at power=-4 |
|---|---|---|
| 4 | 0.7071 | 4.0 |
| 8 | 0.5946 | 8.0 |
| 12 | 0.5373 | 12.0 |
| 36 | 0.4082 | 36.0 |

The attenuation and the amplification cancel, leaving **PC rates that scale
linearly with depth**. That is a principled setting rather than a tuned one, and
it is a natural place for the axis to either resolve or stop.

## Two variables move, deliberately

Clip 1000 -> 20000 alongside power -2 -> -4. This breaks one-variable
discipline on purpose: at power=-2 the clip already engaged on **93%** of steps
at a gradient median of 3591, so it had stopped being a control and become the
dominant term. Amplifying to 8x would leave it binding essentially always, and
the run would measure the clip.

20000 is intended as a catastrophic-runaway backstop that does not shape
ordinary steps. **Engagement rate will be reported.** If it binds often, this
run is confounded and will be reported as confounded rather than as a result.

The divergence guards (loss vs frozen baseline; held-out NMSE > 2.0) are what
make a loose clip affordable -- they did not exist two days ago.

## Registered prediction

Primary metric: within-batch pairwise cosine on the final checkpoint.

Ladder so far: power 0 -> 0.9704, -1 -> 0.6384, -2 -> 0.4127.

- **CONFIRMED (axis still paying):** <= **0.30**
- **REFUTED (saturated or reversed):** >= **0.42** (no better than power=-2)
- **AMBIGUOUS:** 0.30 - 0.42 — treated as refuted.

Recorded separately: **<= 0.10 would put depth 8 within reach of the muPC-off
regime** (0.0111) and depth 4 (0.0231). That would resolve the
scale-control-versus-collapse bind and is the outcome that matters.

## Capability warning, carried forward

At power=-2 capability finally moved with geometry (NMSE 1.5054 -> 1.1742, lift
1.03x -> 2.04x). The target is muPC-off's NMSE 0.5569 / lift 4.19x.

**If the cosine improves again while NMSE regresses, the geometry gain is
cosmetic** and will be reported as such. Registered in advance for the second
time because it is the failure mode most likely to look like success.

## No interim readings

Unchanged: interim windows on these runs are three-for-three wrong. Endpoint
only.

---

# VERDICT, stage 20 — and a METRIC INVALIDATION

**Time:** ~07:30.

## Scored as registered: REFUTED. Re-scored on real text: CONFIRMED.

| arm | cos RANDOM tokens | cos REAL text | NMSE | probe lift |
|---|---|---|---|---|
| depth 4 healthy | 0.0231 | -0.0008 | 0.5215 | 4.80x |
| d8 muPC off | 0.0111 | 0.0038 | 0.5569 | 4.19x |
| d8 power 0 | 0.9704 | 0.3333 | 1.5054 | 1.03x |
| d8 power -1 | 0.6384 | 0.5667 | 1.7070 | 1.35x |
| d8 power -2 | 0.4127 | 0.1319 | 1.1742 | 2.04x |
| **d8 power -4** | **0.9528** | **-0.0294** | **0.8919** | **2.21x** |

## The metric was measured out of distribution

`within-batch pairwise cosine` was computed with `torch.randint` over the
vocabulary -- uniformly random token IDs -- against a model trained on
Gutenberg. That is far out of distribution, and it inverted the ranking.

On real text, power=-4 is the LEAST collapsed muPC-on arm and is
indistinguishable from both healthy references. On random tokens it read the
MOST collapsed. Four consecutive gates (stages 17, 18, 19, 20) were scored on
the random-token version.

**Why it looked trustworthy:** the depth-4 control read 0.0231 on random tokens,
matching expectations, so the instrument appeared validated. But a model can map
random garbage to a single point while discriminating real text perfectly well.
The control passing does not validate the input distribution.

**Real-text cosine orders capability monotonically** and the random-token
version does not:

| real cos | -0.029 | 0.132 | 0.333 | 0.567 |
|---|---|---|---|---|
| NMSE | 0.892 | 1.174 | 1.505 | 1.707 |

Perfect ordering across four arms. That is what a working geometry metric should
do, and it is the strongest evidence that the real-text version is the right
instrument and the random-token version was not.

## Re-scored verdicts

- **Stage 20 (power=-4): CONFIRMED.** Real-text cosine -0.0294 clears the
  registered <= 0.30 bound and also the separately-recorded <= 0.10 threshold
  for "within reach of the muPC-off regime".
- **Stage 19 (power=-2): still CONFIRMED** (real 0.1319).
- **Stage 18 (power=-1): still REFUTED** (real 0.5667 -- and it is the WORST arm
  on real text, which the random metric ranked second-best).

The ladder is not monotonic on the correct metric either: power -1 is worse than
power 0 on real text (0.567 vs 0.333). The axis has structure that neither
metric revealed cleanly, and only power -2 and -4 improve on the control.

## What power=-4 actually is

`residual_scale ** -4 == n_blocks` exactly, at every depth. muPC's attenuation
and this amplification cancel, leaving PC rates scaling LINEARLY WITH DEPTH.
Brian proposed it as "the next step"; it is the point where the two effects
annihilate.

It is the first configuration that keeps muPC's depth-scale control AND has
real-text geometry at healthy levels: cosine -0.0294 against depth 4's -0.0008,
with `residual_scale` still 0.5946 doing its job.

Its gradient median is 469 with the clip at 20000 engaging on **0% of steps** --
so unlike power -1 and -2 (73% and 93% clipped), this run is NOT clip-confounded.

## What is still NOT resolved

Capability has not caught up. NMSE 0.892 and lift 2.21x against muPC-off's
0.5569 and 4.19x, and depth 4's 0.5215 and 4.80x. Power=-4 is the best muPC-on
configuration by a clear margin and remains materially worse than simply
disabling muPC.

So the bind is narrowed, not resolved: there is now a configuration with both
depth-scale control and healthy geometry, but it still costs ~60% higher NMSE
than the configuration that surrenders scale control.

**Confound to state plainly:** stage 20 moved TWO variables (power -2 -> -4 and
clip 1000 -> 20000). The clip change was necessary -- at 93% engagement the old
clip was the dominant term -- but it means power=-4 versus power=-2 is not a
clean single-variable comparison. Power=-4 at clip 1000, or power=-2 at clip
20000, would separate them.

## Immediate consequence

Every geometry claim in this document and in
`2026-07-30_mupc-verdict.md` that rests on random-token cosine needs re-reading.
The muPC verdict's headline (muPC off 0.0111 vs muPC on 0.9704) survives -- both
reproduce on real text (0.0038 vs 0.3333) with the same direction -- but the
magnitudes were inflated by the out-of-distribution input.

---

# Stage 21 (power=-8): registered before the run

**Time:** ~07:50. **Run:** `probe_surprise_d8_amp8_512d_seed88`, 3000 steps.
**One variable vs stage 20:** `mu_pc_rate_power` -4 -> -8. Clip held at 20000.

Multiplier becomes `n_blocks**2` -- 64x at depth 8 (pc_rate 0.001 -> 0.064),
1296x at depth 36. Where power=-4 made PC rates scale linearly with depth, -8
makes them scale quadratically.

## PRIMARY METRIC CHANGED, deliberately: held-out NMSE

Real-text cosine is **already resolved** at power=-4: -0.0294, against depth 4's
-0.0008 and muPC-off's 0.0038. There is no headroom left on geometry -- further
amplification can only break it, not improve it. Scoring on a saturated metric
would make any result look like noise.

Capability is what still has room: NMSE 0.8919 at power=-4 against muPC-off's
0.5569 and depth 4's 0.5215.

So the gate scores the quantity that can still move, and guards the one that is
already won:

- **CONFIRMED (still paying):** held-out NMSE <= **0.80**
- **REFUTED (saturated or reversed):** >= **0.90** (no better than power=-4)
- **AMBIGUOUS:** 0.80 - 0.90 -- treated as refuted.

**Prize threshold:** NMSE <= **0.60** would match muPC-off (0.5569) while KEEPING
muPC's depth-scale control, which is the outcome that resolves the
scale-control-versus-collapse bind outright.

## GUARD: real-text cosine must stay healthy

Registered as a guard, not a target: **real-text within-batch cosine must remain
<= 0.10.** If NMSE improves while the cosine degrades past that, this is a
trade rather than a win and will be reported as one.

Measured with `scripts/measure_input_sensitivity.py`, which exists because the
random-token version of this measurement inverted four gates' worth of verdicts.

## Stability note

64x on `pc_rate` is a large step and instability is a real possibility. The
divergence guards (loss vs frozen baseline, held-out NMSE > 2.0) are the net; if
they trip, that is a result, not a failure. Clip engagement will be reported --
at power=-4 it was 0%, so if -8 pushes gradients into the clip the comparison
becomes confounded and will be reported as such.
