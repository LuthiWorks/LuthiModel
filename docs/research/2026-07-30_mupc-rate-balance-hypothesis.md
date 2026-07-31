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
