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
