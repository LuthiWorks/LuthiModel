# The depth-8 offset, localized: block 0 cannot cancel what it is not allowed to scale

**Date:** 2026-07-31, ~08:50
**Script:** `scripts/localize_offset_in_block.py` (real text, forward hooks only)
**Runs:** muPC-off (healthy), power 0 (collapsed), power -4 (best muPC-on)

## Two earlier claims of mine are corrected here

1. **"The embedding carries almost no batch-constant component (0.065)."** Wrong.
   That was measured on random tokens. On **real text it is 0.58**, in every run.
   Real passages share enormous common structure; uniformly random token IDs do
   not. The offset is **in the input**, not manufactured by the trunk.
2. **"The blocks manufacture the offset; it is total at block 0."** Also wrong,
   and for the same reason. A healthy trunk *removes* the offset. What varies
   between runs is whether it succeeds.

## The measurement

Offset dominance at `x0` (block input), `x1` (post-attention residual), `x2`
(post-FFN residual = block output).

**muPC off — NMSE 0.5569.** Strips it in block 0, then holds flat:

| block | x0 | x1 post-attn | x2 post-ffn |
|---|---|---|---|
| 0 | 0.5845 | 0.2686 (**-0.316**) | 0.1158 (**-0.153**) |
| 1 | 0.1158 | 0.0938 | 0.0702 |
| 2-7 | ~0.06 | ~0.06 | ~0.06 |

**power 0 — NMSE 1.5054.** Block 0 does nothing; the offset then grows:

| block | x0 | x1 post-attn | x2 post-ffn |
|---|---|---|---|
| 0 | 0.5789 | 0.5864 (**+0.008**) | 0.5890 (**+0.003**) |
| 2 | 0.4319 | 0.4684 | 0.7893 (+0.321) |
| 4 | 0.9398 | 0.8243 | 0.9543 |
| 7 | 0.7551 | 0.7070 | 0.7080 |

**power -4 — NMSE 0.8919.** Block 0 *adds*; blocks 6-7 strip it late:

| block | x0 | x1 post-attn | x2 post-ffn |
|---|---|---|---|
| 0 | 0.5711 | 0.8226 (**+0.252**) | 0.9108 (+0.088) |
| 4 | 0.9870 | 0.9785 | 0.9899 |
| 7 | 0.8783 | 0.5041 (**-0.374**) | 0.3885 (-0.116) |

## The mechanism, from the arithmetic

    x1 = x0 + residual_scale * attn_out

To cancel an offset carried in `x0`, the attention output must supply
`-offset / residual_scale`. At `s = 1.0` that is a fair fight. At `s = 0.5946`
the corrector must be **1.68x larger** to achieve the same cancellation, because
**`x0` enters unattenuated while everything that could correct it is
attenuated**.

That is why block 0 specifically fails: it is the first place a large shared
input component meets an attenuated corrector, and it is the only block whose
input offset comes straight from the embedding rather than from an already-
processed stream.

It also explains why `mu_pc_rate_power` helps later blocks and not block 0:
block 0's canceller is **attention**, which is backprop-trained and completely
untouched by the PC rate knob. Amplifying PC plasticity gives the deeper blocks
enough authority to strip the offset at the end (power -4, blocks 6-7), which is
a workaround rather than a fix -- the representation carries a ~0.99 shared
component through five blocks before anything removes it.

## Proposed fix, untested

**Scale the embedding by `residual_scale` before the trunk**, so the input and
its correctors live at the same scale:

    x = residual_scale * (token_emb + pos_emb)
    x = x + residual_scale * attn_out      # unchanged
    ...

Then `x1 = s * (x0 + attn_out)` and cancellation is scale-matched exactly as at
`s = 1.0`. Offset dominance is a ratio and is invariant to the overall factor,
so this should reproduce muPC-off's block-0 stripping **while keeping muPC's
per-block attenuation and its depth-scale control** -- which the depth ladder
showed is real and non-negotiable (activation growth 1.14 flat to 36 blocks with
muPC; 3.92 and climbing without it).

Alternatives not preferred: exempting block 0 from attenuation (arbitrary, and
leaves the same mismatch at every later block whose input carries structure);
normalizing `x0` (LayerNorm removes the per-token feature mean, not a
batch-shared direction, so it does not address this).

**This is a hypothesis with a mechanism and an arithmetic derivation, not a
result.** Five mechanisms have been proposed and refuted across 2026-07-29/30
by exactly this kind of reasoning. It is cheap to test: one flag, one
3000-step run at depth 8, scored on real-text NMSE against power=-4's 0.8919
and muPC-off's 0.5569.

---

# Stage 22 (embedding scaling): registered before the run

**Time:** ~09:05. **Run:** `probe_surprise_d8_embscale_512d_seed87`, 3000 steps.
**One variable vs stage 20** (`probe_surprise_d8_amp4`, the best configuration
found): `mu_pc_scale_embedding=True`. Clip held at 20000.

`h = residual_scale * (token_emb + pos_emb + modality_emb)`, applied once where
the trunk stream is assembled, so `x1 = s * (x0 + attn_out)` and cancellation is
scale-matched as at `s = 1.0`. muPC's per-block attenuation is untouched.

## Registered prediction

**Primary: held-out NMSE.**

| reference | NMSE |
|---|---|
| depth 4 | 0.5215 |
| muPC off (surrenders depth-scale control) | 0.5569 |
| power -4 (base for this run) | 0.8919 |

- **CONFIRMED:** <= **0.70** — a clear move from 0.8919 toward the muPC-off
  regime while keeping attenuation.
- **REFUTED:** >= **0.85** — no better than power=-4 alone.
- **AMBIGUOUS:** 0.70 - 0.85 — treated as refuted.
- **Prize:** <= **0.60** matches muPC-off *with* depth-scale control intact,
  which closes the bind.

**Guard:** real-text within-batch cosine must stay <= 0.10 (power -4 is at
-0.0294). Improvement in NMSE bought by degraded geometry is a trade, not a win.

## Independent mechanism check on the same run

The derivation says the offset should be stripped **in block 0**, as it is with
muPC off (attn -0.316, ffn -0.153), rather than late in the trunk as at
power=-4 (block 0 attn **+0.252**, stripping deferred to blocks 6-7).

So `scripts/localize_offset_in_block.py` on the resulting checkpoint is a second,
independent test of the mechanism -- and it can fail even if NMSE improves. If
NMSE improves while block 0 still fails to strip, the mechanism is wrong and the
gain came from somewhere else, and that will be reported.

That is the part worth watching: it is the first time in this sequence that the
proposed mechanism makes a prediction about *where inside the model* something
should change, rather than only about an outcome number.

## What this cannot establish

Depth 8 only. The depth ladder (`scripts/depth_ladder_probe.py`) and a run at 12
blocks are still required before anything is claimed about production depth 36.
Generalizing from the depth I happened to test is the error that produced this
whole thread.

---

# VERDICT, stage 22 (embedding scaling): REFUTED on both counts

**Time:** ~09:40.

**Outcome:** NMSE 2.1100, killed at 2.4170 by the divergence guard,
`admissible: False`. Registered CONFIRMED <= 0.70. Against power=-4's 0.8919
this made things worse. Probe lift 1.00x -- no information above floor.

**Mechanism prediction: FAILED, but measurably half-right.**

| block 0 | attn delta | ffn delta |
|---|---|---|
| muPC off (healthy) | **-0.316** | -0.153 |
| power -4 | +0.252 | +0.088 |
| embedding-scaled | **+0.146** | +0.175 |

The prediction was that stripping would move into block 0. It did not. The attn
delta shrank from +0.252 to +0.146 -- roughly the 1.68x the arithmetic predicted,
in the predicted direction -- **but it never changed sign.**

## Why the derivation was right and useless

`norm1` is a LayerNorm, so `attn_out` is invariant to scaling `x0`. Scaling the
embedding therefore does exactly what the derivation said: `x1 = s*(x0 +
attn_out)`, and the ratio between the input and its corrector is rebalanced by
1/s. The measured 42% reduction in the attn delta is that rebalancing, visible
and about the right size.

What it cannot do is change **what attention learned to output**. In the healthy
run block-0 attention produces a component that OPPOSES the offset; in every
muPC-on run it produces one that REINFORCES it. Rescaling changes magnitude, not
sign.

So the problem was never the arithmetic of cancellation. **Attenuation changes
what attention learns**, and no rescaling of the input fixes a learned function
of the wrong sign. The target moves from the residual stream's geometry to the
gradient signal attention receives -- which is attenuated by `residual_scale`
for every backprop-trained parameter in the block.

## Tally

Six mechanisms proposed and refuted, 2026-07-29 to 07-31: SIGReg-winning;
gradient-magnitude-as-cause; the PC/backprop rate ratio; the unscaled-x0
constant; the projection-head bias; and now embedding scaling.

This one failed better than the others: it made a quantitative prediction about
a specific location inside the model, that prediction was measurably half-right,
and the half that failed points somewhere specific rather than nowhere.

## What still stands

**power=-4 remains the best configuration found** -- real-text cosine -0.0294,
NMSE 0.8919, lift 2.21x, clip engaged 0%, muPC's depth-scale control intact. It
strips the offset late (blocks 6-7) rather than early, which is a workaround
rather than a repair, and it is still short of muPC-off's 0.5569.

`mu_pc_scale_embedding` is left in the codebase, defaulting False. It is a
correct implementation of a refuted idea; the flag and its comment record why
scaling the input does not fix a sign error in a learned function.

## Operational note

The periodic absolute divergence guard added this morning **fired for the first
time in anger**, at ~26 minutes instead of the 45 the run would have taken to
reach its epoch-end check. First live catch; ~19 minutes saved. The guard hole
found by stage 21 is closed in practice, not just in tests.

---

# Stage 23 (backprop-LR compensation): registered before the run

**Time:** ~10:00. **Run:** `probe_surprise_d8_bplr_512d_seed86`, 3000 steps.
**Base:** stage 20 (`probe_surprise_d8_amp4`), the best configuration found.
**Model config is IDENTICAL** -- the only difference is optimizer param groups.
Cleanest single variable in this whole sequence.

## What it does and why

Every block computes `x = x + residual_scale * f(x)`, so by the chain rule every
BACKPROP-trained parameter inside a block receives a gradient scaled by
`residual_scale`. Parameters outside the blocks -- embeddings, predictor,
projection heads -- do not, because `x0` reaches the output through the
unattenuated skip path.

**muPC therefore applies a smaller effective learning rate to the trunk than to
everything around it, and the gap widens with depth.** At 8 blocks the trunk
learns at 0.5946x the rate of the rest of the model; at 36 blocks it would be
0.4082x.

Stage 22 established that this is a *learning* problem, not a geometry one:
rescaling the input changed the magnitude of block 0's attention delta by the
predicted 1.68x and never changed its sign. Healthy block-0 attention learns an
output that OPPOSES the shared component (-0.316); every muPC-on run learns one
that REINFORCES it (+0.252 at power -4, +0.146 with embedding scaling).

This run multiplies the block parameters' learning rate by `1/residual_scale`
(1.682x at depth 8), restoring parity. It is the exact counterpart of
`mu_pc_rate_power` -- which did this for the PC side and produced the best
configuration found -- applied to the side that actually owns block 0's
canceller.

Verified: 64 block tensors at lr 5.045e-4, 67 others at 3.000e-4. With
compensation off the optimizer receives a single group, byte-identical to every
prior run.

## Registered prediction

**Primary: held-out NMSE.**

| reference | NMSE |
|---|---|
| depth 4 | 0.5215 |
| muPC off | 0.5569 |
| power -4 (base) | 0.8919 |

- **CONFIRMED:** <= **0.70**
- **REFUTED:** >= **0.85** (no better than the base)
- **AMBIGUOUS:** 0.70 - 0.85 -- treated as refuted.
- **Prize:** <= **0.60** matches muPC-off with depth-scale control intact.

**Guard:** real-text within-batch cosine <= 0.10 (base is -0.0294).

## Mechanism check, and this one is sharper than stage 22's

The claim is specifically that block-0 attention learns the wrong SIGN because
its gradient is attenuated. So the prediction is not "the offset gets smaller"
-- it is **block 0's attention delta should go NEGATIVE**, as it is with muPC off.

| block 0 attn delta | |
|---|---|
| muPC off (healthy) | -0.316 |
| power -4 (base) | +0.252 |
| embedding-scaled (stage 22) | +0.146 |
| **this run: predicted** | **< 0** |

A smaller positive number is NOT a pass. Stage 22 already produced that and it
was refuted. Only a sign change confirms the mechanism.

This is the sharpest prediction in the sequence: a specific number, at a
specific location, that must cross zero rather than merely shrink.

## Honest prior

Six mechanisms proposed and refuted, 2026-07-29 to 07-31. This one is better
grounded than most -- it follows from a chain-rule fact rather than a story, and
it is the untested half of an intervention whose other half demonstrably worked.
That is not evidence, and the last one also followed from an arithmetic identity
and still failed.

---

# VERDICT, stage 23: MECHANISM CONFIRMED, INTERVENTION REFUTED

**Time:** ~10:20. Both criteria were registered before the run; both scored.

## Mechanism check: CONFIRMED

| block 0 attn delta | |
|---|---|
| muPC off (healthy) | -0.316 |
| power -4 (base) | +0.252 |
| embedding-scaled (stage 22) | +0.146 |
| **backprop-LR compensated** | **-0.1156** |

**It crossed zero.** For the first time in any muPC-on configuration, block-0
attention learned to OPPOSE the shared component instead of reinforcing it. The
registration specified in advance that a smaller positive number would NOT pass
-- stage 22 had already produced that -- and that only a sign change would
confirm.

So the diagnosis holds: **gradient attenuation is what makes block 0 learn the
wrong sign.**

## Primary metric: REFUTED

NMSE 556.77, killed by the periodic divergence guard at 12 minutes.
Registered CONFIRMED <= 0.70. Loss went 458 -> 85,000; gradient median 4,610,
max 1.4e4 against a 20,000 clip, so clipping was not involved. Block 0's FFN
immediately re-added what attention removed (+0.3578) and the trunk diverged.

## What this establishes, and it is the useful part

**The bind is mechanistic, not incidental.** muPC's attenuation of the trunk's
effective learning rate is simultaneously:

  * what keeps deep training stable, and
  * what prevents block 0 from learning to strip the offset.

Those are not two problems. They are one mechanism with two consequences, and
this is the first run that demonstrates it rather than inferring it. Removing
the attenuation fixes the sign and destroys stability, in the same run, at the
same time.

Every prior result is consistent with this reading:
  * muPC off -- healthy block 0, unbounded activation growth (3.92x at 36 blocks)
  * muPC on -- stable, block 0 learns the wrong sign, offset climbs to 0.99
  * power -4 -- keeps attenuation, buys deep-block plasticity, strips late; best
    compromise found (NMSE 0.8919) and still short of muPC-off's 0.5569

## Next candidate, and it follows directly

Compensation was applied to **all 64 block tensors at once**. But the offset only
needs stripping ONCE, in block 0, and deeper blocks appear to need the
attenuation for stability. **Compensating block 0 alone** -- or the first two --
would give the sign flip where it is needed without un-damping the whole trunk.

That is a one-line change to `_param_groups` and it is the natural reading of
this result: the compensation was correct in kind and far too broad in scope.

## Tally

Seven mechanisms proposed across 2026-07-29 to 07-31. Six refuted outright.
This one is the first to be **confirmed as a diagnosis while failing as a
treatment** -- which is a more useful outcome than either a clean pass or a
clean failure, because it converts the depth problem from "something is wrong
in the trunk" into a named tradeoff with a measured mechanism on both sides.
