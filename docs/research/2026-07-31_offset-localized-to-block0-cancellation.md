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
