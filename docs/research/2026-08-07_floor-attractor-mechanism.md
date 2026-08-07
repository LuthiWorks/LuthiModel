# Why the floor holds: the collapse gets consolidated into the attention write-path

**Date:** 2026-08-07, afternoon
**Author:** Fable 5, at Brian's request ("representation stayed on the
floor. Why?")
**Method:** read-only spectral analysis of existing checkpoints — no GPU
time, no new runs. Effective rank = exp spectral entropy of a weight
matrix's singular values; stable rank = ||W||_F² / σ_max². Init reference
for a 512×512 projection: stable rank ≈ 130.

## The question

Seed 97 completed 3000 steps with offset healed (0.36-0.59) and SIGReg
moderate — the *forces* that should re-inflate the space were present —
yet every block sat at activation rank ~2. What pins it?

## What the checkpoints show

**Not the input, not the living weights.** The embedding table is full
rank (eff 511, indistinguishable from init) in every run — floor,
recovered, and healthy alike. `living_ffn.weight` stays broad everywhere
(stable rank 39-130). The lock is not upstream and not in the substrate's
own channel.

**The attention write-path (v_proj/o_proj) separates the populations
perfectly.** Block-0 stable rank, from init ≈ 130:

| run | outcome | block-0 v/o stable rank | blocks 1-7 range |
|---|---|---|---|
| d4 healthy (24k steps) | healthy | 35 / 47 | — |
| d8 warmup seed 46 | RECOVERED | 27 / 20 | 32 – 108 |
| d8 warmup seed 95 | floor (killed 1800) | 10.7 / 11.7 | — |
| d8 warmup seed 97 | floor (completed) | **4.0 / 4.1** | 5.5 – 37 |
| d8 stable-collapse s84 | floor (bundle on) | 6.9 / 4.6 | — |
| d8 naked @ step 100 | activations already collapsed | **51.6 / 32.7 (still broad)** | — |

**The sequence, from the naked-run row:** at step 100 the activations are
already at rank ~1 while the weights are still healthy-broad. The carving
comes *after*. Seed 97's rolling checkpoints show it locked in by step
~1461 (block-0 v/o at 4.9/4.9) and static thereafter.

## The mechanism, stated

1. The transit collapses **activations** (dynamics — the offset/equilibrium
   story of the July arc; happens even at 1/3 LR, pinned to early
   steps/data).
2. Training continues on the collapsed activations, and plain
   backprop **carves the attention write-path to match**: within ~1400
   steps the floor runs' block-0 v/o projections amplify only ~4
   directions (from an initial ~130).
3. From then on the floor is **locked in weights, not activations**:
   SIGReg can push the latents, the offset can heal, prediction can be
   fine — but the only pathway that writes into the residual stream has
   gain along a handful of directions. Re-inflation has no amplifier.
   That is why seed 97 completed with healed offset and dead rank.
4. Recovery (seed 46) is the path where re-inflation started before the
   carving deepened — its block 0 never went below ~20, and its blocks
   1-7 stayed at 32-108. Consistent with warmup's measured effect
   (shallower carve rate at lower LR-integral) shifting the odds without
   guaranteeing escape.

The irony belongs in the record: this is precisely the mechanism the
2026-08-05 rank doc warned an importance-weighted consolidation feature
would implement deliberately — "the surviving directions carry large,
stable activation… score them as maximally important and harden them."
No such feature exists; **AdamW does it for free** on whatever the
activations collapse to. (AdamW's default weight decay 0.01 is present
and evidently no match for it.)

This also closes the loop on stage 24's "stripping is a system-level
equilibrium": the equilibrium has a weight-side memory. Move the
activations without moving the carved projections and the projections
pull them back.

## Limitations, honestly

- Three rolling checkpoints per run; no floor-era checkpoint survives for
  seed 46 (its earliest is step 1404, already recovering). The
  activations-first sequence rests on the naked run's step-100 snapshot
  plus the monotone carving in seed 97 — strong, not airtight.
- Correlational population (n=6 runs). No intervention has yet moved the
  carve and shown the floor release — that is the falsifiable next test.
- Stable rank of weights, like all our observables, needs a
  reproducibility read before anything is registered on it (07-27 rule).

## What this enables (design seats' call, not run tonight)

1. **A weight-side leading indicator, nearly free:** log per-block
   v/o stable rank at deep cadence (one 512×512 SVD per block — trivial).
   It separates floor from recovery at 5x and moves *before* outcome; it
   would also finally give the guards something rank-shaped that NMSE
   cannot flatter.
2. **Anti-carving candidates for the ladder era,** in rising order of
   invasiveness: stronger/targeted weight decay on attention projections;
   spectral/soft-rank regularization on v/o; carve-rate management (the
   warmup result reinterpreted — it slows carving, which explains the
   shifted floors and occasional escape).
3. **A falsification test for this whole mechanism:** take seed 97's
   final checkpoint, re-inflate the carved projections (e.g., blend
   toward init or decay σ₁..σ₄ specifically), resume training, and watch
   whether rank escapes the floor. If it does not, this doc's mechanism
   is wrong and says so.

The depth question — why the d8 transit happens at all — is untouched by
tonight. This answers the *attractor* half: what happened is that the
trunk's own write-path was sculpted into the shape of its collapse, and
after that, everything downstream was pushing against weights.

---

# SURGERY: registered before the run

**Run:** `probe_v5_d8_surgery_512d_seed97`, stage 33 — resumes seed 97's
completed-collapsed checkpoint (step 3000) for 3000 further steps with:
v/o projections re-broadened by shrink-and-perturb (0.6, 0.8·std,
repeated per-matrix until stable_rank ≥ 20 — the recovered run's
block-0 level; block 0 took 2 passes, final 22-48 everywhere) and **all
Adam moments reset** (the moments carry the carve; uniform reset chosen
over fragile per-param surgery, documented as a co-intervention).
Guards held to global step 4000 (the perturbed projections transiently
predict worse; 1000-step grace), live thereafter. Data continues from
the loader's saved position.

**Gates, frozen:**
- **RELEASE (lock confirmed, remedy viable):** pooled effective rank
  ≥ 100 AND every block ≥ 50 at the final firing (the seed-46 recovery
  criterion).
- **HOLD (mechanism insufficient):** pooled eff < 20 at final — the
  floor re-forms despite broadened weights, implicating activation-side
  dynamics and favoring objective-level fixes (TC-SIGReg family).
- Between: NO VERDICT, reported. Also frozen: per-block v/o stable-rank
  trajectory post-surgery (does the carve re-form, and how fast — the
  re-carve RATE is informative even under HOLD).

**Discriminating role:** Brian's three candidate remedies (TC-SIGReg /
per-block Weak-SIGReg / orthogonal penalty — arXiv 2607.26924,
2603.05924, classic) split into objective-side and weight-side families.
RELEASE favors the weight-side pair; HOLD favors the objective-side.

**Confounds:** single seed; two interventions at once (weight surgery +
moment reset — a RELEASE cannot attribute between them without a
follow-up); the parent run completed via NMSE flattery so its guards-live
survival says nothing about health.
