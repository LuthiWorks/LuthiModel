# Naked trunk at depth 8: who owns the divergence?

**Date:** 2026-08-06
**Author:** Fable 5. Control 1 from the rung-1 verdict
(`docs/research/2026-08-05_bundleoff-at-depth-hypothesis.md`), ordered by Brian.
**Run:** `probe_d8_naked_512d_seed95`, stage 27, 3000 steps, ~45 min.
**Registered BEFORE the run.** Not started at registration time.

## The question

Rung 1 showed the bundle-off muPC-on depth-8 trunk diverges in ~150 steps
(SIGReg 1763 against a healthy 50-110 band). Two hypotheses fit:

- **muPC destabilizes the naked trunk.** The residual attenuation that
  stabilizes the *bundled* trunk (stage 23's finding) does the opposite when
  the bundle's regulators are absent — or the asymmetry it creates (trunk
  learning slower than everything around it) runs away without the bundle
  to damp it.
- **The naked trunk is unstable at depth 8 regardless.** The PC substrate's
  self-modification without its regulators (band, trust weighting, gain cap,
  drive gating) is intrinsically unstable, and muPC is irrelevant to rung
  1's divergence.

This run is one variable against rung 1: `mu_pc_enabled=False` (removing
residual scaling and depth-scaled init together, per the stage-16 caveat —
a clean result implicates or exonerates muPC as a whole). Same seed, same
deterministic data order, so the early steps are directly comparable.

## The full factorial, for orientation

| cell | outcome |
|---|---|
| d8, bundle ON, muPC ON | stable collapse (rank ~2, every variant) |
| d8, bundle ON, muPC OFF | stable, healthy (stage 16) |
| d8, bundle OFF, muPC ON | divergence in ~150 steps (rung 1) |
| **d8, bundle OFF, muPC OFF** | **this run** |

## Registered prediction

**Primary gate: run outcome.**

- **COMPLETES (not killed by any guard):** muPC is implicated in rung 1's
  divergence — the naked trunk trains without it. The bundle and muPC are
  then *each* capable of stabilizing the other's absence, and the collapse
  investigation becomes: what does muPC do to a trunk that the bundle must
  then contain?
- **KILLED by the divergence guard:** the naked depth-8 trunk is unstable
  regardless; muPC is exonerated for the *divergence* (its role in the
  stable *collapse* is untouched either way). The regulators in the bundle
  become the primary stabilization story, and the add-back ladder (band
  first) is the next step.

No ambiguous band: the outcome field is binary. A kill by any *other*
guard (kill-criteria rather than divergence) is scored as REGISTRATION
MISS and reported without a verdict.

**Secondary, scored only if the run completes — the rank gate, unchanged
from rung 1's registration:** block-0 effective rank >= 20 sustained at
two consecutive deep firings AND at the final firing = the naked trunk
*acquires*; every block-0 reading < 20 after step 1000 = the naked trunk
collapses without muPC, which would reopen the muPC-causes-collapse
question entirely. Cadence is 100, so there are ~30 observations.

**Recorded, not scored:** held-out NMSE, SIGReg trajectory, grad-norm
median, clip engagement, activation growth first-to-last block (the depth
ladder's muPC-off concern — 1.47 at 4 blocks climbing to 3.92 at 36; depth
8 sits early on that curve and this run measures it with the bundle absent).

**Prior, stated honestly:** weakly toward COMPLETES. Stage 16 (bundle ON,
muPC OFF) was stable at depth 8, and the divergence signature in rung 1 was
scale-runaway of exactly the kind muPC's attenuation asymmetry could feed.
But no naked run exists at any depth in the JEPA era, so this is a guess
with one supporting cell and zero direct precedent.

## Confounds, stated in advance

1. **muPC-off removes two things at once** (residual scaling + depth-scaled
   init), by design; the stage-16 caveat carries. Init was shown to wash
   out by step 3000 at bundle-on; that measurement does not automatically
   transfer to bundle-off.
2. **Clip 1000 carried** (identical to rung 1 and stage 16); engagement
   reported.
3. **Single seed (95).** Divergence can be seed-sensitive. If the outcome
   is surprising, the follow-up is seed 96 before any conclusion hardens.
4. **The eval harness runs post-kill numbers on a diverged model** (rung 1's
   probe_top1 0.1488 is meaningless); post-kill capability numbers are not
   evidence and will not be cited.

## What this run cannot settle

- Which regulator in the bundle stabilizes (that is the add-back ladder).
- Which half of muPC acts, if muPC is implicated (`mu_pc_exponent=0.0`).
- Anything about depth 4 (control 2 stays specced and unrun).

## Launch recipe

```
python scripts/jepa_pilot_driver.py --stage 27 --seeds 95 \
    --epochs 1 --max-batches-per-epoch 3000 --heldout-batches 5
```

---

# VERDICT: KILLED — the registered gate fires, and the forensics reverse the expected story

**Time:** 2026-08-06. Killed at `nmse=2.7195 > 2.00` (periodic guard), 0.04 h,
one deep firing recorded, ~150 steps. `admissible: False` for capability.

## Registered scoring, exactly as frozen

**Primary gate (run outcome): KILLED by the divergence guard.** The
registered reading fires as written: **the naked depth-8 trunk is unstable
regardless, and muPC is exonerated for rung 1's divergence** (its role in
the stable collapse is untouched). The bundle's regulators become the
primary stabilization story; the add-back ladder, band first, is next.

**Secondary rank gate: not scored** — it was registered "only if the run
completes."

**Prior check:** the registration leaned weakly toward COMPLETES. Wrong
again, second rung in a row. Both wrong priors are in the record unedited.

## The step-100 forensics — same seed, same data order as rung 1

| step 100 | naked (this run) | bundle-off, muPC ON (rung 1) |
|---|---|---|
| block-0 rank | **29.6** | 237.5 |
| blocks 1-7 rank | **1.0 - 1.6 (collapsed)** | 191 - 227 |
| L_pred | **31.45 (broken)** | 0.38 (fine) |
| L_sigreg (healthy 50-110) | **4388** | 1763 |
| off-diag correlation | 0.907 | — |
| offset dominance (target) | 0.845 | — |
| guard NMSE at kill | 2.72 | 41.69 |

Three things fall out, stated with one-record humility:

1. **Rank collapse does not require muPC.** The naked trunk hit rank ~1 in
   blocks 1-7 within 100 steps, with the full offset-dominance signature.
   In the *bundled* system muPC-off is healthy (stage 16), so the clean
   statement is now: **collapse is a failure mode of the depth-8 trunk
   itself; what varies across cells is what prevents it and what it costs.**
2. **muPC held the trunk open.** On identical data, muPC-on at step 100 has
   init-like rank everywhere and working prediction. muPC's residual
   scaling appears to *slow the collapse* — its original stabilization
   purpose — while producing a scale runaway the bundle would otherwise
   damp. The 07-31 framing survives another form: every lever here buys
   one pathology with another.
3. **The guard's NMSE inverted the health ordering.** 2.72 (naked) vs 41.69
   (muPC-on) reads as "naked is nearer health"; every substrate measure
   says the opposite. NMSE normalizes by target variance, and a collapsed
   target flatters it — the *same* flattery that gamed stage 25, now in
   the divergence guard. An instrument correction is warranted: the guard
   should trip on SIGReg or rank alongside NMSE. (Recorded here; not
   unilaterally changed — the guard's thresholds are registered surface.)

## Where the factorial now stands

| cell | outcome |
|---|---|
| d8, bundle ON, muPC ON | stable collapse (rank ~2) |
| d8, bundle ON, muPC OFF | stable, healthy |
| d8, bundle OFF, muPC ON | diverges; rank held open at kill |
| d8, bundle OFF, muPC OFF | diverges; already collapsed at step 100 |

The bundle is necessary for *stability* at depth 8 in every observed cell.
Health then depends on muPC being OFF; with muPC ON the stabilized trunk
lands in rank-2 collapse instead. The next discriminating step is the
inverted add-back: **which single regulator, added to the naked trunk,
restores stability — and does rank-2 collapse arrive with it?** Band first,
per the rung-1 verdict. Control 2 (d4 naked) remains specced and cheap.

## Confounds carried

Single seed; single deep firing per run (the kill lands before firing 2 —
if the add-back rungs also die early, drop cadence to 25 or lower the
guard's first-check latency so the registered gates get at least two
observations); clip 1000 engaged hard here (pre-clip grad 3364 at step
100); post-kill eval numbers cited nowhere.
