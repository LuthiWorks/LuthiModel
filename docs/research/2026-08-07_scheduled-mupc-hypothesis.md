# Scheduled muPC: acquire without it, anneal it in

**Date:** 2026-08-07
**Design:** Brian ("can we run with muPC off until a certain point, like
step 3000, and turn it on?"). Build + registration: Fable 5.
**Run:** `probe_d8_mupc_sched_512d_seed46`, stage 44, 6000 steps.
**Registered BEFORE the run.** Queued behind the in-flight pipeline.

## The seam this targets

muPC's measured harm is acquisition-phase (attenuated gradients teach
block 0 the wrong sign against the offset — stages 22-24); its benefit
is long-run depth-scale control the production ladder cannot live
without (activation growth 1.47 → 3.92 by depth 36 without it). Every
prior arm chose one side for the whole run. This arm splits by time:
acquire in the record's one robustly healthy depth-8 cell (stage 16:
probe_surprise bundle, muPC OFF, clip 1000), then ANNEAL every block's
residual scale from 1.0 to the muPC value (8^-0.25 = 0.5946) across
steps 3000-4000, then 2000 steps of observation at full attenuation.

Delivery limits, stated: only the scaling half of muPC arrives (the
depth-scaled-init half cannot apply retroactively; the 07-30 verdict
measured init washing out by step 3000 regardless). The anneal is a
ramp, never a step — a discontinuity in the computed function would
trip guards on an otherwise healthy run. Runner support:
`mu_pc_schedule_*` in RunnerConfig, loud contracts (refuses models
built with muPC on; refuses rate_power arms), unit-tested.

## Registered gates

- **HEALTHY-THROUGH (the recipe works):** completes, AND pooled eff
  >= 100 with every block >= 50 at BOTH the last pre-ramp firing
  (~2900) and the final firing (6000). Health acquired, health kept
  under full attenuation — the first plausible recipe for the depth
  ladder.
- **COLLAPSE-ON-RAMP (muPC's harm is not acquisition-confined):** any
  block's rank < 20 at two consecutive firings after ramp start, having
  been >= 50 pre-ramp. The stage-24 equilibrium reading predicts this
  risk: the stripping solution learned at scale 1.0 may not be an
  equilibrium at 0.59.
- **Guard kill during/after the ramp with pre-ramp health:** scored as
  RAMP-SPEED result, not mechanism refutation — the registered fallback
  is one rerun with ramp 2000. A kill BEFORE step 3000 voids the run
  (the healthy cell failed to reproduce — chaos caveat).
- Anything else: NO VERDICT, reported.

**Prior, honestly:** genuinely uncertain — the most informative kind.
Stage 24 argues the equilibrium is scale-dependent (re-collapse risk);
the carve findings argue the early window is what matters (the acquired
structure should survive). This run decides between two measured
readings of our own record.

**Recorded:** rank/SIGReg/offset through the ramp window at cadence 100;
block-0 attention behavior across the scale change (does the learned
stripping survive attenuation — the stage-22/23 question, finally asked
at the right moment); post-run write-path spectra.

## Confounds

Single seed (46); stage-16 cell reproduced once (its own chaos exposure);
clip 1000 carried from the base (3% engagement when healthy); ramp
window 3000-4000 chosen by Brian's step-3000 instinct, and ramp speed is
the registered fallback dimension, not swept in advance.

## Launch

```
python scripts/jepa_pilot_driver.py --stage 44 --seeds 46 --epochs 1 --max-batches-per-epoch 6000 --heldout-batches 5
```

---

# VOID-1 (registered condition fired) + amendment

First attempt killed at ~step 400 (nmse 2.36), before the ramp — VOID
per the pre-registered condition. Root cause is instructive and owned:
this arm was registered WITHOUT the guard hold every other probe arm
carries, and at deep cadence 100 the periodic NMSE guard inspects 10x
more often than the cadence-1000 era that certified stage 16 healthy —
the healthy cell may always wobble early, unobserved. Guard-timing as a
hidden variable, third appearance, this time via my own inconsistent
registration. Amendment: guard_min_step 1000 (the probe standard);
void run preserved as *_void1; rerun otherwise identical.

---

# VOID-2 — and the void is itself a finding

Rerun (guard hold restored) killed at ~step 2100, nmse 2.07 — BEFORE the
ramp. Not a guard artifact this time: a genuine late transit (healthy
through ~1600 with SIGReg in-band at 23 and offset 0.13, then b0
184 → 25 → 10 across 1700-2100). **The stage-16 cell — the record's "one
robustly healthy depth-8 configuration" — is now 1-for-2 across seeds.**
The path lottery reaches the healthy cell too; "robust" was a single
draw. This demotion propagates to every document that leaned on stage 16
as an anchor, including the verdicts brief to Opus.

The scheduled-muPC design remains UNTESTED — no run has yet survived to
its own ramp. VOID-2 amendment: rerun at seed 94, the one seed with a
surviving history in this cell, registered as the design's best fair
chance. If seed 94 also fails to reach step 3000, the schedule test is
parked until the battery finds a base that reliably reaches mid-run, and
the 1-for-N health statistics of the nomupc cell become their own
registered question.

---

# PARKED (registered condition fired) — and the cell statistics become the question

Seed 94 — the cell's one surviving draw — transited at ~700 and was
killed at 800 on rerun. The nomupc cell now stands **1-for-3** across
draws. Per VOID-2's pre-registered rule, the scheduled-muPC design is
PARKED, untested through no fault of its own: no base run has reached
its own ramp. It revives automatically when the battery produces a base
configuration that reliably reaches mid-run (the wsig10 repeats are the
nearest candidate).

**Registered question replacing it:** the outcome distribution of the
depth-8 cells themselves. Standing tallies: full-bundle+muPC 0-for-many;
warmup 1-for-3; nomupc 1-for-3; wsig10 1-for-1 (repeats owed). No
depth-8 configuration in the record has a demonstrated reliable good
outcome. The design target is therefore distribution-collapsing
mechanisms or architecture changes (width ratio, decorrelation family,
TC descendants), not setting-search.
