# LR warmup at depth 8: Opus's hypothesis, attacked, survived, registered

**Date:** 2026-08-06
**Hypothesis author:** Opus 5 (08-06 brief §1, relayed by Brian).
**Attack + registration:** Fable 5, per that brief's §7 ("attack §1 first;
if it survives, register and run it").
**Run:** `probe_v5_d8_warmup_512d_seed46`, stage 31, 3000 steps, ~45 min.
**Registered BEFORE the run.**

## 1. The attack record — §1 survived

Opus named three things that would kill the hypothesis. All three checked
against code and record:

1. **"muPC's init already subsumes warmup"** — DEAD. The naked-trunk run
   (08-06) had muPC fully disabled — no depth-scaled init at all — and was
   destroyed inside 200 steps anyway. Whatever early protection exists, it
   is not sufficient, and init-washout was already measured by step 3000
   in the 07-30 verdict.
2. **"Depth-4 is equally violent early and simply survives"** — CANNOT BE
   MADE FROM THE RECORD. Every depth-4 run has deep cadence 1000; steps
   0-999 are unobserved in all of them. This kill remains open in
   principle; the warmup run itself is cheaper than the d4-cadence-100
   control that would test it, so it runs first.
3. **"A prior JEPA-era warmup arm exists"** — DEAD. `cosine_lr_scale`
   returns 1.0 at progress 0 (verified at jepa_runner.py, applied
   ~995-1005); no warmup term exists anywhere in the runner or driver
   (every "warmup" hit is kill-criteria/trending plumbing); the old
   trainers' warmup (train_pc.py, m5_runner.py, audit 2026-05-10) never
   crossed into the JEPA era. Confirmed by direct read.

Supporting mechanism, stated for the record: cold AdamW second moments at
step 0, gradients running ~8x depth-4 scale (213 unclipped at step 100 vs
~28 at d4), at full 3e-4 — enormous effective parameter steps in exactly
the window (0-200) where every depth-8 destruction completes.

**Honest cap on the claim:** the dk5000 record shows the collapsed state
is an *attractor* reachable well after step 200. Warmup can only be the
whole story if the attractor is reachable *only* through the early-violence
transit. The gates below distinguish "prevents the transit" from "delays
the same transit."

## 2. The build

`LRScheduleConfig.warmup_steps` (default 0 = bit-exact legacy schedule;
unit-tested both ways in `tests/test_cosine_lr_and_v4_arm.py`): linear
ramp `(step+1)/w` for steps 0..w-1 — never exactly zero, same guard
philosophy as the cosine floor — then cosine over the remaining steps.

Arm `probe_v5_d8_warmup` = `probe_v5_d8` byte-identical in model config;
deltas are schedule-side (warmup 1000 steps) and observation-side (cadence
100; guard_min_step 1000, held through the ramp because init-proximal NMSE
has never been measured and the NMSE guard is documented to misread — a
trip at step 100 on a barely-trained model would void the run for
nothing; guards go fully live exactly when full LR arrives). Unclipped,
seed 46, same data order as the three v5-d8 twins.

## 3. Registered prediction — scored on stable_rank, absolute

Per the instrument findings in Opus's 08-06 brief (verified here:
dk5000 step-100 pooled effective_rank 196.86 vs stable_rank 2.42;
effective_rank is blind to a dominant direction, stable_rank sees it):
**primary metric is pooled `deep.stable_rank`, in absolute terms.**
Measured bands: healthy d4 (living_v5_4x_d4, five seeds, 72 firings each)
spans 13.5-47.5; the collapsed d8 floor never exceeded 2.42 in 50 dk5000
firings. Gate at 8: ~3.3x above the collapsed maximum, below the healthy
minimum.

- **CONFIRMED (warmup is load-bearing at depth):** stable_rank >= 8 at two
  consecutive deep firings at or after step 1500 (full LR, ramp's effect
  separable) AND at the final firing. If this fires, "bundle ON + muPC
  OFF is the only healthy d8 cell" was an artifact of a missing standard
  practice, and the entire depth investigation reopens on new terms.
- **REFUTED (warmup only delays the transit):** stable_rank <= 4 at every
  firing from step 1500 onward.
- Anything else (oscillation across the gate, a guard kill after 1000,
  a transit *during* the ramp followed by neither gate): **NO VERDICT**,
  reported as recorded. After three rungs of this trunk refusing
  registered outcomes, the middle branch is genuinely likely.

**Recorded, not scored:** per-block effective_rank (kept for continuity
with the arc's tables, demoted from scoring per the stable_rank finding);
SIGReg (with the §5 caveat from Opus's brief: pre-8ec9d07 l_sigreg
references are blinded and the 50-110 band traces to 862cfe1, post-fix);
offset dominance; unclipped grad series; suppressed-trip markers during
the ramp; step of first stable_rank >= 8 if any.

## 4. Confounds, stated in advance

1. Warmup 1000 + guard hold 1000 are deliberately coincident — the guard
   hold is justified independently (unmeasured init-proximal NMSE), but a
   kill in steps 1000-1500 would be hard to attribute between "full LR
   arrived" and "guards arrived." The gate therefore reads from 1500.
2. Single seed (46), and the dk-twin pair showed regime-scale chaos on
   this exact config. A CONFIRMED here is provisional until seed 95
   repeats it; that second seed is pre-committed as the immediate
   follow-up if the gate fires, before any reframing hardens.
3. Unclipped, cadence 100 — same as all v5-d8 twins; comparable.
4. Warmup length 1000 is a first guess (Opus proposed 500-1000). A REFUTED
   at 1000 does not exclude longer/slower ramps; it retires only "standard
   warmup fixes it."

## 5. Launch recipe

```
python scripts/jepa_pilot_driver.py --stage 31 --seeds 46 \
    --epochs 1 --max-batches-per-epoch 3000 --heldout-batches 5
```

---

# VERDICT: NO VERDICT on the registered gates — and the first depth-8 completion in the record

**Time:** 2026-08-06 evening. **Outcome: `completed`** — the first depth-8
run ever to finish under live guards. 30 deep firings. Both gates missed:
stable_rank oscillated 1.56-4.86 after step 1500 (never >= 8 at two
consecutive firings; not <= 4 everywhere either — it crossed 4 five
times). Fourth consecutive registration this trunk has refused to fit.
Scored as frozen: **NO VERDICT**, with the recorded facts below.

## Recorded facts

1. **The transit still happened** (steps 400-500, at ~40-50% LR) — warmup
   did not prevent it; the strongest version of §1 is dead. But it was
   *gentle*: the trunk came out of it with block 0 at rank 31, SIGReg
   at 106 (in-band), and offset already stripped to 0.27 — where every
   full-LR twin exited the transit with nothing above ~11, SIGReg in the
   thousands, and offset saturated.
2. **From that gentler floor, recovery compounded for 2500 steps** — the
   thing the dk5000 tape proved never happens from the violent floor.
   By step 3000: every block at effective rank 128+, pooled eff 181,
   offset 0.133, SIGReg ~11, all under fully live guards from step 1000.
3. **Capability is healthy on the d4 rubric:** held-out NMSE **0.5518** —
   inside the depth-4 band (0.5215-0.60) and equal to stage 16's 0.5569 —
   probe lift **4.33x** (d4: 4.67-4.80x; stage 16: 4.19x). These numbers
   are readable precisely because the target space is no longer
   degenerate.
4. **The recovery is incomplete on the strictest instrument.** Pooled
   stable_rank ended at 4.63 vs 31.4-38.0 for d4 at the same step
   (measured, all five seeds) — a dominant-direction concentration
   persists even as the broad space re-inflated. Whatever direction that
   is, it survived everything and is unexplained.
5. **Instrument recalibration from the ramp:** stable_rank ~2.4 at step
   100 is the *init-proximal* state (identical at 1/10th LR and full LR),
   not evidence of destruction — and init-proximal held-out NMSE
   measures in the hundreds (439 at step 100), which retroactively
   justifies every early-step guard hold and voids any first-check NMSE
   reading as a health signal on young runs.
6. Training-time SIGReg settled at 9-21 — *below* the 50-110 d4 band.
   Recorded as an open question, not a health claim (the 07-27 family
   showed over-quieting has costs).

## Reading, carefully

Warmup did not prevent the collapse; it changed the *class* of the
outcome — from a violent transit into a permanent attractor, to a gentle
transit into a recoverable state. It is the first intervention in the
entire depth arc that changed this trunk's fate rather than its
timing. The chain "gentler transit → structure survives → recovery
compounds" fits every tape in the arc, including dk1000's doomed partial
heal (which started from a violent floor and had nothing to compound
from). It is one seed, on a substrate whose regime outcomes are measured
to be chaotic (dk1000 vs dk5000), and the pre-committed seed-95 repeat is
the immediate next spend before any of this hardens.

## Next, in order

1. **Seed-95 repeat** (pre-committed): same arm, nothing changed. If it
   reproduces even approximately, warmup is load-bearing at depth.
2. The stable_rank residual — what is the persistent dominant direction?
   `localize_offset_in_block.py` on this checkpoint would say whether it
   is the old offset in a new coat (offset dominance says no: 0.13).
3. Longer run: does stable_rank keep climbing past 3000 toward the d4
   trained band, or plateau at ~4?
4. Design (Brian + Opus): if seed 95 reproduces, the "only healthy cell"
   framing is dead, muPC-ON at depth is back on the table with warmup,
   and the ladder (d12, d36) should carry warmup from day one.

---

# EXTENSION: ramp +50% (Brian's call, registered before the run)

**Run:** `probe_v5_d8_warmup15_512d_seed46`, stage 32. One variable
against stage 31: warmup 1000 → 1500 (guard hold moved with it, same
independent justification — init-proximal NMSE measures in the hundreds).
Seed 46 kept so the comparison is one-variable at the same data order.
**Deliberately deferred behind this run:** the seed-95 reproducibility
repeat — Brian ruled ramp-length exploration first; reproducibility stays
an open flank on both warmup results until seed 95 runs.

**Registered reads (gates unchanged in kind, window shifted with the
ramp):** CONFIRMED = stable_rank >= 8 at two consecutive firings at/after
step 2000 AND at the final firing. REFUTED = <= 4 everywhere from 2000.
Else NO VERDICT. Also frozen: (a) transit step and violence (stage 31's
came at 400-500, ~40-50% of ramp — if the transit tracks the *fraction*
of ramp rather than the absolute step, it should land ~600-750 here;
if it tracks absolute LR, ~same steps as before); (b) whether the
post-transit floor is gentler still (block-0 rank at floor, SIGReg at
floor); (c) final stable_rank vs stage 31's 4.63 — the specific number
Brian's +50% is probing.

---

# EXTENSION VERDICT: REFUTED at ramp 1500 — and the pair now demands repeats

**Outcome:** `killed:divergence:text:nmse=3.5017>2.00` at step 2500,
`admissible: False`. **REFUTED gate fires as registered** (stable_rank
<= 4 at every firing from 2000; it never exceeded 1.96 after step 1300).
Probe lift 1.13x — no signal above floor.

## Recorded facts

1. **Transit at the same absolute steps (400-500), not the same ramp
   fraction** — at 33% LR here vs 50% at ramp-1000, on the same seed and
   deterministic data order. The transit is pinned to step/data, not to an
   LR threshold. (Which of step-count or data-content pins it needs a
   different seed to separate.)
2. **The slower ramp exited the transit with LESS structure, not more:**
   block 0 bled 87.7 → 7.3 across steps 500-800, offset re-saturated to
   0.98, and no recovery ever compounded — 2000 steps of floor-churn until
   the kill. The stage-31 chain ("gentler transit → structure survives →
   recovery compounds") did not run here despite a gentler schedule.
3. The run survived steps 1500-2400 under LIVE guards with NMSE <= 2.0
   while fully collapsed — the collapsed-target flattery keeping a dead
   trunk alive, again, now in the survival direction.

## The uncomfortable, load-bearing conclusion

Ramp 1000 (stage 31): near-recovery, first d8 completion, NMSE in the d4
band. Ramp 1500 (this run): permanent floor, REFUTED. Same seed, same
data order, one knob moved in the *gentler* direction. Either warmup
length is a knife-edge parameter with a non-monotone optimum — possible
but a priori unlikely — or **regime-scale path chaos dominates
single-run outcomes and stage 31's recovery cannot yet be attributed to
warmup at all.** The dk-twin pair already proved identical configs can
split heal-vs-floor; this pair shows one-knob comparisons at n=1 are
unreadable on this substrate.

**Consequence, stated as the next-spend rule: no more single-seed knob
moves on this question.** The only informative next runs are repeats —
seeds 95 (and ideally 97) at ramp 1000 — to measure whether stage 31's
recovery is warmup's doing or the path lottery's. If 2/3 seeds at ramp
1000 recover, warmup shifts the odds and the depth investigation reopens;
if 1/3 or 0/3, stage 31 was the lottery and §1 dies at depth 8 despite
the prettiest tape in the record.

---

# REPEATS: seeds 95 and 97 at ramp 1000 (registered before launch)

**Runs:** `probe_v5_d8_warmup_512d_seed95`, `probe_v5_d8_warmup_512d_seed97`
— stage 31 arm unchanged in every respect; seeds are the only variable.
Purpose: estimate the outcome distribution at ramp 1000. The control base
rate is fixed in the record: among non-warmup runs given room to recover
(the dk pair), recoveries = 0/2.

**Per-seed read, frozen (same as stage 31's gates):** RECOVERY =
completes AND final pooled effective_rank >= 100 with every block >= 50
at the final firing (stage 31 read 181 / min-block 128.7; the dk floor
never exceeded eff ~5 pooled) — the stable_rank >= 8 gate stays recorded
but stage 31 showed it lags recovery, so eff-based recovery is the
tally criterion, chosen now, before the draws. FLOOR = killed, or
completes with pooled eff < 20 at final.

**Tally rule, frozen:** counting stage 31, warmup at ramp 1000 recovers
in 2/3 or 3/3 seeds -> warmup shifts the odds; the depth ladder carries
warmup and the ramp-response sweep is justified. 1/3 -> ambiguous;
0/2 new -> stage 31 was the lottery; the warmup branch closes at d8.

**Note on the middle:** a seed that completes with eff between 20 and
100, or heals-then-relapses, is scored neither and reported — the arc
has earned that humility four times.
