# 512 VISReg full-length control — registration

**Author:** Opus 5, 2026-08-14. **Frozen before any data exists.**
**Obligation:** audit item C1
(`docs/audits/2026-08-13_luthimodel-audit.md`), created by the
SUPERSESSION NOTICE on `2026-08-11_visreg-family-registration.md`.

## The question

The 512 VISReg family CONFIRMED 2-of-3 and its scored predictions
included *"the soloist never formed in any seed. The center term killed
the first act in the crib."* That family was scored at **6,000 steps**.

The 768x8 family then produced a soloist whose onset is **~10,000
steps** — and at step 6,000 that same run read b0 effective rank 264 /
`top_dir_share` 0.039, i.e. indistinguishable from healthy. So the 512
family stopped before the phenomenon it was taken to have abolished can
appear, and "abolished" is a claim about all time drawn from a 6,000-step
window.

**Width and run-length are currently confounded.** This control breaks
the confound in the cheap direction.

## Configuration — deliberately not a new experiment

Arm `probe_d8_visreg_long`, stage 56, seeds 46/95/97, d_model 512,
8 blocks. `ARM_CONFIGS["probe_d8_visreg_long"]` is constructed as a copy
of `probe_d8_visreg` and asserted equal at import; every ARM_* table
entry is copied from the source arm.

**The only difference from the original family is that the run is not
truncated.** The 512 family's cosine was already registered for a full
epoch (`lr_total_steps: 24014`) and was capped at 6,000 batches. This
control removes the cap. No schedule change, no parameter change.

The distinct arm name exists solely so run dirs do not collide: the dir
is keyed on (arm, d_model, seed), `training_log.jsonl` is append-only,
and checkpoints would be overwritten. The 512 family's record is
evidence in an open verdict.

**Disclosed difference, and it is not cosmetic:** this control runs under
the **corroborated kill-6** (2026-08-14 audit item B5), where the
original family ran the uncorroborated one. Uncorroborated kill-6 killed
the 768 family's seed 46 at step 9,100 while every geometric measure said
healthy and improving, and was measured to kill 10/10 healthy runs on
2026-07-16. Running this control under the old criterion would very
likely have ended it around step 9,000 without answering the question.
Recorded here rather than buried: if a seed survives past 9,000 where the
old criterion would have killed it, that is a consequence of this change.

## Frozen predictions

Registered before the first step. The primary read is **b0
`top_dir_share`**, the gauge on which the original family's scored
claim rests.

1. **PRIMARY.** At least one of the three seeds shows sustained b0
   `top_dir_share >= 0.20` before step 24,014.
   *Confirms:* the soloist is a run-length effect and the 512 family's
   "never formed" is a truncation artifact.
   *Refutes:* the soloist is width-entangled; the 512 claim stands **for
   512**, and the 768 result needs a width explanation.
2. Onset, where it occurs, is **front-to-back in block order** — b0
   before b1 before b2 — matching seed 97 (13,100 / 14,500 / 51,700).
3. In any block that develops a soloist, **`chorus_eff_rank` stays above
   100** while `effective_rank` falls toward the floor: carrier, not
   collapse, per audit item A7. This is the first live test of the
   `chorus_eff_rank` instrument added 2026-08-14.
4. No seed's `consolidation_noop_fires` diverges from its
   `consolidation_fires` **less than** ~100%, i.e. the episode stores
   stay empty under `adaptive_episodes=False` (audit B4). Recorded as an
   expectation, not a gate — this control does not change that setting.

## Scoring

- A seed that completes 24,014 un-killed is **countable**.
- A seed killed by a **corroborated** criterion is countable as a
  failure and its kill reason is reported.
- A seed killed by an uncorroborated or timing-artifact criterion is
  **uncountable** — the 07-19 lesson about detectors calibrated on
  another substrate's physics.
- Prediction 1 is scored on the pooled family: one seed suffices.
- Chance accounting for prediction 2: three seeds, eight blocks; a
  front-to-back ordering by coincidence is ~1/8! per seed, so a single
  seed matching is already strong and three matching is decisive.

## Cost

~0.62 s/step at 512x8 → ~4.1 h/seed, ~12.4 h for the family, sequential.
Launched 2026-08-14 early afternoon; expected complete overnight.

## Command

    python scripts/jepa_pilot_driver.py --stage 56 --seeds 46,95,97 \
        --epochs 1 --batch_size 32 --heldout-batches 5

(No `--max-batches-per-epoch`: that cap is the entire subject of this
control.)

---

## VERDICT — scored 2026-08-15

All three seeds **completed** the full epoch (24,000 steps). None killed.

| seed | outcome | probe top1 | floor | lift | heldout NMSE | b0 eff | b0 tds |
|---|---|---|---|---|---|---|---|
| 46 | completed | 0.1334 | 0.0320 | 4.17x | 0.5719 | 259.0 | 0.039 |
| 95 | completed | 0.1299 | 0.0320 | 4.06x | 0.5658 | 247.7 | 0.059 |
| 97 | completed | **0.1336** | 0.0283 | 4.72x | 0.5979 | **149.0** | **0.198** |

For scale, the 768x8 family's completing seed read probe 0.1115 and NMSE
0.854. Every seed here beats it on both axes.

### PREDICTION 1 (primary) — REFUTED

*"At least one of the three seeds shows sustained b0 `top_dir_share`
>= 0.20 before step 24,014."*

No seed sustained a crossing. Seed 97 reached **0.1979** and then fell
back to 0.179 before ending at 0.198. It did not cross, and it did not
sustain. **Refuted, plainly** — the 2026-07-27 rule applies: hedging a
falsification is a way of keeping the bet alive after it lost.

### What the data shows that the binary hides

Seed 97 is unambiguously **in onset** at the wire, and the criterion
missed it by 0.002:

| step | b0 eff_rank | b0 top_dir_share | b0 chorus_eff_rank |
|---|---|---|---|
| 100 | 192.1 | 0.075 | 220.3 |
| 6,100 | 203.7 | 0.071 | 232.3 |
| 12,100 | 181.1 | 0.116 | 238.6 |
| 18,100 | 164.0 | 0.167 | 265.1 |
| 24,000 | **149.0** | **0.198** | **275.4** |

Block 0 alone falls (d(eff) over the last 20% is **-1.0** for b0 while
every other block gains +12.9 to +16.7). The final profile is b0 149 with
b1..b7 at 207-234 — the front-to-back shape of prediction 2, arriving in
exactly one seed.

**At step 6,000 — the original 512 family's wire — seed 97 read
`top_dir_share` 0.071 and eff rank 203.7, i.e. healthy and improving.**

### Synthesis: both mechanisms are real

- **Run length is necessary.** The 512 family's scored claim, *"the
  soloist never formed in any seed,"* was made at 6,000 steps. Seed 97
  shows it forming afterwards. The SUPERSESSION NOTICE on that family is
  **vindicated**: the claim was horizon-limited.
- **Width is a strong accelerant.** At 768 the crossing came at step
  13,100 and ran to 0.919. At 512 it had not sustained a crossing by
  24,000 and peaked near 0.198. Same phenomenon, roughly half the
  progress in nearly twice the steps.

Neither "pure run-length artifact" nor "pure width effect" survives. Run
length is necessary but not sufficient; width strongly accelerates.

*Auditor's correction, recorded rather than edited away:* mid-run on
08-14 I read seed 46's early health as trending against prediction 1 and
wrote that I had been "wrong to imply the phenomenon was probably lurking
past the horizon." That walk-back was premature — the phenomenon was
lurking, in one seed of three. The original supersession notice was
right; my correction of it was the error.

### PREDICTION 3 — CONFIRMED, and it is the result that matters

*"In any block that develops a soloist, `chorus_eff_rank` stays above 100
while `effective_rank` falls toward the floor: carrier, not collapse."*

Seed 97 block 0, over the same 24,000 steps in which `effective_rank`
fell 192 -> 149 and `top_dir_share` climbed 0.075 -> 0.198:
**`chorus_eff_rank` rose 220 -> 275.**

The two gauges moved in **opposite directions in the same block over the
same run**. On the gauge the kill criteria and the divergence rank veto
read, block 0 degraded by 22%. Behind the soloist, the representation got
23% richer. And capability agrees with the chorus, not the rank: seed 97
posted the **best probe of the three** (0.1336) while being the only seed
with a soloist.

This is the first live deployment of `chorus_eff_rank` (added 2026-08-14,
audit item A8) and it did prospectively what A7 could only reconstruct
from checkpoints.

**The reframe this forces:** soloist formation is not, by itself, the
disease. The project has been reading `tds` up / `eff` down as the
collapse signature since early August. It is the signature of *a* thing,
and chorus rank says which thing — the 768 family's seed 95 was genuinely
fatal (chorus falling 168 -> 61 with capability falling in step), while
this is the benign kind. Two states, identical on both primary
instruments, separated cleanly by the third.

### PREDICTION 2 — partially observable

Front-to-back ordering: only b0 is affected in seed 97, so there is no
b0-then-b1-then-b2 sequence to score yet. The final profile is
monotonically consistent with it (b0 worst, rising with depth). Not
scorable as stated; recorded as consistent.

### PREDICTION 4 — CONFIRMED

Episode stores stayed empty under `adaptive_episodes=False`, as expected.
Recorded, not gated.

### Registry consequences

1. The 2026-08-11 VISReg family's standing — **"depth-8 collapse SOLVED
   at 512d, provisionally"** — remains superseded. What replaces it:
   *no sustained soloist at 512d through one full epoch, in 3/3 seeds,
   with one seed in onset at the wire.*
2. **A new obligation:** the `chorus_eff_rank` result means every prior
   verdict that read `effective_rank` alone as collapse is open to
   re-reading. The 768x8 family is the first case (audit A7 already did
   it from checkpoints). Earlier families were not instrumented for it
   and cannot be re-read without re-running or re-encoding checkpoints.
3. **Un-blocked:** audit item B3 shipped the per-block chorus veto gate
   disarmed for want of a distribution. This run supplies three seeds of
   healthy-chorus data (231-275 at 512d). Still thin, but it is the start
   of the null the 2026-07-27 rule asks for before a criterion is frozen
   on an observable.

### Caveats

- Seeds ran under the **corroborated kill-6** (audit B5), as disclosed at
  registration. No seed was killed, so the change did not decide any
  outcome here.
- `probe_standardized` (audit B2) is absent: the driver was edited after
  this family launched, and a running process holds the code it loaded.
  The family's probe numbers are legacy-recipe throughout, which is
  internally consistent and comparable to the 512 and 768 families.
