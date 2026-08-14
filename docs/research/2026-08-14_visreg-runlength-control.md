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
