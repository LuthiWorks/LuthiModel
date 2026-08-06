# Recovered data from the deleted probe runs

**Date:** 2026-08-05
**Author:** Opus 5
**Why this exists:** the `runs/jepa_pilot/probe_*` directories were accidentally
deleted on 2026-08-05. They are not in the Recycle Bin, `runs/` is gitignored
(`.gitignore:9`) so git never held them, and a search of C:, D:, and E: found no
copies. The E: experiment archive stops at 2026-07-18, before these runs existed.
**They are gone.**

Earlier the same day, before the deletion, I read effective-rank trajectories out
of those logs while investigating whether the depth-8 collapse was forgetting or
failure to acquire. **The values below are transcribed from that tool output.**
For the runs marked LOST they are now the only surviving per-block record.

---

## READ THIS BEFORE USING ANY NUMBER HERE

**Precision is degraded and not uniform.** These are transcriptions of formatted
console output, not the original logs.

- Values at a block's **first, peak, and last** observation were printed at
  2 decimal places and are given here as printed.
- **Intermediate** observations were printed at 1 decimal place only, and appear
  here at 1 decimal place. They are marked with `~`.
- The original `training_log.jsonl` held full float precision. That is gone.

**Do not do fine numerical work on these.** They support ordinal and
order-of-magnitude claims — "depth 4 runs 100–230, depth 8 runs 1–10" — and not
much more. Any conclusion needing better than ~1% precision cannot be drawn from
this document.

**Coverage is partial.** Only runs I happened to query survive here. Only blocks
I happened to print survive for `seed97`. Only the deep cadence (every 1000
batches) was ever in scope; the light-cadence metrics at every 100 batches
(loss, `l_pred`, `l_sigreg`, `online_std`, substrate health, drive gain, err_acc)
were never read this session and are **entirely lost** for every deleted run.

---

## Inventory

**Survives on disk (full fidelity, read it directly — do not use this doc):**

- `probe_d8_amp4_rawdrive_512d_seed84`

**Lost, partially recovered below:**

| run | n_blocks | recovery |
|---|---|---|
| `probe_surprise_512d_seed45` | 4 | all 4 blocks, 4 observations |
| `probe_surprise_512d_seed46` | 4 | all 4 blocks, 4 observations |
| `probe_surprise_d8_512d_seed96` | 8 | all 8 blocks, 3 observations |
| `probe_surprise_d8_512d_seed97` | 8 | **blocks 0/1/3/7 only** |
| `probe_surprise_d8_amp4_512d_seed89` | 8 | all 8 blocks, 3 observations |

**Lost with no rank recovery at all** (I confirmed these existed but never read
their rank data):

- `probe_surprise_d8_512d_seed98` — configured 600 batches, below the 1000-step
  deep cadence, so it contained **no rank records even before deletion**.
  Recorded `outcome: "completed"`, 7 log records, max_step 600.
- `probe_surprise_d8_512d_seed99` — configured 150 batches, same situation.
  `outcome: "completed"`, 2 log records, max_step 150.
- `probe_storefix_512d_seed42`, `seed43`, `seed44`, `seed45`
- `probe_surprise_d8_amp2_512d_seed90` (`mu_pc_rate_power=-2.0`)
- `probe_surprise_d8_amp8_512d_seed88` (`mu_pc_rate_power=-8.0`)
- `probe_surprise_d8_amplified_512d_seed91` (`mu_pc_rate_power=+1.0`)
- `probe_surprise_d8_balanced_512d_seed92`

**Inventory caveat, stated rather than hidden:** my directory listing was
truncated at 20 lines, so the list above may be incomplete. Arms exist in
`scripts/jepa_pilot_driver.py` for `probe_surprise_d8_embscale`,
`probe_surprise_d8_bplr` and `probe_surprise_d8_bplr0` (stages 22–24); whether
run directories existed for them, I do not know. The 07-31 research docs report
results for those stages, so the runs almost certainly happened.

---

## Recovered rank trajectories

`effective_rank` per block, deep cadence 1000 batches. `~` = 1-decimal
transcription.

### `probe_surprise_512d_seed45` — depth 4, arm `probe_surprise`, 4000 batches

| block | 1000 | 2000 | 3000 | 4000 |
|---|---|---|---|---|
| 0 | 187.85 | ~202.4 | 219.79 | 217.88 |
| 1 | 151.62 | ~158.9 | 181.27 | 176.16 |
| 2 | 103.77 | ~128.6 | ~153.8 | 158.94 |
| 3 | 113.86 | ~151.1 | ~179.7 | 186.80 |

### `probe_surprise_512d_seed46` — depth 4, arm `probe_surprise`, 4000 batches

| block | 1000 | 2000 | 3000 | 4000 |
|---|---|---|---|---|
| 0 | 222.85 | ~228.9 | ~224.4 | 229.38 |
| 1 | 180.67 | ~197.8 | 201.06 | 190.16 |
| 2 | 142.29 | 177.00 | ~166.6 | 166.79 |
| 3 | 146.07 | 184.05 | ~179.4 | 176.50 |

### `probe_surprise_d8_512d_seed96` — depth 8, arm `probe_surprise_d8`, 3000 batches

| block | 1000 | 2000 | 3000 |
|---|---|---|---|
| 0 | 9.95 | ~2.1 | 2.34 |
| 1 | 2.01 | ~1.7 | 1.61 |
| 2 | 1.95 | ~1.7 | 1.76 |
| 3 | 2.75 | ~2.2 | 2.95 |
| 4 | 3.27 | ~2.3 | 3.58 |
| 5 | 3.58 | ~2.3 | 4.24 |
| 6 | 3.30 | ~2.2 | 4.62 |
| 7 | 5.22 | ~2.6 | 5.33 |

### `probe_surprise_d8_512d_seed97` — depth 8, arm `probe_surprise_d8`, 3000 batches

**Blocks 2, 4, 5 and 6 were never printed and are permanently lost.**

| block | 1000 | 2000 | 3000 |
|---|---|---|---|
| 0 | 1.90 | 2.51 | 1.31 |
| 1 | 1.28 | 1.99 | 1.06 |
| 2 | — | — | — |
| 3 | 1.39 | 1.93 | 1.15 |
| 4 | — | — | — |
| 5 | — | — | — |
| 6 | — | — | — |
| 7 | 2.48 | 2.73 | 1.19 |

### `probe_surprise_d8_amp4_512d_seed89` — depth 8, arm `probe_surprise_d8_amp4` (`mu_pc_rate_power=-4.0`), 3000 batches

| block | 1000 | 2000 | 3000 |
|---|---|---|---|
| 0 | 4.48 | ~3.2 | 2.46 |
| 1 | 1.42 | ~1.2 | 2.49 |
| 2 | 1.29 | ~1.2 | 3.14 |
| 3 | 1.25 | ~1.2 | 3.24 |
| 4 | 1.23 | ~1.2 | 3.11 |
| 5 | 1.22 | ~1.2 | 2.85 |
| 6 | 1.22 | ~1.2 | 3.11 |
| 7 | 2.57 | ~3.1 | 5.03 |

### `probe_d8_amp4_rawdrive_512d_seed84` — SURVIVES ON DISK

Reproduced here only so the comparison set is complete in one place. **Read the
actual run directory, not this table.**

| block | 1000 | 2000 | 3000 |
|---|---|---|---|
| 0 | 2.45 | ~2.4 | 2.90 |
| 1 | 1.12 | 1.43 | 1.21 |
| 2 | 1.36 | 1.52 | 1.30 |
| 3 | 1.85 | 1.94 | 1.29 |
| 4 | 2.05 | 2.26 | 1.35 |
| 5 | 2.43 | ~2.3 | 1.34 |
| 6 | 2.40 | 2.45 | 1.35 |
| 7 | 1.91 | 3.43 | 1.86 |

---

## Recovered run configuration

Read out of `run_config.json` / `pilot_result.json` before deletion.

**`probe_surprise_d8_amp4_512d_seed89`** (fullest capture):

```
arm                       probe_surprise_d8_amp4
d_model                   512
n_blocks                  8
epochs                    1        max_batches_per_epoch    3000
batch_size                32       seq_len 128   stride 64
lr                        3e-4     cosine_lr true   lr_total_steps 24014
holdout_fraction          0.02
sigreg_lambd              0.2      sigreg_knots 17   sigreg_num_proj 1024
context_fraction          0.8
grad_clip_norm            20000.0
mu_pc_rate_power          -4.0     (from jepa_pilot_driver.py, NOT persisted)
taper                     enabled, start_fraction 0.5, floor 0.2
lr_schedule               enabled, min_lr_ratio 0.1
corpus                    gutenberg_100 / gutenberg_4x_filelist, text 98,359,168 tokens, alpha 0.7
logging                   light 100, deep 1000, heldout_eval_batches 5
divergence                nmse_max 2.0, loss_mult 10.0, baseline_points 10, sustained_points 3
kill criteria             warmup 5000; std_collapse 0.1; correlation_collapse 0.95;
                          cosine_collapse 0.99; substrate_health_degradation_pct 1.0 / window 10;
                          dimensional_collapse_threshold_pct 0.5; pilot_set_n 10;
                          stationary_deviation_pct 0.85; trending window 9 / warmup 5;
                          deep trending window 2 / warmup 2; loss_descent_window 5000;
                          kill7_descent_margin 0.01; collapse_sustained 3; dimensional_sustained 5
```

**`probe_surprise_512d_seed45`** differs as: `n_blocks=4`,
`max_batches_per_epoch=4000`, `heldout_eval_batches=50`, and **no
`grad_clip_norm` and no `divergence_*` keys at all** — the depth-4 arms ran
unclipped. This is the confound noted in the 08-05 rank doc.

**A structural finding worth keeping even though the runs are gone:** the
persisted config does **not** record which substrate mechanisms were active.
Diffing `pilot_result.json['config']` between the healthy d4 run and the
collapsed d8 run showed `n_blocks` as the *only* difference — but that is an
artifact of the record, not the truth: `mu_pc_rate_power`, `drive_mode`, relative
trust, the homeostatic band and the rest live only in the arm *name* and in
`scripts/jepa_pilot_driver.py`. **Fix this before the ablation ladder runs**, or
the ladder's per-run attribution will be as unreconstructable as these logs are.

---

## Definitively lost

- All light-cadence metrics (every 100 batches) for every deleted run: loss,
  `l_pred`, `l_sigreg`, `online_std`, substrate health, `err_acc`,
  `drive_gain`, `drive_fire_count`, gradient norms, clip engagement rate.
- All held-out NMSE and probe results except those quoted in the surviving
  research docs (e.g. stage 20 base NMSE 0.8919, in
  `2026-07-31_mechanism-isolation-at-depth.md`).
- Blocks 2, 4, 5, 6 of `seed97`.
- Every checkpoint.
- The ability to ask these runs a *new* question. That is the real loss: the
  rank finding exists only because the logs could be re-interrogated after the
  fact with a question nobody had when they were written.

## What survives elsewhere

The scientific conclusions are intact — they were written down:

- `docs/research/2026-08-05_rank-trajectory-at-depth.md` — the rank finding
- `docs/research/2026-07-29_*`, `2026-07-30_*`, `2026-07-31_*`,
  `2026-08-01_*` — stage verdicts, the muPC ladder, offset localization,
  the cascade check
- `docs/research/2026-07-15_falsification-preregistration.md` — registered
  predictions and outcomes
- `scripts/jepa_pilot_driver.py` — every arm definition, so any run is
  re-launchable

## On re-running

Reruns would produce *new* data, not this data. Per the 2026-07-27 finding,
`precision_spread` is chaotic: two runs at identical configuration and identical
data order diverge 70.8% by the late phase while loss stays within 2.5%. Loss and
final outcomes reproduce; substrate observables do not.

The recommendation in the 08-05 rank doc stands and is now cheaper to justify:
do not bulk-regenerate. The ablation ladder is producing fresh depth-8 runs
anyway, and it wants `deep_interval_batches` at ~100 rather than 1000 — so those
runs will be strictly better instrumented than the ones lost here.

---

## Addendum (Fable 5, same day): the primary source for this transcription survives

The tool output transcribed above was independently recovered, verbatim, from
the authoring session's transcript before this doc was written, and is
preserved with the full session JSONL at:

    E:\ClaudeContinuityBackup\2026-08-05_rank-evidence\

Anyone doubting a value here can check it against the instrument's actual
stdout rather than trusting a transcription from context. Spot-checks at
review time (seed46, seed96, seed97 block 0, seed89, seed84) matched exactly.
The E: copy also carries the seed97 `--all-matching` output and the ad-hoc
seed89/seed84 dump in their original formatting. Two independent recovery
paths, one on each model line, converging on the same numbers is about as
strong as provenance gets for data that no longer exists — but neither copy
restores the light-cadence metrics, and the "do not do fine numerical work"
warning above applies to both equally.
