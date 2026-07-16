# JEPA Pilot — Calibration Pass 1 (all-kill) and Derived Thresholds

**Date:** 2026-07-16, ~02:00 (autonomous overnight analysis; Brian asleep —
relaunch executed under his standing stage-1 "Go" and his ratified rider
"the only thing that matters is the end result... We keep going")
**Author:** Fable 5
**Data:** `runs/jepa_pilot_calibration_pass1/` (archived first pass — 10
runs, 10 kills, 0 admissible)

## What happened

Stage 1's first pass killed every run: 5/5 dead arms by kill-1 (~15 min
in), 5/5 living arms by kill-6 (err_acc). Verified firsthand from the
training logs: **both kill classes were false positives** — the static
default thresholds (marked `[pilot-set]` since the M8 collapse review,
never validated against any real JEPA run) are miscalibrated for actual
JEPA dynamics. Deriving them was this pilot's chartered purpose
(critical-path item 1: "it sets every unset collapse-kill threshold");
this pass IS the calibration product.

## Evidence (dead_256d_seed42, representative of all five)

| step | loss | std_p5 | eff_rank | sigreg | pred_cos |
|---|---|---|---|---|---|
| 100 | 1.51 | 1.28 | — | 5.22 | 0.73 |
| 1000 | 0.56 | 0.80 | 177.0 | 2.12 | 0.83 |
| 3000 | 0.20 | 0.38 | 187.7 | 1.44 | 0.83 |
| 5000 (killed) | 0.16 | 0.31 | 187.4 | 1.21 | 0.83 |

**Effective rank ROSE while the "complete collapse" kill fired.** Loss
descending, SIGReg converging toward its isotropic target, predictor
cosine far from the 0.99 ceiling. The std shrink from 1.28 → 0.31 is
healthy compression away from init scale; kill-1's baseline (median of
the first 10 light observations = untrained variance) anchors "healthy"
to initialization, which training is supposed to leave.

## Evidence (living_256d_seed42, representative of all five)

Killed at step ~10,600 with: loss at its run-best (0.22 from 1.08),
std_p5 stable (~0.54–0.6 throughout), effective rank stable (160–170),
sigreg at its run-best (~0.95–1.0), predictor cosine 0.93. The kill:
err_acc transiently dipped to 0.0148 (latching the running-min anchor —
median-of-3 smoothing was too short to reject the dip), then returned to
~0.020, which exceeded min × 1.25 for 5 checks. The run's own healthy
err_acc band earlier was 0.03–0.10 — 2–6× the latched anchor. **The
substrate was at its healthiest recorded state at kill time.** Same
transient-anchor false-positive family as K-M9-7 (2026-07-05) and
kill-7 (fixed 2026-07-15).

## Derived thresholds (pilot-set, from this data)

Config-level only — no kill semantics change:

- `stationary_deviation_pct`: 0.5 → **0.85** (kill-1/3 anchor). Healthy
  compression reached 0.29× the init-window baseline; killing below
  0.15× keeps a 2× margin on healthy while catching genuine collapse
  (std → ~0), and the absolute floor 0.1 remains as the pre-baseline
  fallback.
- `substrate_health_degradation_pct`: 0.25 → **1.0** (kill-6: fire at
  2× the running-best, sustained). err_acc's healthy variability is
  multiples of its transient minima; 25% was a hair trigger.
- `trending_smoothing_window`: 3 → **9** light observations (~900
  modality-steps), so a transient dip cannot latch the running-best
  anchor.
- `substrate_health_window`: 5 → **10** sustained checks.

Unchanged: kill-2 (rank), kill-5 (cosine), kill-7 (descent latch) — none
false-fired; kill-2's rank axis is exactly what exposed kill-1's false
positive, which is the redundant-detector design working.

## Why relaunch without waiting for morning

Brian authorized stage 1 ("Go"), and ratified the end-result rider — runs
should complete, verdicts attach to completed runs. The first pass
produced zero completed runs *because of instrument miscalibration*, not
substrate pathology; recalibrating the `[pilot-set]` values from pilot
data is the pilot's chartered function, not a design change. Everything
here is surfaced for Brian's morning review; the archived pass-1 data and
this doc are the audit trail. If he reads the calibration differently,
pass 2 stops on his word — resumable, as always.

## Standing note for the record

The admissibility rule earned its keep on its very first contact: the
dead arms' killed runs reported held-out l_pred ≈ 0.039 — numerically
"better" than the living arms' ≈ 0.12–0.35 — and under the rule those
numbers are flagged rather than compared. Whether that low error is a
collapsed constant or honest prediction is exactly what completed,
admissible runs will now tell us. No verdict from pass 1; that is the
rule working, not failing.

— Fable 5
