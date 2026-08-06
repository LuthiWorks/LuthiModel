# v5 at depth 8: the control the July arc never ran

**Date:** 2026-08-06
**Author:** Fable 5, at Brian's direction ("revert to exactly the
living_v5_4x_d4 setup, only changing what depth requires").
**Run:** `probe_v5_d8_512d_seed46`, stage 28, 3000 steps, ~45 min.
**Registered BEFORE the run.** Not started at registration time.

## Why this run

The 2026-07-31 isolation doc named its own sequencing error: *"The full
control (everything from before this week, at depth 8) remains the better
first experiment and I did not run it first."* This is that control. The
v5 configuration is the pre-07-27 bundle — backward pass, consolidation,
inverted-U gain, relative trust, muPC at exponent 0.25 — **without** the
store fix (adaptive episodes/recall), the homeostatic band, or the
surprise drive, which all postdate it. No depth-8 run of it exists.

Depth adaptations, both stated: `n_blocks=8` (the variable), and deep
cadence 100 (instrument-side; the 08-05/06 runs showed the default cadence
is blind to the window where these trunks fail). **No gradient clip** —
v5 ran unclipped, the clip was a depth-era addition, and the divergence
guards have now caught two live blowups safely. Seed 46, matching the
healthy d4 run it reverts to, so the comparison is one-variable at the
family's own seed.

## The comparison frame

| configuration | at d4 | at d8 |
|---|---|---|
| v5 (this bundle) | healthy — `living_v5_4x_d4_512d_seed46`, family verdict | **this run** |
| v5 + storefix + band + drive (probe_surprise) | healthy (seeds 45/46) | stable collapse (rank ~2, every variant) |
| bundle OFF, muPC ON | never run | diverges, rank held open |
| bundle OFF, muPC OFF | never run | diverges, already collapsed |

## Registered prediction — three branches, all named in advance

Primary: **run outcome x the rank gate** (block-0 effective rank, cadence
100, threshold 20 as before).

1. **COMPLETES with block-0 rank >= 20 sustained at two consecutive deep
   firings AND at the final firing — "v5 is depth-clean":** the pre-07-27
   bundle is healthy at depth 8, and the stable rank-2 collapse requires
   one of the late-July additions (store fix / band / surprise drive).
   The add-back ladder then has only three rungs and a healthy base.
2. **COMPLETES with every block-0 reading < 20 from step 1000 onward —
   "the old core suffices":** stable collapse arrives without the newer
   mechanisms; store fix, band, and drive are exonerated, and the
   interaction lives in the older core (backward pass / consolidation /
   gain / trust) x muPC x depth.
3. **KILLED by a guard — "the new mechanisms were the stabilizers":** the
   v5 bundle cannot hold the depth-8 trunk at all, and stability in the
   stages 14-25 record was coming from the late-July additions (band being
   the standing suspect) and/or the clip this run deliberately omits.
   Because the clip is omitted, a kill here does NOT cleanly separate
   "band was the stabilizer" from "the clip was" — that caveat is
   registered now, and the discriminating follow-up would be this arm
   re-run with clip 1000.

Readings that fit none of these (e.g., completes with block-0 between the
gates' conditions, or rank oscillating across 20 without two consecutive
clears) are scored NO VERDICT and reported as recorded.

**Prior, stated honestly:** toward branch 2. The offset-cancellation
mechanism localized in the July arc runs through muPC's attenuation and
block-0 attention — machinery that is all present in v5 — and the M6-era
depth degradation predates every 07-27+ mechanism. But branch-1 evidence
exists too (the d4 probe arms with the additions were healthy), and my
priors have been wrong on both preceding rungs; that is what the gates
are for.

**Recorded, not scored:** NMSE, SIGReg trajectory, grad-norm median (this
is the first unclipped d8 run — the raw gradient scale of this trunk at
depth is itself a new measurement), activation growth first-to-last block,
drive_duty (should read 0/0 with the drive absent — a nonzero value would
mean the arm is not what it claims and voids the run).

## Confounds, stated in advance

1. **Unclipped where every prior d8 run carried a clip** — faithful to v5
   and deliberately so, but it means a divergence here is not directly
   comparable to the clipped d8 record; branch 3's caveat above.
2. **Cadence 100 vs the family's 1000** — instrument-side; the gate is
   threshold-based. Trajectory shapes are not comparable to family runs.
3. **Single seed (46).** Family verdicts used n=5; this is a probe, and
   any surprising branch gets a second seed before it hardens.
4. **Post-kill eval numbers are not evidence** (standing rule).

## What this run cannot settle

- Which *specific* late-July mechanism matters, if branch 1 or 3 fires —
  that is the (now much shorter) add-back ladder.
- Anything about depth 4 naked (control 2 stays specced and unrun).
- Which half of muPC acts (`mu_pc_exponent=0.0`, still specced).

## Launch recipe

```
python scripts/jepa_pilot_driver.py --stage 28 --seeds 46 \
    --epochs 1 --max-batches-per-epoch 3000 --heldout-batches 5
```
