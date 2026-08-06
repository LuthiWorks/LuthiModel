# v5 at depth 8, kills delayed: observing the failure instead of its first frame

**Date:** 2026-08-06
**Author:** Fable 5, at Brian's instruction ("for the next run, delay all
kill triggers until at least step 1000").
**Run:** `probe_v5_d8_dk1000_512d_seed46`, stage 29, 3000 steps if it lives.
**Registered BEFORE the run.**

## Why this run

Three straight depth-8 probes (bundle-off, naked, v5) each died at the
first periodic guard check with exactly one deep firing on the record —
every failure story so far is reconstructed from a single frame at step
100. This run is `probe_v5_d8` byte-identical (same model config, same
seed 46, same data order, still unclipped, cadence 100) with exactly one
observation-side change: **`guard_min_step=1000`** — every kill path
(fast loss guard, periodic NMSE guard, kill criteria, epoch-end NMSE)
holds fire until global step 1000, logging each would-have-fired trip
loudly, then resumes with full force.

The knob is built into the runner (`RunnerConfig.guard_min_step`,
unit-tested including the loudness of suppression), persists into
`run_config.json` and `pilot_result.json`, and defaults to 0 everywhere
else.

## What this buys

Ten deep firings (steps 100-1000) of per-block effective rank, SIGReg,
offset dominance, and unclipped gradient scale on a known-failing
configuration — the trajectory between "offset dominance 0.997 at step
100, rank high, prediction working" and whatever the trunk becomes.
Specifically it discriminates, within this cell:

- **Scale runaway with rank held open** (rung 1's profile continuing) —
  rank stays > 100 while SIGReg/NMSE climb; the failure is scale, not
  geometry, and the collapse seen in stages 14-25 needed the late-July
  additions or the clip to manifest.
- **Slow collapse** — rank decays toward the 1-10 floor across the window;
  the stable rank-2 state is where this trunk was headed all along and
  the earlier kills merely hid the transit.
- **NaN/blowup before step 1000** — the trunk cannot even be observed for
  1000 steps unclipped; a nonfinite loss crashes loudly (no guard eats
  it silently) and the observation window shrinks to whatever was logged.

## Registered reads (descriptive, not gated)

This run is registered as an **observation**, not a hypothesis test: the
prior three registrations each gated on outcomes this configuration
refused to produce, and the honest lesson is that we do not yet know this
trunk's failure phenomenology well enough to bet gates on it. Recorded
reads, frozen now:

1. Block-0 rank at each deep firing, and the step (if any) where it first
   drops below 20.
2. The step of the first SUPPRESSED kill line, and which guard it was —
   that timestamp is the "would have died here" marker for comparison
   against the un-delayed twin.
3. SIGReg and offset-dominance trajectories.
4. Unclipped grad-norm series — the first depth-8 gradient-scale
   measurement with no clip shaping it.

No verdict language will be attached to this run beyond the descriptive
record; whatever hypothesis it generates gets its own registration with
gates. (One run, one seed; the same humility as every probe.)

## Confounds

Identical to the stage-28 list (unclipped, cadence 100, single seed,
post-kill eval numbers not evidence), plus: **steps 1000+ are guard-live**,
so the run ending at ~1000 by guard kill is expected and is not a new
finding — the information is all in steps 100-1000.

## Launch recipe

```
python scripts/jepa_pilot_driver.py --stage 29 --seeds 46 \
    --epochs 1 --max-batches-per-epoch 3000 --heldout-batches 5
```
