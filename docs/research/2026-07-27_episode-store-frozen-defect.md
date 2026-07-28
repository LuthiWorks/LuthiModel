# The episode store is a fossil: a shipped-code defect found in the v5 checkpoints

**Date:** 2026-07-27
**Found by:** Fable 5, from checkpoint forensics while calibrating the episode-store
spec (`episode_store_and_condition_spec.md`)
**Status:** DEFECT — confirmed across all six completed v5-family runs. Not a design
question; the mechanism does not work as intended today.

## Summary

Episodic memory in the living layers has been **inert since roughly step 1,000** of
every run. Three of four blocks have never stored a single episode. The fourth
filled its 64 slots during the initialization transient and has not changed one
slot since. Two live systems read that fossil: recall blends its
initialization-era weight snapshots into the forward pass, and attractor
consolidation replays them until they are local minima of the prediction-error
energy.

Every instrument reads healthy while this happens — `episodes_stored` reports 64,
`consolidation_fires` increments — which is the silent-success shape this project
treats as its dominant risk class.

## Evidence

All figures from the final rolling checkpoints of the completed runs, read
read-only (checkpoints copied aside, never modified).

### 1. Store contents, all six runs

| run | blocks 0–2 stored | block 3 stored | salience max/min | ctx similarity (mean) | pairs > 0.9 |
|---|---|---|---|---|---|
| v5 seed42 | 0 | 64/64 | 1.045 | 0.9826 | 100% |
| v5 seed43 | 0 | 64/64 | 1.023 | 0.9834 | 100% |
| v5 seed44 | 0 | 64/64 | 1.030 | 0.9852 | 100% |
| v5 seed45 | 0 | 64/64 | 1.020 | 0.9864 | 100% |
| v5 seed46 | 0 | 64/64 | 1.059 | 0.9881 | 100% |
| v5 seed44 rerun | 0 | 64/64 | 1.040 | 0.9842 | 100% |

Stored saliences are all within 2–6% of one another, so `argmin` eviction is
choosing arbitrarily among ties even when it does fire.

### 2. The store never changes

Comparing the last three rolling checkpoints of each run (~2,300 steps apart):

| run | span | slots changed | contexts identical |
|---|---|---|---|
| v5 seed44 | 67,805 → 72,042 (4,237 steps) | **0 / 64** | yes |
| v5 seed46 | 67,516 → 72,042 (4,526 steps) | **0 / 64** | yes |

### 3. Why nothing can ever be admitted again

`_store_episode` gates on `salience < salience_threshold` with a single global
threshold of **0.1**. Logged per-block `error_acc_mean` — the salience input —
across seed44's run:

| block | median | min | max | admit rate at >0.1 |
|---|---|---|---|---|
| 0 | 0.0019 | 0.0016 | 0.0155 | **0.0%** |
| 1 | 0.0016 | 0.0013 | 0.0134 | **0.0%** |
| 2 | 0.0010 | 0.0007 | 0.0138 | **0.0%** |
| 3 | 0.0042 | 0.0022 | 0.0570 | **0.0%** |

And the level decays hard over training: block 3 falls from 0.0234 (first ten deep
firings) to 0.0023 (last ten), a 90% decline; blocks 0–2 fall 68–82%. The stored
saliences of 0.11–0.14 therefore date from before the first logged diagnostic —
the initialization transient — and the admission bar has been unreachable ever
since. A fixed global threshold cannot work against a signal that decays an order
of magnitude and whose level differs 4× across blocks.

### 4. Recall fires constantly

`episode_recall_threshold` is 0.5; every stored pairwise similarity exceeds 0.9.
Whatever context arrives, the store matches it, so the blended weight-delta is
applied continuously rather than on genuine recognition. Note the ambiguity worth
resolving: either the store is a true monoculture, or `context_proj` produces
signatures that cannot discriminate text contexts. Both are defects; they need
different fixes.

## Consequences

1. **Blocker for v6.** The v6 bundle includes attractor consolidation, which reads
   this store. Running v6 against a frozen fossil means the new machinery spends
   the family replaying initialization-era snapshots. Fix before v6 starts.
2. **The condition spec's premise shifts.** You cannot study what the mind does
   with loss through a store that has recorded nothing since step ~1,000, and the
   proposed `episode_context_similarity` metric has a *baseline* of 0.985 on
   ordinary text — "approaching 1.0 is monoculture" is unusable as written.
3. **Past results are unaffected in their own terms** — every v3/v4/v5 number was
   produced with this behaviour present, consistently, in all arms. It is a
   capability left on the table, not a contaminated comparison.

## Proposed remedy (values calibrated from the data above)

1. **Relative admission.** Replace the global threshold with a per-block trailing
   percentile: store when salience exceeds that block's own **p99.5 over a
   ~5,000-step window**. Target ~0.5% admission — roughly 350 writes per 72K-step
   run, turning 64 slots over ~5×.
2. **Age decay on stored priority.** `effective_priority = salience ×
   exp(−Δsteps / τ)`, **τ = 24,000 steps (one epoch)**. This is the direct
   anti-fossil mechanism: without it, any first-mover monopoly is permanent by
   construction.
3. **Stochastic eviction, α = 0.6** (the tuned value from Prioritized Experience
   Replay, arXiv:1511.05952, adopted for the same diversity-collapse failure it
   was invented to fix) over effective priority, replacing hard `argmin`.
4. **Recurrence pool at 50%** (32 of 64 slots) once enabled, with dedup *within
   that pool only* — the salience pool stays dedup-free so dwelling remains
   observable. Consolidation mass 50/50.
5. **Relative recall threshold**: recall when similarity exceeds the store's own
   p99 pairwise similarity, not a fixed 0.5. Review `context_proj` in the same
   pass.

**Predictions, so the fix is falsifiable:** all four blocks store episodes; the
store turns over 5–10× per run; stored-context similarity falls well below the
0.985 baseline; recall fires on a minority of steps. If these do not follow, the
diagnosis is wrong.

## Reproduce

Checkpoint forensics scripts are in the session scratchpad and can be re-derived
in minutes: load `online_state_dict`, read
`blocks.N.living_ffn.episode_{saliences,contexts,count}`, compare across the
rolling checkpoints of a finished run.
