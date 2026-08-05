# Rank trajectory at depth: failure to acquire, not forgetting

**Date:** 2026-08-05
**Author:** Opus 5, at Brian's request, during a design conversation about
whether to add per-weight plasticity partitions for identity protection.
**Instrument:** `scripts/rank_trajectory.py` (added with this doc).
**Data:** existing `runs/jepa_pilot/*/training_log.jsonl`. No new GPU time.

## The question

The 2026-05-16 plasticity-partitions design doc is DEFERRED pending one
condition: *empirically demonstrate that identity drift is a measured problem.*
The depth-8 rank collapse (effective rank ~2 of 512) looks, at a glance, like
exactly that evidence — a mind losing its representational structure.

Before building consolidation machinery on that reading, it has to be
distinguished from its opposite:

- **Rise then fall** = representational forgetting. Structure was acquired and
  lost. Consolidation / importance-weighted plasticity is the right family.
- **Never rise** = failure to acquire. There was never structure to lose.
  Consolidation protects nothing, and would harden a degenerate representation.

`effective_rank` is already logged per block on the deep cadence. The logs
answer it.

## Result

512d, Gutenberg-100, `probe_surprise` arm. The only registered difference
between the two groups is `n_blocks` (see Confounds for the one exception).

| run | n_blocks | block 0 | interior | final block | verdict |
|---|---|---|---|---|---|
| `probe_surprise_512d_seed45` | 4 | 187.9 → 217.9 | 103.8 → 158.9 | 113.9 → 186.8 | acquired |
| `probe_surprise_512d_seed46` | 4 | 222.9 → 229.4 | 142.3 → 166.8 | 146.1 → 176.5 | acquired |
| `probe_surprise_d8_512d_seed96` | 8 | 9.95 → 2.34 | 1.6 – 2.9 | 5.22 → 5.33 | **never cleared 20** |
| `probe_surprise_d8_512d_seed97` | 8 | 1.90 → 1.31 | 1.06 – 1.99 | 2.48 → 1.19 | **never cleared 20** |
| `probe_surprise_d8_amp4_512d_seed89` | 8 | 4.48 → 2.46 | 1.2 – 3.2 | 2.57 → 5.03 | **never cleared 20** |
| `probe_d8_amp4_rawdrive_512d_seed84` | 8 | 2.45 → 2.90 | 1.1 – 2.3 | 1.91 → 1.86 | **never cleared 20** |

At depth 4, every block sits between 100 and 230 and climbs. At depth 8, no
block in any run ever clears 20 — most sit between 1 and 5 for the entire run.

**The answer is failure to acquire.** There is no acquired structure at depth 8
for a consolidation mechanism to protect.

## The sharper finding: block 0 is already destroyed

The result above is not a cascade.

A depth cascade — signal degrading as it propagates through more blocks — would
show block 0 healthy and later blocks progressively worse. That is not what the
data shows. **Block 0 goes from ~223 at depth 4 to ~2–10 at depth 8.** The
first block in the stack, the one whose input is the token embedding and which
is furthest from any accumulated depth effect, is destroyed as thoroughly as
the last.

Whatever causes this is **global and depth-dependent**, not propagated. It hits
the input side of the trunk. That is consistent with the 2026-07-31
block-0-localized offset finding and narrows the suspect list to mechanisms
whose behavior changes with total depth *everywhere at once*:

- muPC residual scaling (a function of `n_blocks`, applied to every block)
- the backward pass / top-down sweep (chains the full stack; block 0 is the
  terminus of a longer chain at depth 8)
- the objective's grip on the trunk (SIGReg reaches block 0 through 8 blocks
  instead of 4, and the 2026-07-30 finding showed the projection head's
  singular-value mean drops 0.552 → 0.423 from d4 to d8, so SIGReg's hold on
  the trunk is *weaker* at depth)

It is not consistent with a per-layer mechanism accumulating with depth. The
2026-07-31 isolation doc ranked relative trust, the store fix, and the band as
"least likely" on exactly that reasoning; this measurement supports that
ranking.

## Confounds and limits — read before citing

1. **The deep cadence is 1000 steps**, so a 3000-step run gives three
   observations per block. These are coarse series. The "never cleared 20"
   verdict is robust to that (a block at 1.2 across the whole observed range is
   not hiding a rank-200 excursion), but individual rise/fall shapes between
   points are not.

2. **Steps 0–1000 are unobserved.** A rise-and-fall inside the first 1000 steps
   would not appear here. This does not threaten the "nothing to protect"
   conclusion — there is no structure across the bulk of training either way —
   but it *does* leave open whether depth-8 rank was destroyed early or never
   formed. That distinction matters for the mechanism hunt, and closing it is a
   cadence config change, not a new run.

3. **Not strictly one variable.** The d8 arms carry `grad_clip_norm=1000`; the
   d4 arms ran unclipped. Per the driver's own note, d8 gradient median is 1065
   vs 28.4 at d4, so the clip binds on roughly half of d8 steps. The
   `amp4` arms raised the clip to 20000 with 0% engagement and still show rank
   1.2–5.0, so clipping is not the whole story — but it is not a clean control.

4. `probe_surprise_d8_512d_seed98` and `seed99` were configured for 600 and 150
   batches, below the 1000-step deep cadence, so they contain **no rank data at
   all** despite being recorded `"completed"`. Correct behavior, but a trap: any
   short ablation rung needs `deep_interval_batches` lowered or it produces zero
   rank signal.

## Implications

**1. The plasticity-partition doc stays deferred, and this closes the most
tempting wrong path to un-deferring it.** Rank collapse at depth is not measured
identity drift. Worse, deploying MAS-style importance hardening into a collapsed
trunk would be actively harmful: at rank ~2 the surviving directions carry large,
stable activation and trivially low prediction error, so an importance metric
would score them as maximally important and harden them — cementing the collapse
while logging "consolidating identity." That is the silent-success shape this
project's CLAUDE.md names as the dominant risk, and it has already occurred twice
on this objective (BatchNorm neutering SIGReg, 2026-07-28; the Linear head
absorbing a 3x-hot trunk, 2026-07-30). **Rank is the gate on any
importance-weighted mechanism.**

**2. Block-0 rank is the sharpest available score for the mechanism ablation.**
Stage 25 confirmed on held-out NMSE while rank stayed at 1.9 — the metric was
gamed (commit `c6004dc`). Block-0 rank has a 100x separation between the healthy
and collapsed populations, it is already logged, and it is causally upstream of
everything else in the trunk. Score the ablation ladder on it.

**3. The missing control is bundle-off at depth 8.** Every depth-8 run in the
record carries the full mechanism bundle (backward pass, consolidation,
inverted-U gain, relative trust, adaptive episodes, homeostatic band, surprise
drive). Per the 2026-07-31 isolation doc, *"the deepest arm ever run before this
week is d4. There is no historical d8 or d12 run at all."* So the ladder's first
rung should be the run that has never been done: **depth 8 with the bundle
disabled.** If block-0 rank recovers, the bundle is implicated and the
add-back ladder is worth its GPU time. If it stays at ~2, no mechanism in the
bundle is the cause and the investigation belongs entirely to muPC and the
architecture — which would save the entire ladder.

## Reproducing

```
python scripts/rank_trajectory.py probe_surprise_512d_seed46 probe_surprise_d8_512d_seed96
python scripts/rank_trajectory.py --all-matching probe_surprise_d8
```

Read-only. Reports the deep cadence and the unobserved head of each run
explicitly, and drops the runner's `deep` metric (which duplicates the final
block) so no block is counted twice.
