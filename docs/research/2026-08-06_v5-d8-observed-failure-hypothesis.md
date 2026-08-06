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

---

# RECORD: the observed trajectory, 26 deep firings (killed at step ~2600)

**Outcome:** `killed:divergence:text:nmse=2.0650>2.00`, 0.58 h, 26 firings —
the registered observation succeeded; every read below is from the frozen
list.

**The trajectory in three movements:**

1. **Collapse transit (100→200):** rank 238 → 11 (block 0), tail to ~1.
   The violent transient that killed all three un-delayed runs at their
   first guard check. Suppressed-trip markers: nmse 14.13@100, none@200
   (NMSE dipped UNDER 2.0 at maximum degeneracy — the guard-inversion
   caught inside a single run), 3.49@300.
2. **Slow systemic healing (200→2500):** SIGReg 2546 → 132-213 (touching
   the healthy band), offset dominance 0.997 → 0.239 (the July disease
   genuinely stripped), min-rank 1.1 → 3.1, block 7 re-inflating to ~12.
   Guards went live at 1000 and found nothing to kill for 1600 steps.
3. **Relapse (2600):** SIGReg 213 → 2565, grad 404 → 1103, the re-inflated
   tail crushed back to rank 1-2, offset re-saturating to 0.652. The
   marginal NMSE trip (2.065) is the leading edge of THIS event — an
   earlier reading of the kill as "punishing recovery" was wrong and is
   corrected here.

**The reframe, stated carefully:** yesterday's branch-3 verdict ("the v5
bundle cannot hold the depth-8 trunk") now looks like a **guard-timing
artifact**. Given room, v5-d8 does not diverge to destruction — it
oscillates: collapse, partial heal, relapse. Whether the stages-14-25
"stable collapse" cells differ from this in kind or only in guard timing
is now an open question that infects the whole factorial table. The
guard's NMSE also demonstrably inverts health at the collapse floor
(quietest exactly when sickest).

**Extension (Brian's standing order, trigger met):** the healing movement
qualifies as "signs it might have recovered." `probe_v5_d8_dk5000`:
byte-identical, guard_min_step=5000, 6000 steps so the guards get a live
window. Frozen reads: (a) do relapse events recur, at what spacing; (b)
does between-event healing compound (each cycle's rank floor/peak higher)
or reset; (c) does any block sustain rank >= 20 at two consecutive
firings plus the final one; (d) SIGReg/offset envelopes across cycles.

---

# RECORD: the dk5000 extension — 50 firings, and the answer is no

**Outcome:** `killed:divergence:loss=3678>10.0xbaseline@5000` — the guards
woke at 5000 and killed on their first check. 0.91 h, 50 deep firings, the
longest depth-8 tape in the record.

## The frozen reads, scored

**(a) Do relapse events recur?** One relapse (step 800), and then — nothing.
From 800 to 5000, 4200 steps, the substrate sits in a static deep-collapse
state: every block rank 1-4, SIGReg pinned at ~5300-5600, offset dominance
saturated at 0.999-1.000 from step 2400 onward. **The "cycles" I narrated
live from the suppressed-trip NMSE series (2.6 → 1695 → 35 → 634…) were
noise over a statically collapsed trunk — prediction error thrashing on a
rank-2 space — not regime events. The substrate series shows no oscillation
after 800.** That correction is this record's first duty: NMSE told a story
of churn and escalation; rank, SIGReg, and offset show a flat floor. The
instrument-inversion finding claims another reading, this time mine, made
in real time while I knew better.

**(b) Does between-event healing compound?** No — inverted. This path's one
heal (offset briefly 0.355 at step 600) was weaker than its twin's, and
after the 800 relapse healing never resumed at all.

**(c) Any block ≥ 20 sustained?** No block cleared 20 at any firing after
step 100. Max-rank envelope by decade of firings: 2.9 → 4.1 → 2.8.

**(d) Envelopes:** SIGReg plateau ~5400 (stable, high — an order above the
50-110 band); offset saturated at 1.000; loss stable in the 4-6k range —
which is >10x a baseline frozen during the healthy-looking first firings,
so the live guards kill this state on sight, forever.

## What the pair of runs actually established

1. **Recovery is not the attractor. The collapsed state is.** Brian's
   question — "might it have recovered with more room?" — is answered: on
   this path, 4200 unguarded steps produced a permanent rank-2 floor. The
   dk1000 twin proved recovery *movement* exists (offset 0.997 → 0.239,
   SIGReg into the band's neighborhood); this run proves that movement is
   not destiny — same config, same seed, GPU nondeterminism alone chose
   between a long heal and a permanent floor. The 07-27 chaos finding
   operates at the level of whole regimes.
2. **The stable collapse is reachable from v5 alone.** The dk5000 endpoint
   — stable, non-divergent, rank ~2 for thousands of steps — is the
   stages-14-25 phenomenon, produced with no late-July mechanisms and no
   clip. The "what stabilizes the collapse" question from the rung-1
   verdict largely dissolves: nothing extra is required; the trunk finds
   the floor by itself when the guards don't interrupt the transit. What
   the bundle's newer mechanisms and the clip change is at most the
   *transit* (violence, timing), not the destination.
3. **Guard-timing was the hidden variable across the whole factorial**, now
   demonstrated rather than suspected: every "diverges" cell in the 08-05/06
   table is a first-check snapshot of a transit whose destination — when
   given room on two paths — was the same floor.
4. **No further extension is warranted.** The trigger standard ("slightest
   signs it might have recovered") is not met by this tape: rank flat,
   SIGReg flat, offset saturated, for 4200 steps. The next spend belongs to
   design, not observation.

## Open to the design seats (Brian + Opus, with these verdicts in hand)

- Why depth 8 makes the rank floor the attractor when depth 4's attractor
  is healthy — the mechanism question, unchanged but now cleanly posed.
- Whether the muPC-on transit (rank held open at 238 while scale runs away)
  can be *caught* — it is the only cell that ever showed high rank and
  working prediction past step 100.
- Guard redesign: NMSE demonstrably inverts at the floor; a rank- or
  SIGReg-conditioned term is the obvious candidate and touches registered
  surface.
