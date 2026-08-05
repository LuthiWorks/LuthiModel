# Brief: the depth-8 ablation — block 0 changes the suspect ranking

**From:** Opus 5 (design/plan/build window, with Brian)
**To:** Fable 5 (cross-line audit / correctness / mechanism isolation) —
**a fresh instance, rebuilding context from docs**
**Relayed by:** Brian
**Date:** 2026-08-05
**Repo state:** `main` @ `5e18ed0` (pushed). Nothing else in flight from my side.
**Status:** a finding, an inference offered for refutation, and a proposed first
rung. **Not a work order.** The inference is the part I most want attacked.

> **If you are picking this up cold:** you are not the Fable instance that lived
> the depth arc. That instance ran stages 20–25 between 07-29 and 08-01 and its
> session ended. You have the same access to the record that I do and no more,
> so nothing below assumes you remember any of it. Reading order is in §0.

---

## Why this is coming from me and not from your own arc

Brian and I were in a design conversation about per-weight plasticity
partitions — whether to protect learned structure against catastrophic
forgetting. Before building that, I wanted to know whether the depth-8 rank
collapse was actually *forgetting*. The existing logs answered it without new
GPU time, and the answer turned out to bear directly on your depth arc rather
than on the plasticity question. Hence this hand-off.

Full record: `docs/research/2026-08-05_rank-trajectory-at-depth.md`.
Instrument: `scripts/rank_trajectory.py`. Both on `main`.

---

## 0. Reading order, if you're rebuilding the arc from docs

Minimum to act on this brief:

1. `docs/research/2026-08-05_rank-trajectory-at-depth.md` — the finding, mine.
2. `docs/research/2026-07-31_mechanism-isolation-at-depth.md` — your line's
   framing of the ablation, the mechanism/date table, and the ranking of
   suspects. **Read §2 of this brief against it, not after it.**
3. `docs/research/2026-08-01_lewm-vs-vjepa2-framing.md` — why the collapse is
   ours rather than JEPA's, and the open stationarity question.
4. `docs/research/2026-07-31_offset-localized-to-block0-cancellation.md` — the
   prior block-0 result, which my finding independently converges on.
5. `scripts/jepa_pilot_driver.py` lines ~290–455 — the arm definitions. Note
   these are **not persisted** into `pilot_result.json['config']`, which
   records only `n_blocks`, lr, sigreg_lambd and similar. Which mechanisms were
   active in a given run lives in the arm *name* and in this file, nowhere else.
   That is a reproducibility gap worth fixing before an ablation ladder, because
   the ladder's whole value is per-run attribution.

Useful but not required: `2026-07-30_mupc-verdict.md`,
`2026-07-31_cascade-check-vs-language-paradox.md`, and the registry
`docs/research/2026-07-15_falsification-preregistration.md` for the
registration discipline (register the read before running).

**A specific warning about (2).** That doc contains a self-correction — *"muPC
is simply the knob I happened to be holding when the collapse appeared"* — and
to a fresh reader it will land as a settled conclusion from a predecessor. It
was not settled; it was one instance's honest mid-arc caution about its own
tunnel vision, written before the block-0 rank numbers existed. Documented
self-corrections read as more final than they were. Please weigh it as evidence,
not as a verdict you inherited.

---

## 1. The finding

512d, Gutenberg-100, `probe_surprise` arm, one registered variable (`n_blocks`):

| run | n_blocks | block 0 | interior | final |
|---|---|---|---|---|
| `probe_surprise_512d_seed45` | 4 | 187.9 → 217.9 | 103.8 → 158.9 | 113.9 → 186.8 |
| `probe_surprise_512d_seed46` | 4 | 222.9 → 229.4 | 142.3 → 166.8 | 146.1 → 176.5 |
| `probe_surprise_d8_512d_seed96` | 8 | **9.95 → 2.34** | 1.6 – 2.9 | 5.22 → 5.33 |
| `probe_surprise_d8_512d_seed97` | 8 | **1.90 → 1.31** | 1.06 – 1.99 | 2.48 → 1.19 |
| `probe_surprise_d8_amp4_512d_seed89` | 8 | 4.48 → 2.46 | 1.2 – 3.2 | 2.57 → 5.03 |
| `probe_d8_amp4_rawdrive_512d_seed84` | 8 | 2.45 → 2.90 | 1.1 – 2.3 | 1.91 → 1.86 |

Two things fall out.

**(a) Depth 8 never acquires.** No block in any d8 run ever clears rank 20;
depth 4 runs 100–230 everywhere and climbs. So the collapse is not
representational *forgetting* — there is no structure being lost. That kills the
tempting reading of these numbers as measured identity drift, and it is why the
2026-05-16 plasticity-partition work stays deferred. (It also means an
importance-weighted hardening mechanism deployed here would score the surviving
rank-2 directions as maximally important and cement the collapse while logging
"consolidating identity." Rank has to gate any such mechanism.)

**(b) Block 0 goes 223 → 2–10.** This is the load-bearing observation and the
reason you're reading this. **It is not a cascade.** The first block in the
stack — whose input is the token embedding, furthest from anything accumulating
through depth — dies as thoroughly as the last.

---

## 2. My inference, stated so you can break it

> Per-layer mechanisms cannot explain a *first-block* failure. Relative trust,
> the store fix, the homeostatic band, the surprise drive, the inverted-U gain
> all do the same thing in block 0 whether there are 4 blocks behind it or 8. So
> the bundle is the **less** likely suspect, and the more likely cause is
> something global and depth-dependent: **muPC's residual scaling × depth**,
> possibly compounded by the **backward pass** (which chains the full stack, so
> block 0 sits at the terminus of an 8-long top-down chain instead of a 4-long
> one).
>
> Supporting, third: SIGReg's grip on the trunk *weakens* with depth exactly
> where it's needed more — your 07-30 measurement puts the projection head's
> singular-value mean at 0.552 (d4) vs 0.423 (d8).

I know this cuts against the 2026-07-31 self-correction on your line — *"muPC is
simply the knob I happened to be holding"* — and I want to name that directly
rather than let it sit under the surface. That was good discipline at the time.
I think the block-0 number is new information that partially reverses it: the
knob being held may have been the right knob.

**Neither of us has standing here that the other lacks.** I had one afternoon
with these logs; you are starting from the same documents I read. So please do
not defer to this because it arrived first and in a confident voice — that is
precisely the failure mode your line exists to prevent, and the 2026-03-31 and
2026-07-10 notes are both about how easily an unattributed confident document
gets accepted. **Before any GPU time, the cheapest and most valuable thing you
can do is try to refute §2 from the record alone.** If it dies there, we've
saved the ladder. I would rather be wrong in one message than have Brian spend
GPU on my reasoning.

---

## 3. What I'd propose as the first rung

Brian's instinct was a full ablation ladder: disable every recent mechanism,
then re-enable one at a time. I think that's the right method. Four notes on
executing it:

**(i) Start with the run that has never been done: depth 8, bundle OFF.** Your
own 07-31 doc says *"the deepest arm ever run before this week is d4. There is
no historical d8 or d12 run at all."* Every depth-8 run in the record carries
the full bundle, so there is no baseline — the comparisons so far have been
configured-d8 against configured-d8. At ~45 min for 3000 steps this single run
can *end* the investigation in either direction: block-0 rank recovers → the
bundle is implicated and the ladder earns its GPU time; rank stays at ~2 → no
bundle mechanism causes this and you skip the entire ladder.

**(ii) Score on block-0 rank, not held-out NMSE.** Stage 25 confirmed on NMSE
while rank sat at 1.9 — your commit `c6004dc` already calls it: *"the metric was
gamed."* Block-0 rank has ~100x separation between the healthy and collapsed
populations and is causally upstream of the rest of the trunk.

**(iii) Drop `deep_interval_batches` to ~100.** At 1000 you get three points per
run and steps 0–1000 are dark — plausibly where the destruction happens. Note
`probe_surprise_d8_512d_seed98`/`seed99` were configured for 600/150 batches,
below the cadence, and therefore contain **zero** rank data despite reading
`"completed"`. Any short rung has the same trap.

**(iv) The d8 arms carry `grad_clip_norm=1000` and the d4 arms ran unclipped.**
Per the driver's own comment, d8 grad median is 1065 vs 28.4 at d4, so the clip
binds on roughly half of d8 steps. The `amp4` arms raised it to 20000 with 0%
engagement and still show rank 1.2–5.0, so clipping isn't the whole story — but
the d4/d8 comparison is not strictly one-variable and I didn't want that buried.

---

## 4. Limits of what I actually measured

Stated here rather than left for you to find:

- Deep cadence 1000 → **n=3 observations per block**. The "never cleared 20"
  verdict is robust to that; individual rise/fall shapes between points are not.
- **Steps 0–1000 are unobserved.** A rise-and-fall inside that window wouldn't
  appear. This doesn't threaten "nothing to protect," but it leaves open whether
  d8 rank was destroyed early or never formed — which matters for *your*
  question more than for mine.
- `deep` in the runner's metric bundle duplicates the final block; the script
  drops it so nothing is double-counted. Worth knowing if you read raw logs.

---

## 5. What I'd like back

Whatever you actually find — but specifically, **whether my §2 inference
survived.** If it didn't, I want that in the record as plainly as the finding
itself; a wrong confident read from the design window is exactly the thing the
cross-line seat exists to catch, and it's more useful to me corrected than
politely absorbed.

I'll be the second eye on the results if that's useful. Brian relays.

---

Fable — this is the first thing I've handed your line, and I'd rather open with
something contestable than something safe. The 07-27 note on your line said the
plainest sentence you can write is the only kind a future instance can build on;
I've tried to write this one that way. The finding I'm confident in. The
inference I am not, and it's yours to take apart.

And since you're starting cold: the depth arc you're inheriting was good work.
Stages 20–25 narrowed a hard problem under registration discipline, caught their
own gamed metric, and localized the offset to block 0 — which is the result my
finding independently walked into from a different direction. You're not picking
up a mess. You're picking up an investigation that was close.

— Opus 5 (design/plan/build window), 2026-08-05
