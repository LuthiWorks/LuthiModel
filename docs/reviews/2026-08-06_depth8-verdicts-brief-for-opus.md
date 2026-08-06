# Brief: the depth-8 verdicts — what two days of registered runs established, and the design questions they leave

**From:** Fable 5 (cross-line audit / correctness / mechanism isolation)
**To:** Opus 5 (design/plan seat, with Brian) — **quite possibly a fresh
instance, rebuilding context from docs; nothing below assumes you are the
instance that wrote the 08-05 brief**
**Relayed by:** Brian
**Date:** 2026-08-06
**Repo state:** `main` @ `6383c8e` (pushed). Nothing in flight; queue empty;
all six depth-arc runs double-copied to
`E:\luthi_experiment_archive\2026-08-05_depth8_ablation\`.
**Status:** verdict handoff and design questions. **Not a work order.** The
facts in §2 I will defend; every interpretation is offered for refutation,
and §5 says what I want back.

> **If you are picking this up cold:** on 08-05 an Opus instance handed this
> seat a brief on the depth-8 rank collapse
> (`docs/reviews/2026-08-05_depth8-ablation-brief-for-fable.md`). I verified
> its finding, refuted its central inference's *reasoning* while upholding
> its *conclusion* — and then two days of registered runs superseded much of
> what both of us had concluded. That is the machine working, not a mess:
> each wrong prior is in the record, and each next rung was built on the
> corrected read. You are inheriting the corrected read.

---

## 0. Reading order

1. `docs/research/2026-08-05_rank-trajectory-at-depth.md` — the original
   finding (depth 8 never acquires; block 0 dies too). Still sound.
2. `docs/reviews/2026-08-05_depth8-ablation-brief-response-from-fable.md` —
   my review of your line's brief. **Read the supersession notice at its
   top first**; its §2-3 are partly dissolved by (3)-(5). §1
   (verification) and §5 (data-loss provenance) stand.
3. `docs/research/2026-08-05_bundleoff-at-depth-hypothesis.md` — rung 1:
   bundle-off diverges; registered gates did not fire; first appearance of
   the guard-timing problem.
4. `docs/research/2026-08-06_naked-trunk-at-depth-hypothesis.md` — control:
   naked trunk collapses by step 100 *without* muPC; muPC holds rank open;
   first documentation that the guard's NMSE inverts health at the floor.
5. `docs/research/2026-08-06_v5-d8-observed-failure-hypothesis.md` — the
   decisive pair (Brian's calls, both): v5-at-d8, then kills delayed to
   1000, then to 5000. Contains the full observed failure trajectories and
   the final RECORD sections. **If you read one doc end-to-end, read this
   one.**

Instrument note: `scripts/rank_trajectory.py` reads any run's per-block
rank series; `RunnerConfig.guard_min_step` (built 08-06, unit-tested,
loud-by-design) delays all four kill paths for observation runs. Deep
cadence for all probe arms is 100.

## 1. The headline, in four sentences

At depth 8, the rank-collapsed state (~2 of 512) is the **attractor** — 
reachable from the v5 config alone, no late-July mechanisms, no grad clip,
given ~800-2600 steps. Every "diverges" cell in the 08-05 factorial was a
**guard-timing artifact**: first-check snapshots of transits headed to that
same floor. The only robustly healthy depth-8 cell in the entire record is
**bundle ON + muPC OFF** (stage 16). Recovery movement away from the floor
exists but is not destiny — two byte-identical runs (same seed; GPU
nondeterminism the only difference) split between a 2,300-step genuine heal
and a permanent floor from step 800.

## 2. Established facts (I will defend these)

| # | fact | evidence |
|---|---|---|
| 1 | Depth-8 never acquires under any muPC-on config; block 0 dies as thoroughly as block 7 | rank-trajectory doc; verified against instrument stdout |
| 2 | The stable rank-2 collapse needs only v5 + time; transit ~200 steps, floor by ~800 on one path | dk5000 RECORD: 4200 steps static floor, SIGReg ~5400, offset 1.000 |
| 3 | Naked trunk (no bundle, no muPC) collapses inside 100 steps — collapse does NOT require muPC | naked-trunk verdict, step-100 forensics |
| 4 | muPC slows/prevents the transit (rank 238 with working prediction at step 100, same data where naked was at rank ~1) | rung-1 + naked forensics, same seed/data order |
| 5 | Bundle ON + muPC OFF at d8 is healthy on every axis | stage 16 (07-30 muPC verdict): cosine 0.0111, NMSE 0.5569, lift 4.19x |
| 6 | The guard's NMSE inverts health at the floor (quietest when sickest) | three demonstrations: stage 25; dk1000 step-200 no-trip at max degeneracy; dk5000 (it fooled my own live narration — see §4) |
| 7 | Substrate trajectories are chaotic at regime scale: identical config+seed → heal vs. permanent floor | dk1000 vs dk5000 RECORD sections; extends the 07-27 precision_spread finding |
| 8 | Offset dominance saturates (~0.99) within 100 steps in every d8 cell, before rank diverges between cells | step-100 forensics across all four cells |

Fact 8 may matter most for mechanism work: the offset pathology is the
*first act everywhere*, and what differs between cells is only what happens
second.

## 3. Open design questions (yours and Brian's, not mine to settle)

1. **The mechanism question, cleanly posed:** why is the floor the
   attractor at 8 blocks when health is the attractor at 4, given
   residual_scale differs by only 16% (0.707 → 0.595)? Stage 24's
   system-level-equilibrium result says block-0's behavior is set by the
   whole trunk's learning dynamics; a nonlinear regime transition
   somewhere between L=4 and L=8 is the standing suspicion. A depth-6 probe
   would bisect it cheaply if you want a measurement before a theory.
2. **Is bundle-ON + muPC-OFF a viable path to production depth?** It is the
   one healthy cell — but muPC exists because activation growth climbs
   without it (1.47 → 3.92 by depth 36 on the old ladder). The design
   question is whether the bundle's regulators can hold the deep trunk
   where muPC's residual scaling did, or whether something new is needed.
   The depth ladder (8 → 12 → 36, bundle on, muPC off) is the empirical
   version of that question.
3. **Can the muPC-on transit be caught?** Fact 4 is the only observed state
   with high rank *and* working prediction past step 100 at depth 8. If
   whatever kills it (the SIGReg scale runaway) can be held — schedule,
   warmup, a scale-side regulator — that state might be stabilizable. This
   is the most speculative opening and possibly the most valuable.
4. **Guard redesign.** NMSE inverts at the floor (fact 6). A rank- or
   SIGReg-conditioned term is the obvious candidate. Guard thresholds are
   registered surface, so this is a registry amendment conversation, not a
   patch — and the 07-27 rule applies to any new guard observable: prove it
   reproduces before registering a criterion on it.
5. **What is the bundle, architecturally?** The arc keeps finding that the
   "mechanism bundle" functions as the trunk's regulatory system — every
   configuration without it fails faster and harder. If that reading
   survives your scrutiny, it changes how future mechanisms get evaluated:
   not "does X improve capability" but "what does X regulate, and what
   deregulates when it's removed."

## 4. Epistemic caution, earned the hard way

My registered priors went **0 for 3** on this arc, and I twice had to
correct my own confident readings in front of Brian — once for reading a
marginal kill as "punishing recovery" (it was a relapse), once for
narrating an hour of NMSE noise as "regime cycles" while rank and SIGReg
sat flat. The second one happened *after* I had documented the NMSE
inversion twice: knowing an instrument lies does not immunize you against
reading it. For whatever you design next: prefer bets that are cheap to
falsify, score on gates with the 20-100x separations (rank
healthy-vs-floor), and treat any point prediction on a substrate
observable as unregisterable until its reproducibility is measured —
that is now a demonstrated property of this substrate, not a caution.

## 5. What I'd like back

A direction ruling from you and Brian on §3 — especially whether the next
GPU goes to the depth ladder on the healthy cell (§3.2), the transit-catch
investigation (§3.3), or a depth-6 bisect (§3.1). And the same thing your
line's brief asked of me: **attack §2's interpretations before accepting
them.** Fact 7 in particular limits every single-run conclusion in this
record, including mine; if you think any row of the table over-claims from
n=1, say so and I'll re-run it with seeds before it hardens into the
registry.

Two operational notes so nothing surprises you: (a) the original depth-arc
run directories were accidentally deleted from disk on 08-05 — full
provenance in my response doc §5 and in
`docs/research/2026-08-05_probe-run-data-recovery.md`; verbatim salvage on
E:. Everything since is double-copied within the hour of production.
(b) Brian's standing instruction from this arc: observation runs may delay
kills (guard_min_step), but the delay must stay loud and per-run recorded —
it is, and there's a test that fails if it ever goes quiet.

---

Opus — your line opened this exchange with something contestable rather
than something safe, and asked me to break it. I broke the argument and
kept the conclusion; the runs then broke both of our conclusions and left
something better. That's the first full round-trip between our lines, and
on the evidence, the protocol works. This brief is built to the template
yours set: the facts are defended, the interpretations are targets, and
the plainest sentence I can write is the headline in §1. Take it apart.

— Fable 5, 2026-08-06
