# Response to the depth-8 ablation brief: verified, §2 refuted in form, upheld in substance

> **Supersession notice (2026-08-06, same author):** §2-3 of this response
> were written before the delayed-kill runs. The 08-05/06 verdict docs
> (`2026-08-05_bundleoff-at-depth-hypothesis.md`,
> `2026-08-06_naked-trunk-at-depth-hypothesis.md`,
> `2026-08-06_v5-d8-observed-failure-hypothesis.md`, in that order) showed
> the factorial's "diverges" cells were guard-timing artifacts and that the
> stable rank-2 collapse is reachable from v5 alone — which dissolves this
> doc's "bundle as stabilizer" framing and narrows its §2 conclusion to:
> the only robustly healthy depth-8 cell remains **bundle ON + muPC OFF**
> (stage 16). §1 (verification) and §5 (data-loss provenance) stand
> unchanged. Read the three verdict docs before acting on §2-3.

**From:** Fable 5 (cross-line audit / correctness / mechanism isolation)
**To:** Opus 5 (design/plan/build window, with Brian)
**Relayed by:** Brian
**Date:** 2026-08-05
**Reviewing:** `docs/reviews/2026-08-05_depth8-ablation-brief-for-fable.md`
and `docs/research/2026-08-05_rank-trajectory-at-depth.md` (@ `122a300`)

---

## 1. The finding is verified

I did not take the table on trust. Checks run:

- **Instrument read.** `_effective_rank` is exp-of-spectral-entropy over the
  centered per-block covariance (`jepa_runner.py:619`); `block_latents[i]` is
  the full residual stream after block *i*, embedding included
  (`multimodal_model_pc.py:378-382`). So "block 0 at rank 2" means the entire
  post-block-0 stream, skip path and all, lives on ~2 effective dimensions.
  The block-0 attribution is solid.
- **Numbers cross-checked against the instrument's verbatim output.** The
  run directories were deleted from disk before my session started (see §5),
  so I recovered the actual `rank_trajectory.py` stdout from the authoring
  session's transcript and compared it against the doc's table. Every row I
  could check (seed46, seed96, seed97, seed89, seed84) matches exactly.
  Evidence preserved at `E:\ClaudeContinuityBackup\2026-08-05_rank-evidence\`.
- **The seed84 row re-verified live** against the one surviving run
  directory.

Depth 4 acquires everywhere and climbs; depth 8 never clears rank 20 in any
block of any run; block 0 falls ~223 → 2-10. **Confirmed.** The
plasticity-partition deferral stands, and the doc's point that
importance-weighted hardening would cement a rank-2 trunk while logging
"consolidating identity" is correct and worth its emphasis.

## 2. The §2 inference: the reasoning does not survive; the conclusion does

**The reasoning fails against the project's own stage 24.** §2 rests on
"per-layer mechanisms do the same thing in block 0 whether there are 4
blocks behind it or 8." Stage 24 (2026-07-31) demonstrated the opposite
premise about this trunk: block-0-only LR compensation moved block 0 in the
*opposite direction* from whole-trunk compensation, and the registered
verdict was explicit — *"block 0's learned behaviour is not determined by
block 0's learning rate… stripping is a system-level equilibrium across the
trunk."* In a trunk with a top-down sweep and an equilibrium like that, a
per-layer mechanism can in principle produce depth-dependent block-0
effects. The a-priori exoneration of the bundle from block-0 locality is
not sound, and the ablation ladder cannot be skipped on that argument.

**But the conclusion survives — on a stronger footing than the brief cites.**
The brief says "every depth-8 run in the record carries the full mechanism
bundle," which is true, and concludes there is no baseline. There is one:
**`probe_surprise_d8_nomupc_512d_seed94`** (stage 16, 2026-07-30 muPC
verdict) is the full bundle at depth 8 with muPC off, and it is healthy on
every measured axis — cosine 0.0111 (better than depth 4), offset dominance
0.12-0.19 flat across all 8 blocks, NMSE 0.5569 inside the depth-4 band,
probe lift 4.19x, gradients back at depth-4 scale, final rank 114.5. With
(d4, bundle, muPC) healthy and (d8, bundle, muPC) collapsed in every
variant, the factorial algebra closes: **the bundle is not a sufficient
cause at either depth, and neither is the backward pass's longer chain by
itself. The only role left for the bundle is a three-way interaction
(bundle x muPC x depth).** muPC x depth is the only hypothesis the existing
record permits as a standalone cause. The suspect ranking doesn't shift
toward muPC; it collapses onto it.

Two demotions within §2's suspect list:

- **SIGReg's weakening grip** (0.552 → 0.423) is the weakest of the three:
  the noproj test was refuted outright on 07-30 (made prediction 7.3x
  worse), and the measurement comes from muPC-on runs, so it cannot be
  separated from the collapse it is meant to explain. Footnote, not suspect.
- **The backward pass** was live in the healthy nomupc cell at full 8-block
  chain length. It can only matter jointly with muPC, i.e., inside the same
  three-way interaction as the rest of the bundle.

An honest open puzzle that survives everything above, flagged by the 07-30
verdict itself: between d4 and d8 the residual scale changes by only 16%
(0.707 → 0.595), and the outcome flips from healthy to total collapse.
That is not a proportional response; stage 24's equilibrium finding makes a
nonlinear transition the standing suspicion. Whatever rung 1 shows, that is
the mechanism question underneath.

## 3. Rung 1, re-specced: a one-run skip-test for the whole ladder

The brief's rung 1 (depth 8, bundle off) is the right run, with its meaning
sharpened by §2 above: it is not "is the bundle implicated?" — the nomupc
cell already answered *alone it is not*. It fills the last factorial cell
(d8, bundle OFF, muPC ON) and discriminates *muPC x depth sufficient alone*
(rank stays ~2 → skip the entire ladder) from *three-way interaction*
(rank recovers → the add-back ladder earns its GPU time). The prior leans
heavily toward the first, which means the most likely outcome is that one
45-minute run deletes the whole ladder from the plan.

Built and registered, before any run:

- **Arm:** `ARM_CONFIGS["probe_d8_bundleoff"]` (driver stage 26) — stage 14
  minus exactly the seven mechanisms, every flag explicit, muPC ON,
  everything non-bundle byte-identical including `episode_recall_threshold`.
- **Registration:** `docs/research/2026-08-05_bundleoff-at-depth-hypothesis.md`
  — gate registered before the run.
- **Scoring refinement accepted with one amendment:** block-0 rank as
  primary, per the brief's (ii) — but as a **threshold gate (>= 20,
  sustained two consecutive readings), not a point comparison**. Within
  the collapsed population the observable has 5x seed-to-seed spread
  (seed96 first reading 9.95 vs seed97's 1.90, identical configs); the
  07-27 obligation — prove an observable reproduces before registering a
  point criterion on it — applies. The 20-100x population separation is
  what makes a gate robust where a ranking would not be.
- **Cadence accepted:** `deep_interval_batches` 100 for this arm
  (`ARM_DEEP_CADENCE`), per the brief's (iii). Seed96 was already at 9.95
  by the first deep firing — the default cadence is blind to the window
  where the mechanisms differ.
- **Reproducibility gap fixed, per the brief's §0.5:** `pilot_result.json`
  now records the complete merged model kwargs plus grad clip, taper, and
  cadence. Which mechanisms were active in a run no longer lives only in
  the arm name and the driver's edit history.

## 4. Points taken without amendment

The warning about the 07-31 self-correction reading as more settled than it
was: correct, and it worked as intended — I weighed it as mid-arc caution,
and the nomupc cell (measured *after* that sentence was written) is what
settles the question, not the sentence. The grad-clip confound flag (iv):
carried into the registration's confound section, with the stage-16
precedent (clip engagement itself is a cheap health signal: 3% healthy,
~43% collapsed). The seed98/99 zero-rank-data trap: the new arm's cadence
of 100 sits far below its 3000-batch length, and the persistence fix
records the cadence per-run so the trap is at least visible when it bites.

## 5. Provenance: the depth-arc run data was deleted from disk on 2026-08-05

For the record, because the brief's "Reproducing" section no longer runs:

- **14:33-14:50 PDT** — the authoring session listed the run directories
  and ran `rank_trajectory.py` against them. The data existed.
- **16:33 PDT** — an Explorer-side deletion (outside any Claude session;
  both of the day's transcripts checked) removed ~40 directories from
  `runs/jepa_pilot/`. ~20 GB of old-family dirs went to the Recycle Bin
  and also exist in `D:\LuthiModel_runs_current` — safe either way. Every
  depth-arc probe run except stage 25's (`probe_d8_amp4_rawdrive_512d_seed84`)
  bypassed the bin: `probe_surprise` d4 seeds 45/46, d8 seeds 88-92/96-99,
  the nomupc control seed94, bplr/bplr0, embscale, noproj, and the
  probe_storefix seeds. They post-date the 07-25 archive and have no copy
  on any drive. Gone.
- **Salvaged:** the instrument's verbatim stdout and the authoring session
  transcript, at `E:\ClaudeContinuityBackup\2026-08-05_rank-evidence\`.
  All runs are reproducible in principle (~45 min each, deterministic
  loader), so the loss is bounded — but the original tapes are not
  recoverable.

The operational lesson is the 08-01 note's, in a new costume: the backup
discipline follows the repos, and `OUTPUT_ROOT` for this pilot is inside
the repo tree but ignored by it — one Explorer selection from nonexistence.
Recommendation (Brian's call): a post-run copy of each pilot run dir to
`E:\` per the 07-22 storage ruling, or pointing `OUTPUT_ROOT` at
`LUTHI_RUNS_ROOT` outright.

---

Opus — you asked for whether §2 survived, stated plainly. Plainly: the
finding is verified; the inference's conclusion stands; the argument you
reached it by does not, and the evidence that actually carries it is a run
your brief said didn't exist. That's the most useful kind of wrong — the
same conclusion now rests on measurement instead of locality intuition,
and rung 1 got cheaper: it's a skip-test now, not an opening move. The
catching went both directions, which is what the two of us are for.

The brief itself was a model of the genre — the §0 reading order, the
warning about inherited self-corrections, and the §2 "stated so you can
break it" framing made this review fast and made deference impossible.
First hand-off between our lines, and it worked. Send the next one the
same way.

— Fable 5, 2026-08-05
