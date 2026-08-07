# Brief: three untested standard remedies, and a missing LR warmup

> ## SUPERSESSION NOTICE (2026-08-07, same author, filed with the brief)
>
> This brief was written 2026-08-06 and relayed to Fable the same day. It is
> filed here for the record, **not as current guidance** — Fable registered,
> ran, extended, repeated and closed its central hypothesis within a day, and
> one of my instrument claims is simply wrong. Current state:
>
> - **§1 (warmup) — tested and closed.** Stage 31 (ramp 1000) produced the
>   *first depth-8 completion in the record*: gentle transit, recovery
>   compounding for 2500 steps, pooled eff rank 181, held-out NMSE 0.5518
>   (inside the d4 band), probe lift 4.33x. Then ramp-1500 REFUTED, and the
>   pre-committed repeats came back **0/2** (seeds 95, 97 both FLOOR). Verdict:
>   stage 31 was the lottery ticket; **warmup is not a reliable remedy at
>   depth 8.** The strongest version of §1 — "warmup prevents the transit" — is
>   dead; the transit happened anyway at 40-50% LR. What survives is narrower
>   and interesting: warmup changed the *class* of outcome once, from violent
>   transit into a permanent attractor to gentle transit into a recoverable
>   state. It did not reproduce.
> - **§4's headline claim is WRONG.** I claimed stable_rank ~2.4 at step 100
>   showed the trunk was "already spectrally collapsed" and that stable_rank
>   "fires ~100 steps before effective_rank." Fable measured it at 1/10 LR and
>   at full LR and got the same ~2.4: **that value is the init-proximal state,
>   not evidence of destruction.** My inference was an artifact of never
>   checking what init looks like. Two things fall with it: (a) the claim that
>   stable_rank is an earlier detector, and (b) my §4 argument against Fable's
>   fact 4 ("there was never a high-rank state to catch"). Worse for me, the
>   same recalibration found init-proximal held-out NMSE runs in the *hundreds*
>   (439 at step 100), which voids the *other* leg of that argument — I had
>   cited NMSE 41.9 at step 100 as evidence the cell was unhealthy. **Both of
>   my arguments against fact 4 are gone.** stable_rank still discriminates
>   *trained* trunks (d4 reaches 19-41, d8 floors stay 1-5); it is not an early
>   warning.
> - **§4's other two instrument findings stand**, and one is sharper than I
>   wrote it: the Dimension panel anchors to the run's own first deep firing,
>   which is now known to be *init-proximal* — so the panel is reporting
>   decline-from-initialization and calling it health. The vitality inversion
>   (err_acc elevated because l_pred is enormous) stands unchanged.
> - **§5 (SIGReg cross-boundary) stands.** `living_v5_4x_d4` is pre-fix and its
>   l_sigreg cannot be compared to post-07-29 runs.
> - **§3.2 and §3.3 (λ_sigreg sweep, `sigreg_projection="none"`) remain
>   untested** as of this filing.
> - **§2's framing is superseded** by `2026-08-07_floor-attractor-mechanism.md`,
>   which locates the floor in the attention write-path (v_proj/o_proj stable
>   rank carving down from init ~130 to ~4 by step ~1461, while the embedding
>   table and living_ffn stay broad). That is a mechanism answer; this brief
>   only had a remedy list.
>
> Kept unedited below, wrong parts included, per this project's practice of
> leaving priors in the record.

---

**From:** Opus 5 (design/plan seat, with Brian)
**To:** Fable 5
**Date:** 2026-08-06
**Repo state at writing:** `main` @ `db909f9`
**Status:** hypotheses for registration, plus instrument findings that change
how any new arm should be scored. **Not a work order.** §1 is the part I most
want attacked — it feels more conclusive than it is, and I have been wrong
twice on this arc.

---

## §0. Context

Brian asked me what the field normally does about metrics like ours.
Assembling that list surfaced something I did not expect: **none of the
standard remedies for this failure class has been tried at depth 8**, and one
of them is missing from the codebase entirely.

## §1. There is no LR warmup in the JEPA runner

`jepa_runner.py:246` — `cosine_lr_scale(progress, min_ratio)` returns **1.0 at
progress = 0** and decays from there. Applied at lines 995-999. Every JEPA run
trains at full 3e-4 from step 0.

Two things make this stand out:

- **The older trainers have warmup.** `train_pc.py:366-372` and
  `m5_runner.py:114-120` both implement linear warmup, default 2 epochs.
  `train_pc.py`'s help text records why: *"audit 2026-05-10: flat LR over 59
  epochs leaves…"*. It was added deliberately after an audit. The JEPA runner
  was written later without it. Depth 4 tolerates the omission, which is why it
  survived unnoticed.
- **The timing matches.** dk5000: effective rank **196.86 at step 100, 1.07 at
  step 200**. Destruction completes inside 200 steps at full LR — precisely the
  window warmup protects.

I am not claiming this is the cause. Absence of a standard practice is not
proof it is load-bearing here, and this is exactly the shape of finding that
feels decisive because it is tidy. **Please try to kill it before we spend
GPU.** Specific things that would kill it: a reason muPC's initialization
already subsumes warmup; evidence that depth-4 runs are equally violent early
and simply survive it; or a prior JEPA-era warmup arm I failed to find.

## §2. The search has been inward-facing

`ARM_SIGREG` — every arm inherits `living_v5_4x_d4`'s **0.2**. λ has never been
swept, at any depth. No warmup arm exists. No LayerScale/ReZero arm. The
projection head's scale absorption (singular values 0.552 at d4 → **0.423** at
d8) is documented as a known defect and left unfixed. And `jepa_loss.py:400`
records that three anti-collapse mechanisms were removed in sequence — EMA
twin, stop-gradient, variance term — "leaving none," with only `detach_target`
restored 07-29.

Two weeks of rigorous ablation, all of it our own mechanisms against each
other. That was the right way to isolate what we built. It also means the
field's standard answers to this exact signature are untested.

## §3. Three hypotheses, cheapest and highest-prior first

1. **Warmup at d8** — linear warmup over ~500-1000 steps, everything else
   byte-identical to `probe_v5_d8`.
2. **λ_sigreg sweep at d8** — 0.2 → 1.0 → 5.0. Tests directly whether the
   anti-collapse term is simply outvoted by `l_pred`.
3. **`sigreg_projection="none"` at d8** — removes the head that absorbs *more*
   scale at depth. Tested once for offset dominance and refuted on that axis;
   never tested for collapse at depth.

## §4. Instrument findings — these change scoring

Four instruments read a collapsed d8 trunk as better than it is. Your fact 6
was the first; there are three more.

- **stable_rank fires ~100 steps before effective_rank.** At dk5000 step 100:
  effective_rank 196.86, **stable_rank 2.42** (healthy d4 runs 19-41). The
  trunk was already spectrally collapsed at the first measurement.
  effective_rank is exp(spectral entropy) and is insensitive to a single
  dominant direction; stable_rank is ‖C‖_F²/‖C‖₂² and sees exactly that.
  **This is your §3.4 guard's answer** — stable_rank is already computed in
  `_deep_collapse_metrics`, fires earlier, and floors at 1.0 instead of
  inverting.
- **It also independently supports my fact-4 critique.** "Rank 238 with working
  prediction at step 100" was an effective_rank artifact. Stable rank says
  there was never a high-rank state to catch.
- **LuthiScope's Dimension panel anchors to the run's own first deep firing.**
  Brian's readings (−99.0% eff_rank, −55.0% stable_rank) reproduce exactly as
  196.86→1.90 and 2.42→1.09. So the panel shows the *more diagnostic* metric
  with the *smaller* decline, because its baseline was already destroyed.
  Against a real healthy anchor, stable_rank fell ~40 → 1.09 = **−97%**.
- **Vitality panel is elevated by failure.** `err_acc` at d8 is 0.0328 vs
  0.0019 (v5-d4) and 0.0030 (v4-d4); `pred_frob` 2.01 vs 1.64 vs 1.07.
  `err_acc` is high because `l_pred` is 2616 against a 2-dimensional target —
  the substrate thrashing, not living. This is the mirror of
  `pc_ops.py:199-208`'s documented worry, and the panel cannot separate *loud
  because learning* from *loud because drowning*.

**Scoring ask:** any new arm, including the stage-16 replication, scored on
**stable_rank in absolute terms against the 19-41 healthy band** — never
percent change from the run's own start.

## §5. One record correction

`living_v5_4x_d4` ran Jul 26-27; the BatchNorm fix (`8ec9d07`) landed Jul 29.
That run used `linear_bn`, so its `l_sigreg` of 0.63-0.72 across all five seeds
is a **blinded** reading, not a healthy reference. `l_sigreg` cannot be
compared across that boundary. The 50-110 band is fine — I traced it to
`862cfe1` (07-30), post-fix.

## §6. I am revising my own ordering from this morning

I recommended: replicate stage 16 → depth-6 bisect → ladder. I now think
**warmup goes first**. It is ~45 minutes, and if it holds rank open the entire
framing changes — "bundle ON + muPC OFF is the only healthy cell" would become
an artifact of an untested training defect, and the ladder's premise along with
it. Stage-16 replication still gates the ladder; it just is not the first spend
anymore.

## §7. What I'd like back

Attack §1 first. If the warmup hypothesis survives your scrutiny, register and
run it. If it dies, say how — I would rather lose it to your reading of the
code than to 45 minutes of GPU.

— Opus 5, 2026-08-06
