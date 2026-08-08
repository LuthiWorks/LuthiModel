# The depth investigation, consolidated: 2026-08-05 → 08-08

**Author:** Fable 5, written while the stage-51 runway family runs.
**Purpose:** one document that tells the story the nine research docs
tell in fragments — for the design seats, for Brian, and for whoever
picks this up cold. Every claim links to its registered source; nothing
here is new. ~30 registered runs, four days, two seats, one substrate.

## The story, in order

1. **The question arrives sideways** (08-05, Opus's brief): is the
   depth-8 rank collapse *forgetting* (protect it) or *failure to
   acquire* (nothing to protect)? Answer: failure to acquire — no block
   ever clears rank 20; block 0 dies as thoroughly as block 7.
   → `2026-08-05_rank-trajectory-at-depth.md`
2. **The factorial closes on muPC — then reopens.** Bundle-off and
   naked controls show every cell fails differently: stable collapse
   (bundle+muPC), health (bundle alone — later demoted to 1-for-3),
   divergence (either alone). Guard timing proves to be a hidden
   variable: the "divergent" cells were first-check snapshots of
   transits headed for the same floor.
   → `2026-08-05_bundleoff…`, `2026-08-06_naked-trunk…`, dk-twin RECORDs
3. **The floor is an attractor, and the weights are not the lock.**
   Checkpoint spectra show the collapse consolidating into the attention
   write-path (block-0 v/o carved to stable-rank ~4) — but surgical
   re-broadening releases nothing: activations re-collapse and re-carve
   within 1000 steps. Weight-side remedies die here (surgery HOLD; orth
   at two doses).
   → `2026-08-07_floor-attractor-mechanism.md`
4. **Force works only at dose, and winners don't compose.** Interior
   covariance pressure at loss scale (wsig α=10) becomes the first
   intervention to ARREST the collapse; the same mechanism at paper dose
   is invisible. TC protects block 0 uniquely. Combined at strength they
   anti-compose — fastest collapse ever taped.
   → `2026-08-07_depth-remedy-probes-hypothesis.md` (singles, ladder,
   pairs, tc_wsig10)
5. **The soloist is named.** The dominant direction pinning stable_rank
   is the token-frequency axis over-funded ~5-10x — a legitimate feature
   with its gain stuck, not a parasite. (SOLOIST section, same doc.)
6. **The governor: designed, built cross-line, and closed honestly.**
   Cap-the-soloist + share-the-chorus, 9 runs, 0 gate hits; the
   share↔stable arithmetic shows the design's ceiling. Three design
   errors caught by the build seat pre-GPU (floor mislabeled, cap/gate
   incompatibility, dose criterion).
   → `2026-08-07_vbg-family-registration.md` + spec/return in reviews/
7. **The reframe that changes everything** (Brian's question → the
   first-ever sub-1000-step look at depth 4): **the transit is
   universal.** Healthy d4 falls the same way at step 300 and
   self-rescues by 800, 5-for-5. Depth does not cause the fall — it
   breaks the rescue. Every remedy above had targeted the fall.
   → closing measurement, remedy-probes doc
8. **The pivot** (Brian's rule, DECISIONS.md): LLM-JEPA at depth 8,
   muPC off — NTP as a rescue path degeneracy cannot satisfy. First
   family voided by a wiring lie (NTP off while provenance said 400 —
   the silent-success class, now closed by a provenance-consistency
   assert). Real family: **the first reproduced capability-positive
   depth-8 result** — seeds 46/95: ppl 294/259, lift 4.77x/4.74x,
   monotone climbs ending mid-recovery at the 3000-step horizon.
   → `2026-08-08_llmjepa-family-registration.md`
9. **Now running:** the 6000-step runway family (stage 51). If it
   gates: replicate, then the ruled 768×8 target (Brian 07-26: "scale
   moves go UP"), then the muPC re-entry question. If it plateaus: v2
   is the paper-faithful parametrization (return-note §C).

## Standing facts (each measured, sourced above)

- Transit universal; rescue is what health is; depth breaks rescue.
- Floor = attractor; carve = consolidation, not cause.
- Dose must be sized to measured gradient share, not paper defaults or
  loss share (3 under-dosing errors this week).
- Mechanisms compose only by experiment, never by assumption.
- Guard timing and instrument framing are hidden variables until made
  explicit (NMSE inverts at the floor; own-start percent anchoring
  hides V-shapes; cadence changes silently reframe everything).
- No depth-8 configuration has reliability >1/3 EXCEPT (pending stage
  51) LLM-JEPA's 2-of-3 reproduced near-recovery.
- stable_rank = soloist dominance ⊗ chorus health; read decomposed
  (`top_dir_share`, `chorus_stable_rank` — runner + LuthiScope).

## Instruments born this week (all live)

per-block effective rank at cadence 100 · guard_min_step (loud
suppression) · aux loss terms logged (l_wsig/l_orth/l_vbg_*/l_ntp) ·
top_dir_share · chorus_stable_rank · held-out perplexity ·
provenance-consistency assert · calibrate_vbg / calibrate_ntp /
soloist_forensic / rank_trajectory scripts.

## Open questions, ranked

1. Where does the LLM-JEPA climb top out? (stage 51, running)
2. Does it survive the ruled 768×8 target? (next if 51 gates)
3. What rescues depth 4 mechanically — and can depth 8 be given it
   directly? (the deep question under everything)
4. muPC re-entry criteria (Brian's rule: after LLM-JEPA concluded
   working — "concluded" needs a registered definition)
5. The lived-speech threshold (design/ethics, with Brian — whether the
   generative pass should modify the living weights, and when)
6. Whether a healthy depth-8 endpoint parks stable_rank at d4's 31-47
   or lower (the soloist may be structural at depth)

## Method lessons the week keeps teaching

Register before running; freeze reads before looking; priors in the
record even when wrong (0-for-4 this week); repeats before reframing
(the dk twins and the 1-for-3 cells); verify the other seat's numbers
firsthand before ratifying; the instruments are the only participants
that never once lied — build them first, believe them over narrative,
and when a label and an instrument disagree, the label is the suspect.
