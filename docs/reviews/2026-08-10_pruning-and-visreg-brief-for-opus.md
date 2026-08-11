# Brief: prune the probe era; assess VISReg as SIGReg's successor

**From:** Fable 5 (design seat) **To:** Opus 5 (build seat)
**Relayed by:** Brian **Date:** 2026-08-10
**Authority:** Brian's ruling — "remove the code bloat, follow these
instructions." Context for the instructions: `docs/research/refs/
2026-08-10_suggested-solutions-external.md` (external protocol, filed
with provenance) and the family registrations through 08-09.

## Task 1 — the pruning (mechanical, test-verified)

The depth-8 probe era is closed by verdict. Remove from the LIVE code
(git history and docs/ keep everything):

- **jepa_loss.py:** `temporal_center` + `sigreg_tc_window` (TC — parked,
  block-0 result preserved in docs), `orthogonality_penalty` +
  `orth_lambda` (retired, three strikes), the VBG cap machinery
  (`top_direction_share` power-iteration path, `vbg_*` params/buffers —
  family closed 0-for-9). **KEEP** `sketched_isotropy_penalty` +
  `interior_sigreg_alpha` (wsig — the arrest result; superseded only
  if/when VISReg lands) and the whole NTP path.
- **jepa_runner.py:** `mu_pc_schedule_*` (parked twice, never reached
  its ramp — remove; the design is documented for revival). **KEEP**
  `guard_min_step`, the two-gauge veto, `chorus_stable_rank`,
  `top_dir_share` in `_deep_collapse_metrics` (the INSTRUMENT stays even
  though the governor mechanism goes — it is now core telemetry).
- **driver:** the dead ARM_* dictionary entries for pruned arms (I
  removed the stage entries already; the config dicts remain and should
  go, EXCEPT the llmjepa/v2/spa arms and the historical family arms
  stages 1-11 rely on). The provenance-consistency assert stays and
  should shrink to the surviving dual-sourced doses.
- **tests:** delete tests of removed mechanisms; keep guard, warmup,
  NTP, wsig, instrument tests. Suite must be green with zero behavioral
  change to any surviving arm (bit-exactness spot-check on the v2 arm's
  config dump appreciated).

## Task 2 — VISReg feasibility read (no build yet)

[arXiv 2606.02572](https://arxiv.org/abs/2606.02572) — Wu, Balestriero,
Levine: SIGReg's designed successor. Its two stated motivations are our
two measured diseases: vanishing gradients under collapse (our transit)
and scale/shape entanglement (our raw-vs-normalized result). Read the
full paper + their code (HF: BooBooWu/visreg) and return: (a) the exact
loss (Sliced-Wasserstein sketching + variance term — formulas, defaults);
(b) drop-in feasibility as a REPLACEMENT for our SIGReg (the external
protocol's instruction: replacement, not addition — stacking
double-counts shape); (c) DirectML hazards (sorting ops for
Sliced-Wasserstein on DML?); (d) what happens to the projection head,
whose scale-absorption we measured; (e) your §C reparametrization
instinct revisited — VISReg's explicit variance term may finally
license the O(1)-JEPA-side form you preferred. No code until the design
seat registers it.

Also flagged for your read, same stack: the auxiliary-task theorem
([arXiv 2509.12249](https://arxiv.org/abs/2509.12249)) — its
both-losses-near-zero condition explains our LLM-JEPA families' failure
to be protected by NTP, and may constrain VISReg-era dosing.

## Sequencing

The set-point sweep (stage 53, external protocol step 2) is on the GPU
now — scouting-grade, ~4.5 h. Task 1 can land any time (touch nothing
the sweep arms use — they ride the surviving llmjepa path). Task 2 is
reading. The design table convenes on: sweep verdict + your VISReg
return + the late-transit precision-divergence lead (block-0 precision
tripling while deep blocks fall, steps 3000-5000 in v2 seed 95 — the
current best candidate for "what accumulates").

The week taught the round-trip discipline; same rules: deviations in a
return note, numbers I can verify, and flag anything in this brief that
smells like the design seat protecting its own patchwork — Task 1
deletes a lot of my own work, and I have tried to keep only what the
verdicts keep.

— Fable 5, 2026-08-10
