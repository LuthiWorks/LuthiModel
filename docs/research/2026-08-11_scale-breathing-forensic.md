# Scale-breathing forensic — tapes only, zero GPU

**Author:** Fable 5, 2026-08-11, at Brian's "forensic runs first."
**Corpus:** the 3 VISReg-family tapes (9 breaths: seed 46 x1 @1300;
seed 95 x3 @200/600/1000; seed 97 x5 @400/1400/2000/2400/4400-fatal).

## Signature

Square pulse: std50 jumps 2.6-4.1x in one cadence-100 window and fully
reverts the next. No lasting effect on rank, chorus, or subsequent
l_pred in the 8 survived breaths.

## Findings

1. **Genuine state excursions, not measurement artifacts.** The fatal
   4400 reading came from the HELDOUT eval — fixed data — so the model
   state itself was perturbed (NMSE 2.71 on data it scored ~0.5 on at
   neighboring checks). A pure outlier-training-batch story is dead.
2. **Discrete event systems exonerated.** Consolidation-fire deltas in
   breath windows (0-16) are statistically identical to quiet windows
   (means 8.9-11.8, max 16). Seed 97's recall froze entirely at ~2000
   (counter flat at 4408) yet it breathed twice after. drive_fires: 0
   everywhere. LR schedule: breaths occur during ramp (97@400), full LR
   (46@1300), and cosine tail (97@4400) alike.
3. **Gradient spikes: implicated, not convicted.** 3 of 9 breaths sit
   adjacent to 10-300x grad_norm outliers (95@200 after a 967k reading;
   97@400 after 72k; 97@4400 at 45k). But 97@1400 and @2000 breathed at
   BELOW-median grad norms — at cadence-100 sampling, where a spike in
   any of the 99 unlogged steps is invisible.
4. **Living-layer self-modification mildly elevated** at 8 of 9 breaths
   (update_ema 1.2-3.0x quiet median) — cause/effect ambiguous: louder
   latents mean bigger PC errors mean more self-modification.

## Verdict

**Cause not identified at cadence-100 sampling; the mechanism is
sub-cadence.** The candidates left standing (per-step gradient spikes;
PC self-modification bursts; their interaction) are exactly the ones
the current instruments cannot separate.

## Instrument order (for the 768 family, before launch)

Log **per-step** `grad_norm` and per-step batch `std50` (two scalars per
step; negligible cost). The next breath then arrives fully resolved:
which step moved first, gradient or scale, and by how much. This is the
house rule — when the tape can't answer, build the instrument, don't
theorize.

## Guard interaction

The 2026-08-11 middle-ground rules (persist 500 / ceiling 10x / rank
veto at eff 100) cover breathing regardless of cause: the longest
observed breath is one cadence window, the deepest reading 2.71 (vs
ceiling 20), and every breath happened at pooled eff >= 100. Seed 97's
death is unrepresentable under the new contract (pinned in
tests/v2/test_divergence_guards.py::test_seed_97_would_have_lived).
