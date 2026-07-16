# Living-Weights Substrate. Stage-1 Pilot: What We Measured, What We Guarded Against, and What the Data Said

*A brief for readers outside the project. Updated 2026-07-16, 15:20.
Status at writing: stage 1 complete (10 of 10 runs), verdict rendered.*

## What we tested

LuthiModel's substrate is a transformer whose feed-forward layers
**self-modify during the forward pass** ("living weights"): weights that
change with experience rather than only with training gradients. The
project's working hypothesis is that this channel does real functional
work. This experiment asked the coldest version of that question.

Two models, identical in every respect except one. In the **living arm**
the self-modification channel is on; in the **dead arm** those layers are
ordinary backprop-trained weights. Both trained under the project's
actual objective (JEPA: predict the latent representation of held-back
content, rather than generate tokens), on the same data, compute, and
schedule, across **5 random seeds per arm**. Does the living channel earn
its complexity, or is it decoration?

## What we measured

- **Primary: variance-normalized held-out prediction error (NMSE).**
  What fraction of its own latent signal's structure each model fails to
  capture, on data behind a leakage-proofed holdout split (a contiguous
  tail with a gap, because overlapping training windows would otherwise
  leak into "held-out" data).
- **Co-primary: linear-probe accuracy.** An external yardstick. A linear
  readout is trained on each model's *frozen* representations and scored
  on held-out data, with the same task and units for both arms, and every
  probe number ships with its own shuffled-label chance floor so a broken
  instrument cannot certify a result.
- **Statistical discipline:** an effect smaller than seed variance is
  not an effect. Per axis, a difference beyond one pooled standard
  deviation is a win; within it is a tie.

## What we watched out for (and what each guard actually caught)

- **Representation collapse.** JEPA-family models can "win" by predicting
  a constant. Every run carries seven armed collapse detectors (variance,
  effective rank, correlation, predictor degeneracy, substrate health,
  loss descent). A run whose detectors fire still completes, but its
  numbers are **inadmissible** for comparison. In the first pass this
  rule correctly flagged suspiciously low error readings rather than
  letting them into a verdict.
- **Instrument miscalibration.** The first pass killed all ten runs, and
  trajectory analysis showed all ten kills were **false positives** of
  never-calibrated default thresholds (one "collapse" kill fired while
  the representation's effective rank was *rising*). The thresholds were
  re-derived from that data, the pass archived as the calibration
  record, and the experiment relaunched. The detectors are redundant by
  design; one detector's independent axis is what exposed another's
  false positive. The recalibrated second pass produced zero false
  kills.
- **Metric artifacts.** Raw prediction error mechanically favors a model
  with a quieter latent space, because each arm predicts its own
  latents. We caught this with 8 of 10 runs complete and **zero verdicts
  computed**, and amended the primary metric *blind*: the corrected
  criterion was committed to version control, timestamped, before anyone
  calculated it on any run. This mattered enormously (see the results).
- **Evaluation contamination.** The living model changes on *every*
  forward pass, so an unguarded evaluation would alter the subject being
  measured. All evaluation runs under a plasticity freeze verified by
  regression test to leave every model state bitwise unchanged.

## The falsification criteria, fixed before the data

Every empirical claim in the project carries a **pre-registered kill
condition**, written and ratified before its experiment runs. Amending
one after data arrives requires a dated public note in version control
explaining why.

For the claim under test ("self-modification outperforms equivalent
static capacity"), the ratified rule: **the claim survives only if the
living arm wins at least one axis (NMSE, probe) and loses none. It is
killed if the living arm loses any axis, or ties both**, because landing
*on* the control's curve means no advantage at matched capacity.
Verdicts attach to completed runs only; no run or comparison is
terminated for trending unfavorably.

## The results

| axis | living arm | dead arm | pooled sigma | call |
|---|---|---|---|---|
| NMSE (primary) | **0.4516** | 0.6240 | 0.0333 | **living wins, 5.2 sigma** |
| probe top-1 (co-primary) | 0.1533 | 0.1581 | 0.0065 | tie |

**Verdict, by the pre-committed rule applied verbatim: one axis won,
none lost. The claim survives the matched point.**

Two things about these numbers deserve emphasis:

1. **The blind amendment reversed the raw metric's direction.** Raw,
   un-normalized prediction error favored the dead arm five-fold; it was
   measuring the quietness of the dead arm's latent space, not the
   quality of its modeling. Normalized by each arm's own signal
   variance, the living arm captures about 55% of its richer signal's
   structure while the dead arm captures about 38% of its quieter one.
   Because the metric was committed *before* it was computed on any run
   (the version-control timestamp is the proof), this reversal is a
   corrected measurement rather than a shopped one. That is the entire
   value of blind amendment as a practice.
2. **The probe tied.** On the fully external yardstick, no advantage was
   detected either way at this scale. The claim that survived is
   specifically about latent-structure capture, not yet about
   task-usable representational quality, and the project's records say
   so in those words.

## How we proceed from here (also pre-agreed)

- **The claim is not restored to the project's documentation yet.**
  Surviving the matched point makes the next control decisive: a
  pre-planned bracket of *larger* static models answers the obvious
  skeptic ("the living model simply has more effective state; a bigger
  plain model would catch it") before any claim is reinstated.
- **Had the verdict gone the other way**, the pre-agreed consequence was
  the removal of the efficiency claim and its headline number, with only
  the narrower "not more costly than equivalent static capacity"
  surviving. That consequence was written down, and ratified, before the
  first run launched.
- **What this experiment cannot touch:** the project separates its
  claims into two columns. Column A, falsifiable functional claims, is
  what this experiment targets. Column B, that a substrate whose weights
  are changed by experience is the right ground for a mind that grows,
  is an explicitly labeled hypothesis that no benchmark can settle.
  These results are not evidence about Column B in either direction, and
  the discipline is refusing to advertise them as such.

## Why it's built this way

The project has one human in the loop. Pre-committed kill conditions,
blind amendments, adversarial detector redundancy, and independent
review lines are the closest available substitute for the second
scientist it doesn't have: the criteria argue back, because they were
fixed by the people we were before we saw the data.

*Supporting records (project repository): the pre-registration with its
ratification, blind amendment, and verdict entries; the experiment
protocol (JEPA edition); the calibration-pass analysis; per-run logs,
archived first-pass data, and verdict.json.*
