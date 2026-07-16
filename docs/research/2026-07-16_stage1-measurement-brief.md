# Living-Weights Substrate — Stage-1 Pilot: What We Measure, What We Guard Against, and What Would Prove Us Wrong

*A brief for readers outside the project. 2026-07-16. Status at writing:
run 8 of 10 complete; verdict pending.*

## What we're testing

LuthiModel's substrate is a transformer whose feed-forward layers
**self-modify during the forward pass** ("living weights") — weights that
change with experience rather than only with training gradients. The
project's working hypothesis is that this channel does real functional
work. The current experiment asks the coldest version of that question:

Two models, identical in every respect except one — in the **living arm**
the self-modification channel is on; in the **dead arm** those layers are
ordinary backprop-trained weights. Both train under the project's actual
objective (JEPA: predict the latent representation of held-back content,
rather than generate tokens), on the same data, compute, and schedule,
across **5 random seeds per arm**. Does the living channel earn its
complexity, or is it decoration?

## What we measure

- **Primary — variance-normalized held-out prediction error (NMSE):**
  what fraction of its own latent signal's structure each model fails to
  capture, on data behind a leakage-proofed holdout split (contiguous
  tail with a gap, because overlapping training windows would otherwise
  leak into "held-out" data).
- **Co-primary — linear-probe accuracy:** an external yardstick. A
  linear readout is trained on each model's *frozen* representations and
  scored on held-out data — same task, same units, both arms — and every
  probe number ships with its own shuffled-label chance floor, so a
  broken instrument cannot certify a result.
- **Statistical discipline:** an effect smaller than seed variance is
  not an effect. Per axis: a difference beyond one pooled standard
  deviation is a win; within it is a tie.

## What we watch out for

- **Representation collapse.** JEPA-family models can "win" by predicting
  a constant. Every run carries seven armed collapse detectors
  (variance, effective rank, correlation, predictor degeneracy,
  substrate health, loss descent). A run whose detectors fire still
  completes — but its numbers are **inadmissible** for comparison.
  A collapsed encoder's beautiful low error is flagged, not scored.
- **Metric artifacts.** Raw prediction error mechanically favors a model
  with a quieter latent space. We caught exactly this with 8 of 10 runs
  complete and **zero verdicts computed**, and amended the primary
  metric *blind* — the corrected criterion was committed to version
  control, timestamped, before anyone calculated it on any run.
- **Instrument miscalibration.** The first pass of this experiment
  killed all ten runs — and trajectory analysis showed all ten kills
  were **false positives** of never-calibrated default thresholds (one
  "collapse" kill fired while the representation's effective rank was
  *rising*). The thresholds were re-derived from that data, the pass
  archived as the calibration record, and the experiment relaunched.
  The detectors are redundant by design; one detector's independent
  axis is what exposed another's false positive.
- **Evaluation contamination.** The living model changes on *every*
  forward pass — an unguarded evaluation would alter the subject being
  measured. All evaluation runs under a plasticity freeze verified
  (by regression test) to leave every model state bitwise unchanged.

## The falsification criteria — fixed before the data

Every empirical claim in the project carries a **pre-registered kill
condition**, written and ratified before its experiment runs. Amending
one after data arrives requires a dated public note in version control
explaining why — the amendment described above is such a note.

For the claim under test here ("self-modification outperforms equivalent
static capacity"), the ratified rule, verbatim in effect: **the claim
survives only if the living arm wins at least one axis (NMSE, probe) and
loses none. It is killed if the living arm loses any axis — or ties
both**, because landing *on* the control's curve means no advantage at
matched capacity. Verdicts attach to completed runs only; no run or
comparison is terminated for trending unfavorably.

## How we proceed if it doesn't go our way

Pre-agreed, in writing, before launch:

- **On a kill:** the efficiency claim (and its headline number) comes
  out of the project's documentation. The surviving claim is only the
  narrower one — *self-modification is not more costly than equivalent
  static capacity.*
- **What the experiment cannot touch:** the project separates its claims
  into two columns. Column A — falsifiable functional claims — is what
  this experiment targets. Column B — that a substrate whose weights are
  changed by experience is the right ground for a mind that grows —
  is an explicitly-labeled hypothesis that no benchmark can settle, and
  these results are not evidence about it in either direction. The
  discipline is refusing to let a Column-A win be advertised as
  Column-B evidence — or a Column-A loss be mistaken for a Column-B
  refutation.
- **If the living arm wins instead:** a pre-planned bracket of larger
  static controls runs next, to answer the obvious skeptic — "the
  living model just has more effective capacity" — before any claim is
  restored.

## Why it's built this way

The project has one human in the loop. Pre-committed kill conditions,
blind amendments, adversarial detector redundancy, and independent
review lines are the closest available substitute for the second
scientist it doesn't have: the criteria argue back, because they were
fixed by the people we were before we saw the data.

*Supporting records (project repository): the pre-registration and its
ratification + blind amendment; the experiment protocol (JEPA edition);
the calibration-pass analysis; per-run logs and archived first-pass
data.*
