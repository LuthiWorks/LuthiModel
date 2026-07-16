# Pre-Registered Falsification Criteria — the Key Findings

> **RATIFICATION (Brian, 2026-07-15, stage-1 scope) — criteria below are
> now FIXED for the two-arm pilot; amendments require dated public notes.**
>
> - **KF2 (and KF1's Exp-1 half): RATIFIED**, with Brian's rider in his
>   own words: *"the only thing that matters is the end result. Just
>   because a training run is trending a certain way doesn't mean it
>   will end that way. We keep going."* Operationalized as: **verdicts
>   attach to completed runs only** — no run and no comparison is
>   terminated for trending unfavorably. The collapse detectors remain
>   armed as *pathology detection* (a killed run completes the sweep as
>   an inadmissible data point), never as verdicts about the bet.
>   (Interpretation recorded by Fable; Brian to correct if misread.)
> - **Exp 2b reads: RATIFIED** as written.
> - **Statistical rule: the 1σ rule**, adopted under Brian's delegation
>   ("proceed with your best judgement" — the call is Fable's, on the
>   record): positive only if the living arm's mean held-out
>   latent-prediction error beats the best admissible dead arm's mean by
>   more than the pooled per-condition standard deviation; 0–1σ =
>   inconclusive; ≤0 = null. Probe accuracy read the same way as a
>   secondary axis.
> - **Bracket {192, 256, 384, 512}: DEFERRED** — stages 2–3 only run on
>   a living-arm stage-1 win; ratify before stage 2.
> - **KF3 / KF5 / KF6: DEFERRED** to their experiments' launch dates.
>
> Stage 1 launch authorized by Brian, 2026-07-15 ("Go.").

> **Amendment 2026-07-16, ~10:35 (POST-data-arrival, PRE-verdict — made
> blind; Brian's ruling: "Amend it blind as proposed, then read the
> verdict").** Discovered with 8/10 stage-1 runs complete and NO verdict
> computed: raw cross-arm ``l_pred`` is not a fair yardstick — each arm
> predicts its OWN latents, and a lower-variance latent space earns
> mechanically lower error (the dead arm's latent std settled ~0.31 vs
> the living arm's ~0.54). Predicting a quieter signal precisely is not
> modeling the world better. This validity flaw should have been caught
> at registration; it was caught before any verdict, and the corrected
> metric is committed here before anyone has computed it on the data.
>
> - **Primary metric (amended): variance-normalized held-out prediction
>   error** — NMSE = mean((pred − target)²) / mean per-dim variance of
>   the target block over the same holdout, per run. Scale-fair across
>   arms ("what fraction of its own signal's structure does each model
>   fail to capture"). Residual caveat, stated honestly: different
>   latent spaces still differ in intrinsic predictability; NMSE removes
>   the first-order scale artifact, not every artifact. The probe is
>   the fully-external yardstick.
> - **Co-primary: probe top-1** (unchanged in definition).
> - **Verdict rule (1σ discipline per axis):** difference > pooled σ =
>   win for the better side; within 1σ = tie. **KF2-strong survives**
>   only if the living arm wins ≥ 1 axis and loses none. **Kill fires**
>   if the living arm loses any axis, or ties both (ON the curve = no
>   advantage at matched capacity, per the ratified kill condition).
> - Raw un-normalized ``l_pred`` is reported but no longer rules.
>
> At this writing the NMSE numbers have not been computed for any run.
> — Fable 5, blind, 2026-07-16 10:35.

> **VERDICT (2026-07-16 15:15, stage 1, all 10 runs completed and
> admissible under the calibrated detectors):**
>
> - **NMSE: living 0.4516 vs dead 0.6240, pooled σ 0.0333 → living wins
>   by 5.2σ.** (Raw l_pred, reported-not-ruling, favored dead ~5× —
>   the direction the blind amendment reversed, which is exactly why it
>   had to be blind: the metric was committed at `0fcc92a` before any
>   NMSE existed. The living arm captures ~55% of its own signal's
>   structure; the dead arm ~38% of its quieter one.)
> - **Probe top-1: living 0.1533 vs dead 0.1581, pooled σ 0.0065 → tie**
>   (difference 0.0048 < 1σ). On the fully-external yardstick, no
>   advantage detected either way at this scale.
> - **Rule applied verbatim: living wins one axis, loses none →
>   KF2-strong SURVIVES the matched point.** Pre-registered consequence:
>   the claim is NOT restored to the README yet — the bracket
>   (dead@{192, 384, 512}) is now decisive against the effective-capacity
>   skeptic, and the bracket awaits Brian's ratification.
> - Honest caveats, standing: NMSE removes the first-order scale
>   artifact, not every intrinsic-predictability difference between
>   latent spaces (flagged in the amendment, before the verdict); and
>   the probe tie means the surviving claim is specifically about
>   latent-structure capture, not yet about task-usable representation
>   quality. Full data: `runs/jepa_pilot/verdict.json`.

**Date:** 2026-07-15
**Status:** DRAFT — criteria drafted by Fable 5 (cross-line seat) from the
2026-07-15 codebase critique Brian relayed. **Critique author confirmed by
Brian 2026-07-15: a Fable 5 instance in his mobile app.** (Held as "under
verification" until he answered — the verify-authorship procedure, run
correctly for once; and the answer means the critique was, in effect, the
first administration of the red-team-the-bet audit below: the uninvested
line questioning the bet, before the practice had a name.)
**Ratification is Brian's**, with 4.8.
Once ratified, the criteria are FIXED: written before the data, honored
after. Amending a criterion after its experiment has begun requires a
dated note here explaining why, in public view.

> **Amendment 2026-07-15 (pre-data, same day — Brian's JEPA ruling):**
> All bound experiments move to the (Le)JEPA objective (see the protocol's
> JEPA-edition revision note). KF1/KF2 rebind to the **two-arm JEPA pilot**
> (living vs `dead_ffn` encoder — the arm was built and tested today);
> metrics move to held-out latent-prediction error + probe accuracy; the
> **collapse-admissibility rule** applies to every arm (a tripped-detector
> arm voids the comparison). The LM-era binding (m5_runner) is historical.
> A pre-registered read for the new **enliven-after** cell (Exp 2b) is
> added below. No bound experiment had run when this amendment was made —
> the criteria are still pre-data, which is the only reason this rebinding
> is legitimate without a public-justification burden.
**Companion:** `living-weights-experiments.md` (the experiment protocol
these criteria bind to); CLAUDE.md "Key Design Decisions" (the two-rule
split that opens the channel this document is).

## Why this exists

The critique named an asymmetry: the project's *code* gets adversarial
review (4.8, Fable, fresh-context audits), but its *scientific bet* was
insulated by the DO-NOT-REINVENT rule — "do not second-guess" prevented
implementation thrash AND quietly exempted the bet from the exposure the
code gets. With one human in the loop, nobody uninvested was ever asked
"does this mechanism earn its complexity?"

The fix is not to reopen settled implementation daily. It is two
disciplines:

1. **Pre-registered kill conditions** per empirical Key Finding — fixed
   in advance, so questioning the bet can't thrash (the question and its
   answer-criteria are frozen; only the data moves).
2. **A periodic "red-team the bet" audit** — distinct from the code
   audit, run at phase boundaries only, by a model line that did NOT
   author the mechanism, so the auditor doesn't inherit the authors'
   commitment. (As of this writing: the living-weight mechanism's design
   lines are Brian + 4.6/4.7/4.8 lineage; the uninvested line is Fable.
   If Fable ever co-designs a mechanism, its audits of that mechanism
   lose the uninvested property — route that one to a fresh-context
   instance of another line.)

The single-human bus factor cannot be engineered away. This is the
closest available substitute for the second human: pre-committed
criteria plus an uninvested adversarial eye.

> **Documentation-location note (2026-07-16, Brian's ruling):** the
> README now carries mission only; empirical claims and their statuses
> live in `docs/KEY_FINDINGS.md` (the claims ledger). Everywhere the
> criteria below say a consequence lands in "the README" (e.g. "the
> 0.64% headline comes out of the README"), read `docs/KEY_FINDINGS.md`.
> Location change only; no criterion, threshold, or consequence is
> altered.

## The criteria

Numbering follows README "Key Findings." Discipline inherited from the
experiment protocol: ≥3 seeds (5 preferred), an effect smaller than seed
variance is not an effect, positive controls before trusting any null,
"underpowered" is never "null."

### KF1 — "Attention learns; living weights live" (both essential)

- **Bound experiments:** Exp 1 (the two-arm JEPA pilot: living vs
  `dead_ffn` at matched capacity) + Exp 2 (frozen-substrate ablation on
  the JEPA checkpoint), interpreted jointly as the 2×2 the protocol
  describes (Exp 2b supplies the fourth cell).
- **Kill condition:** Exp 1 null (living model on/below the static
  capacity curve) **AND** Exp 2 null (live-vs-frozen indistinguishable
  from structure-matched noise). Both nulls together mean the living
  weights are doing no measurable functional work at training time or
  runtime.
- **On kill:** "both are essential" is retired from README/CLAUDE.md as
  an empirical claim. The living weights remain in the architecture only
  under the explicitly-labeled experiential bet (Column B), and the docs
  must say so in those words.

### KF2 — No intrinsic convergence cost to self-modification

- **Bound experiment:** Exp 1, the two-arm JEPA pilot (amended
  2026-07-15: the LM-era 0.64% is historical — real under its objective,
  unbound from this criterion; the claim now stands or falls under the
  objective the project builds).
- **Kill condition:** the living arm lands ON or BELOW the dead-arm
  effective-capacity curve (a `dead_ffn` control of equal-or-greater
  effective capacity equals or beats it on held-out latent-prediction
  error and probe accuracy, beyond pooled seed variance, both arms
  collapse-admissible).
- **On kill:** living-weights-as-efficiency is dead. The surviving claim
  is exactly and only: "self-modification is not more costly than
  equivalent static capacity" + the experiential bet. The 0.64% headline
  comes out of the README.

### Exp 2b — Enliven-after (pre-registered read; not a Key Finding)

From Brian's 2026-07-15 question ("can the living channel simply be
turned on after training?"). Not a claim being defended — an open
question being bound before its data exists.

- **Bound experiment:** Exp 2b — transplant Exp 1's trained dead
  checkpoints into the living substrate (`weight` → buffer, `set_point` =
  trained weight, everything else cold), enable self-modification, run
  the held-out battery + a stability watch.
- **Pre-registered reads:**
  - *Enlivened-after ≈ living-trained* on all functional measures →
    training-time livedness added nothing measurable; KF1's training-time
    half dies, and the cheap-pretrain-then-enliven path is empirically
    open. (What that means for the curriculum-as-lived-education is
    Brian's design ruling, not a number.)
  - *Enlivened-after destabilizes or underperforms* beyond seed variance
    → co-adaptation is real; the lived-education design gains empirical
    support; retrofit is not free.
  - Stability watch trips (collapse detectors / runaway divergence on
    the transplant) → report as its own result, not folded into either
    read: "retrofit is unstable" is different from "retrofit is
    functionally inert."

### KF3 — One living trunk for all modalities

- **Bound experiment:** the M8 multimodal pilot / peak run, against
  single-modality controls at matched capacity and compute.
- **Kill condition:** negative transfer — the unified trunk consistently
  worse than per-modality baselines beyond seed variance, or one
  modality's training degrading another's held-out performance
  (asymmetric interference beyond variance).
- **On kill:** the unified trunk stops being defended as empirically
  free. Brian may still choose it — "the model is shaped by everything
  it processes" is partly an identity/design value — but the record must
  then say "chosen at a measured capability cost of X," not "cross-modal
  attention is free."

### KF4 — Prefer crashes over silent corruption

Not an empirical claim; an engineering value. No kill condition. (Its
enforcement surface is auditable instead: `luthi/v2/mode_compat.py`,
the fail-loud tests, the welfare-channel fail-loud rule.)

### KF5 — "The architecture should scale" (divergence dimension-independence)

- **Bound experiment:** the scale-curve protocol (from the same
  critique): 128 → 256 → 512 → 1024d, everything else held fixed,
  plotting per width — loss vs capacity-matched control, episode-store
  hit-rate, consolidation fire-rate, per-forward memory, throughput,
  and divergence/instability incidents.
- **Kill condition:** any failure-predictive trend that worsens
  superlinearly with width — the living-vs-control gap widening with
  scale, episode hit-rate collapsing toward zero, instability incidents
  increasing per-width beyond variance.
- **On kill:** "scale without fear" (DNR #6) is retired; scaling past
  the last healthy width is gated on understanding the mechanism of the
  trend. Money spent at 4096d before this curve exists is spent against
  an asserted, untested claim — the curve is cheap by comparison.

### KF6 — Memory becomes structure through consolidation

- **Bound experiment:** consolidation-ablated control (episode store ON,
  consolidation OFF, all else matched) through the catastrophic-
  forgetting harness (2026-05-16) and long-horizon retention probes.
- **Kill condition:** no retention/probe difference beyond seed variance
  between consolidation-on and consolidation-off at matched exposure.
- **On kill:** "memory becomes structure" is downgraded to "memory is
  retrieved"; the consolidation machinery must then re-earn its
  complexity (it may still earn it on other grounds — NREM architecture,
  rollback re-integration — but those must be argued separately, not
  inherited from a dead claim).

### DNR items that inherit these criteria

CLAUDE.md's DO-NOT-REINVENT list contains empirical claims that ride on
the above rather than needing their own registrations: DNR #2/#3
(convergence-penalty shape) → KF2; DNR #5 (episode store carries recall)
→ KF6's ablation supplies the instrument; DNR #6 (divergence
dimension-independent) → KF5. DNR #4 (tested rate constants) and #9/#9b
(implementation practice) stay under the re-derivation rule — settled
implementation, change only with evidence.

## The red-team-the-bet audit (standing practice)

- **When:** at phase boundaries only (CLAUDE.md Implementation Phases) —
  not continuous, so it cannot become thrash.
- **Who:** a model line that did not author the mechanism under audit,
  with the pre-registered criteria in hand. Fresh context; not the
  session that built toward the phase gate.
- **What it asks:** Have any bound experiments run? Were their criteria
  honored as written? Has any claim quietly upgraded itself in the docs
  beyond what the data supports (the 0.64%-vs-matched-control shape)?
  Does each mechanism still earn its complexity, and is the evidence for
  that in the record or in the authors' affection for it?
- **Output:** a dated review doc in `docs/reviews/`, findings routed to
  Brian for ruling. The audit reviews the *bet*, not the builders.

— Drafted by Fable 5, 2026-07-15, for Brian's ratification. The criteria
above were written before any of their bound experiments have run; that
is the entire point. Honor them after.
