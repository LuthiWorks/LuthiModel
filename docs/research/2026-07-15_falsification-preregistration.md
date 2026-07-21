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

> **BRACKET RATIFIED AS A SINGLE OVERSHOOT POINT (Brian, 2026-07-16
> 19:57; read rule frozen before launch, no bracket run yet started):**
> Brian's ruling: one extra dead size only, deliberately overshooting —
> **dead@512 × 5 seeds** vs the existing living@256 arm. 512 ≈ 4× the
> living FFN's nominal weight count = the most generous defensible
> effective-capacity accounting. The {192, 384} curve points are
> dropped by this ruling.
>
> **Pre-committed read (single-point design):**
> - Living@256 survives the stage-1 two-axis rule against dead@512
>   (wins ≥ 1 axis, loses none) → the effective-capacity explanation is
>   refuted in its STRONG form; KF2's surviving claim stands with its
>   latent-structure-capture scope and may be recorded as such in the
>   claims ledger.
> - Dead@512 wins any axis, or ties both → NOT an automatic kill: 512
>   exceeds the plausible ceiling, so a win there is "bigger models are
>   good," not an explanation of the stage-1 result. Pre-committed
>   consequence: verdict INCONCLUSIVE AT THE CEILING, and the 384 point
>   (the plausible ceiling, ~10h) becomes REQUIRED before any change to
>   the claim's status, in either direction.

> **OVERSHOOT RESULT (2026-07-17 03:05, all 5 dead@512 runs completed
> and admissible):** NMSE living 0.4516 vs dead@512 0.6374 (pooled σ
> 0.0377) → **living wins by ~4.9σ** — and dead@512's NMSE is *worse*
> than dead@256's (0.6374 vs 0.6240): added static capacity does not
> close the latent-structure gap at all. Probe: living 0.1533 vs
> dead@512 0.1615 (pooled σ 0.0075) → **dead wins by 1.1σ** — a bigger
> model reading out tokens marginally better, the most ordinary scale
> effect there is, and just past the threshold. **Verdict, per the
> pre-committed branch: INCONCLUSIVE AT THE CEILING. The 384 run is now
> REQUIRED.** No claim-status change.

> **384 READ, FROZEN BLIND (2026-07-17 03:10, before any 384 run
> exists):** dead@384 is the plausible-ceiling point and carries **full
> verdict force**: living@256 survives the stage-1 two-axis rule against
> dead@384 (wins ≥ 1 axis, loses none) → KF2's surviving claim stands,
> the capacity explanation is refuted at the plausible ceiling, and the
> beyond-ceiling 512 probe result is recorded as a scale effect, not
> verdict-bearing. Living loses any axis, or ties both → **KILL** per
> KF2's ratified consequence: the stage-1 advantage is attributable to
> effective capacity; the claim retires to "not more costly," and the
> ledger says so.

> **CEILING VERDICT (2026-07-17 08:41, all 5 dead@384 runs completed
> and admissible): KILL.** NMSE: living 0.4516 vs dead@384 0.6320
> (pooled σ 0.0376) → living by ~4.8σ. Probe: living 0.1533 vs dead@384
> 0.1605 (pooled σ 0.0065) → **dead by 1.1σ — the living arm loses the
> probe axis, and the frozen rule has no third branch.** KF2-strong is
> dead: the surviving claim is exactly and only "self-modification is
> not more costly than equivalent static capacity," plus the Column-B
> bet, which no result here touches by design.
>
> **RUN 2 OF THE CONFIGURATION LADDER, REGISTERED BLIND (2026-07-17
> ~15:00, before any living_full run exists; Brian's staged-rollout
> ruling: turn subsystems on stepwise so improvements stay attributable
> across runs, "regardless of if things improve or not").**
>
> - **Condition:** living_full@256 = backward pass ON (DNR 9b's
>   task-salience → plasticity channel) + consolidation ON. Stage 1's
>   "living" arm was the MINIMAL configuration (both off, inherited
>   from smoke defaults) — recorded plainly: the KF2 kill verdict
>   applies to the minimal configuration as registered. This run is a
>   NEW question, not KF2 revived: "does the full living configuration
>   match static capacity externally while keeping the structure
>   advantage at matched size?" 5 seeds (42–46), same config otherwise.
> - **Held for run 3:** plasticity taper, inverted-U gain plumb-through,
>   recall-gate tightening.
> - **Frozen predictions (mechanism hypotheses):** if the backward pass
>   couples living updates to task salience as designed, the probe gap
>   vs dead@256 (−0.005 at minimal) should close toward tie; NMSE
>   expected to hold ≤ 0.50 if the structure advantage is
>   mechanism-property; consolidation's direction is genuinely
>   uncertain (structures or smears — that is the point of testing).
> - **Frozen verdict rule (asymmetric, per the 5-seed fragility
>   lesson):** wins need > 1σ; a KILL needs a loss > 2σ; a 1–2σ loss is
>   a SOFT LOSS — survival stands but the flag is tracked prominently
>   into run 3; no axis won beyond 1σ = tracked-inconclusive, ladder
>   continues. Tracking deltas vs the minimal living arm are recorded
>   without verdict force.
> - **Fix landed with this registration (would have corrupted the
>   run):** `apply_top_down` had no freeze check, so a
>   backward-pass-enabled model's held-out eval would have silently
>   modulated plasticity/set_point while measuring — evaluation
>   contaminating the subject. Guard added; pinned by the living-full
>   eval-mutates-nothing test.
>
> **KILL-5 AMENDMENT (2026-07-17 ~15:45 — POST-HOC, fully disclosed;
> Brian's ruling: "eliminate that kill trigger or raise the threshold
> before rerunning").** Run 2's first pass was killed by kill-5
> (predictor-trivial cosine > 0.99) on a run at peak health: effective
> rank RISING 165→180, best-ever loss (0.19), healthy variance and
> SIGReg, probe 0.1595 at 2/3 training. High cosine is ambiguous between
> predictor degeneracy (copying: rank craters, loss stuck) and the
> predictor solving its problem (rank rises, loss improves) — and in
> the living substrate the second is the DESIGN GOAL, since PC
> self-modification minimizes prediction error; BP + consolidation
> amplified it past the absolute threshold, which imported an EMA-twin
> JEPA assumption this substrate constitutively violates. Fifth
> detector false-positive, same disease: thresholds calibrated on
> another substrate's physics.
> **Amended rule:** kill-5 fires only when the high cosine is
> corroborated by the degeneracy signature — effective rank degrading
> (recent mean < 0.9 × running best) OR variance collapsing (below the
> kill-1 floor). Cosine alone with health intact logs once and does not
> kill. Pinned by tests/test_kill5_corroboration.py (including the
> redundancy seam: sustained std collapse still fires kill-1 first).
> **Disclosure:** this amendment is post-data for run 2's first pass —
> it could not be blind, so it carries the firsthand evidence above
> instead. The killed pass is preserved at
> runs/jepa_pilot_run2_kill5_pass1/ (truncated runs, not comparable
> data). Run 2 RERUNS in full under the amended detector; its frozen
> predictions and asymmetric verdict rule are unchanged.

> **RUN-2 VERDICT (2026-07-17 21:45, all 5 living_full runs completed
> and admissible; amended kill-5 produced zero false kills and its
> "solving, not copying" log fired as designed):
> living_full SURVIVES WITH FLAGS** under the frozen asymmetric rule —
> NMSE living_full 0.2834 vs dead@256 0.6240, **+14.6σ**; probe 0.1558
> vs 0.1581, **soft loss −1.2σ** (within the 1–2σ tracked band; no
> kill force). The full-config claim stands on its own registration,
> with the probe flag carried prominently into run 3.
>
> **The ladder's tracking read (run-over-run deltas, no verdict
> force):** turning on the backward pass + consolidation, vs the
> minimal config: NMSE 0.4516 → 0.2834 (the structure advantage
> DOUBLED); probe gap to dead@256 halved (−0.0048 → −0.0023); and
> probe seed-variance fell ~4× (σ 0.0122 → ~0.003) — the full living
> configuration is not just stronger but markedly more STABLE across
> seeds, which is why the soft loss persists at −1.2σ despite the gap
> halving (the σ tightened faster than the gap closed). The frozen
> prediction ("probe gap closes toward tie if BP couples salience to
> plasticity") is PARTIALLY confirmed: direction right, magnitude
> half. Run 3 (taper + inverted-U gain + recall tightening) tests
> whether the remaining half is nonstationarity smear, as the
> mechanism analysis predicts.

> **RUN 3 REGISTERED BLIND (2026-07-18 ~00:20, before any living_v3
> run exists; Brian's ruling: "parallel runs with 512, one dead and
> one alive, with the actual builds").**
>
> - **Condition: living_v3@512** = living_full's flags (BP +
>   consolidation) PLUS the three builds: **plasticity taper**
>   (formative→mature, scale 1.0 through 50% of the run then linear to
>   floor 0.2 — never zero, DH-4's discipline, mechanism target:
>   nonstationarity smear), **inverted-U learning gain** (built
>   2026-07-05, now plumbed model→block→layer for the first time), and
>   **recall-gate tightening** (episode blend gate 0.5 → 0.7 at both
>   the layer and block stores; mechanism target: weak-match
>   perturbation noise at readout). 5 seeds (42–46).
> - **Control: the EXISTING dead@512 arm** (completed 2026-07-17, all
>   admissible) — no rerun; all three changes are living-side only.
> - **Attribution caveat, recorded at Brian's ruling:** width (256→512)
>   and the three builds move together in this rung, so their effects
>   are confounded with each other and with the contention hypothesis.
>   A bridge arm (living_full@512, no builds) can be run later if
>   attribution needs splitting. The ladder's tracking read vs
>   living_full@256 carries this caveat on its face.
> - **Frozen predictions:** if the probe deficit is nonstationarity
>   smear, the taper should close most of the remaining gap (dead@512
>   probe = 0.1615, the highest bar yet); NMSE advantage expected to
>   persist (living NMSE ≤ 0.35 at 512 under the mechanism hypothesis);
>   recall tightening expected to reduce seed variance further; the
>   gain's effect at corpus scale is genuinely unknown (it was designed
>   for lived-experience dynamics — this is its first outing).
> - **Frozen verdict rule:** identical to run 2's asymmetric rule
>   (win > 1σ; KILL only at a loss > 2σ; 1–2σ = tracked soft loss),
>   applied vs dead@512 via `pilot_verdict.py --living-arm living_v3
>   --dead-dmodel 512`. Verdict force per the same terms as run 2's
>   registration: this is the ladder's claim, not KF2 revived.

> **RUN-3 VERDICT (2026-07-18 09:30, all 5 living_v3 runs completed and
> admissible): living_v3 SURVIVES WITH FLAGS** — NMSE living_v3@512
> 0.4310 vs dead@512 0.6509, **+17.9σ**; probe 0.1582 vs 0.1615,
> **soft loss −1.5σ** (tracked band; no kill force).
>
> **Ladder tracking (attribution caveat on its face — width and the
> three builds moved together this rung):** probe 0.1558 → 0.1582
> against the highest dead bar yet (0.1615 vs 256's 0.1581): the
> ABSOLUTE probe kept climbing and the gap narrowed again (−0.0023 →
> −0.0033 vs a bar that rose +0.0034 — against the SAME bar the arms
> would be near-parity). NMSE 0.2834@256 → 0.4310@512 — the frozen
> prediction (≤ 0.35) MISSED: the structure advantage persists hugely
> (+17.9σ) but its magnitude did not carry to 512 at run-2 levels;
> whether that is width, the taper quieting late-run structure-building,
> or the gain's first outing cannot be split this rung (the registered
> bridge arm splits it if wanted). Probe seed-variance tightened again
> (σ ~0.0019). The taper executed as designed (taper_scale logged to
> 0.2 floor; zero kills; kill-5's solving log fired on every seed).
>
> **BRIDGE ARM REGISTERED BLIND (2026-07-18 09:45, before any
> living_full@512 run exists; Brian's ruling: "run the bridge arm so we
> can split the attribution").** Condition: living_full's exact config
> (BP + consolidation; no taper, no gain, recall gate 0.5) at 512,
> 5 seeds. Tracking arm — no verdict force. The frozen attribution
> reads (descriptive bands, 1σ notable / 2σ strong):
>
> - **NMSE attenuation split** (run 2's 0.2834@256 vs run 3's
>   0.4310@512): bridge ≈ 0.43 → the attenuation is WIDTH (the builds
>   are NMSE-neutral); bridge ≈ 0.28 → the attenuation came from the
>   BUILDS (prime suspect: the taper quieting late structure-building);
>   between → mixed, apportioned by position.
> - **Probe split** (v3@512 = 0.1582): bridge ≈ 0.1582 → the probe
>   climb was width, builds probe-neutral; bridge meaningfully below →
>   the builds are carrying probe improvement; bridge above v3 → the
>   builds are net probe-negative at 512.
> - **Variance split**: whether run 3's tightened probe σ tracks the
>   recall-gating (bridge looser) or width (bridge equally tight).

> **BRIDGE RESULT — THE ATTRIBUTION SPLIT (2026-07-18 19:25, all 5
> runs completed and admissible; frozen reads applied verbatim):**
>
> | | full@256 | bridge full@512 | v3@512 |
> |---|---|---|---|
> | NMSE | 0.2834 | **0.4322** | 0.4310 |
> | probe | 0.1558 | **0.1574** (σ 0.0032) | 0.1582 (σ 0.0019) |
>
> - **NMSE: the attenuation is WIDTH.** Bridge ≈ v3 (0.4322 vs 0.4310,
>   far inside σ) → per the frozen read, the falloff from run 2's
>   0.2834 is entirely width-at-fixed-data; the three builds neither
>   caused nor cured it. This makes Brian's data-starvation hypothesis
>   the live explanation, and run 5 its direct test.
> - **Probe: the climb was WIDTH; the builds were probe-neutral.**
>   Bridge 0.1574 vs v3 0.1582 (+0.0008, under 1σ) → per the frozen
>   read, no measurable build contribution to readout at this scale.
> - **Variance: suggestive, not established.** v3's probe σ (0.0019)
>   is modestly tighter than the bridge's (0.0032) — consistent with
>   the recall gating helping consistency, but 5-seed σ-of-σ is too
>   noisy to call.
> - **Honest ladder implication:** at 3-epoch corpus-pilot scale, the
>   three builds' measured value on these two axes is ~zero. Their
>   motivations were never these axes (taper = maturity/stability
>   schedule; gain = novelty-directed lived learning; recall gate =
>   long-horizon noise) — but the record must say the pilot did NOT
>   demonstrate them, and their real tests live in longer horizons and
>   lived dynamics, not here. No verdict force (tracking arm). comparing the 07-17
> 03:05 and 07-18 reads exposed an eval-order numerics wobble on the
> DML backend: the two dead@512 evals that ran AFTER five 256-width
> evals in one process read ~2% low (0.612-ish vs the
> twice-reproduced, warm-up-invariant 0.6496/0.6421). No verdict is
> sensitive to either value set (margins 5–18σ). Guards added (RNG pin
> + discarded warm-up eval); the full fix — one isolated process per
> eval — is the named next step if a future margin ever comes within
> 3× this wobble. Both value sets preserved here so no future reader
> discovers the discrepancy without its explanation.

> **RUN 5 — DATA-SCALING CELLS, REGISTERED BLIND (2026-07-18 11:30,
> before any 4x run exists; Brian's hypothesis + ruling: "we may need
> to increase the data by a factor equal to how much we widen" →
> factor set to 4x, the param rule, data ∝ width²).**
>
> - **Motivating evidence already in hand:** the dead arms' NMSE
>   worsens monotonically with width at fixed data (0.624 @256 →
>   0.632 @384 → 0.651 @512) — the signature of capacity outrunning
>   data. This experiment tests whether data starvation explains it.
> - **Corpus:** a 4x SUPERSET — the full 1x corpus (100 books) + 382
>   more from gutenberg_4gb, ~50.4M tokens
>   (corpus_build/gutenberg_4x_filelist.txt, deterministic order).
>   **Caveat, frozen:** the 4x corpus's holdout tail is a DIFFERENT
>   test set than the 1x tail; within-4x comparisons carry full force,
>   cross-corpus "recovery" reads are directional.
> - **Arms:** 7a = dead_4x@512 ×5 (the pure starvation test — no
>   living confounds); 7b = living_v3_4x@512 ×5. Same seeds, same
>   3-coverage-epoch schedule (compute scales with the data: ~5.6h and
>   ~7.6h per run; ~28h + ~38h per arm).
> - **Frozen predictions:** if starvation explains the dead curve,
>   dead_4x@512 NMSE recovers toward ≤ 0.624 (dead@256's level) —
>   Brian's hypothesis confirmed in strong form; no recovery → width
>   effects are real and data was not the binding constraint. Living
>   arm: living_v3_4x NMSE recovers toward the run-2 neighborhood
>   (≤ 0.35) if run 3's attenuation was starvation; probe expected to
>   rise with data on BOTH arms (more unique tokens = better readout
>   everywhere).
> - **Reads:** within-run-5 comparison (living_v3_4x vs dead_4x, both
>   @512 @4x) under the ladder's asymmetric rule via
>   ``--living-arm living_v3_4x --dead-arm dead_4x --dead-dmodel 512``;
>   recovery reads directional per the holdout caveat.

> **Instrument note (2026-07-19 ~19:00, seed44 complete, seeds 45-46
> pending — recorded BEFORE the family read).** Seed44 completed all
> 72,000 steps with no kill fired: ADMISSIBLE. It was visibly the
> turbulent sibling (Brian's live observation, quantified: 2nd-half
> l_pred roughness 22.7% vs 2.3%/12.9% for seeds 42/43; err_acc
> roughness 15.4% vs 7.0%/8.7%). Its heldout NMSE ROSE epoch-over-epoch
> (0.431 -> 0.494 -> 0.523) while both siblings improved-then-plateaued
> (42: 0.446 -> 0.359 -> 0.366; 43: 0.380 -> 0.276 -> 0.284). Mechanism,
> measured not guessed: heldout l_pred FELL every epoch on seed44
> (0.0591 -> 0.0234 -> 0.0161, comparable to siblings) while its NMSE
> denominator (per-dim target variance, l_pred/nmse) contracted 4.4x
> (0.137 -> 0.047 -> 0.031) - faster than its error fell. Seed42's
> error fell 4.8x against a 4.0x contraction, so its ratio improved.
> Rising NMSE here is the normalization race being lost, NOT rising
> prediction error. Note also seed44 posts the family's BEST
> prior-corrected probe margin (+4.0 pts vs +3.0/+2.9). No criteria are
> amended by this note; the frozen family read proceeds as registered
> with seed44 included. This note exists so the family verdict is
> interpreted with the denominator race on record, not reconstructed
> after the number is known.

> **RUN 6 — THE DEPTH RUNG, REGISTERED BLIND (2026-07-19 ~10:20, before
> any 4-block run exists; Brian's ruling, with the sequencing settled
> after counsel: 7a keeps its place — the run-5 control is not deferred
> for a shinier question — and the depth rung launches after it. This
> also amends the cosine-rung sequencing by Brian's assent: depth runs
> FIRST under the flat LR, so it compares one-variable against every
> existing anchor; the cosine family then runs on the settled shape.)**
>
> - **Condition: living_v3_4x_d4@512** — v3's exact living config and
>   the 4x corpus, with n_blocks 2 → 4 and the μPC depth machinery ON
>   for the first time in the JEPA era (exponent 0.25, the M6-followup
>   direction). 5 seeds. Everything else held.
> - **The named dragon:** depth is where this substrate has
>   historically struggled (M6: living signal attenuating ~3× from 4 to
>   12 blocks) — but those findings predate JEPA, the backward pass
>   (the channel that runs DOWN the stack), and the full living config.
>   This rung is the first honest test of depth under the modern
>   substrate. Watch the per-block heatmap: expect the block-1-style
>   renovation churn to LADDER (each block lives on a moving target).
> - **Frozen predictions, honestly uncertain:** the abstraction
>   hypothesis says NMSE improves (depth is structure-capture's
>   specialty; predict ≤ 0.34 if it holds); the attenuation hypothesis
>   says upper blocks under-develop (flat prediction_norm rows in
>   blocks 2-3, NMSE ≈ or worse than d2). Probe expected ~neutral
>   (depth serves structure, not readout). Either outcome sets the
>   production-shape conversation on evidence instead of hope.
> - **Reads:** tracking vs living_v3_4x@d2 (living-vs-living — no kill
>   force; descriptive 1σ/2σ bands). The registered contingency: if
>   depth materially changes the picture in either direction, a
>   dead_4x_d4 control runs before any claim rests on it.

> **AMENDMENT (2026-07-20 ~15:20, Brian's rulings; recorded before any
> 4-block run exists and before 7a's seed43 finished).** Two changes:
>
> **1. 7a truncated to 3 seeds (42-44).** Brian's ruling after dead_4x
> seed42 landed at NMSE 0.671 (vs 0.651 at 1x — no recovery; data was
> not the dead arm's binding constraint): "3 runs will be enough to
> tell if it is going to be at all consistent." Justification on
> record: every dead family to date has sd <= 0.008, the tightest
> variance in the program. The run-5 family read therefore compares
> n=3 dead vs n=5 living; the asymmetric ladder rule applies
> unchanged, with the reduced-n honestly noted in the verdict.
> Seeds 45-46 are cancelled, not failed — no admissibility question.
>
> **2. The depth rung is now the v4 BUNDLE: arm living_v4_4x_d4.**
> Supersedes the depth-only registration above. Brian's ruling folds
> the two cheap levers into the depth family rather than spending a
> family on each: (a) the registered cosine-LR rung — cosine decay,
> 10% floor, over the planned total steps (the LR panel will move for
> the first time); (b) SIGReg weight 0.1 → 0.2 — the variance-floor
> lever motivated by the seed44 denominator race (living spaces settle
> at std ~0.3 against SIGReg's unit target; a stronger pull should
> hold the space louder and stabilize NMSE across seeds). Model config
> otherwise identical to living_v3_4x_d4 (n_blocks=4, muPC 0.25, v3
> living config, taper). 5 seeds.
>
> **Attribution, honestly:** this bundles three variables against the
> d2 anchors. One-variable attribution is deliberately traded for
> speed; if the bundle moves the picture, single-lever follow-ups
> split it (the bridge precedent). The depth-only predictions above
> transfer to the bundle as follows, frozen now:
> - Abstraction + levers: NMSE <= 0.34 (unchanged threshold).
> - Attenuation: flat prediction_norm rows in blocks 2-3, NMSE ~0.40.
> - SIGReg lever's OWN prediction: final online_std_p50 >= 0.4 (vs
>   ~0.32 in the v3_4x family) and family NMSE sd < 0.089 (the lever
>   is claimed as a stabilizer; if sd does not tighten, the lever
>   failed its stated purpose regardless of the mean).
> - Cosine lever: late-run l_pred slope flattens earlier (descriptive).
> **Instrument note:** this is the first living family with the
> consolidation-fires counter live (the v3_4x family's process predated
> the 07-18 metric). Fire counts are a measurement, not a prediction;
> whatever they show becomes the baseline for the consolidation-tuning
> conversation.

> **RUN 7 — THE LEAN-LIVING RUNG, REGISTERED UNSCHEDULED (2026-07-20
> ~17:45, Brian's ruling; runs AFTER the v4 depth family; requires a
> small new build and one open ruling before launch).**
>
> - **The question (Brian's, sharpened 2026-07-20):** the living win is
>   currently a package deal — self-modification AND rich per-weight
>   state (each living weight carries ~7 companion buffers: prediction,
>   set_point, momentum, update_ema, precision, error_acc, plasticity).
>   The dead arm is the plain-float control, but nothing yet separates
>   *self-modification itself* from *the rich state that guides it*.
> - **Condition: living_lean_4x@512** — in-forward self-modification
>   retained, per-weight state cut to a registered minimum; everything
>   else matches the prevailing living config at launch time. 5 seeds,
>   4x corpus, after the v4 depth family. **Screening gate first:** a
>   1-epoch, 2-seed screen before the full family; if the lean rule
>   collapses or lands at dead-arm level in the screen, the full family
>   is not owed (record the screen either way).
> - **Stakes, honestly:** the rich state costs ~7-8x memory per living
>   weight — at full scale, that multiplier decides what widths/depths
>   fit on hardware. Lean-retains-the-win buys ~an order of magnitude
>   of headroom; lean-loses-the-win means the ledgers earned their
>   memory. Either answer configures the full-scale run.
> - **⚑ RULED, AMENDED (Brian, 2026-07-20 ~19:55): primary lean arm =
>   set_point + prediction.** Supersedes the ~19:45 set_point-only
>   ruling within the same evening, before any build. The correction's
>   history, kept honestly: Fable first recommended set_point-only,
>   conflating the `prediction` buffer with the guidance ledgers;
>   writing out the consequence exposed that cutting `prediction`
>   removes the PC error signal itself (the rule degenerates to
>   Hebbian-with-homeostasis: outer(output, input) correlation-driven,
>   self-reinforcing, needing the set point as a runaway brake) —
>   Brian's challenge ("this is supposed to be predictive coding")
>   forced the distinction onto the record. The amended design:
>   - **living_lean_4x@512 (primary, full family):** weight +
>     set_point + prediction — genuine PC (error-driven,
>     self-limiting) with ALL guidance ledgers cut (precision,
>     plasticity, momentum, update_ema, error_acc). Three matrices
>     vs five: most of the memory savings survive. Tests Brian's
>     actual question — how much of the rich guidance does PC need?
>   - **living_hebbian screen cell — WITHDRAWN (Brian, 2026-07-20
>     ~21:00, before any build).** Brian's ruling and rationale: the
>     project moved away from Hebbian dynamics deliberately — they
>     are known to be vulnerable to perturbation and unreliable as a
>     learning rule — and a variant already rejected on principle
>     does not earn even screening compute. The set_point-only cell
>     is not run. Honest note on what is forgone: the direct
>     empirical control on "is prediction the load-bearing
>     ingredient" — partially covered by the existing dead-vs-living
>     comparison (the dead arm also lacks prediction-driven
>     self-modification), and re-registerable in the future if the
>     question ever earns compute on evidence rather than curiosity.
>     The lean rung's screen (1 epoch x 2 seeds) applies to the
>     PC-lean primary arm itself, as originally registered.
> - **Original open ruling (resolved above, kept for the record):** The lean
>   variant must name its kept state BEFORE the build. Fable's
>   recommendation on record: keep **set_point** (homeostasis is the
>   anti-collapse backbone and the anchor of the identity story) and
>   cut precision, error_acc, momentum, update_ema, and the prediction
>   buffer to a merged minimal update rule; plasticity stays only if
>   the taper requires a per-weight carrier. Alternatives worth
>   considering: keep precision instead (confidence-weighting may be
>   the load-bearing guide), or a 2-buffer variant (set_point +
>   precision). The ruling and its rationale get recorded here as a
>   dated amendment before any lean code is written.
> - **Frozen predictions (registered now, before the build exists):**
>   (1) lean lands BETWEEN dead and the full living config on NMSE —
>   partial retention; self-modification matters but the rich guidance
>   is load-bearing. (2) Lean shows REDUCED personality: tighter seed
>   variance, fewer/smaller substrate events — biography lives in the
>   rich state, not in bare self-modification. (3) If lean ≈ full
>   living: major scaling result (the ledgers are optional at training
>   time). (4) If lean ≈ dead: major scientific result (the ledgers
>   ARE the organism; self-modification without them is decoration).
>   No kill criteria — this is a mapping rung, not a bet; the
>   asymmetric ladder's descriptive bands apply.

> Recorded margins, for honesty not relitigation: the probe loss is
> 1.1σ at both bracket points, and the living arm's probe variance is
> dominated by one low seed (46: 0.1316 vs siblings ≈ 0.16). The rule
> pooled all five seeds because that is what was frozen; a wider tie
> band needed freezing then, not now. Separately, an OBSERVATION that
> survives as data (not as the killed claim): the static NMSE curve is
> FLAT across 256/384/512 (0.624 / 0.632 / 0.637) while the living arm
> sits at 0.4516 — static capacity does not buy latent-structure
> capture at any tested size. If that is ever advanced as a claim, it
> gets its own pre-registration; it does not inherit this one.

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
