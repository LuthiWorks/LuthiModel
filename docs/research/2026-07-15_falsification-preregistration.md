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

> **RUN-5 FAMILY VERDICT (computed 2026-07-21 ~05:20, frozen read,
> n=3 dead per amendment; verdict.json in runs/jepa_pilot/):**
> - **NMSE: living by 4.8σ** — living_v3_4x 0.3974 (0.366/0.284/
>   0.523/0.438/0.377) vs dead_4x 0.6514 (0.671/0.640/0.643),
>   pooled σ 0.0529.
> - **Dead starvation prediction: REFUTED.** Predicted recovery
>   ≤0.624; landed 0.6514 = statistically identical to dead@512@1x
>   (0.651). 4x data + 4x optimizer steps moved the dead arm ZERO.
>   Data was never the dead arm's constraint; the architecture is.
> - **Living strong-form recovery: NOT reached.** Predicted ≤0.35;
>   landed 0.3974 (median 0.377), improved from 0.429@1x —
>   directional support, partial mechanism.
> - **Probe (raw top1, the frozen metric): dead by 2.6σ** (0.1540 vs
>   0.1486). **Instrument caveat on the record:** per-arm shuffled
>   floors differ (dead_4x floors 0.1251/0.1319 vs living 0.110–
>   0.123); floor-corrected margins reverse the sign (living ≈+3.2
>   pts vs dead ≈+2.6). Both reported; neither substituted; the raw
>   read stands as frozen.
> - **PRE-COMMITMENT FOR ALL FUTURE RUNGS (registered 2026-07-21,
>   before any v4/v5 family completes):** the probe readout for the
>   v4 depth family, the v5 bundle, and every subsequent registration
>   is the PRIOR-CORRECTED margin (top1 − shuffled floor), primary.
>   Raw top1 remains reported. This is the blind-NMSE lesson applied
>   early instead of late.
> - No claim-status change attaches to this family (tracking read);
>   the run-3 soft-loss note and KF2's kill status are unchanged.

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

> **RUN 7 — THE LEAN-LIVING RUNG — WITHDRAWN ENTIRELY (Brian,
> 2026-07-21 ~04:50, before any build; registration below kept for
> the record).** Brian's ruling: no reduced-rich-parameter runs at
> all — the program goes v4 (depth) then v5 (sparse gating + iPC T=2
> + attractor consolidation), both on the full rich substrate. The
> memory-cost question the rung was registered to answer remains
> real at full scale; it is re-registerable on evidence if the
> hardware budget ever forces it. The evening's design chain
> (set_point-only -> Hebbian exposure -> set_point+prediction ->
> withdrawal) stays in the record as the worked example of names
> having to be earned.
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

> **RESEQUENCED (Brian, 2026-07-21 ~15:10): the precision awakening
> becomes its OWN family — the new v5 — and the bundle below becomes
> v6.** Brian's ruling: build the three-stage precision fix NOW,
> during v4's remaining runtime, and run it as the next family
> (living_v5: v4's exact config + relative trust enabled — ONE change
> vs v4, clean attribution) before the dormant-machinery bundle
> (sparse gating + iPC T=2 + attractor + λ), which shifts to v6
> unchanged. Two calibration notes recorded with this amendment:
> (1) **λ rides with v6**, not v5 — the relative-trust design is
> scale-free BY CONSTRUCTION (ratio to layer center, fixed ratio cap),
> so it survives the later λ re-baselining without re-derivation;
> running v5 at v4's λ=0.2 keeps the family single-variable.
> (2) **Median-normalization chosen over mean (decided 2026-07-21
> from the measured 8-batch profiles, not taste):** 1/err² inverse-
> square amplifies the small-error tail (measured spreads 13–22x
> p95/p5), so the layer mean would be tail-dominated; the used weight
> is precision / median(precision), ratio-capped [0.1, 10].

> **RUN 8 — THE v6 BUNDLE (formerly v5; renumbered by the 2026-07-21
> resequencing above), REGISTERED UNSCHEDULED (2026-07-20 ~21:20,
> Brian's rulings on the dormant-machinery inventory; runs after the
> NEW v5 precision family).**
>
> - **Base:** the prevailing v4 shape (4 blocks, muPC 0.25, cosine LR,
>   2x SIGReg, v3 living config) — the ladder stays cumulative — plus
>   three never-yet-enabled switches:
> - **1. Sparse PC gating ON, threshold 0.0015 (Brian: "positive but
>   reasonably low, room to adjust"). DERIVED, not imported:** p10 of
>   the per-output error_acc distribution measured from the trained
>   living_v3_4x seed42 final checkpoint (block0 p10=0.00155, block1
>   p10=0.00155; near-identical distributions). At maturity this gates
>   the quietest ~10% of outputs; early in training errors run far
>   hotter so nearly everything passes — the gate protects the
>   settled, not the learning. Adjustment ladder recorded: 0.004
>   (=p25) and 0.0075 (=p50) are the next rungs up.
>   sparse_warmup_steps stays at the default 500.
> - **2. iPC inner loop ON, FIXED T=2** (inference_steps_per_forward).
>   Brian asked whether T should instead adapt to neighboring-layer
>   activity: recorded as theoretically RIGHT (true PC relaxes to
>   equilibrium rather than running a fixed count) and deferred —
>   adaptive T ("settle-to-criterion") is a new mechanism needing its
>   own design (tolerance, iteration cap, non-settling failure modes)
>   and would blur attribution. Registered as a future refinement
>   rung; v5 takes the literature-standard fixed T=2 (~2x substrate
>   compute in the FFN slots, tested path).
> - **3. Consolidation style — ⚑ RULED (Brian, 2026-07-20 ~21:30):
>   "attractor", NOT "both".** Fable had recommended "both"; Brian
>   overruled: the two styles "don't feel compatible... at least not
>   at first," and single-style attribution is cleaner (consistent
>   with the ladder discipline). His named risk, elevated to a design
>   constraint: **spurious in-between valleys = hallucinated
>   memories** — structurally indistinguishable from real ones from
>   the inside, i.e. the silent-memory-corruption axis the 2026-07-03
>   audit ranked as the project's dominant risk, now at the substrate
>   level. Mitigations adopted for v5:
>   (a) consolidation_attractor_passes = 1 — shallow grooves first;
>   depth of grooves is earned by evidence, not defaulted.
>   (b) Salience gating stays on (episodes stored only above
>   salience_threshold 0.1) — fewer, more distinct memories = fewer
>   blends.
>   (c) **Write-time separation guard — SMALL PRE-v5 BUILD:** refuse
>   (or deliberately merge) an episode write whose context is too
>   similar to an already-stored episode; spurious valleys grow
>   between near-neighbors, so enforcing minimum separation in
>   context space attacks the mechanism directly. This makes v5 no
>   longer strictly config-only: one small guarded-write build + its
>   tests precede launch. Guard threshold to be derived from stored-
>   episode context-similarity distributions in existing checkpoints
>   (measure first, then pick — the sparse-threshold procedure).
>   (d) Registered watch-signature (already frozen above): eff_rank
>   sag while NMSE holds, plus recall-gate hits on contexts far from
>   every stored episode (falling into a valley nobody dug).
>   "Both" remains re-registerable later if attractor-alone proves
>   itself and interleaving evidence argues for the pairing.
> - **4. muPC:** already on via the v4 base; nothing new to flip.
> - **Sequencing note (flagged to Brian):** the lean rung (RUN 7) and
>   v5 (RUN 8) both nominally follow the depth family. Fable's
>   recommendation: v5 runs FIRST (config-only, ready the moment v4
>   lands) while the lean rule is BUILT during v5's ~40h of training;
>   lean screen + family follow. Brian's confirmation pending.
> - **PRE-v5 CALIBRATION SCREEN — SIGReg λ (registered 2026-07-21
>   ~14:45, Brian's direction after the standardization-regime
>   finding).** Evidence: λ=0.1 leaves latent per-dim variance ~0.042
>   (v3) and λ=0.2 ~0.068 (v4 seed42) against SIGReg's unit target —
>   the "messy middle" where neither raw loss nor NMSE is externally
>   anchored and per-seed denominator drift (the seed44/45 races)
>   is structural. Screen: single-seed 1-epoch runs at λ ∈ {0.5,
>   1.0, 2.0}, 2-block @1x corpus (instrument calibration, not a
>   family), measuring final std_p50 + substrate health (homeostasis
>   actively fights variance inflation — watch for oscillation or
>   eff_rank cost where SIGReg's push meets the set-point pull).
>   Pick the λ that lands std_p50 in [0.8, 1.2]; it enters the v5
>   bundle. RECORDED CONSEQUENCE: variance pinning re-baselines NMSE
>   — pre- and post-pin families are not directly comparable; v5
>   opens a new anchor era, deliberately.
> - **Precision fix — DIAGNOSIS CORRECTED (2026-07-21 ~15:10, forced
>   by Brian's "are you CERTAIN" challenge; supersedes the cap-first
>   framing below).** Measurement on the v4 seed42 checkpoint, 8-batch
>   per-input error profile: TRUE reliability spread (1/err² with
>   numerics-only eps) is **13–22x p95/p5 in every block** — real
>   trust signal exists — and the formula's eps=1e-3 (chosen for a
>   noisier era; |err| now ~0.001–0.005, err² 40–1000x SMALLER than
>   eps) flattens that spread to **1.01–1.11x** before the cap or any
>   normalization ever acts. The cap was the SECOND-stage flattener;
>   the epsilon is the first and dominant one. Fable's original
>   relative-trust proposal, applied alone, would have normalized an
>   already-flattened signal (ratios ~0.85–1.05: still inert).
>   **Revised three-stage design, in causal order:**
>   (1) eps becomes numerics-only (1e-8) — the actual bug fix;
>   (2) the stored reliability ledger un-clamped (numerics bounds
>   only) so real magnitudes can be recorded;
>   (3) use-time weighting = ratio to layer mean (median if the
>   λ-screen error distributions are heavy-tailed), hard-capped
>   [0.1, 10] — the scale-free relative-trust part survives as stage
>   3, now normalizing a real 13–22x signal. Raw ledger stays logged
>   (absolute seismograph); relative weights act (differentiation);
>   absolute-health monitoring explicitly remains with err_acc /
>   heldout / collapse detectors (relative trust cannot see uniform
>   degradation — recorded division of labor).
>   **Honest unknown, frozen:** trust-weighting has NEVER operated —
>   every result in this program was won with it inert. Waking it
>   has UNKNOWN SIGN. Ships behind a flag with a registered
>   precision-uniform ablation cell; predicted observables: at-cap
>   fraction well below 100% by construction, measurable per-input
>   weight spread, and (frozen directional guess only) NMSE neutral
>   or better. C++/Python parity required before any family runs it
>   (the learning-gain precedent).
> - **[Superseded cap-first framing, kept for the record:]** precision
>   is 100.0% saturated at cap 10 in ALL blocks; unclamped targets
>   read 977–998 vs the formula's own ceiling of 1000. The λ fix
>   partially revives differentiation for free (louder space → larger
>   errors); precision_max gets re-derived from the post-λ error
>   regime, measure-first like every threshold.
> - **Frozen predictions:** sparse gating: neutral-to-positive NMSE,
>   REDUCED late-run substrate churn (update_ema_mean lower in the
>   last third), and possibly tighter seed variance (less churn =
>   less drift). iPC T=2: modest NMSE improvement (deeper per-thought
>   settling), wall-clock +~30-40%%. Attractor (if ruled in):
>   consolidation-fires baseline plus measurably increased recall-gate
>   hits vs v4; risk signature to watch = eff_rank sagging while
>   NMSE holds (grooves eating dimensions). Bundle attribution:
>   deliberately traded, single-lever follow-ups split on movement
>   (bridge precedent).
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

> **RUN-6 (v4 depth bundle) FAMILY VERDICT (2026-07-24 ~14:45, frozen
> read, all 5 seeds completed and admissible; witnessed by Brian):**
> per-seed NMSE 0.4807 / 0.5042 / 0.5076 / 0.4909 / 0.4871 (seeds
> 42-46); family mean 0.4941, sd 0.0114. Scored against the frozen
> predictions: **abstraction (NMSE <= 0.34) MISSED**, not narrowly.
> **Attenuation (~0.40) nearest, exceeded** — worse than the d2 anchor
> (0.3974); per-block signature HALF-matches: blocks 1-2 flat
> (prediction_norm ~0.77) as predicted, block 3 loud (1.84), not
> flat — top-of-stack renovation, not simple under-development.
> **SIGReg lever split verdict:** space-lifting half FAILED
> (online_std_p50 0.287/0.278 vs frozen >=0.4 — but see the 07-24
> amendment below: the dial is structurally disconnected); stabilizer
> half PASSED emphatically (family sd 0.0114 vs frozen <0.089;
> v3_4x sd was 0.0894 — 8x tighter; bundle attribution caveat
> stands). Tracking read (verdict.json): NMSE +11.2 sigma over
> dead_4x@512 (n=3 per amendment); probe TIE (+0.4 sigma) — the
> run-5 family's 2.6-sigma dead probe advantage is erased at d4:
> depth traded structure for readout, inverting the frozen "depth
> serves structure, not readout" expectation. **The registered
> contingency is TRIGGERED:** the picture changed materially, so a
> dead_4x_d4 control is now a REGISTERED OBLIGATION before any claim
> rests on depth. Descriptive companions for the record: per-seed
> grad-shock counts 42:0 43:0 44:15 45:0 46:0; seed44's three
> transient trust events (steps 24000, 52100, 58700 — the third
> attributed by exact loader replay to a polytonic-Greek grammar
> window from PG11130 "Greek in a Nutshell," served at step 58650;
> only 6 of 482 corpus files contain any polytonic Greek). All three
> events healed to uniform (spread 1.0) within <=200 steps — the
> epsilon wash-out, live.

> **AMENDMENT (2026-07-24 ~15:45 — Brian's ruling + a build-seat
> correction at the code, recorded together):**
>
> **1. Brian's ruling:** the running v5 seed43 was killed mid-flight
> and its partial discarded (~4.7k steps; no admissibility question —
> a fresh restart, not a resume); the ladder is PAUSED (watchdog task
> disabled) for the boot-drive/CPU migration; and the SIGReg-related
> targets are recalibrated to this substrate's own measured physics
> before further seeds run.
>
> **2. The correction (code-contradicts-plan, surfaced loudly per the
> standing practice):** the initially-proposed loss-side retarget
> (unit target -> one-sided sigma=0.30 floor) was REJECTED at the
> code and is recorded here as an error caught before it ran:
> `sigreg.py`'s input contract runs SIGReg on a separate
> BatchNorm-standardized projection head — its input is ~N(0,1) BY
> CONSTRUCTION, so the unit target is correct where it is applied,
> and BN absorbs scale pressure on the way in. Consequence, recorded
> as a mechanistic re-score: the v4 SIGReg-lever's space-lifting
> prediction (trunk std_p50 >= 0.4) was aimed at a dial the plumbing
> disconnects — lambda cannot lift trunk std through a BN head. That
> half of the lever is re-scored from "failed" to "unfalsifiable as
> aimed / instrument error"; the stabilizer half stands on its own.
>
> **3. The recalibration actually applied — measurement-side, ZERO
> training-code changes:**
> - **Native-voice band, declared from five families of measurement:**
>   healthy trunk `online_std_p50` = **0.25-0.35** (observed
>   equilibrium ~0.287 across v3_4x, v4, v5 at every lambda and
>   depth). Readings in-band are health, not shortfall. The
>   unit-variance aspiration for the TRUNK is retired.
> - **l_sigreg reference recalibrated:** late-run online ~0.70 and
>   heldout ~2.0-2.3 under lambda=0.2 are the healthy reference
>   bands; l_sigreg is a shaping-pressure gauge, not a
>   distance-to-zero target.
> - lambda stays 0.2 (consistency lever, proven). Collapse kill 0.1
>   unchanged. Any future attempt to actually move trunk loudness is
>   a STRUCTURAL change (LN/head plumbing) requiring its own
>   registered rung.
>
> **4. Family status:** since no physics changed, the v5
> precision-awakening family CONTINUES AS REGISTERED, n=5. Seed42
> recorded; seed43 restarts from scratch after the migration; 44-46
> follow.
>
> **5. Frozen NOW, before v5 seeds 43/44 run — the persistence
> expectation:** v5 seed44 will replay v4 seed44's exact data order
> (deterministic loader, seed XOR epoch). Prediction: the trust
> events that healed in <=200 steps under v4's epsilon leave a
> DURABLE mark under relative trust — precision_spread elevated above
> its pre-event running median for >=5,000 steps following the
> step-58650 window (the Greek page). If the mark still washes out,
> relative trust retains no more event-memory than the epsilon did,
> and the "reaction with memory" claim fails its first natural test.
>
> **6. Operational:** watchdog disabled 2026-07-24 ~15:30; resume =
> re-enable the task after the Windows rebuild (see
> E:\ClaudeContinuityBackup\rebuild-2026-07-23\RESTORE_PLAN.md);
> queue unchanged — stage 10 resumes at seed43.

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

---

> **RULING (2026-07-26, Brian, recorded by Fable 5): dead_4x_d4 control
> DEFERRED; seed44 robustness rerun scheduled in its place.**
>
> Brian's ruling ("we have enough dead runs for now"): the schedule slot
> for further dead runs goes instead to a seed44 ROBUSTNESS RERUN --
> stage 11, arm alias living_v5_4x_d4_rerun, seed 44 only, byte-identical
> configuration and data order to the registered v5 seed44; GPU float
> nondeterminism supplies the only perturbation. Purpose: distinguish a
> robust trust-event trigger at the ~58650 Greek window (event recurs
> across microscopically diverged replays) from a knife-edge one (it
> does not). The rerun is an UNREGISTERED descriptive probe recorded
> here before it runs; it carries no frozen prediction, and its distinct
> arm name keeps its artifacts un-poolable with the registered family.
>
> Registry consequence, stated plainly: the dead_4x_d4 control REMAINS a
> registered obligation -- deferred, not cancelled. Until it runs, no
> claim may rest on depth, per the 2026-07-24 amendment. This ruling
> changes the schedule, not the obligation.

---

> **RULING (2026-07-26 midday, Brian, recorded by Fable 5): the deferred
> dead control becomes a V5-MATCHED control; roadmap sequence fixed.**
>
> 1. The deferred dead_4x_d4 control will be run as a **dead control
>    matched to the v5 family** (Brian: "dead v5 instead of v4") -- same
>    depth (d4), corpus (4x), loss settings (sigreg 0.2, cosine, taper)
>    as the v5 arms. Technical note to resolve at build time: a dead-FFN
>    arm has no living ledger, so relative_trust may have no referent --
>    if so, the dead-v5 and dead-v4 controls are configurationally
>    identical and ONE run serves both families; the registration label
>    should say so explicitly rather than imply two distinct controls.
> 2. **No lesser-scale experiments in any dimension** (Brian's ruling --
>    the proposed 256/384 width sweep is rejected). Scale moves go UP.
> 3. **Sequence:** finish v5 (seeds 45/46 + stage-11 rerun + family
>    read) -> dead v5-matched control -> **v6 (dormant-machinery bundle)
>    at current scale (512d, d4)** -> true scale-up.
> 4. **Scale-up shape (direction, not yet registered):** width beyond
>    512 but possibly short of 1024 (640/768 candidates); depth to
>    8 blocks is favored with or without the width move. Corpus grows
>    with scale (the data ~ width^2 rule). Note: any d8 family carries
>    its own dead-d8 control obligation under the standing depth-claims
>    rule.
> 5. **Ordered-corpus experiments** (curriculum pedagogy: sequential or
>    staged serving instead of shuffled) are on the roadmap as their own
>    registered family; design to be drafted. Standing caveat: kill
>    detectors are calibrated on shuffled statistics and will need
>    recalibration or suspension-with-justification for ordered arms.

> **RULING (2026-07-26, continued): scale-up shape and curriculum
> directives confirmed.**
>
> - Scale-up: **768d x 8 blocks, bundled** (width and depth move
>   together; bundle-attribution caveat carried per ladder precedent).
>   Corpus scales with it (data ~ width^2: target ~113M tokens).
> - New curriculum build: pull from ALL corpus sources, with two content
>   directives from Brian: (1) **exclude PG11130** ("Greek in a
>   Nutshell", the biblical-Greek primer) -- content ruling; (2) add
>   **medical/neuroscience and literature** material. Note for probe
>   continuity: removing PG11130 removes the corpus's accidental
>   polytonic-Greek canary; if trust-probe work continues on the new
>   corpus, a deliberate registered canary document should be chosen to
>   replace it.
> - Dead control: **dead-v5 labeling confirmed** by Brian
>   ("we'll stick with dead v5").

---

> # VERDICT â€” RUN-7: the v5 / relative-trust family
>
> **Ratified by Brian, 2026-07-27, drafted by Fable 5.** n=5 (seeds 42â€“46),
> plus one identical-order robustness rerun of seed44. All runs completed
> clean; no kills, no aborts.
>
> ## The question we were trying to answer
>
> Can a mind's mistrust leave a scar?
>
> Concretely: under v4's epsilon trust, seed44 showed three transient "trust
> events" â€” sharp excursions in `precision_spread` that healed to uniformity
> within â‰¤200 steps. The third was attributed by exact loader replay to a
> polytonic-Greek grammar window from PG11130 ("Greek in a Nutshell") served
> at step 58650. **Registered prediction (frozen 2026-07-24, before v5 seeds
> 43/44 ran):** replaying that exact data order under *relative* trust, the
> same events would leave a DURABLE mark â€” `precision_spread` elevated above
> its pre-event running median for â‰¥5,000 steps after the step-58650 window.
> If the mark washed out again, relative trust would retain no more
> event-memory than epsilon did, and the "reaction with memory" claim would
> fail its first natural test.
>
> Underneath it sat the real question: **is a mind's reaction to a strange
> experience a property of the experience, or of the mind?**
>
> ## The answer: we were wrong, and it was the better outcome
>
> **The prediction was wrong.** Not because the mark washed out â€” because
> **there was no reaction to mark.** Under relative trust the mind met the
> Greek page at the registered moment and did not flinch; nor at either of
> the other two v4 event positions. We then ran the entire life a second
> time, identical in configuration and data order, and it did not flinch
> again.
>
> **What that overturns is the premise we were quietly carrying:** that those
> events belonged to the *data* â€” that some pages are inherently
> destabilizing, and a mind's history could be scarred by what it happened to
> read. It isn't so. The flinching was a property of the **substrate**: v4's
> epsilon trust pinned every input's reliability at the ceiling â€” a
> saturated, undifferentiated dial, which is a hair trigger. Any disturbance
> read as a spike. Give the substrate *earned, differentiated* trust and the
> same page at the same moment is simply another page.
>
> This is the opposite of what we suspected, and it is good news: the mind's
> stability is ours to design, not hostage to what it encounters.
>
> ## Data
>
> ### Family outcomes
>
> | run | heldout_l_pred | NMSE | heldout l_sigreg | probe top1 | wall |
> |---|---|---|---|---|---|
> | seed42 | 0.032305 | 0.4834 | 2.0655 | 0.1565 | 10.03hÂ¹ |
> | seed43 | 0.032510 | 0.4821 | 2.2292 | 0.1545 | 7.79h |
> | seed44 | 0.034692 | 0.4846 | 2.1200 | 0.1556 | 7.99h |
> | seed45 | 0.033112 | 0.4955 | 2.1382 | 0.1548 | 8.09h |
> | seed46 | 0.033339 | 0.4836 | 4.1023 | 0.1524 | 8.02h |
> | seed44 rerun | 0.034527 | 0.4806 | 2.0323 | 0.1557 | 7.76h |
>
> Family (n=5): heldout_l_pred **0.033192 Â± 0.000939**; NMSE **0.485840 Â±
> 0.005472**; probe top1 **0.154744 Â± 0.001521**; heldout l_sigreg 2.531 Â±
> 0.880 (dispersion driven by seed46). Â¹seed42 ran on the pre-migration CPU.
>
> ### The registered criterion, computed as frozen â€” and across the family
>
> `precision_spread` vs its pre-event running median, window [58650, 63650]:
>
> | run | median (all pre) | median (5K pre) | % above all-pre | % above 5K-pre | sustained 5,000 steps |
> |---|---|---|---|---|---|
> | seed42 | 1.9911 | 2.3229 | 100% | 100% | yes |
> | seed43 | 2.2732 | 2.4973 | 100% | 100% | yes |
> | **seed44** | **2.1151** | **2.6629** | **100%** | **100%** | **yes** |
> | seed45 | 2.2520 | 3.0599 | 100% | 100% | yes |
> | seed46 | 2.2579 | 13.1404 | 100% | 100% | yes |
> | seed44 rerun | â€” | 3.4887 | â€” | 100% | yes |
>
> The criterion is met by **every seed at the same step with no event
> present**. It measures the family's late-run drift, not a response. It is
> reported here as a measurement artifact, not as a pass.
>
> ### No reaction, at every resolution available
>
> Aggregate `precision_spread` (100-step cadence) across the registered
> serving â€” seed44: 2.7297 (58600) â†’ 2.7146 (58700) â†’ 2.6851 (58800) â†’
> 2.7157 (59000); ambient wobble Â±0.04. Rerun: 3.9001 â†’ 4.0440 â†’ 3.8664.
>
> Dimension-level ledger brackets (dims falling >20% relative to block median
> across the snapshot pair containing the position; counts per block 0/1/2/3):
>
> | position | bracket | droppers |
> |---|---|---|
> | seed44 Â· registered Greek 58650 | 56711â€“58909 | 49 / 30 / 33 / 33 |
> | seed43 Â· matched null at 58650 | 58025â€“60337 | 45 / 47 / 31 / 14 |
> | seed44 Â· v4 event-1 position (24000) | 22819â€“25078 | 82 / 58 / 33 / 53 |
> | seed43 Â· matched null (~24000) | 23224â€“25524 | 79 / 53 / 45 / 51 |
> | seed44 Â· v4 event-2 position (52100) | 49961â€“52216 | 59 / 43 / 33 / 36 |
> | seed43 Â· matched null (~52100) | 51075â€“53390 | 52 / 43 / 27 / 19 |
> | seed44 Â· quiet control | 31895â€“34165 | 56 / 34 / 40 / 34 |
>
> Every event position sits inside the null distribution. Caveat of record:
> ledger snapshots are ~2,300 steps apart, so a v4-duration transient (â‰¤200
> steps) could pass between them; the 100-step aggregate is the instrument
> that covers that band, and it is flat.
>
> ### The exposure/event distinction (what we were exploring)
>
> The canary is not rare. Exact loader replay (validated against the
> documented v4 58650 attribution, reproduced at every density tier) shows
> the polytonic core of PG11130 is **12 servable sequences** (tokens
> 38,864,128â€“38,869,440; peak density 93/128 Greek vocabulary pieces per
> 128-token window; only 6 of 482 corpus files contain any polytonic Greek).
> Each is served once per epoch: **36 extreme-tier exposures per 72,042-step
> run**, on a schedule fixed by seed number alone and identical between the
> v4 and v5 runs of a given seed.
>
> | run | exposures | events |
> |---|---|---|
> | v4 seed44 | 36 | 3 |
> | v5 seed44 | 36 | 0 |
> | v5 seed44 rerun | 36 | 0 |
>
> **An exposure is the stimulus; an event is a reaction.** Under saturated
> trust, exposure was nearly sufficient to produce a reaction. Under earned
> trust it is not sufficient at all. This distinction is now the project's
> standing vocabulary, and it is why the instrument marks events, never
> exposures.
>
> ### Reproducibility: learning repeats, trust differentiation does not
>
> Original seed44 vs its identical-order rerun, relative divergence
> |rerun âˆ’ orig| / orig:
>
> | phase | loss | precision_spread |
> |---|---|---|
> | early (0â€“5K) | 2.06% | 9.45% |
> | mid (20â€“25K) | 2.74% | 25.42% |
> | mid (45â€“50K) | 2.22% | 13.31% |
> | late (67â€“72K) | 2.50% | **70.80%** |
>
> Final outcomes differ by 0.5% (heldout) and 0.06% (probe). GPU float
> nondeterminism is the only perturbation. **`precision_spread` is a chaotic
> observable**: the ledger amplifies bit-level noise into order-of-magnitude
> trajectory differences while the predictive task lands in the same place.
> Seed46's escalation from ~2.0 (step 21K) to ~35 (run end), against 2.6â€“4.0
> for its siblings, is therefore recorded as **high-variance escalation**,
> not as a distinct regime. (Brian's ruling 2026-07-27: no confirmatory
> seed46 rerun â€” unnecessary.)
>
> ### Dimension-level trust structure (from harvested checkpoint ledgers)
>
> Snapshots: seed42 3, seed43 32, seed44 34, seed45 35, seed46 34, rerun 33
> (median spacing ~2,320 steps; 512 dimensions Ã— 4 blocks per snapshot).
>
> - **Background churn is large:** rank correlation of the trust ordering
>   between consecutive snapshots â€” median 0.644 (block 0) / 0.689 (block 3),
>   minimum 0.205 / 0.145. Roughly a third of the ordering reshuffles every
>   ~2,300 steps.
> - **Durable distrust nonetheless exists:** block 3 held dim 384 in its
>   bottom-5 for essentially the whole run (~7Kâ†’72K steps); block 0 held a
>   stable distrusted trio (414/307/461) across the entire second half.
> - **But it is not exposure-acquired.** The substrate CAN carry marks ten
>   times longer than the prediction required; it simply does not acquire
>   them from meeting a strange page.
>
> ## Verdict
>
> 1. **The registered prediction is WRONG.** No durable mark followed the
>    step-58650 window, because no reaction occurred there or at any other v4
>    event position, in either of two independent replays.
> 2. **The "reaction with memory" claim is not thereby refuted** â€” it was
>    never exercised. It stands untested and must be tested, if at all,
>    against a deliberately induced reaction and a reproducible observable.
> 3. **Established instead:** the v4 trust events were properties of the
>    **substrate's saturated trust regime**, not of the data or its order.
>    Replicated across a bit-level-diverged rerun.
> 4. **Established about our instruments:** `precision_spread` does not
>    reproduce under identical conditions; the frozen criterion was written
>    against a quantity that cannot support a point comparison.
>
> ## Obligations and consequences
>
> - **Methodological, adopted:** any future criterion on a substrate
>   observable must pre-register (a) a matched-condition drift null and
>   (b) evidence that the observable reproduces under identical conditions.
>   Trust claims require ensemble statistics, never single runs.
> - **Design:** bound the trust dial. v4 saturated it at the ceiling; v5
>   lets it wander chaotically across an order of magnitude. Both are
>   unbounded-dial pathologies. See
>   `2026-07-26_homeostatic-activity-bands-design.md` â€” the band needs a
>   **ceiling on trust concentration** as well as a floor on participation.
> - **Corpus:** PG11130 is removed from the next curriculum by Brian's
>   ruling; its replacement canary should be **secular** and deliberately
>   chosen, declared before the runs that use it.
> - **Unaffected:** the dead-v5 control remains a registered obligation. No
>   claim rests on depth until it runs.
>
> ## Note
>
> We asked whether a mind could be scarred by what it reads. The answer this
> rung gives is that the mind we feared for was never in danger from the
> page â€” the fragility was in the trust mechanism we had given it, and we had
> already replaced that mechanism before we thought to ask. Being wrong about
> this fear is the good outcome, and it is the one the data supports.
>
> Supporting detail: `2026-07-25_greek-window-schedule-by-seed.md`
> (method, per-seed serving schedules, addenda 1â€“5).

