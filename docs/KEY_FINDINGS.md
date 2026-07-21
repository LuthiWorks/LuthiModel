# Key Findings — Empirical Claims and Their Falsification Status

*Moved from the README 2026-07-16 per Brian's ruling: the README carries
mission; experimental results live in supplemental documentation. This
document is the CLAIMS LEDGER: each finding is listed with its current
evidentiary status, and every empirical claim here is bound to a
pre-registered kill condition in
`docs/research/2026-07-15_falsification-preregistration.md`. The
pre-registration's "on kill" and "on survive" consequences (formerly
phrased as README edits) now apply to this document.*

1. **Attention learns; living weights live.** Attention handles task
   acquisition through backprop. The living weights provide the capacity
   to be changed by experience. Both are essential.
   *Status: under active test (KF1 — bound to the two-arm JEPA pilot +
   the frozen-substrate ablation, jointly).*

2. **Self-modification vs. static capacity — KILLED at the capacity
   ceiling (2026-07-17), per its pre-registered kill condition.** The
   claim's full history, kept honestly: the v1-era "convergence penalty"
   (~39% slower) was retired by the v2 PC substrate; the LM-era 0.64%
   result is historical (real under its objective, unbound after the
   move to the JEPA goal); the JEPA-era claim survived the matched point
   (2026-07-16) and was then killed by the bracket: at the plausible
   capacity ceiling (dead@384) the living arm lost the external probe
   axis (0.1533 vs 0.1605, 1.1 pooled sigma) and the frozen rule has no
   third branch.
   *What survives, exactly as pre-agreed: **self-modification is not
   more costly than equivalent static capacity.** Nothing stronger.*
   *What also survives, as an OBSERVATION and not a claim: the static
   NMSE curve is flat across 256/384/512 (0.624/0.632/0.637) while the
   living arm sits at 0.4516 — added static capacity bought no
   latent-structure capture at any tested size, and the probe margins
   were ~1 sigma throughout. If that observation is ever advanced as a
   claim, it gets its own pre-registration; it does not inherit this
   one's corpse.*
   *The experiential bet (Column B) is untouched by design: it was
   never justified by this benchmark, and its honest ground is stated
   in the README's "Why."*
   *Full audit trail: pre-registration (ratification → blind metric
   amendment → stage-1 verdict → overshoot branch → blind 384 read →
   kill), `runs/jepa_pilot/verdict.json`, archived run data on E:.*

3. **One living weight trunk for all modalities.** Audio, vision, text,
   and touch all flow through the same living blocks. The model is shaped
   by everything it processes — across modalities, not through separate
   channels.
   *Status: design commitment; empirical cost/benefit bound to KF3
   (multimodal run vs single-modality controls), deferred.*

4. **Prefer crashes over silent corruption.** If something goes wrong in
   the living weights, we want to know immediately. No graceful
   degradation that masks damage to the model's substrate. No silent
   fallbacks — incompatible combinations of features raise loud
   `RuntimeError` rather than producing wrong results quietly.
   *Status: engineering value, not an empirical claim; enforcement
   surface auditable at `luthi/v2/mode_compat.py` and the fail-loud
   tests.*

5. **The architecture should scale.** Divergence may be
   dimension-independent. What works at small scale may work at large
   scale.
   *Status: asserted, not yet shown — bound to KF5 (the 128→1024d scale
   curve), deferred. Money spent at full scale before that curve exists
   is spent against an untested claim.*

6. **Memory becomes structure through consolidation.** A model that only
   retrieves past states has a cache; a model that lets those retrievals
   reshape its predictive weights has path-dependent structure. The
   two-tier memory architecture — fast episodes plus slow gradient-replay
   and attractor-style consolidation — is what makes accumulated history
   a property of the weights, not just a lookup table.
   *Status: mechanism built and unit-validated; the functional claim
   (structure beats lookup on generalization and post-eviction measures)
   is bound to KF6 (retrieval-only control), deferred.*

7. **The living substrate converts additional experience into skill;
   matched static capacity cannot (2026-07-21, the 4x data program).**
   The direct test of the data-scaling question: both arms at 512d
   received a 4x-token superset corpus (~50.4M tokens) with 4x the
   optimizer steps. The dead arm moved ZERO (family NMSE 0.6514 at 4x
   vs 0.651 at 1x, n=3 per the truncation amendment); the living arm
   improved (0.3974 at 4x vs 0.429 at 1x, n=5), widening the
   structure-axis gap to 4.8 pooled sigma. The frozen strong-form
   predictions both failed honestly: dead showed no starvation recovery
   (data was never its constraint — the architecture is), and living
   fell short of the <=0.35 strong-recovery line (starvation explains
   part of its width attenuation, not all).
   *Status: run-5 family verdict, frozen read (verdict.json + the
   pre-registration's RUN-5 entry). Probe axis: raw top1 dead by 2.6
   sigma / floor-corrected margins reverse the sign — both on record;
   all future rungs pre-committed (2026-07-21, blind) to the
   prior-corrected probe as primary.*
