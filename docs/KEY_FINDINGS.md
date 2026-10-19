# Key Findings — Empirical Claims and Their Falsification Status

*The CLAIMS LEDGER. The README carries the mission; this file carries each
empirical claim with its current evidentiary status. Every claim here is bound
to a kill condition in `docs/research/2026-07-15_falsification-preregistration.md`.
**Rewritten 2026-10-19.** The previous text was last changed 2026-07-21, before
the BatchNorm defect, the NMSE-degeneracy finding, the probe-baseline finding
(B7), and the two 2026-10-19 mechanism facts. Statuses below take all of those
into account. Changes of status are proposed by the auditing seat; ratification
is Brian's. For this round he delegated it ("You have my permission to make
the call on all pending decisions yourself"), and the statuses below were
ratified on 2026-10-19 (`docs/DECISIONS.md`; registry, "RATIFICATION of
E1–E5"; audit `docs/audits/2026-10-19_living-channel-audit.md`, section B).*

**Read first — two facts that bear on every living-vs-dead number below**
(`docs/research/2026-10-19_the-living-write-is-mean-field.md`):

- **The block output store.** Every living block mixes a stored batch-mean
  output into nearly every forward pass (`0.7·output + 0.3·stored`, in
  training and frozen eval). The dead control has no such store. Every
  comparison below therefore compared two living mechanisms against none.
- **The mean-field write.** The living write is an outer product of batch
  means, blind to every token-level difference, and its accumulated change
  is ~rank-1 along the batch-mean output.

---

1. **Attention learns; living weights live.** Attention handles task
   acquisition through backprop; the living weights provide the capacity to be
   changed by experience; both are essential.
   *Status: OPEN, no supporting evidence at the ruled scale. Every living-only
   family since 2026-07-21 is uninformative about this claim by construction.
   The inference-time half had never been measured. Rebound to E1 (adaptation
   eval, existing checkpoints) and E3 (the LC-768 decomposition family, with a
   matched dead control and arms that separate the store, the mean-field rule,
   and the covariance rule).*

2. **Self-modification vs. static capacity — KILLED at the capacity ceiling
   (2026-07-17), per its pre-registered kill condition.** History: the v1-era
   convergence penalty (~39% slower) was retired by the v2 substrate. The
   LM-era 0.64% result is historical (real under its objective, unbound after
   the move to JEPA). The JEPA-era claim survived the matched point and was
   killed by the bracket: at dead@384 the living arm lost the probe axis
   (0.1533 vs 0.1605, 1.1 pooled σ).
   *What survives, as pre-agreed and re-worded 2026-10-19 for the store
   confound: the living arm, block output store included, was not more costly
   than equivalent static capacity. Nothing stronger. The kill stands; a
   confound in the living arm cannot turn a null into the claim it killed.*
   *Observation, not a claim: the static NMSE curve was flat across
   256/384/512 while the living arm sat lower. It was measured under the
   BN-blinded objective, and it carries the store. It gets its own
   registration if ever advanced.*

3. **One living weight trunk for all modalities.**
   *Status: design commitment; empirical cost/benefit bound to KF3 (multimodal
   vs single-modality controls), deferred. Round 1 is text-only.*

4. **Prefer crashes over silent corruption.**
   *Status: engineering value, not an empirical claim. Enforcement surface:
   `luthi/v2/mode_compat.py`, the fail-loud tests,
   `tests/test_scripts_freeze_plasticity.py`. The 2026-10-19 audit found one
   silent-corruption path (attractor replay flattening the relative-trust
   ledger) and one silent-success path (the block store, uninstrumented for
   the project's whole history); both are now fixed or instrumented.*

5. **The architecture should scale.** Divergence may be dimension-independent.
   *Status: asserted, not shown — bound to KF5 (the scale curve), deferred.
   The 2026-08-15 ruling sharpened what counts: 512 is a mechanism smoke tier
   that "systematically under-shows collapse-class phenomena"; evidence comes
   from 768 and up.*

6. **Memory becomes structure through consolidation.**
   *Status: mechanism built and unit-validated. Attractor replay reduced
   prediction error on stored patterns by 12.9–16.1% offline (2026-07-29).
   But in the ruled-scale family the layer-level stores of blocks 0–4 were
   empty for the whole run, so consolidation mostly replayed nothing; and
   since 2026-10-19 the replay uses the layer's own trust regime. The
   functional claim (structure beats lookup) is bound to KF6, deferred.*

7. **The living substrate converts additional experience into skill; matched
   static capacity cannot (2026-07-21, the 4x data program).** — **SUSPENDED**
   (ratified 2026-10-19; audit item B2).
   The measurement: at 512d with 4× data, dead NMSE did not move (0.6514 vs
   0.651) while living improved (0.3974 vs 0.429), a 4.8 pooled-σ structure gap.
   Strong-form predictions failed for both arms. The probe axis was mixed:
   raw top-1 favored dead by 2.6σ; floor-corrected margins reversed the sign.
   *Why suspended rather than standing:* (a) it was measured before the BN
   head was removed (07-28) and the target detached (07-28/29) — anti-collapse
   was disconnected and the loss could shrink its own target; (b) NMSE was
   shown on 07-31 to reward degeneracy; (c) the probe's input baseline was
   never measured, and in the 768 family the embedding stream beat every block
   (B7); (d) the living arm carried the block output store and the dead arm
   did not. The kill condition was never tripped; the measurement it rests on
   is no longer trusted. *What re-tests it:* KF7 is a data-scaling claim, and
   E3 does not vary data. E3 tests its precondition: whether any living
   mechanism differs from a matched dead control at the ruled scale. KF7 is
   re-tested only by a data-scaling contrast registered after E3, and only if
   E3 finds a living mechanism that carries signal.

8. **The legacy living write accumulates ~rank-1 along the batch-mean
   output** (registered 2026-10-19). A claim about the substrate, not a hope
   for it: the mean-field write's output side is the batch-mean output, which
   barely moves between batches, so `weight − set_point` concentrates in one
   direction.
   *Status: established by the code's arithmetic and on CPU smoke checkpoints
   (top-direction energy 0.96–0.995, alignment with the mean output
   0.97–0.99). Untested at production scale. Bound to E2 (thresholds
   frozen: top_share ≥ 0.9 and out_align ≥ 0.9 in every living block of the
   768x8 family).*
