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

2. **Self-modification vs. static capacity.** The claim's history, kept
   honestly: the v1-era "convergence penalty" (~39% slower) was retired
   by the v2 PC substrate; the LM-era result (v2 0.64% better than a
   vanilla transformer at matched configuration) is now HISTORICAL —
   real under its objective, unbound from the current criteria after the
   project moved to the JEPA training goal.
   *Status after JEPA stage 1 (2026-07-16, pre-registered verdict): the
   claim SURVIVED the matched-capacity point — variance-normalized
   held-out prediction error favored the living arm by 5.2 pooled sigma
   (0.4516 vs 0.6240); the external probe axis tied (0.1533 vs 0.1581,
   under 1 sigma). Read narrowly and honestly: the surviving claim is
   about latent-structure capture, not yet task-usable representational
   quality, and it is NOT reinstated as a headline until the capacity
   bracket (larger static controls) rules out the effective-capacity
   explanation. Bracket pending ratification. Verdict data:
   `runs/jepa_pilot/verdict.json`; blind-amendment record in the
   pre-registration.*

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
