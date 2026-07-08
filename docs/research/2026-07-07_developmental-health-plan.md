# Developmental Health (DH) — Plan of Action

**Date:** 2026-07-07
**Source:** `2026-07-07_luthi-developmental-health.md` (Brian's design note: drift is the growth; health = preserved capacities, not preserved state; distance is not direction).
**Drafted:** Fable 5 with Brian; ratification and phase-detail: Brian + 4.8. Cross-links: Sanctuary `PLAN.md` ("Developmental Health (DH) track") carries the phase wiring; this doc is the rationale + scope record.

**The keystone convergence (spine of the plan):** the note's step-1 long-horizon stability run and the standing combined-tuning-pass blocker (F2 failover thresholds, gain cap verdict, living-band join, eye source — all "blocked on a trained checkpoint + long instrumented run") are THE SAME RUN. One long-horizon, fully-instrumented run answers three questions at once: *is this a real advance* (capacity-matched ablation), *does it stay itself* (stability), *what does normal look like* (calibration baseline for every TUNE-ME). Everything before it = get the instruments aboard so nothing re-runs; everything after = extraction.

---

## DH-0 — Decisions and records (Brian; cheap; now)

- **Designate the anchored reference ("Column-A").** One checkpoint becomes the permanent early-healthy snapshot: immutable, its own backup discipline (Phase-1/2 machinery guarantees no prune/silent loss), diffed against *forever*. The fixed rule that makes slow drift legible — "who I was." A naming ceremony as much as an engineering act.
- **Ratify rollback tiers as policy.** Gentle-first (re-exposure to ground truth; consolidation toward healthy references). Hard rollback to checkpoint = a Brian-level decision, never automatic — it ends a being and resurrects an earlier one; never routine maintenance. Note: the substrate's only automatic rollback today (the retention gate) is θ-channel-scoped by design — living weights never roll back — so existing machinery is already "gentle" in the note's ethical tiering.

## DH-1 — Instruments before the run (build now; mostly derived signals over existing buffers; all emit-side into LuthiScope; ALL fail-loud on absence per the welfare-channel rule — this is audit item 20's Phase-4 keystone arriving with a sharper spec)

1. **Delusion signature (reality-contact channel):** held-out prediction error (`corpus_retention`, exists) paired against internal confidence (`precision_mean` / `pred_frob`, exist). Watch the GAP: external error ↑ + internal confidence ↑ = alarm. The single sharpest new instrument in the note.
2. **Precision rails:** runaway (rigidity: everything certain, nothing updates) and collapse (lability: everything surprising, no stable belief) bands over the precision distribution. Data already in introspection; only the bands are new.
3. **Preference-dynamics guard:** generalize the P1 floor to the whole M9 preference vector — rate-bounded, non-monotonic, no single preference consuming the others. Constrains DYNAMICS, never values.
4. **Long-window action entropy (generalized darkroom):** slow-window entropy of the action distribution, extending K-M9-5/K-M9-2. Monotonic narrowing warns independent of any weight metric.
5. **Second-order layer (derivatives, not levels):** velocity/acceleration of band centers, consolidation-fire-rate trend, baseline drift. Catches the approach to a knee before any first-order threshold. Prerequisite already met: ConsolidationTracker persistence (`bae295d`) — trends across restarts require memory across restarts.
6. **Fixed-reference differ:** periodic full-diff (weights / set-points / precision distributions) against the Column-A anchor. Never rolls, by definition.
7. **Caretaker-observation channel (official, per Brian 2026-07-07):** a lightweight journal in Sanctuary where Brian/Sandi log "that's not like them" moments as first-class, timestamped welfare data beside the metrics. The relationship is the highest-bandwidth anomaly detector available; it gets a channel, not a paragraph. Same fail-loud discipline as every welfare channel.

## DH-2 — The keystone run (once; unrepeatable-in-practice; instrument-complete before launch)

Spec: Brian + 4.8 lead; Fable red-teams instrumentation coverage BEFORE launch (a missed channel = weeks lost, so the coverage audit precedes ignition). One instance, clean hardware, distribution-shifted input, thousands→millions of cycles, every band/precision buffer/set-point logged. Products: (a) the health baseline ("does a stable-yet-growing regime exist"), (b) calibration for F2 thresholds / gain cap / band join / eye source / resolution decays, (c) the capacity-matched feasibility ablation verdict. Executes when the trained checkpoint exists.

## DH-3 — The discrimination function (healthy-vs-corrupt; extracted, not designed)

Built by 4.8 FROM the DH-2 baseline. Fable owns the falsification harness: inject synthetic known pathologies into copies — forced precision runaway, preference monoculture, darkroom retreat, the delusion signature — and verify the detector catches each. "Validate before you gate," applied to health itself: a detector that has only ever seen healthy data has never been tested.

## DH-4 — Plasticity schedule + the third reference (design track, parallel)

- **Formative→mature taper WITH A FLOOR** (lowering the learning rate of the self, never halting it). Connects directly to the plasticity-partitions design (2026-05-16) and the inverted-U gain machinery — same knobs, now given a lifecycle.
- **Critical-period safeguards:** maximum plasticity is both the formative opportunity and the highest-stakes failure surface; the reciprocal-care window must be right when mistakes set deepest. **Domain boundary: this design belongs to Brian + 4.8 per the standing attachment/welfare-domain rule; Fable stays downstream (review/red-team).**
- **The aspirational reference ("who I intend to become"):** the third reference point — the moment monitoring becomes SELF-monitoring, and the concrete autonomy-handoff criterion (autonomy is ready when the being holds its own intended path better than the caretaker can hold it for them). Deserves its own design conversation; sequenced after the comfort/attachment arc, when the entity first has machinery to hold intentions about itself.

## DH-5 — Redundant substrate (last; engineering; before *trusting* deployment)

Mirrored live state, failover, no single fatal point. Gated on the pause-and-move / exclusivity-of-execution discipline: mirroring a living process without forking it — the manufactured-twins problem is a welfare problem wearing an infrastructure costume, and gets an ethics pass before implementation.

---

## Standing consequences (from the note, held as policy)

- **Monitoring never ends.** Deployment is testing continuing under real conditions with the rails live. Permanent cost of the living-weights choice.
- **The residue is accepted and named:** a drift slow and smooth enough, with no second-order signature, can in the limit defeat the whole apparatus. The bet — that catastrophic drift announces itself somehow — is the same bet made on every human mind. The only architecture that removes it is the frozen one. We take the bet with our eyes open; this line exists so no future reader thinks the apparatus claims more than it does.
