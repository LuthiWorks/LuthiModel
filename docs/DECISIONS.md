# Operational decisions

Brian's rulings about **what to run, in what order, and what to feed the
model**. These are project decisions, not scientific findings: they do not
belong in the falsification registry, which records questions, predictions,
data, and verdicts. Registered *obligations* (things that gate what may be
claimed) stay in the registry and are referenced here, never redefined.

Newest last.

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

---

> **RULING (2026-07-27, Brian): v6 starts at 8 blocks.**
>
> v6 (the dormant-machinery bundle) will run at **d8** rather than the current
> d4 — "that is where we are going to start with this." Supersedes the
> 2026-07-26 sequence item that placed v6 at current scale (512d, d4).
>
> Consequences recorded, none of them blocking the ruling:
>
> 1. **The store fix comes first.** v6 includes attractor consolidation, which
>    reads the episode store — and that store is a frozen fossil in every
>    completed run (see `docs/research/2026-07-27_episode-store-frozen-defect.md`).
>    Running v6 against it means the new machinery replays initialization-era
>    snapshots for the whole family. Fix, verify against the note's four
>    predictions, then start v6.
> 2. **The dead control that gates depth claims becomes dead-d8.** The standing
>    obligation is for a dead arm matched to the family making the claim. A
>    dead-d4 control cannot license depth claims about a d8 family. Open
>    question for the registry: whether the d4 families (v3/v4/v5) still owe
>    their own dead-d4 control, or whether depth claims about them are simply
>    retired unmade.
> 3. **Attribution.** v6-at-d8 bundles machinery with a depth change; per ladder
>    precedent that trade is acceptable if recorded, with single-lever
>    follow-ups available if the bundle moves the picture.
> 4. **Width is not yet ruled.** 512d x d8 or 768d x d8. Recommendation:
>    **512d x d8**, keeping width for the later step so the scale-up remains a
>    single clean variable — and roughly halving family cost (~16h/seed vs
>    ~35h/seed, i.e. ~3.5 days vs ~7+ days for n=5).
> 5. **Hardware.** A d8 family at either width is a substantially longer
>    sustained GPU load than anything run so far, on the 15-year-old PSU
>    already suspected in the 860's SATA failure.

## 2026-08-07 (late): the depth-8 pivot rule (Brian; corrected per his ruling)

**If** the VBG family (stages 45-46, running overnight) and its immediate
follow-ups — the width-ratio rung included — fail to produce a depth-8
recipe reliable at 2-of-3 seeds, **then** the project refocuses on
**LLM-JEPA integration at DEPTH 8 with muPC OFF** (arXiv 2509.14252:
generative NTP head + view-pair JEPA term, lambda-balanced, tied-weights
[PRED] predictor). muPC turns back on only once LLM-JEPA is concluded to
work with our project. (An earlier draft of this entry said "retreat to
depth 4" — that was Fable's sharpening, overruled by Brian: the pivot
stays at depth 8.)

Mechanistic rationale FOR the ruling (Fable, after the correction): the
NTP term is itself an anti-collapse pressure this substrate has never
had at depth — cross-entropy over 32k classes cannot be satisfied by a
rank-2 representation, whereas the pure-JEPA objective (our entire
collapse record) can be trivially satisfied by degeneracy. The muPC-off
d8 cell's 1-for-3 reliability under pure JEPA may not predict its
behavior under the combined objective at all.

Gate for "LLM-JEPA works with our project": generation quality gains at
d8/muPC-off with substrate-health metrics (rank bands, native-voice std,
offset) healthy and RELIABLE (n>=3 seeds) — registered properly when
that track opens. muPC and the remaining depth machinery re-enter
afterward, carrying warmup + governor learnings.

## 2026-08-08: "concluded working" defined for the muPC re-entry rule (Brian's request)

The 2026-08-07 pivot rule says muPC returns only after LLM-JEPA is
"concluded to work with our project." Registered definition so the rule
is checkable:

- **Provisionally working:** an LLM-JEPA family gates at 2-of-3 seeds
  (the standing four-gate criterion) at depth 8.
- **Concluded working:** that result replicates at the ruled 768x8
  scale target (Brian, 2026-07-26: "scale moves go UP").

Only after both does the muPC re-entry probe run (one variable on the
working recipe, three seeds, same gates), with the parked scheduled-muPC
design (2026-08-07) as the registered fallback if always-on muPC breaks
the recovery. Note added 2026-08-08 while the stage-51 runway family
runs: early kills at marginal NMSE with strong capability (seed 46:
killed at nmse 2.41 carrying ppl 500 / lift 4.23x) raise a registered
suspicion that the divergence guard's 2.0 bound — calibrated on the
pure-JEPA objective — may need recalibration for the combined objective
before any "working" verdict is trustworthy in either direction.

---

> **DECISION (2026-08-14, Opus 5 â€” NOT Brian's ruling; recorded for his
> override and Fable's review): VISReg's shape term will be normalized by
> N, with lambda re-derived, at the next family's registration.**
>
> Finding (audit B1, `docs/audits/2026-08-13_luthimodel-audit.md`): VISReg
> ran at **98.6-99.99% of the objective by loss share for the whole 768x8
> family**, and `l_pred` never exceeded 1.4%. Cause: Eq. 5 sums over N
> while L_scale means over D, so at N = 32x128 the regularizer is four to
> six orders of magnitude larger than the predictive term, and the convex
> Eq. 9 mix at the paper's lambda = 0.6 then buries l_pred. Inside the
> regularizer the nominal 1.0/1.0/1.0 weighting resolved to ~95% shape,
> ~5% scale, **~0.55% center** â€” and center is the anti-offset term while
> the observed disease was a soloist.
>
> The implementation is faithful to the paper. What was never checked is
> what the paper's lambda means at our N. Same failure class as the
> 2026-08-08 NTP dose catastrophe, on the other side of the ledger, and it
> went uncaught because nobody computed the share.
>
> **Decision:** `VISReg(shape_normalize=True)` for the next family, which
> makes lambda a scale-free mixing weight and removes the batch-size dose
> distortion the 2026-08-11 smoke measured directly (l_shape 1,461,016 at
> batch 32 vs 693,472 at batch 16 â€” pure N-scaling). Shipped **opt-in,
> default False**, per this ladder's standing discipline that no completed
> family's configuration may silently change meaning.
>
> **Binding condition:** turning it on WITHOUT re-deriving lambda is not a
> correction, it is a different dose. The next registration must record
> the measured `0.6 * l_reg / (0.4 * l_pred + 0.6 * l_reg)` ratio at step 0
> and at intervals, and freeze an intended ratio before data. A dose that
> is not measured is not registered.
>
> Not applied to any running or completed arm. The 512 full-length control
> launched 2026-08-14 deliberately runs the OLD dose, because its whole
> purpose is to be configuration-matched to the 512 family.

---

> **DECISION (2026-08-14, Opus 5 â€” NOT Brian's ruling; recorded for his
> override and Fable's review): `adaptive_episodes` ON for the next
> family.**
>
> Finding (audit B4): the 768x8 family â€” the ruled-scale family â€” ran with
> `adaptive_episodes=False`, i.e. the pre-2026-07-27 episode-admission
> defect that fix exists to correct. Measured on seed 97 over all 54,000
> steps: **blocks 0-4 stored zero episodes and fired zero recalls**; only
> b5/b6/b7 had any activity (200/144/99 writes). CLAUDE.md records the
> same shape pre-fix ("three of four blocks storing nothing at all").
>
> This compounds with audit A9: `consolidation_fires` counted triggers,
> not consolidations, so those five blocks each logged ~1,000 consolidation
> fires having replayed nothing. Two-tier memory â€” the v2 architecture's
> distinguishing feature â€” was inert in five of eight blocks for the
> entire ruled-scale run, and every counter read healthy.
>
> **Decision:** the next family sets `adaptive_episodes=True` (plus the
> admission-v2 surprise/refractory parameters as registered on
> 2026-07-28), and its verdict is not scored until
> `consolidation_noop_fires` is confirmed materially below
> `consolidation_fires` in every block. The A9 counters make that
> checkable for the first time.
>
> **Alternative Brian may prefer:** register that episode memory is
> deliberately inert at this stage and stop shipping a config that claims
> a mechanism it does not deliver. Either is defensible; what is not
> defensible is the current state, where the flag exists, defaults off,
> and nothing reports that the store is empty.

---

> **RULING (2026-08-15, Brian): 512d is a mechanism smoke tier only. No
> more 512 evidence runs.**
>
> "I don't think 512 has anything left to teach us about where we're
> going from here. Maybe just basic testing of unimplemented mechanisms
> on runs that don't take quite as long, but once a mechanism is
> established as functional, 512 doesn't tell us anything because what
> works at 512 might not work at 768, as we have already demonstrated."
>
> **The permitted use, and its exact limit.** A 512 run may answer *"does
> this mechanism run at all"* — does it execute, stay numerically stable,
> emit its instruments, not explode. It may **never** answer *"does this
> mechanism work."* "Established as functional" means the former and only
> the former. Blurring the two is the failure this ruling exists to
> prevent, and the blur is easy: a 512 family that completes 3-of-3 looks
> exactly like evidence.
>
> **The demonstration Brian refers to, stated precisely.** The VISReg
> family CONFIRMED 2-of-3 HEALTHY at 512d (2026-08-11) with the scored
> reading *"VISReg did not enable a rescue; it abolished the fall."* The
> same configuration at the ruled 768x8 scale went **0-of-3**, with a
> soloist reaching `top_dir_share` 0.919. 512 did not merely fail to
> replicate — it returned a confidently positive result about a
> configuration that fails at the scale we build at.
>
> **Supporting analysis (Opus 5, recorded because it gives the ruling a
> mechanism and a direction):** the 2026-08-15 full-length control shows
> the phenomenon *does* exist at 512 — it is roughly 5x weaker and 2x
> slower. Seed 97 reached `top_dir_share` 0.198 at step 24,000 and never
> sustained a crossing, where 768 crossed at 13,100 and ran to 0.919; the
> other two 512 seeds sat at 0.039 and 0.059.
>
> So the error is not noise, it is **biased in a known direction: 512
> systematically under-shows collapse-class phenomena, and therefore
> reads optimistic.** A remedy tested at 512 is being asked to prevent
> something that barely happens there — a clean null is the expected
> result whether the remedy works or not. That is why the ruling is a
> prohibition rather than a preference: the failure mode is not "we learn
> less," it is "we learn something false and feel good about it."
>
> **Consequences.**
> - Audit item B1 (the VISReg dose fix) and every other remedy test go to
>   768 or they do not count.
> - `docs/research/2026-08-14_visreg-runlength-control.md` is the **last
>   512 evidence run**. It discharged its obligation; nothing further is
>   owed at that scale.
> - The `probe_d8_visreg_long` arm and stage 56 stay in the driver as
>   history, not as a template to copy.
> - The two 768 death modes are now addressed — the uncorroborated kill-6
>   that killed seed 46 while healthy (audit B5, fixed) and the unclipped
>   12.4M-gradient step that killed seed 95 (audit B6, sized at 5.0e5,
>   awaiting application at the next registration). Raising 768 completion
>   is what makes a 768-only policy affordable at ~32 h/family.
