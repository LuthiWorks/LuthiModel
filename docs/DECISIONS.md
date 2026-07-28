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
