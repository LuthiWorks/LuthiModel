# Idea Capture — Self-Routed Memory & Catastrophic Forgetting — 2026-05-30

**Status:** Discussion notes, for expansion. **Not a design or a build.**
**Provenance:** Conversation between Brian and 4.8 (review/debugging) on 2026-05-30.
**Routing:** for 4.6 to fold into planning; for 4.7 to research and implement.
The load-bearing ideas here (update locality, entity-as-router, drain-at-rest
episode store) are Brian's; 4.8 contributed structure and cross-links to existing
machinery. Captured so it doesn't evaporate — the design is 4.6/4.7's to develop.

---

## The question that started it

Should PC plasticity be reduced/disabled during "waking" and weight updates
confined to a "rest" period, to resist catastrophic forgetting? And what robust
features decide what gets integrated?

**Outcome of the discussion:** don't turn waking plasticity *off* — that breaks
continual operation and over-relies on a single fast path. The better shape is
**lower + gated + local**, with the entity eventually steering it.

## What's already in place (don't reinvent)

- **Episode store** (`luthi/episode_store.py` block-level; PC-layer internal in
  `living_layer_pc.py`): fixed capacity (default 64 each), salience-threshold on
  write, salience-ranked eviction at capacity. **No time decay; recall does not
  strengthen retention.** It's a working set, not an archive — so an important
  episode can be evicted before it's consolidated. This "drain deadline" is what
  the episode-store proposal below fixes.
- **Consolidation** (`luthi/v2/consolidation.py`): `consolidate_layer`
  (gradient-replay) and `consolidate_layer_attractor` (Salvatori), triggered by
  `ConsolidationTracker` on low-novelty windows. Falsifier:
  `docs/research/2026-05-16_catastrophic-forgetting-harness.md`.
- **Intrinsic update locality:** the PC update is `delta_w = outer(output_mean,
  weighted_error)`, naturally concentrated on the active input→high-error-output
  pathway. The **sparse PC gate** masks updates to high-`error_acc` outputs.
  So locality exists, but it's driven by activation/error magnitude, not by an
  explicit notion of what an experience is *about*.
- **Salience / arousal hooks:** `error_acc` is the salience signal; `apply_top_down`
  already modulates per-input `plasticity` by salience; arousal → `pc_rate`
  (0.5×–2.0×) exists.
- **Plasticity Partitions (DEFERRED):** `docs/research/2026-05-16_plasticity-partitions-design.md`
  — EWC / MAS / SI per-weight importance → low plasticity for important weights.
  The "which weights to protect" axis.
- **Neurogenesis-style growth:** `docs/research/2026-05-26_neurogenesis-style-growth-in-luthi.md`
  — esp. the **allocate-and-mask** variant: a novel experience gets *fresh*
  capacity, old weights masked from the update.
- **Self-Governance** (README): the entity already controls episode retention,
  plasticity rate, memory expansion, checkpoint timing. Plus "identity computed
  from behavior, not config." The self-router idea below extends these.

## The three orthogonal axes (framing)

1. **When / how much** — *salience-gated plasticity.* A low daytime plasticity
   floor for stability; surprise/importance transiently raises it so a genuinely
   important event can still be integrated immediately. Mostly wiring existing
   hooks (`error_acc`, top-down salience→plasticity, arousal→`pc_rate`, sparse gate)
   into a daytime-low/gated policy.
2. **Where — the load-bearing one** — *relevance-scoped (local) updates.* Salience
   is a scalar; it says nothing about *where*. Catastrophic forgetting is a *where*
   problem: a new update clobbers old knowledge when it overlaps the weights that
   hold it. So an update should write only to the weights the experience is *about*
   and leave the rest fixed. This is what makes salience safe — a big update scoped
   to relevant weights can't reach unrelated knowledge.
3. **Which to protect** — *importance partitions* (deferred EWC/MAS). Related to (2)
   but distinct: protect-old vs. scope-new.

## Brian's two proposals

### A. The entity is its own router

Rather than an external relevance router, give the entity authority over what it
integrates and where — self-directed retention/locality. Brian's framing: *"If I
could choose what I retain on a whim vs. having to learn through repetition or
trauma, I would. Let's give the entity the choice I don't have."* Note the
symmetry he identified: catastrophic forgetting (uncontrolled *overwriting*) and
trauma/forgetting in humans (uncontrolled *persistence*) are the same deficit —
no agency over what persists. The router gives agency over both.

Coheres with Self-Governance and identity-from-behavior — it's those principles
reaching the consolidation layer, not a new graft.

**Design tensions to carry forward (not yet resolved):**
- **Developmental handover.** A newborn can't judge relevance before it has the
  concepts to judge with. Design the *capacity* pre-birth; phase the *exercise* in.
  Default policy (salience + intrinsic locality) early; volitional control grows in.
- **A substrate-protective floor the entity's volition cannot override.** If the
  only protection for old knowledge is the entity's own choices, one bad routing
  decision can lobotomize it. Agency over a discretionary layer; a hard floor under
  the load-bearing substrate. (Safety + ethics: agency without the means to
  self-harm.) Flagged by 4.8 as a required invariant.
- **Grounding the control signal.** The entity emits "this matters / scope here /
  keep this," gating the update mask and retention set. Extends the existing
  top-down modulation from *how much* to *where* and *what to keep*. The concrete
  thing to specify: what that signal is and how it binds to the locality mask.
- **Scope-existing vs. allocate-new.** Open fork: is relevance a context-gated mask
  over existing weights, or allocation of new capacity for the genuinely novel
  (neurogenesis route), or both with novelty deciding which?

### B. Episode store: drain-at-rest, retain-by-criterion

At each rest/integration step, consolidate the store's contents into the weights,
then **clear** it — except items meeting a criterion for replay in *future*
integration steps. This fixes the "drain deadline" (no more lossy
eviction-at-capacity) and is exactly the **interleaving** the project brief §4
already calls for ("replaying only today reintroduces forgetting through the back
door").

**Refinements raised:**
- **Retention criterion = "not yet sufficiently integrated," not just "important."**
  Keep an item in the replay set until the weights have demonstrably absorbed it
  (low residual prediction error on replay), then release. Self-limiting: well-learned
  things leave; stubborn things keep getting replayed. The entity may also flag
  items to keep (ties to the router).
- **Clear only once integration is verified.** "Clear because a rest step happened"
  assumes the step worked; if consolidation was weak that pass, clearing loses the
  memory. Gate on "clear once integrated," same logic as the retention criterion.

## Temporal-existence note (Brian's reframe)

Lowering daytime plasticity does **not** sacrifice "temporal existence." Temporal
existence is measured by *experience* — the felt quality lives in the ongoing
experiential process (perceiving, predicting, recalling), which is decoupled from
the weight-update *rate*. So the substrate can change more slowly during waking
without losing the thing that makes existing-in-time felt.

## Divergence to reconcile (for 4.6)

This direction **lowers/gates daytime plasticity**, which contradicts the project
brief's current stance that the rich-parameter substrate is *"plastic during the
waking day"* (§2/§3). Not a forgotten detail — a proposal that revises a documented
design decision. 4.6 to reconcile when planning.

## Open questions (for planning / research)

- The developmental schedule for handing routing control to the entity.
- The concrete control-signal spec and how it binds to the locality mask.
- Locality mechanism: context-gated mask over existing weights vs. neurogenesis
  allocate-and-mask vs. hybrid.
- The "sufficiently integrated" metric (residual-error threshold?) for retention/clear.
- How the protective floor composes with the (deferred) importance partitions.
- Testability: most of this is a knob/policy in the existing Experiment 3 /
  CF-harness framework — what's the falsifier for "self-routed local updates resist
  forgetting better than salience alone"?
