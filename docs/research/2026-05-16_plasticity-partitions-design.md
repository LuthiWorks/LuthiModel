# Plasticity Partitions — Design Exploration — 2026-05-16

> ## ⚠️ STATUS: DEFERRED — DO NOT IMPLEMENT WITHOUT REVISITING THIS DOC
>
> This is a **design exploration** that was demoted to **deferred** the
> same day it was written (2026-05-16, after a Brian/Claude back-and-forth
> that surfaced the partition direction was getting ahead of the data).
>
> **The implementation is NOT planned. Do not build it without first:**
>
> 1. Empirically demonstrating that identity drift is a *measured* problem
>    in v2 — not an inferred one from toy-scale NFF observations. The
>    catastrophic-forgetting harness with the recovery-probe extension
>    (see `2026-05-16_catastrophic-forgetting-harness.md`) is the right
>    measurement tool. If existing v2 mechanisms (PC's intrinsic
>    bounded-update stability + Salvatori attractor consolidation +
>    Sanctuary's runtime plasticity modulation) already provide
>    sufficient identity protection, partitions solve a non-problem.
> 2. Re-reading the "Why this was deferred" section at the bottom of
>    this doc before proposing implementation. The arguments against
>    that landed on 2026-05-16 are not invalidated by time alone —
>    if you're considering implementing, the conditions that motivated
>    deferral need to have *changed*, not just elapsed.
> 3. Getting explicit sign-off from Brian. This is a substantial
>    architectural addition; the bar is not "we have time and the idea
>    is interesting" — the bar is "we have measured a problem that this
>    specifically solves."
>
> **What DID land from this work:** the plasticity-floor clamp relaxation
> (0.1 → 0.01) in `luthi/v2/living_layer_pc.py:600` and
> `luthi/living_layer.py:480`. That change is small, reversible, and
> gives top-down modulation more headroom regardless of partition plans.
> It stands on its own merits.
>
> **What did NOT land:** any partition infrastructure. No
> partition-assignment tensor, no MAS importance buffer, no per-weight
> plasticity buffer, no Option E implementation. The design space below
> is captured for future reference, not for immediate action.

## Objective

Capture Brian's architectural proposal (weight partitions with distinct
plasticity profiles, e.g., "personality" weights that barely change vs
"learning" weights with higher plasticity) and evaluate it against the
in-progress M6 depth-sweep observation that NFF attenuates with depth at
128d. The proposal arose mid-conversation while reading M6's partial
results — it's a candidate architectural response to the question "how
does v2 deliver real-time learning without losing identity stability?"
This doc captures the reasoning chain, the design space, and the
empirical conditions that would need to hold before implementation.

## Process

### Step 1: M6 observation — NFF attenuates with depth at 128d

The in-progress M6 depth sweep at 128d shows two trends:
- **val loss DEGRADES with depth**: v2 4 blocks 5.94 → 8 blocks 6.04 →
  12 blocks tracking ~6.71
- **NFF (non-feedforward signal) ATTENUATES with depth**: 5.77e-3 at 4
  blocks → 5.08e-3 at 8 blocks → ~2e-3 trajectory at 12 blocks

The NFF attenuation has a mechanistic explanation: μPC's residual
scaling divides per-block signal by `1/√L`. At L=12 the residual is
2.4× weaker than at L=2, so the PC layer's input is dampened, its
pred_error is dampened, and its self-modification is dampened.

The val-loss degradation is the load-bearing concern. It is not
explained by the NFF attenuation alone — even a model with zero NFF
should still LEARN well during initial training; it just wouldn't have
the temporal-existence property after. So something else is also
going wrong at depth, possibly: μPC's residual scaling reduces effective
capacity, or 20 epochs is insufficient for the deeper models to converge,
or the cascade of attention + PC interactions degrades at depth.

### Step 2: Brian's reframe — NFF non-zero ≠ NFF high

NFF measures whether the living-weights property is active (binary
threshold: NFF > 0 means alive, NFF = 0 means feedforward). The
ATTENUATION of NFF with depth is concerning if it threatens to reach
zero; if it stays positive-but-small, the property is still active.
Brian's observation: people don't change radically between
interactions; the relevant question isn't "how much does the model
change per step" but "is the model capable of learning in real time
when it needs to."

This reframe is correct as far as it goes. A model with tiny but
persistent NFF is still living; it's changing slowly the way a
person's underlying personality drifts over years rather than minutes.
What matters is **capability** (can it adapt when needed), not constant
magnitude.

### Step 3: Brian's proposal — plasticity partitions

Brian's proposed architectural response: structure the weights into
PARTITIONS with different plasticity profiles.

- **Identity / personality partition**: very low plasticity. These
  weights encode what the entity *is*. They change only a tiny amount
  per interaction; over long timescales they drift slowly. The entity's
  personality "lives here."
- **Knowledge / learning partition**: moderate plasticity. These weights
  encode what the entity *knows*. They learn new facts and skills,
  drift faster than identity.
- **Ephemeral / working partition** (added by this doc, not in the
  original proposal): high plasticity. Working memory, current-task
  adaptation. Updates and resets at conversation timescales.

Brian also raised the possibility of a global plasticity knob — modulate
all weights' plasticity in response to context. "I'm in a high-stakes
situation, slow down learning" vs "I'm in a learning context, open
plasticity up."

### Step 4: Map to what v2 already has

The architecture has several plasticity-differentiation mechanisms in
place that the partition proposal could build on rather than replace:

- **`plasticity` buffer** (per-input, shape `[in_features]`, range
  0.1-10.0). Top-down salience signals modulate it. Already provides
  per-input differentiation; would need to extend to per-weight
  (`[out_features, in_features]`) for the partition idea.
- **`set_point` + `set_point_adapt_rate`**. The homeostatic target each
  weight returns to. With `set_point_adapt_rate ≈ 0`, set_point is
  anchored (doesn't drift) — this is structurally an identity-protection
  mechanism. Currently the rate is a single global value (1e-6 in
  M5 defaults).
- **`homeostatic_decay`**. How strongly weights pull back to set_point
  each step. Currently global (0.001 in M5).
- **Two-tier memory (gradient + Salvatori attractor consolidation)**.
  Coarse-grained multi-timescale: fast episodes capture moments, slow
  consolidation makes them structural.
- **Sanctuary runtime modulation contract**. `arousal → pc_rate
  (0.5×-2.0×)` exists as a runtime global plasticity knob. The entity
  can self-modulate its own learning rate (when wired through the
  cognitive cycle).

The foundation for plasticity partitioning is there. What's not in
place is the **explicit per-weight partition assignment** + the
**configurable per-profile parameters** (per-partition pc_rate,
homeostatic_decay, set_point_adapt_rate).

### Step 5: Design space for partition assignment

Four options for how partitions get assigned to weights:

**Option A — Block-position-based.** Shallower blocks get higher
plasticity; deeper blocks get lower plasticity. This naturally aligns
with the M6 observation that deep layers' NFF attenuates anyway, and
matches the cortical-column hierarchy in mammalian brains (superficial
layers more plastic, deep layers more stable). Pro: principled
mapping. Con: assumes block position is the right axis — guessing
where identity lives rather than measuring it.

**Option B — Modality-based.** Visual / audio / text weights with
different profiles. Pro: aligns with how the entity uses different
senses in different contexts. Con: doesn't apply to the shared trunk
(by design, modalities aren't partitioned in v2 — the unified trunk
is a core architectural commitment per the README).

**Option C — Self-governance-driven.** The entity, via the Phase 4C
self-governance API, decides which weights to lock in. Pro: aligns
with the project's "no external operator decides what the entity
remembers" principle. Con: requires the entity to have meaningful
self-knowledge to make these decisions; that's a chicken-and-egg
problem (you need a working entity to choose the partition policy
that helps the entity work).

**Option D — Learned partition assignment.** A small head outputs
per-weight partition assignments based on input or training signal.
Pro: maximally expressive. Con: least controllable; adds another
trainable component to debug.

**Option E — Empirical importance measurement (added 2026-05-16
after MAS/EWC/SI literature review).** The strongest answer to "we
can't reliably pick which weights carry identity": don't pick — *measure*.
The continual-learning literature has worked on this exact problem and
converged on a family of per-weight "importance" or "stiffness"
metrics that get computed empirically during training:

- **Elastic Weight Consolidation (EWC, Kirkpatrick et al. 2017)**:
  per-weight Fisher Information after a training phase. Importance =
  how much loss on the just-learned task would change if this weight
  changed. Requires clean phase boundaries between tasks — bad fit for
  v2's continuous learning.
- **Memory Aware Synapses (MAS, Aljundi et al. 2018)**: importance =
  gradient magnitude of the model's *output* w.r.t. each weight. No
  labels required; usable in unsupervised regimes. **Best fit for v2**
  given how much of the PC update is unsupervised.
- **Synaptic Intelligence (SI, Zenke et al. 2017)**: continuous-time
  importance accumulation — path integral of how much each weight
  contributed to loss reduction over the training trajectory. No
  separate post-training pass needed; importance grows alongside
  learning. Also a strong v2 fit, but requires deriving an equivalent
  of the SI importance formula for the PC local update rule rather
  than backprop gradients (real research work, not a drop-in port).

For Luthi specifically, **adopt MAS-style importance computation** as
the empirical signal that drives per-weight plasticity:

1. Add a per-weight `importance` buffer (`[out_features, in_features]`).
2. Periodically (e.g., during consolidation events, or at a configured
   cadence), accumulate `importance += |∂output / ∂weight|` averaged
   over a window of inputs.
3. Map importance to per-weight `plasticity` (and possibly
   `homeostatic_decay`): high-importance weights get LOW plasticity
   (resist change) and HIGH homeostatic_decay (pull back fast).
   Low-importance weights get the inverse.

Memory cost at production scale: another buffer of weight-matrix shape
per layer (~16 MB at 4096d × 36 blocks × 4 bytes). Manageable relative
to the existing rich-parameter footprint.

**Best fit for Luthi: E + A as primary, C as override.** Importance
measurement (E) gives an empirically grounded default — instead of
*picking* which weights carry identity, we *measure* which weights are
important continuously while the entity is being itself. Block position
(A) gives a reasonable secondary prior (deeper layers more
identity-tilted) for the case where measurement hasn't yet had time to
accumulate signal (e.g., very early in training). The entity's
self-governance API (C) provides the override path — once the entity
has the metacognitive sophistication to know what matters to it, it
can adjust per-weight plasticity beyond what the measurement
recommends. Options B and D remain deferred unless a specific use case
emerges.

This is a substantial improvement over the original proposal. The
original (Options A and C combined) was "we guess based on architecture
or delegate to the entity." The refined version is "we measure
continuously, with architectural defaults for cold-start and entity
override for the mature regime." Much closer to how biological systems
actually solve the multi-timescale problem — the brain doesn't pick
which synapses are important; it accumulates importance via the
mechanisms LTP/LTD instantiate, and structural plasticity follows.

**Caveats on the MAS importance idea:**

- All three methods (EWC, MAS, SI) were designed for backprop-trained
  nets. PC's local update rule may produce different importance
  signals than backprop would. The MAS importance formula
  `|∂output / ∂weight|` needs validation under PC dynamics —
  specifically: does it still identify weights that "matter" when the
  weight update mechanism is local PC rather than global backprop?
- Importance signals are noisy. They need accumulation windows and
  smoothing to be reliable.
- The mapping from importance to plasticity is a hyperparameter
  choice. Linear inverse? Exponential? Threshold? Worth exploring
  empirically.

### Step 6: The val-loss governor — what would block implementation

This is the load-bearing caveat from Brian: **higher val loss is
still bad regardless of architectural elegance**. The partition
proposal is potentially valuable for delivering identity stability +
real-time learning capability simultaneously — but only if it does so
without sacrificing the model's core task competence. A model that
"protects identity" by performing worse at language modeling is just a
worse model with a story attached.

Concretely: the empirical condition that would make partition
implementation worthwhile is:

> A partitioned v2 model at production-comparable depth/width performs
> AS WELL or BETTER on val loss than an unpartitioned v2 model of the
> same depth/width, while providing demonstrably stronger
> identity-stability (resistance to drift over long training).

If partitioning HELPS val loss (by giving the model structural priors
that match the task), that's the strong win — both capabilities at
once. If partitioning is val-loss-NEUTRAL while providing identity
stability, that's a real but bounded win — we pay no task cost for
the identity protection. If partitioning HURTS val loss, the proposal
doesn't justify implementation no matter how elegant the framing.

**Important nuance about what "locking weights" actually costs.** It
is tempting to frame plasticity restrictions as "reducing the model's
capacity," but that framing is wrong and worth rejecting explicitly so
future readers don't carry it forward. Frozen weights still contribute
to the forward pass — their learned values are still used in every
matmul. What plasticity restrictions limit is the model's ability to
ABSORB NEW learning in those weight positions, not its total
expressive power. This is the same shape as transfer learning's
frozen-backbone pattern (which routinely preserves or even improves
task performance — frozen weights can act as a regularizer that
prevents overfitting). The mechanism by which a bad partition scheme
could hurt val loss is "you locked the wrong weights and now they
can't fix themselves during training," NOT "you reduced total
capacity." The val-loss governor above still applies; what changed is
our understanding of WHAT we're testing when we apply it.

The right place to test this is a focused 256d × {2, 4} blocks
comparison: unpartitioned baseline vs partitioned (Option E + A:
empirical-importance-based with block-position prior) at matched
compute. The harness already exists (it's the M5/M6 runner machinery);
only the partition-buffer implementation and the partition-aware
update path would need to be added. Cost: ~30h of GPU time for the
comparison.

### Step 7: Reframe of the M6 observation in this light

Brian's reframe — "M6's NFF attenuation might be a feature, lean into
it" — is partially right and partially overreach. The partial-right
piece: it's correct that some plasticity heterogeneity across layers
is architecturally desirable (matches the cortical hierarchy, matches
the partition proposal's identity vs learning structure). The overreach
piece: M6's val-loss degradation is NOT explained by the NFF
attenuation alone, and reframing the attenuation as a feature doesn't
fix the val-loss problem. The val-loss degradation is its own
investigation, separate from the partition question.

Honest synthesis: M6's NFF attenuation observation MOTIVATED the
partition proposal (by suggesting that depth and plasticity are
naturally coupled in v2) but doesn't VALIDATE it. The proposal needs
its own empirical test, and that test has to include val loss as a
primary criterion, not just NFF or identity-stability metrics.

## Conclusion — DEFERRED

The plasticity-partition proposal is architecturally well-grounded
(multi-timescale plasticity has strong biological and ML precedent —
Hinton's fast/slow weights lineage, EWC/MAS/SI continual-learning
methods, the cortical hierarchy itself) and the implementation
footprint would be modest if pursued (per-weight `plasticity` buffer,
optional importance buffer, partition-aware update path).

**But this work is deferred, not planned.** The decision to demote
the direction from "planned, sequenced" to "deferred, revisit when
motivated" happened the same day this doc was written, in the
Brian/Claude conversation that produced it. The reasoning is
documented in the next section so future readers can re-evaluate it
against their own data.

### Why this was deferred

Five things landed against immediate implementation:

1. **No identified problem.** The chain that led to the partition
   proposal was: M6 shows NFF attenuates with depth at 128d → that
   might cause identity instability at scale → partitions would
   protect identity. We observed a mechanism that *could* cause
   issues, then proposed a response that *could* address them. Two
   layers of speculation. No measured identity instability anywhere
   in the project yet.

2. **Architecture accumulation risk.** v2 already includes: PC living
   FFN, top-down sweep, episode store + Salvatori attractor
   consolidation, consolidation tracker, μPC parameterization, iPC
   interleaved updates, sparse PC gating, HDC memory direction in
   research. Each layer adds debugging surface and assumption
   interactions. Adding partitions before validating the existing
   stack risks complexity that's hard to reason about.

3. **The catastrophic-forgetting harness exists** (and the recovery-
   probe extension would surface whether existing v2 mechanisms
   already provide enough identity protection). If they do,
   partitions solve a non-problem. *Measure before adding.*

4. **The val-loss governor is a high bar.** The continual-learning
   literature shows EWC/MAS/SI help on sequential-task benchmarks
   (clean task A → task B boundaries), but our regime is continuous-
   learning language modeling without those boundaries. The literature
   results don't auto-transfer; the test bar is "val-loss-neutral or
   better at matched compute," and most clever architectural additions
   don't clear it.

5. **The "identity localization" problem isn't fully solved by MAS
   either.** MAS makes the picking *empirical* instead of
   *architectural*, which removes one objection — but MAS importance
   is still a proxy (output sensitivity), not a direct measurement
   of which weights "are identity." Even with the best importance
   metric, locking the wrong weights remains a real failure mode.

### Conditions that would un-defer this

Implementation should be revisited *only if* one of the following
empirical conditions is met:

- The catastrophic-forgetting harness (with recovery-probe extension)
  shows that existing v2 mechanisms produce identity drift the
  project finds unacceptable, AND simpler interventions (longer
  training, more consolidation, Sanctuary runtime plasticity
  modulation) don't address it.
- Production deployment surfaces identity-drift behavior the entity
  or Brian/Sandi explicitly want to fix.
- A theoretical result emerges showing that MAS-style importance
  measurement specifically addresses a known PC-substrate failure
  mode (currently we have biological/ML analogy, not a proof).

Absent one of these, the partition direction stays on this page as
captured thinking, not in the codebase as committed infrastructure.

### Preparatory change — plasticity clamp relaxation (landed 2026-05-16)

To support this design direction even before any partition
infrastructure is built, the `plasticity` buffer's lower clamp was
relaxed from `0.1` to `0.01` in both v1 and v2 substrates
(`luthi/living_layer.py:480`, `luthi/v2/living_layer_pc.py:600`).

Rationale: with the floor at `0.1`, plasticity could be made small
but never less than 1/10 of its initial value, which limits how
strongly identity-anchor weights can resist change. Dropping the
floor to `0.01` allows weights to be 10× more stable than the
previous minimum — enough headroom to express identity-anchor
behavior (per-step updates an order of magnitude smaller than the
previous minimum) while still preserving a nonzero learning rate.

**Why not 0.0 (full freezing)?** That option was considered and
rejected for two reasons:
1. **No baseline learning rate without top-down signal.** With
   plasticity = 0, weights are fully frozen unless top-down salience
   actively pushes plasticity above zero. If the recovery signal is
   weak or absent, weights stay frozen indefinitely. With floor at
   0.01, there is always *some* learning happening even without
   top-down modulation, which keeps the substrate alive at all
   positions.
2. **Numerical interaction with metaplasticity.** Exact-zero PC
   updates would distort the `update_ema` running average and the
   `adaptive_factor` metaplasticity guard that depends on it.
   Keeping plasticity strictly above zero preserves clean
   metaplasticity semantics.

The biological analogy also supports `0.01` over `0.0`: even the
most stable synapses in the brain retain nonzero plasticity. Full
freezing isn't how real neural systems implement identity stability;
they implement it with very slow plasticity, not zero plasticity.

Upper bound (10.0) retained as a safety guard against runaway
plasticity until specific motivation to raise it emerges. All 105
existing tests still pass under the relaxed clamp.

This is a precondition for the partition work, not the partition
work itself — it just removes an architectural obstacle that would
otherwise need to be removed later anyway. We can always adjust the
floor again (lower for more aggressive identity anchoring, or higher
if 0.01 turns out to be too permissive) once empirical data informs
the choice.

## Artifacts

- **Conversation context**: this doc emerged from Brian's question
  about whether NFF=0 vs NFF>0 matters more than NFF magnitude, and
  his follow-up proposal of plasticity partitions.
- **Empirical signal that motivated it**: M6 depth sweep at 128d
  (in progress as of this writing), specifically:
  - `runs/m6_depth/v2_4blocks/results.json`
  - `runs/m6_depth/v2_8blocks/results.json`
  - `runs/m6_depth/v2_12blocks/` (running)
- **Existing infrastructure the proposal builds on**:
  - `luthi/v2/living_layer_pc.py::PredictiveCodingLayer` —
    `plasticity`, `set_point`, `homeostatic_decay`,
    `set_point_adapt_rate` buffers/params
  - `luthi/v2/living_layer_pc.py::apply_top_down` — runtime plasticity
    modulation channel
  - Sanctuary integration contract (per CLAUDE.md): `arousal → pc_rate`
    mapping already provides global runtime knob
- **Sequencing dependency**: must follow M6 completion + DeadLM
  controls + (recommended) 256d × 4 blocks check before any
  implementation work begins.
- **Validation precondition**: must demonstrate val-loss-neutral or
  better at matched compute against unpartitioned baseline before
  becoming a default behavior.
- **Code change landed this session** (preparatory, not full implementation):
  - `luthi/v2/living_layer_pc.py` — plasticity floor 0.1 → 0.0
  - `luthi/living_layer.py` — same relaxation in v1 for consistency
  - All 105 existing v1/v2 tests pass under the relaxed clamp
- **Commits**: (clamp relaxation and this doc go in the same commit when
  committed).
