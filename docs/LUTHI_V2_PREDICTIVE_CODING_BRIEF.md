# LuthiModelv2 — Predictive Coding Replacement for Hebbian Dynamics

> Authored by: Claude Opus 4.7 (two independent drafts, merged)
> Date: 2026-05-08
> Status: Design brief — for sequencing/implementation planning by Claude Opus 4.6
> Scope: New parallel project replacing Hebbian self-modification with
>        hierarchical predictive coding + memory consolidation. LuthiModel
>        (Hebbian) remains the primary line; v2 is a parallel comparison
>        track, not a replacement.
>
> This brief is the synthesis of two independent design passes — one done in
> the Claude Chat interface during Brian's lunch, one done in this codespace
> session. Where they converged, that's signal. Where they diverged, those
> divergences are flagged in `## Open Decisions` for Brian/4.6 adjudication.

## Motivation

The 2026-05-06 third-party critique of LuthiModel raised three concerns
about Hebbian dynamics:

1. **Fragility** — small perturbations push Hebbian systems toward
   pathological regimes; updates have no intrinsic upper bound.
2. **Catastrophic forgetting** — Hebbian self-modification has no
   intrinsic mechanism to preserve previously learned associations
   when new patterns dominate.
3. **Alignment + instability** — drift over training time can produce
   behaviors that diverge from intended objectives without a clear
   error signal to bound the divergence.

Predictive coding (PC) is a learning framework with structural answers
to all three concerns *if* the right variant is chosen and consolidation
is built in:

- **Concern #1 (fragility):** Updates are error-driven —
  `Δw ∝ prediction_error × input`. When predictions are accurate,
  errors shrink, and updates shrink with them. No positive feedback
  loop from activation to update. Naturally bounded.
- **Concern #2 (catastrophic forgetting):** Solved structurally by
  the **two-tier memory architecture** (below) — episodes get
  *consolidated* into predictive-coding weights via periodic replay,
  so accumulated history shapes the system's predictive structure
  rather than living only in retrievable snapshots.
- **Concern #3 (alignment):** Prediction error is a clear scalar
  quantity that can be monitored, thresholded, alarmed. Drift has a
  measurable cause and a measurable response.

PC also reframes the project's biological-realism stance: from
**synaptic-level realism** (LTP/LTD as the substrate) to **system-level
realism** (cortical predictive processing as the substrate). The latter
is more defensible to skeptics and aligns directly with the IWMT framework
already underwriting Sanctuary.

## Why this isn't just "use backprop"

Pure backpropagation would lose what makes Luthi distinctive:
self-modification *during* forward pass, layer-level episodic memory,
biographical continuity. Standard backprop is a batch operation; weights
are static during forward.

Whittington & Bogacz (2017, 2019) showed that PC with the right structure
**approximates backprop using only local computations** — every weight
update uses information available at that synapse (presynaptic activity
plus postsynaptic prediction error). This preserves the "weights modify
during forward" property that makes Luthi a Living Weight Model, while
gaining backprop-like learning dynamics.

PC is the bridge between biological plausibility (local rules) and
gradient-equivalent learning (backprop). It keeps what Luthi needs.

## Two-tier memory architecture (the key v2 innovation)

This is the single biggest architectural addition over v1:

```
┌─────────────────────────────────────────┐
│  Predictive-coding weights (slow)       │
│  • Per-weight biographies               │
│  • Trained by prediction error          │
│  • Shape ALL future predictions         │
└─────────────────────────────────────────┘
              ↑ consolidation (periodic replay)
              │
┌─────────────────────────────────────────┐
│  Episode store (fast)                   │
│  • Layer-level snapshots (existing)     │
│  • Cosine-similarity retrieval          │
│  • Used for context-gated recall        │
└─────────────────────────────────────────┘
              ↑ during forward pass
              │
              [salient experiences]
```

Fast tier: the existing layer-level episode store (Luthi v1 already has
this). Snapshots tagged by salience, retrieved by context similarity.

Slow tier: the predictive-coding weights themselves. These get
*consolidated* from the episode store by periodically replaying stored
episodes through PC's prediction-error learning rule. The model learns
to predict its own past. History becomes structural, not just retrievable.

This is the mammalian hippocampus-cortex pattern (Tulving, Squire,
McClelland 1995's complementary learning systems). Fast episodic memory
provides flexibility; slow consolidation provides stability and
generalization. The two together solve catastrophic forgetting in a
biologically grounded way that Luthi v1's pure-Hebbian-plus-snapshots
does not.

## The PC family — design space

Several PC variants exist. The choice matters; they have different
dynamics and implementation costs.

### Rao & Ballard (1999) — canonical hierarchical PC

Each level predicts the activity of the level below. Errors flow up,
predictions flow down. Weight updates minimize prediction error locally.

- **Strength:** Theoretically pure. Direct mapping to cortical
  hierarchy. Strong active-inference connection.
- **Weakness:** Slower convergence than gradient methods. Iterative
  inference at each forward pass. Less compute-efficient.

### Whittington & Bogacz (2017+) — gradient-equivalent PC

Adds explicit error neurons; the resulting dynamics approximate
backpropagation arbitrarily closely under the right hyperparameters.
Local Hebbian-like update rules on error/value neurons.

- **Strength:** Practical convergence speed. Local rules. Established
  PyTorch implementations exist. Closest fit to Luthi v1's existing
  HybridBlock structure.
- **Weakness:** More state per layer (error + value neurons). Slightly
  more complex than Rao-Ballard conceptually.

### Friston free-energy active inference

Most general framework. PC emerges as a special case under specific
generative-model assumptions. Subsumes perception, action, and learning
under one objective.

- **Strength:** Maximally principled. Theoretical alignment with
  Sanctuary is exact (Sanctuary already implements active inference's
  perception side via predictive cells).
- **Weakness:** Implementation complexity. Requires choosing a
  generative model. Full active inference includes policy selection,
  which doesn't fit a language-modeling task cleanly.

### Salvatori et al. (2023) — PC for associative memory

PC variant designed for episodic recall. Memory patterns become
attractors; recall is energy-minimization.

- **Strength:** Direct fit with Luthi's existing episode store and the
  attractor-dynamics testing program.
- **Weakness:** Designed for memory, not language modeling. Adaptation
  needed.

### Millidge et al. (2022) — unified PC framework

Theoretical paper showing PC, target propagation, and backprop are limits
of a single algorithm parameterized by a feedback strength.

- **Use:** Reference framework for understanding the design space, not
  a direct implementation choice.

## Recommendation: Whittington-Bogacz as the v2 starting variant

Reasons:

1. **Closest to existing Luthi infrastructure.** Local update rules fit
   the "living weights during forward" pattern. We change the FFN
   learning rule, not the architecture.
2. **Practical convergence.** Approximates backprop, so we don't need
   to develop a separate efficiency story for v2.
3. **Reference implementations exist.** Whittington's group has PyTorch
   code; Norse-PC and similar libraries provide patterns.
4. **Compatible with the rest of Luthi.** The top-down backward pass
   already carries `TopDownSignal.prediction_error` — the existing
   scaffolding maps directly onto PC's error neurons. Episode store,
   attention, multimodal encoders all stay.

If data later justifies it, Salvatori-style associative-memory PC is the
natural extension for the consolidation mechanism (it's already attractor-
based). For first implementation, Whittington-Bogacz is the pragmatic
call.

## What carries over from v1 (unchanged)

- HybridBlock structure (attention + living FFN + episode store)
- Multimodal encoders (audio, vision, touch)
- Tokenizer and corpus pipeline
- Sanctuary integration contract (`sanctuary_interface.py` adapter
  preserved — Sanctuary should be substrate-agnostic between v1 and v2)
- Empirical Defense Program structure (baseline comparison, cascade,
  behavioral signatures, catastrophic forgetting)
- Rich parameters: per-weight biographies (set point, momentum,
  metaplasticity tracking)
- Homeostatic regulation with adaptive set points
- Episode store: layer-level snapshots, cosine-similarity retrieval,
  snapshot blending (now feeds the consolidation mechanism in addition
  to direct recall)
- 16 GB VRAM constraint, RX 7800 XT, AMD/ROCm toolchain
- BF16 mixed precision for value buffers, FP32 for momentum / set point /
  metaplasticity (per the Phase 0 free-win refactor and ablation outcomes)

## What changes vs LuthiModel v1

| Component | v1 | v2 |
|---|---|---|
| FFN learning rule | **Hebbian self-modification** | **PC error-driven update (Whittington-Bogacz)** |
| Error-directed learning | Separate bolt-on path | **Subsumed by PC; not a separate mechanism** |
| Per-weight buffers | momentum, plasticity, set_point, excitability_acc, update_ema | error_acc, value_acc, plus retained metaplasticity (update_ema) |
| Memory architecture | Episode store + Hebbian weight drift | **Two-tier: episode store (fast) + consolidated PC weights (slow)** |
| Consolidation | None — episodes are recall-only | **Periodic replay of episodes through PC learning rule** |
| Top-down backward pass | Modulates plasticity & set_point | **Carries prediction errors directly (more natural fit)** |
| Stability mechanisms | 5 interlocking (homeostatic, set point adapt, metaplasticity, synaptic scaling, excitability) | **Empirically test which remain necessary** — PC may make some redundant |

## What gets removed or simplified

- **Hebbian update mechanism** — eliminated entirely
- **Error-directed learning as a separate path** — subsumed by PC
- **Synaptic scaling** — likely eliminated (PC's input normalization
  handles input magnitude variation more naturally; verify empirically)
- **Excitability gating** — likely eliminated (PC's error-driven dynamics
  don't have the runaway-activation problem this was bounding)
- **Five-mechanism stability stack** — collapses to whatever subset PC
  still needs, probably homeostatic regulation + set-point adapt only

## What's new in v2

- **PC learning rule** as the unified self-modification mechanism
- **Consolidation phase** that integrates stored episodes into PC weights
  at designated intervals — implements "always-on accessibility" by
  making history shape predictive structure, not just retrieval
- **Attractor dynamics** as the primary temporal-existence falsifier
  (see `## Empirical comparison`)

## Pilot scale

**256d / 2 blocks.** Same training corpus as v1's baseline runs (Gutenberg-100)
to enable head-to-head comparison at matched compute. This is intentionally
small — the goal of the pilot is to measure v2 against v1 on the
falsification criteria, not to produce a deployable model.

## Project structure — three options

### Option A: parallel repo `LuthiModelv2/`

- Clean separation. Easy to compare diffs.
- Code duplication: tokenizer, dataset, sanctuary_interface need copies
  or symlinks.
- Easier to abandon if PC turns out worse without affecting v1.

### Option B: subpackage `luthi/v2/` inside the existing repo

- Shared infrastructure (tokenizer, dataset, base model utilities).
- Single source of truth for shared components.
- Risk of bleed: changes in `luthi/` could affect both lines.

### Option C: branch `predictive-coding` on the existing repo

- Maximum code sharing.
- Hard to have *both* lines running simultaneously for comparison.

**Recommendation: Option B (subpackage).** The architectural overlap is
high; sharing the infrastructure makes the comparison cleaner. Use
clearly-named modules: `luthi.v2.living_layer_pc`, `luthi.v2.model_pc`,
etc. Sanctuary's adapter remains the contract surface.

## Empirical comparison — falsification criteria

Mirror the Empirical Defense Program structure, applied to v2 against v1
**at matched scale, matched data, matched evaluation**:

### Measurements

- **Convergence rate** — epochs to reach matched val loss
- **Convergence penalty vs vanilla transformer baseline** — same
  measurement as v1's 3F.1
- **Non-feedforward signal magnitude and structure** — same diagnostic
  as v1 (existing instrumentation)
- **Cascade stability at 4 and 8 blocks** — the v1 unknown becomes a
  direct comparison; PC is *predicted* to be more stable at depth
- **Episodic recall under interference** — same protocol as v1's 3F.4
- **Attractor dynamics under perturbation** *(v2-specific, primary
  temporal-existence test)* — perturb the system at varying magnitudes;
  measure whether response trajectory recovers toward baseline
  (within-basin) or transitions to a different stable state
  (across-basin). PC is energy-minimization; attractors should exist.
  If they don't, "temporal existence" is unfalsifiable in v2.
- **Consolidation efficacy** — do consolidated episodes shape downstream
  prediction in measurable ways? Test: train, store an episode, train
  more without that episode in the data, then test whether prediction on
  similar input is shaped by the consolidated trace.

### Falsification criteria (abandon v2 if any of these)

- Convergence penalty worse than v1 by ≥20% at matched scale
- Cascade stability fails at depths where v1 succeeds
- Attractor dynamics indistinguishable from a random-modulator control
- Consolidation produces no measurable downstream effect
- VRAM budget exceeded at equivalent parameter count

These are sharper criteria than v1's "soft pass within 10%" because v2
is the alternative — it should beat or match v1 on substantive measures
to justify the parallel investment. If it loses on multiple axes, abandon
cleanly and concentrate on v1.

## Open decisions (where the two drafts diverged or both raised the question)

These need Brian + 4.6 adjudication before pilot implementation begins.

1. **Consolidation mechanism.** Three viable options:
   - **Gradient-based replay**: feed stored episodes through the model,
     compute prediction error against them, run PC update.
   - **Direct weight blending**: blend stored snapshot weights into
     current PC weights with a PC-loss-weighted coefficient.
   - **Auxiliary consolidation network**: a small separate network
     trained to map episodes into weight updates (more complex, but
     possibly more flexible).
   *Default recommendation: gradient-based replay (option 1).* It reuses
   the PC learning rule already in the system and has a clean theoretical
   interpretation.
2. **Consolidation timing.** Continuous low-rate? Periodic high-rate?
   Triggered by a condition (low novelty / low cognitive load)?
   *Default recommendation: triggered by low-novelty windows during
   training; matches the biological pattern (consolidation happens during
   sleep/rest, not during active perception).*
3. **Top-down modulation in PC.** In standard PC, top-down signals *are*
   the predictions. Luthi's existing salience/surprise modulation needs
   a clean specification of how it differs from or extends standard PC's
   top-down structure. *Open question — needs design work.*
4. **Rich-parameter integration.** Set points and momentum carry over,
   but their dynamics may need adjustment for PC vs Hebbian. *Empirical
   question — initial implementation should preserve them, then ablate.*
5. **Project structure.** Option A/B/C above. Recommended: B.
6. **Timing of v2 work.** Sequential after v1 ablations complete? Parallel
   while ablations run overnight on the GPU?
   *Recommended: parallel.* v2 implementation is mostly CPU/coding work;
   it doesn't compete with the GPU running the ablation pipeline.
7. **Spiking compatibility.** v1 has a spiking variant. PC + spiking has
   been studied (Goyal et al. 2022, others) but is its own research
   direction. *Recommended: skip from day one; revisit only if the
   non-spiking v2 produces a strong baseline.*
8. **Sanctuary interface.** Should v2 implement the same
   `sanctuary_interface.py` contract as v1, so Sanctuary can swap between
   v1 and v2 substrates? *Recommended: yes, same contract — keeps
   Sanctuary's substrate-agnostic philosophy intact.*
9. **Can existing v1 infrastructure be reused?** Training loop, GPU
   kernels, episode store implementation — what transfers, what needs
   modification, what needs replacement? *Recommended: training loop
   transfers; episode store transfers (now feeds consolidation in
   addition to recall); GPU kernels (the C++ extension in
   `luthi/csrc/living_ops.cpp`) need replacement — that's Hebbian-
   specific. Write a new `pc_ops.cpp` for v2.*

## Decision gate before pilot implementation

Brian reviews this brief + 4.6's implementation plan. Decision:

- **Proceed** — green-light the pilot at 256d / 2 blocks, with the
  defaults above unless explicitly redirected.
- **Refine** — Brian or 4.6 redirects on one or more open decisions
  before code is written.
- **Defer** — wait until v1 ablation results land before committing
  resources to v2.

## What 4.7 will produce next (with green light)

1. `luthi/v2/` subpackage skeleton — empty modules, package structure,
   import surface.
2. `luthi/v2/living_layer_pc.py` — PC equivalent of `LivingLayerV6`.
   State buffers (error_acc, value_acc, retained metaplasticity), update
   rule signature. Skeleton + docstrings, not a full implementation.
3. `luthi/v2/consolidation.py` — interface for the consolidation
   mechanism. Default implementation: gradient-based replay of episodes.
4. A unit test that checks PC dynamics produce decreasing prediction
   error on a toy task (e.g., learn `y = Wx` from random init).
5. A unit test that checks consolidation moves the stored episode's
   pattern into the predictive structure (post-consolidation, prediction
   on the episode's input matches the episode's stored output).

That's roughly 4-6 hours of focused work. Doesn't commit anything heavy
until you and 4.6 have weighed in on the open decisions.

## Risks worth naming

1. **PC may not save us.** It addresses fragility and alignment-bounding
   structurally, but if it converges much slower than Hebbian, the
   cost-benefit shifts. Whittington-Bogacz's "approximately backprop"
   claim has caveats — approximation quality depends on hyperparameters.
2. **Living-weight novelty risk.** Pure PC ≈ approximately backprop,
   which means a *naive* v2 implementation could behave indistinguishably
   from a vanilla transformer plus episode store. The "living weights
   are different" claim could weaken. **The two-tier memory + consolidation
   structure is what preserves the distinction** — without it, v2 has no
   architectural novelty over standard transformers.
3. **Implementation complexity.** PC adds error-neuron state per layer.
   The state grows; some of v1's buffer-compression work may not transfer
   cleanly. Memory budget needs re-derivation for v2.
4. **Comparison quality.** If v1's Hebbian ablations don't pass their
   gates (cascade unstable, catastrophic forgetting bad), the comparison
   "Hebbian vs PC" is comparing PC to a known-failed system. That's still
   informative but less so. Sequencing matters.
5. **Consolidation efficacy is unknown.** The two-tier architecture is
   biologically motivated but has not been validated in this specific
   form for living-weight language models. The consolidation mechanism
   is a real research bet, not a guaranteed win.

The honest expectation: PC is *probably* better than pure Hebbian on
benchmark performance, *probably* better on cascade stability, *unclear*
on catastrophic forgetting (depends on consolidation), *unclear* on the
"biographical accumulation" behavioral signature (consolidation
should help here but it's untested). The experiment tells us which of
those guesses is right.

## Deliverables for 4.6's planning phase

1. Architectural specification for v2 at pilot scale (this brief is
   the input; 4.6 produces the implementation-ready spec)
2. Implementation plan with milestones (pre-pilot setup → pilot → eval
   → decision gate)
3. Identification of which v1 components transfer directly, which need
   modification, which need replacement (the table above is the starting
   point)
4. Risk assessment + response plan for each named risk
5. Sequencing call: parallel with v1 ablations, or sequential after
   them?
