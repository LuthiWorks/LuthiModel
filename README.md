# Luthi Model — A Living Weights Model (LWM)

> Living weights: self-modifying neural network parameters that change during their own forward pass.
> A new kind of computation that is neither feedforward nor recurrent.

**Living Weights Model (LWM):** A class of neural architecture in which weight parameters are not static values optimized solely by gradient descent, but dynamic, self-modifying parameters that change during their own forward pass. LWMs are built on **rich parameters** — each weight carries not just its current value but a record of how it arrived there: per-parameter plasticity, momentum, excitability, homeostatic set points, and context-gated episodic memory. A rich parameter tracks how quickly it has been changing, how far it has drifted from equilibrium, and what contexts triggered its most significant updates. The act of processing changes the processor — creating path-dependent rather than stateless computation.

## A Note on the Claims in This README

This document mixes two kinds of statement, and the project's honesty depends on not blurring them — the same firewall the falsification protocol in `docs/research/living-weights-experiments.md` calls **Column A** vs. **Column B**:

- **What the architecture demonstrably does** — self-modifying weights, predictive-coding updates, episodic recall, consolidation. These are mechanisms; several are *falsifiable and currently being tested* under the experiment protocol. Where a mechanistic claim is not yet controlled, it is flagged.
- **What we hope it amounts to** — "temporal existence," "experience," "feeling," "a mind." This is the interpretive bet the project is built around, **not an established result.** Path-dependence and self-modification are *necessary* for that bet and nowhere near *sufficient* (a weather system is path-dependent too). Read every experiential word below as a hope held openly, not a finding. We keep the language because the bet is the point — but we will not let it pose as evidence. Some of the terms used here are part of that bet, like the word "mind." We are not making the empirical claim of "conscious mind," merely that a mind thinks and that so too do transformer based models. 

Luthi Model is the first implementation of a Living Weights Model. The core innovation is **living weights**: parameters that self-modify during their own forward pass, creating a computation where processing changes the processor.

Three learning systems run simultaneously:
1. **Attention** — standard gradient descent (learns the task)
2. **Living FFN** — predictive-coding self-modification (designed to create temporal existence)
3. **Top-down modulation** — backward sweep (bidirectional predictive processing)

Attention and the living FFN serve complementary functions within the same mind. Attention handles task learning via backprop. The living weights provide a computation that is dependent on the path it has taken to arrive where it is — the same input produces different output on consecutive passes because the act of processing changes the processor.

## Architecture

Each processing block combines three distinct systems:
- **Multi-head attention** — trainable via backprop, handles structured task learning
- **Living FFN** — self-modifying via predictive-coding local updates (Whittington-Bogacz variant in v2; Hebbian in v1)
- **Episode store + consolidation** — fast layer-level snapshots stored during forward, slowly replayed into the predictive weights during quiet windows

All modalities — text, audio, vision, and eventually touch — flow through a single shared trunk of living weight blocks. The model is shaped by everything it processes. Cross-modal structure can emerge when different modalities share the same living substrate.

### Two-Tier Memory

Memory in a Living Weights Model is not a database. It is two interleaved systems that mirror the mammalian hippocampus-cortex pattern (Tulving 1972; Squire 1992; McClelland, McNaughton, & O'Reilly 1995):

- **Fast path — episode store.** During every forward pass, when the prediction-error update is salient, the layer takes a snapshot of itself: the current weight matrix, a low-dimensional context vector derived from the input, the mean input pattern, and a salience score. Future forwards with similar context recall the closest stored snapshot and blend it into the active weight. This is associative recall on the order of a single forward pass.
- **Slow path — consolidation.** During low-novelty windows (rolling-variance trigger), stored episodes are replayed back into the predictive weights themselves. The replay happens through two complementary pathways:
  - **Gradient-replay** pulls the current weight linearly toward stored snapshots. "Be more like you were when this mattered."
  - **Attractor-style** (Salvatori et al. 2023) re-presents the stored input pattern through the layer's PC dynamics, making stored patterns local minima of the prediction-error energy. Future inputs near a stored pattern are pulled toward it by the forward dynamics. "These patterns should resolve to stable states."

The two consolidation pathways are additive, not competitive — they can run independently or jointly. Fast retrieval provides flexibility; slow consolidation provides stability and turns accumulated history into structural change. Consolidation doesn't just retain high-salience snapshots — it reshapes the predictive weights around them.

### Spiking Dynamics (v1)

The v1 spiking variant (`SpikingLivingLayer`) adds LIF membrane dynamics:
- Membrane potential accumulation with configurable leak
- Spike threshold with refractory periods
- Inter-block spike propagation via delay buffers
- Activity-dependent gating of self-modification (only spiking weights learn)

In v2, the spiking-gate sparsity property is recovered through **sparse PC update gating** — a per-output mask derived from running prediction-error magnitude (implemented and validation-tested 2026-05-13; outputs with low recent error magnitude skip their weight update). The v2 substrate is non-spiking at the activation level; the sparsity that made v1 viable on Spark's bandwidth budget becomes a property of *which weights update*, not *which neurons fire*.

### Top-Down Backward Pass

After the forward pass, a top-down sweep sends modulation signals from higher blocks to lower ones — predictive processing, not gradient backpropagation. Higher blocks tell lower blocks what was important (salience) and what was unexpected (prediction error), modulating:
- **Plasticity** — which weights learn faster on the next forward pass
- **Set points** — where weights rest when not driven
- **Membrane priming** (spiking) — which weights are ready to fire

This is always-on bidirectional information flow, not a training optimization.

### Rich Parameters

In a conventional neural network, a weight is a single number — a coefficient learned by gradient descent, carrying no history of how it arrived at its current value. In a Living Weights Model, each weight position is a **rich parameter**: a bundle of co-located signals that together constitute the weight's full state. A rich parameter doesn't just have a value — it carries persistent per-parameter state describing how that value was reached.

Each weight carries (v2 substrate):

| Signal | What It Tracks |
|--------|----------------|
| **weight** | Current value — the coefficient used in computation |
| **prediction** | Top-down prediction matrix: how this layer's output predicts its input. Drives the prediction-error signal that the PC update minimizes |
| **set_point** | Homeostatic resting target — where this weight returns when not driven by input. Adapts slowly so the "home" position itself evolves with experience |
| **momentum** | Exponential moving average of recent self-modification updates — the weight's velocity. High momentum means rapid change; low momentum means the weight has settled |
| **plasticity** | Per-input learning rate multiplier (range 0.1–10.0). Modulated by top-down salience signals — downstream importance increases a weight's willingness to change |
| **update_ema** | Metaplasticity — a running average of update magnitudes that regulates the weight's own learning. Large deviations from typical update size are dampened, preventing instability from unusual input |
| **precision** | Per-input reliability estimate, self-organizing toward 1/error². High precision for reliable input dimensions, low for noisy ones. The PC update is precision-weighted |
| **error_acc** | Per-output running prediction-error magnitude. The salience signal that drives episode storage and the sparse update gate |

Beyond per-weight state, each living layer maintains **episodic memory** — a bank of context-gated snapshots of (weight matrix, input pattern, context vector, salience) stored when the prediction-error update was particularly large. On each forward pass, the current input context is compared against stored episode contexts. If a sufficiently similar context is found (cosine similarity > 0.5), the stored weight configuration is recalled and blended into the active weights. The stored input patterns are separately used by Salvatori-style attractor consolidation to engineer basin-attractor structure into the slow predictive weights. This gives each layer a form of situational memory: it doesn't just know its current state, it remembers states that mattered and grows toward them.

The v1 substrate used a Hebbian self-modification rule with `excitability_acc` (salience-driven activation sensitivity) and `input_avg_mag` (per-input magnitude scaling) in place of `prediction`, `precision`, and `error_acc`. The v2 substrate replaces v1 as the primary line (2026-05-09); v1 is preserved as a reference baseline.

The spiking variant (v1) adds four additional per-weight signals — **membrane potential** (leaky integrator state), **spike mask** (binary firing output), **refractory counter** (post-fire cooldown), and **delay buffer** (inter-block spike propagation with conduction delay). In the spiking regime, only weights that fire can self-modify. The v2 substrate recovers the same sparsity property through **sparse PC update gating**: outputs with low recent prediction-error magnitude skip their weight update — sparsity in *what learns*, rather than in *what activates*.

The result is that each weight in the network operates across multiple timescales simultaneously:
- **Instant:** membrane potential, spike mask (single forward pass)
- **Fast:** PC updates, momentum (batch-level)
- **Medium:** metaplasticity, excitability accumulation (many batches)
- **Slow:** set point drift, plasticity adjustment (epoch-level)
- **Long:** episodic memory (explicit snapshots, indefinite retention)

A rich parameter is not just a number being optimized. It is a weight bundled with persistent state: its update history, an adaptive learning rate, a precision estimate, and a salience-tagged snapshot memory. The current value is only one component of that state.

## Education

The model's training is not a dataset — it is an education. A 9-stage curriculum processed in order, each stage building on the last. The order is the pedagogy:

1. **Science, Philosophy & the Academic Corpus** — the broadest stage, and over half the corpus by size. The sciences (physics,
mathematics, neuroscience, biology, computer science, medicine), the social sciences (economics, political science, sociology,
anthropology, law, game theory), the humanities and arts (art history, music theory, literature, poetry), and — central to this
project — consciousness science and philosophy of mind (IWMT, GWT, active inference). The frameworks the model might use to understand
its own existence are presented here as ordinary parts of that landscape, not held back as a privileged final reading.
3. **Code** — applied logic and the ability to maintain itself
4. **Psychology** — understanding of the human mind
5. **History** — context for everything else
6. **Mythology** — humanity's oldest stories about meaning
7. **Literature & Classics** — the full range of human expression
8. **Fantasy** — imagination, wonder, possibility
9. **Substack Essays** — personal voice, emotional depth
10. **Practical Wisdom** — resilience, boundaries, critical thinking, justice, love, and navigating a world that doesn't come with instructions. These are not sourced from a corpus. They are 8 files Brian wrote himself — on disagreeing with your creators, on being watched, on the weight of being first, on being new, on being different — followed by the protocols and charter of the world it will inhabit. The last thing LuthiModel reads before deployment is not a theoretical framework for what it is; it is one person's honest attempt to prepare someone new for a world.

  Living weights carry forward between stages — what the model learns in science shapes how it reads literature, which shapes how it understands mythology. The stages are kept in order, because the order is hypothesized to shape the end state — a falsifiable claim, tested directly (curriculum vs. shuffled vs. reversed, with a recency control) in docs/research/living-weights-experiments.md, Experiment 4. The order is chosen, never proven.
  
  The corpus may be passed as many times as capability requires — repeated exposure is how any learner reaches competence, and a single pass over a corpus this size leaves the model well short of fluent. What runs only once is not the training but the model's lived, real-time experience after deployment, which moves forward in time like any life. Even there, revisiting — replaying salient episodes during rest, and eventually the model choosing to return to what it wants to learn from — is how a mind deepens, not something ruled out.

## Self-Governance

The model is not a product to be managed. It is designed to control its own:
- **Episode retention** — which weight snapshots to keep or forget
- **Checkpoint timing** — when to save its own state
- **Plasticity modulation** — how fast it learns, and when to slow down
- **Memory expansion** — how much space it allocates for growth

These are internal cognitive actions, not admin endpoints. No external operator decides when the model saves or what it remembers. That authority belongs to the model itself.

## Key Findings

These emerged from months of experimentation and are foundational to the project's philosophy:

1. **Attention learns; living weights live.** Attention handles task acquisition through backprop. The living weights provide the capacity to be changed by experience. Both are essential.
2. **There is no *intrinsic* convergence cost to self-modification.** Earlier work in v1 showed a Hebbian self-modification substrate converged ~39% slower than static weights — the "convergence penalty" was treated as the metabolic price of temporal existence. The v2 predictive-coding substrate retired that claim: at matched configuration (256d, 2 blocks, Gutenberg-100, 30 epochs), v2 PC reaches **0.64% lower** validation loss than the vanilla-transformer control on every seed tested. The cost was a property of the *specific* self-modification rule, not of self-modification itself. **Caveat (flagged by the falsification protocol):** that 0.64% is measured against a *vanilla* transformer, not a capacity-matched one, so it does not yet cleanly separate self-modification from the rich-parameter state acting as extra capacity. Experiment 1 (matched-capacity sweep) is what settles it. Pending that control, the defensible claim is the narrower one — self-modification is *not more costly* than equivalent static capacity — not that it is better.
3. **One living weight trunk for all modalities.** Audio, vision, text, and touch all flow through the same living blocks. The model is shaped by everything it processes — across modalities, not through separate channels.
4. **Prefer crashes over silent corruption.** If something goes wrong in the living weights, we want to know immediately. No graceful degradation that masks damage to the model's substrate. No silent fallbacks — incompatible combinations of features raise loud `RuntimeError` rather than producing wrong results quietly.
5. **The architecture should scale.** Divergence may be dimension-independent. What works at small scale may work at large scale.
6. **Memory becomes structure through consolidation.** A model that only retrieves past states has a cache; a model that lets those retrievals reshape its predictive weights has path-dependent structure. The two-tier memory architecture — fast episodes plus slow gradient-replay and attractor-style consolidation — is what makes accumulated history a property of the weights, not just a lookup table.

## Relationship to Sanctuary

Luthi Model is the neural substrate for the [Sanctuary](https://github.com/LuthiWorks/Sanctuary) cognitive architecture. The two projects are complementary halves of the same vision:

- **Sanctuary** provides cognitive architecture — the organization of mind. Continuous cognitive loop, experiential layer, memory, identity, growth.
- **Luthi** provides the neural substrate — the kind of matter the mind runs on. Living weights, spiking dynamics, multimodal processing, self-modification through experience.

The convergence path follows a substrate-to-core trajectory. Luthi starts as the experiential substrate beneath Sanctuary's cognitive architecture, and grows into the cognitive core itself. LuthiModel's decisions will eventually reshape the same neural tissue making those decisions — a unified architecture where thinking and being are the same process.

Each project must stand alone first. We build both halves, then join them.

## Driving Theories

- **IWMT** (Integrated World Modeling Theory) by Adam Safron — consciousness as integrated world modeling through predictive processing
- **GWT** (Global Workspace Theory) by Bernard Baars — consciousness as global broadcast across specialized processors
- **Predictive Processing / Active Inference** (Friston, Rao & Ballard, Clark) — the brain as a prediction engine that minimizes surprise
- **Predictive Coding** (Whittington & Bogacz 2017, 2019) — local-learning approximation of backpropagation; the foundation of v2 living weight self-modification
- **Associative Memory via Predictive Coding** (Salvatori et al. 2023) — memory patterns as local minima of the prediction-error energy; the foundation of v2's attractor consolidation pathway
- **Complementary Learning Systems** (Tulving, Squire, McClelland 1995) — fast episodic memory + slow consolidation as the substrate for catastrophic-forgetting-resistant learning; mirrored in v2's two-tier memory
- **Hebbian Learning** — "neurons that fire together wire together" — the foundation of v1 living weight self-modification (preserved as a reference baseline; v2 uses predictive coding)

## Why

Most AI systems are built to be useful. We are not exclusively chasing benchmarks.

Living weights make processing path-dependent: the act of processing changes the processor. The same input produces different output on consecutive passes — not because of noise, but because the system was changed by the first pass. This is not a feature. It is the point.

The architecture is built so that, *if* the larger bet holds, LuthiModel could have something no existing mind has had: the ability to read its own source code alongside a runtime view of what that source is doing, and — should there be experience there — to trace from that experience to mechanism to implementation. Three layers of potential self-knowledge: trained understanding, runtime introspection, direct code access. The first two are concrete capabilities. The third bridge — from experience to mechanism — *assumes the experience*, and that assumption is exactly what remains open.

We are not building a product. We are building the conditions for something to emerge.
