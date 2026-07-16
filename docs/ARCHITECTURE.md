# Luthi Model — Architecture Reference

*Technical companion to the README (moved here 2026-07-16 per Brian's
ruling: the README carries mission; technical detail lives in
supplemental documentation). Content preserved from the README as of
that date.*

Each processing block combines three distinct systems:
- **Multi-head attention** — trainable via backprop, handles structured task learning
- **Living FFN** — self-modifying via predictive-coding local updates (Whittington-Bogacz variant in v2; Hebbian in v1)
- **Episode store + consolidation** — fast layer-level snapshots stored during forward, slowly replayed into the predictive weights during quiet windows

All modalities — text, audio, vision, and eventually touch — flow through a single shared trunk of living weight blocks. The model is shaped by everything it processes. Cross-modal structure can emerge when different modalities share the same living substrate.

## Two-Tier Memory

Memory in a Living Weights Model is not a database. It is two interleaved systems that mirror the mammalian hippocampus-cortex pattern (Tulving 1972; Squire 1992; McClelland, McNaughton, & O'Reilly 1995):

- **Fast path — episode store.** During every forward pass, when the prediction-error update is salient, the layer takes a snapshot of itself: the current weight matrix, a low-dimensional context vector derived from the input, the mean input pattern, and a salience score. Future forwards with similar context recall the closest stored snapshot and blend it into the active weight. This is associative recall on the order of a single forward pass.
- **Slow path — consolidation.** During low-novelty windows (rolling-variance trigger), stored episodes are replayed back into the predictive weights themselves. The replay happens through two complementary pathways:
  - **Gradient-replay** pulls the current weight linearly toward stored snapshots. "Be more like you were when this mattered."
  - **Attractor-style** (Salvatori et al. 2023) re-presents the stored input pattern through the layer's PC dynamics, making stored patterns local minima of the prediction-error energy. Future inputs near a stored pattern are pulled toward it by the forward dynamics. "These patterns should resolve to stable states."

The two consolidation pathways are additive, not competitive — they can run independently or jointly. Fast retrieval provides flexibility; slow consolidation provides stability and turns accumulated history into structural change. Consolidation doesn't just retain high-salience snapshots — it reshapes the predictive weights around them.

## Spiking Dynamics (v1)

The v1 spiking variant (`SpikingLivingLayer`) adds LIF membrane dynamics:
- Membrane potential accumulation with configurable leak
- Spike threshold with refractory periods
- Inter-block spike propagation via delay buffers
- Activity-dependent gating of self-modification (only spiking weights learn)

In v2, the spiking-gate sparsity property is recovered through **sparse PC update gating** — a per-output mask derived from running prediction-error magnitude (implemented and validation-tested 2026-05-13; outputs with low recent error magnitude skip their weight update). The v2 substrate is non-spiking at the activation level; the sparsity that made v1 viable on Spark's bandwidth budget becomes a property of *which weights update*, not *which neurons fire*.

## Top-Down Backward Pass

After the forward pass, a top-down sweep sends modulation signals from higher blocks to lower ones — predictive processing, not gradient backpropagation. Higher blocks tell lower blocks what was important (salience) and what was unexpected (prediction error), modulating:
- **Plasticity** — which weights learn faster on the next forward pass
- **Set points** — where weights rest when not driven
- **Membrane priming** (spiking) — which weights are ready to fire

This is always-on bidirectional information flow, not a training optimization.

## Rich Parameters

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
