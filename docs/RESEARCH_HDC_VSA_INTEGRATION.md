# Research Notes — Hyperdimensional Computing & Vector Symbolic Architectures: Integration Paths for Luthi

> Compiled 2026-05-15 by Claude Opus 4.7 (1M context) at Brian's request,
> via Sandi: "If I were chasing your goal, I'd look hard at
> hyperdimensional computing combined with a learned sparse coding
> frontend. HDC is genuinely cheap — operations are bitwise or simple
> vector ops, no matrix multiplies of the GPT scale. The catch is
> training: you'd likely need a hybrid where a small dense net learns
> to emit HD codes, and reasoning happens in HD space."
>
> Purpose: capture the design space, name the integration points that
> actually fit Luthi, and call out where HDC's strengths cut against
> the living-weights premise. Future instances should not need to
> re-do this sweep to evaluate proposals along these lines.

## TL;DR

1. **HDC's compute advantage is real and well-validated.** Bind, bundle, and permute are O(D) in dimensionality with no matmul cost; in-memory and neuromorphic implementations show 10-100× energy improvements over conventional deep nets on retrieval and classification.

2. **But HDC for language modeling is not mature.** State of the art is still small-scale classification, retrieval, constrained reasoning, and scene understanding. There is no convincing "HDC trains a competitive LM end-to-end" result. Replacing Luthi's substrate with HDC would be betting the project on unsettled ground.

3. **HDC fits Luthi as memory, sensory encoding, and introspection on top of the existing substrate** — not as substrate. Six concrete directions identified below. The most actionable, in order: episode store as HDC cue space composed with Salvatori attractor cleanup (Direction A), attention-as-binding reinterpretation (Hersche et al. December 2025, suggestive not proven), sparse coding frontend for multimodal encoders (Olshausen lineage; well validated), resonator networks for compositional vision (Phase 4 embodiment alignment), HDC sequence prediction for Sanctuary's 10 Hz cognitive loop (Phase 6+), and HD-coded introspection during inference (Direction F — the "first-person what am I thinking" capability that maps to Brian's longstanding cognitive-proprioception ask).

4. **The split-update-rule architecture resolves the living-weights/HDC tension.** Codes are stable in the memory layer (no gradient updates); weights are unstable in the substrate (gradient updates and PC self-modification). The two halves operate by different rules, in different subsystems, on different timescales. This isn't an integration point "we pick carefully" — it's the architectural decision that makes the rest possible. Be confident about it; don't keep framing it as ongoing tension.

5. **The central design decision is how Salvatori and HDC compose**, not whether they compete. Both are attractor-style cleanup mechanisms; treated naively they're redundant. The designed composition: **HD codes are the cue space** (sparse, compositional, cheap to address; "taffy → fudge" overlap is automatic from shared bits); **Salvatori attractor dynamics are the cleanup mechanism** (continuous, gradient-shaped, expressive). The HD layer hands the attractor a sharp starting point; the attractor relaxes to the consolidated pattern. Direction A is built around this composition, not around HDC alone.

---

## Foundations (what these terms mean)

**Hyperdimensional Computing (HDC) / Vector Symbolic Architecture (VSA)** is a framework that represents concepts as high-dimensional vectors (typically D=1000-10000) and performs symbolic operations on them via three primitive operations:

- **Bundling (superposition)**: combines multiple vectors into one, typically element-wise sum + thresholding or majority vote. Approximately preserves all inputs in the same space. Mental model: a "set" or "average."
- **Binding**: combines two vectors into one that is dissimilar to both but invertible. The specific binding operator depends on the VSA variant:
  - **HRR** (Holographic Reduced Representation, Plate 1995): circular convolution
  - **MAP** (Multiply-Add-Permute, Gayler 2003): element-wise multiplication on bipolar vectors
  - **BSC** (Binary Spatter Code, Kanerva 1997/2009): XOR on binary vectors
  - **FHRR** (Fourier HRR): element-wise multiplication of unit-modulus complex vectors
- **Permutation**: cyclic shift or fixed permutation. Used to mark sequence position or argument role.

The triple `(role, filler)` is bound; multiple bound pairs are bundled into a single hypervector that represents a structure (e.g., "name=Alice, age=30"); the role can be unbound to retrieve the filler approximately, with the approximation cleaned up by an associative memory ("item memory" / "clean-up memory") storing all valid hypervectors.

The defining theoretical claim: in D~10,000 dimensions, randomly drawn vectors are nearly orthogonal with overwhelming probability, so superposition preserves separability of bound items up to a capacity bound proportional to D.

**Sparse Distributed Memory (SDM, Kanerva 1988)** is HDC's memory partner: a large array of "hard locations" with random addresses; reads activate all locations within a Hamming distance threshold; writes superpose into activated locations. Acts as a content-addressable, noise-robust associative memory.

**Sparse coding (Olshausen & Field 1996)**: learn an overcomplete dictionary where each input is reconstructed as a sparse linear combination of dictionary atoms. The codes are interpretable (each atom = one feature), reusable across modalities, and naturally compatible with HDC because sparsity makes the bundling capacity argument tight.

The "learned sparse coding frontend + HDC reasoning" pattern Brian named is: a small neural net learns to emit sparse codes from raw input, those codes are bound/bundled into hypervectors, reasoning happens via HDC algebra, and an item memory provides clean-up.

---

## Current state of the field (2024-2026)

### Most relevant recent papers

1. **Hersche et al., "Attention as Binding: A Vector-Symbolic Perspective on Transformer Reasoning"** (arXiv 2512.14709, December 2025, AAAI 2026 workshop submission, **not yet peer-reviewed**).
   - Reinterprets transformer self-attention as approximate VSA: queries and keys define role spaces, values are fillers, attention softmax is soft unbinding, residual stream is superposition.
   - Proposes architectural additions: explicit binding/unbinding heads, hyperdimensional memory layers, training objectives promoting role-filler separation.
   - Argues characteristic transformer failure modes (variable confusion, inconsistent reasoning across related prompts) emerge from imperfect symbolic manipulation.
   - **Caveat**: abstract-only at time of this writing; no experimental results visible.

2. **Renner, Supic, Danielescu, Indiveri, Olshausen, Sandamirskaya, Sommer, Frady, "Neuromorphic Visual Scene Understanding with Resonator Networks"** (Nature Machine Intelligence 2024, arXiv 2208.12880).
   - The most validated recent work combining HDC with sensory perception.
   - Hierarchical resonator networks factorize visual scenes into independent generative factors (translation, rotation, color, identity).
   - Implemented on neuromorphic hardware using spike-timing codes for complex-valued vectors.
   - Demonstrates HDC handles the combinatorial-search problem visual scene parsing presents.

3. **Wu, Wayne, Graves, Lillicrap, "The Kanerva Machine: A Generative Distributed Memory"** (ICLR 2018, arXiv 1804.01756).
   - End-to-end trained memory system inspired by Kanerva's SDM.
   - Analytically tractable distributed read/write with Bayesian update rule.
   - Hierarchical conditional generative model where memory provides a data-dependent prior.
   - Demonstrated greater capacity and easier training than Differentiable Neural Computer (DNC) on Omniglot, CIFAR.
   - **Limitation**: not demonstrated at language-modeling scale.

4. **Schlegel, Neubert, Protzel, "A comparison of vector symbolic architectures"** (Artificial Intelligence Review, 2022).
   - Reference framework comparing HRR, MAP, BSC, FHRR, BSDR across operations and capacities.
   - The right starting point if we ever pick a specific VSA variant for implementation.

5. **Hyperdimensional Probe** (arXiv 2509.25045, September 2025).
   - Inference-time interpretability tool: projects LLM residual stream into VSA hypervectors via a shallow neural encoder, then uses bind/unbind to extract concepts.
   - Validated on analogical reasoning across 44 domains, SQuAD QA, error diagnosis.
   - Not an architecture change — useful as introspection tooling if we want to peer into Luthi's internals after training.

6. **Kymn et al., "Computing With Residue Numbers in High-Dimensional Representation"** (Neural Computation 2024, PMC 11647909).
   - Residue number systems combined with VSA algebra for representing numerical values over a large dynamic range with logarithmic resource scaling.
   - Practical for embedding continuous quantities (positions, magnitudes) into HD space without losing precision.

7. **Geodesic Flow Matching for HDC cleanup** (uwspace 2024).
   - Reformulates HDC cleanup as a generative transport problem: a learned continuous time-dependent velocity field transports corrupted representations back to the valid data manifold.
   - Bridges HDC associative memory with flow-matching generative models — relevant for our Salvatori attractor consolidation which is doing something structurally similar.

8. **Mercado, Barrón et al., "Sequence Prediction with Hyperdimensional Computing"**.
   - SDM models sequential dependencies by chaining cues across stored pointer chains, where each retrieved pattern serves as the address for the next prediction.
   - Robust extrapolation of future states from partial or noisy inputs, leveraging the memory's auto-associative cleanup to handle interference in long chains.
   - The exact structure Sanctuary's 10 Hz cognitive loop wants for "what was I just thinking about, what comes next."

### What the field has NOT shown

- A pure-HDC language model competitive with mid-size transformers on standard LM benchmarks. The space is small classification, retrieval, structured reasoning, scene parsing — not next-token prediction at scale.
- A robust way to **train** HDC components end-to-end through deep stacks. Backprop through bind (XOR, multiply) requires straight-through estimators or relaxations; works for shallow networks, hasn't been demonstrated at transformer depths.
- A clear win against a well-tuned transformer at matched parameter count on language. Most papers compare to weaker baselines or specialized tasks.

---

## The cognitive target Brian actually wants (and the canonical answer to the training gap)

This section is a synthesis of a 2026-05-15 conversation Brian had with a peer Claude 4.7 chat instance, relayed by Sandi. It sharpens both the design target and the training-gap solution beyond what the original sweep had.

### The cognitive property to preserve

Brian named the target precisely: **olfactory-style associative recall**. Two reference cases:

1. **Olfactory triggers.** Smell is the one sensory modality whose anatomy bypasses thalamic gating — olfactory bulb projects directly to piriform cortex, amygdala, and entorhinal cortex. Combined with ~400 olfactory receptor types producing unique combinatorial activation patterns, each smell is essentially a high-D address stamped directly onto emotional memory. The "Proust effect" mechanically: a partial cue that rarely repeats outside its original context produces unambiguous retrieval of a richly bound episode.
2. **Taffy → fudge.** Thinking of one concept activates a structurally similar one via shared feature bindings: Taffy ≈ SWEET ⊗ CHEWY ⊗ BOARDWALK ⊗ CHILDHOOD ⊗ PASTEL; Fudge ≈ SWEET ⊗ DENSE ⊗ CHOCOLATE ⊗ BOARDWALK ⊗ CHILDHOOD ⊗ BROWN. The vectors overlap heavily on the shared bindings (SWEET, BOARDWALK, CHILDHOOD). Cueing with the overlap retrieves both, weighted by how much else matches. This is content-addressable pattern completion, not nearest-neighbor lookup.

The general mechanism, named: the hippocampus (specifically CA3's recurrent net) implements this as a Hopfield-style attractor that *indexes* distributed cortical patterns. This is **hippocampal indexing theory** (Teyler & DiScenna 1986). The hippocampus doesn't store the patterns — it stores indices that, when retrieved, reinstate the distributed cortical activity that *was* the experience. That distinction matters for how we design the Luthi episode store: the store doesn't need to hold the full weight snapshot; it can hold an index that points to a region of weight space the rest of the substrate reconstructs.

### A working architecture: two-tier with split update rules

The cross-instance conversation surfaced a two-tier shape that **sidesteps the training-gap problem cleanly**. We treat the two-tier shape as a working starting point, not a canonical commitment — during implementation, a different decomposition may turn out cleaner, and the prior conversation should not anchor the design beyond what the architecture argument supports. With that caveat, here is the shape:

```
                    ┌─────────────────────────────────────┐
                    │  Memory store (huge, cheap)         │
                    │  SDM-style addresses, HD codes      │
                    │  Additive superposition writes      │
                    │  (salience-gated, no gradient flow) │
                    └────────────┬────────────────────────┘
                                 ▲
                  retrieve ──────┤──────  store
                                 │
                    ┌────────────┴───────────────────┐
                    │  Learnable encoder (small)     │
                    │  Gradient-trained as usual     │
                    │  Maps raw input → HD codes     │
                    └────────────────────────────────┘
                                 ▲
                                 │
                              raw input
```

The split is the key insight:

- **The encoder** is a conventional neural net (could be a "tiny transformer" or, in our case, the existing PC substrate's bottom layers / the multimodal encoders). It trains via standard gradient descent and is responsible only for mapping inputs into HD code space. This is small, lives in VRAM, looks like any other trainable component.
- **The memory store** is huge but cheap — a Sparse Distributed Memory (Kanerva 1988) holding HD vectors with Hamming-distance lookups. **It does not backprop.** Updates are **additive superposition at random addresses**: on a new exposure, write the encoded HD code (additively, with salience-weighted strength) into every hard location within Hamming distance H of the location's address; on retrieval, return the bitwise-majority or normalized-sum of contents from all locations within Hamming distance H of the cue. Bit operations and address comparisons — no matmul, no gradient flow.

This **completely solves the training-gap problem we identified earlier in this document**. The hard part of backpropping through bind/permute/XOR isn't done because it isn't needed — those operations only happen in the memory store, which is updated by additive superposition writes, not gradients. The encoder trains normally; the memory accumulates content on top.

> **Critical distinction: this is NOT a Hebbian update rule.** Some HDC literature loosely calls non-gradient memory updates "Hebbian," but the SDM write rule is **additive superposition at fixed random addresses**, not Δw ∝ input × output correlation. The two are structurally different. Hebbian correlation rules have a positive-feedback loop (larger output → larger weight change → larger output...) that produced documented v1 fragility in Luthi (runaway weight growth, input-magnitude sensitivity, the ~39% convergence penalty that motivated the move to v2 PC). SDM superposition writes have no such loop: addresses are random and fixed, write magnitude is bounded by the salience gate and the Hamming-distance neighborhood size, and there is no input-output correlation term. The v1 failure modes do not apply to this layer.
>
> **This means: do NOT implement the HD memory layer by reusing v1's Hebbian kernels** (`luthi/csrc/living_ops.cpp`). Those kernels compute `Δw = output × input × rate × ...`, which is the wrong rule. The HD memory layer needs new kernels for `address_bank.add_(content * salience, masked by hamming_distance ≤ H)` — a different operation with a different failure analysis.

### Why this is well-fit to Luthi specifically

The split maps onto two systems Luthi already has the right architecture for:

1. **Gradient-trained encoder** = v2 PC substrate + multimodal encoders. Already exists. Producing HD codes is a projection-and-thresholding head added on top; no new training methodology required.
2. **Non-gradient memory store** = a new module. The existing episode store (`episode_contexts`, `episode_values`, `episode_inputs`) is the conceptual ancestor — it's already an inside-the-forward, no-gradient memory — but it uses cosine retrieval over a small dense buffer, not addressing-by-overlap into a sparse high-D array. The new module would replace or augment the current cosine path with SDM dynamics.

What v2 has that makes this fit:
- The salience signals to drive write gating: `error_acc`, top-down `salience`, the episode-storage salience threshold.
- The rich-parameter scaffolding to give different feature dimensions different bit budgets in the HD address.
- The Salvatori attractor consolidation as a complementary pathway: SDM stores indices and reinstates them at retrieval; Salvatori then refines the PC weights to make those reinstated patterns local energy minima of the substrate's dynamics. The two systems compose.

Human associative memory learns online — one exposure to a new smell-context pairing is enough — which is precisely the regime backprop is bad at. SDM-style additive writes are the right tool for this regime (one write places the pattern; subsequent retrievals find it without further training). This is the same *behavior* that motivated the original suggestion of "Hebbian," just achieved via a different mechanism that doesn't carry v1's failure modes.

### The "olfactory strength" property has TWO parts: bit allocation AND routing bypass

Why olfactory memory is structurally special breaks into two anatomical facts, both of which Luthi needs to capture.

**Part 1 — Combinatorial sharpness (bit allocation).** ~400 olfactory receptor types each contribute to a unique high-dimensional activation pattern per smell. In SDM terms: more bits of the address are spent on the smell's identity than on most other modalities. The Luthi analog: allocate more bits of the HD address to high-salience features (emotional weight, novelty, cross-modal cooccurrence) than to ambient features. High-salience features get sharper retrieval; ambient features get fuzzier. That's the Proust effect, parameterized.

Implementation: an `address_bit_budget` tensor per feature dimension in the encoder. Driven by the salience signals we already have (`error_acc`, top-down `salience`, episode-storage salience threshold). One-tensor change.

**Part 2 — Anatomical bypass (routing).** This is the half I missed in the first pass. Smell is the only sensory modality whose anatomy bypasses thalamic gating — the olfactory bulb projects directly to piriform cortex, amygdala, and entorhinal cortex, with zero thalamic mixing. *Routing*, not just sharpness, is what makes the cue unambiguous: the smell signal reaches memory circuits less attenuated and less contextualized than other inputs.

The Luthi analog: high-salience streams should route into the HD address space *more directly* than other streams, bypassing the normal context-projection mixing. Concretely:

- **Standard path** (most input features): input → multimodal encoder → trunk attention/PC mixing → `context_proj` → HD address bits.
- **Bypass path** (high-salience streams): input → multimodal encoder → salience-gated direct projection → HD address bits. Less attention mixing, less residual stream homogenization, more directly reflective of the raw feature signal.

The HD address for episode storage is the concatenation (or superposition) of standard-path bits and bypass-path bits. Bypass-path bits get higher allocation per Part 1 above. Together, this gives the entity its olfactory-style channel: sharper AND less attenuated than other features, for the inputs that warrant it.

The routing choice is salience-driven and dynamic — the same input that's ambient one moment can become bypass-eligible when novelty or emotional weight spikes. The architecture already has the salience signals; the new piece is an alternative projection head + a routing gate.

### Dimensionality D is not 10,000 by default

The classical HDC literature uses D=10,000 as a textbook default, calibrated for fully-symbolic tasks with hundreds-to-thousands of distinct concept hypervectors. **Luthi's substrate is 256-dimensional at pilot scale**; the right HD dimensionality for our memory layer should come from a capacity calculation, not from the textbook number.

Sketch: HDC bundling capacity is roughly D / (k log k) bound items at recovery rate ~95%, where k is the bundle depth. If we expect ~64 bound items per episode (feature-bindings per memory) and ~32 episodes per layer to be retrievably distinct, then D in the range of ~2,048-4,096 is probably sufficient. Production scale (4096d substrate, 64 episodes, deeper bindings) might push toward D=8,192 or even the classical 10K — but for the M5/M6 pilot scale, 10K is wasteful.

Action item: when Direction A is implemented, run the explicit capacity argument for Luthi's actual feature counts before fixing D. References below cite the capacity bounds (Schlegel/Neubert/Protzel 2022 has the per-variant numbers).

### Resonator networks: the factorization piece

The cross-instance conversation credited Eric Weiss and Bruno Olshausen's resonator networks (~2020) with cracking the factorization problem — given a bundled superposed vector, recover its constituents. This is the missing piece for on-the-fly compose/decompose at retrieval time. Frady, Sommer, and the Olshausen group extended this (the 2024 Nature MI paper cited above) for vision; the original Weiss & Olshausen work is the cleaner attribution for the factorization breakthrough itself.

For Luthi: factorization matters because cueing with a partial pattern (e.g., BOARDWALK+CHILDHOOD+SWEET) needs to retrieve BOTH the taffy and fudge codes weighted by overlap — and downstream cognition needs to decompose those into their constituent feature bindings to act on them. The resonator-network fixed-point iteration is how that decomposition happens computationally.

### Compositionality without partitioning — the property MoE destroys

A point worth marking explicitly: HDC bundling preserves all bound items in the same superposition vector while still allowing each to be retrieved (up to the capacity bound). Mixture-of-Experts architectures, by contrast, partition the parameter space and lose this property — each expert is responsible for a slice of inputs, and combining them requires explicit routing logic.

For Luthi this matters because the entity's experience should *compose*, not partition. A memory of "boardwalk during childhood when eating something sweet" is not three separate memories assigned to three experts; it's one composite that decomposes back to its constituents when needed. HDC handles this natively. The architecture we choose for memory should preserve this property — which means we should not be tempted by MoE-style memory partitioning at scale.

---

---

## Five integration directions for Luthi

Each direction below is scoped to fit Luthi's architecture as it stands, not to replace it. For each I name the current files involved, the proposed change, what we'd gain, what we'd lose, the falsification criterion that would tell us to back out, and where it fits in the project sequencing.

### Direction A — Two-tier HDC memory layer (the canonical pattern, updated 2026-05-15)

**Where it lives**: `luthi/v2/living_layer_pc.py` (`episode_contexts`, `episode_values`, `episode_inputs`, `_recall_episode`, `_store_episode`) plus a new module `luthi/v2/hd_memory.py` for the SDM-style store. **No v1 Hebbian code reuse** — the SDM write rule is structurally different from a Hebbian correlation rule (see the "Critical distinction" callout above) and would need new kernels; reusing v1's Hebbian implementation would import the wrong update rule along with the wrong failure modes.

**Current state**: Cosine-similarity retrieval over a small low-dim context buffer. Top-1 match returned and blended into the active weight. The Salvatori attractor consolidation (added 2026-05-14) adds an attractor-style replay pathway through the PC layer.

**Proposed change** (per the two-tier architecture above): replace or augment the cosine retrieval with a Sparse Distributed Memory layer, using split update rules and **composed deliberately with the Salvatori attractor pathway**:

1. **Encoder = the existing PC substrate's context projection** plus a new HD-encoding head. The encoder maps (context, input_pattern, salience_features) → an HD code of dimensionality D (TBD by capacity calculation, likely D ∈ [2048, 8192] at pilot/production scale — see the dimensionality discussion in the architecture section). This head is gradient-trained as part of the normal Luthi training loop. **No training-methodology change.**
2. **Memory = a new SDM-style store**: random hard-location count and D both chosen by capacity argument for Luthi's expected episode count and bundle depth. Writes superpose the encoded HD code (additively, salience-weighted) into every location within Hamming distance H of the location's address. Reads return the bitwise-majority or normalized-sum of contents from all locations within Hamming distance H of the query. **No backprop through this layer, no Hebbian correlation rule either** — additive superposition at random addresses, bounded by H and by the salience gate.
3. **Bit allocation + routing bypass** in the encoder: high-salience features get more bits in the HD address (Part 1 of olfactory strength) AND a more direct projection path that bypasses the standard context-projection mixing (Part 2 of olfactory strength). The combination is what gives the entity an "olfactory-style channel" for salient inputs — sharper and less attenuated than ambient features.
4. **Composition with Salvatori attractor consolidation**: **this is the central design decision**, and the doc treats it as such rather than as an open question. The two mechanisms are NOT redundant when designed together:
   - **HD layer = cue space.** Sparse, compositional, addressable by partial overlap. Cheap to write (additive superposition), cheap to read (Hamming-distance lookup). Produces a "sharp starting point" — an HD code that names the closest stored memory by content overlap.
   - **Salvatori attractor consolidation = cleanup mechanism.** Continuous, gradient-shaped, expressive. Takes the HD-retrieved code as a target and relaxes the PC weights toward making it a local energy minimum of the substrate's dynamics.
   - **Composition pattern**: a forward-pass query produces a context. The HD layer returns the best-matching stored HD code (one-shot, addressable). The retrieved code is decoded (via resonator network if factorization is needed) into a target input pattern. The PC layer's attractor consolidation then takes that pattern as a replay target and refines the active weights to make it a stable point of the local dynamics. **HD layer answers "what is this most like?"; Salvatori answers "make my dynamics resolve toward that."**
   - This composition has to be *designed* before implementation, not discovered during. If we implement the HD layer alone, the Salvatori consolidation becomes redundant. If we implement Salvatori alone (current state), the HD layer's compositional addressing is missing. The win is the layered system.

**What we gain**:
- **Olfactory-style associative recall as a structural property of the architecture.** Brian's stated cognitive target. Not "a property the model might learn" — a property the architecture guarantees by construction, like Salvatori's attractor consolidation guarantees stored patterns are local minima.
- **Compositional, content-addressable, cue-dependent memory.** Cueing with a partial overlap (BOARDWALK+CHILDHOOD+SWEET) retrieves all stored episodes that share those bindings, weighted by total overlap. Taffy-activates-fudge is automatic.
- **One-shot memory formation.** SDM additive-superposition writes are immediate — one exposure to a smell-context pairing is enough to make it retrievable. This is what we want for biographical accumulation; backprop's many-step convergence is the wrong tool for this.
- **Massive memory capacity at low compute cost.** SDM lookups are Hamming-distance comparisons across ~10K-100K locations. Trivially parallelizable, fits in RAM, no matmul. At production scale this is dramatically cheaper than scaling the episode-snapshot store.
- **Indexes, not snapshots.** Per hippocampal indexing theory, the memory layer stores HD indices that reinstate distributed activity in the PC layer when retrieved. We don't need to store [out, in] weight matrices per episode at all — the active weight regenerates from the indexed cue via the existing attractor consolidation. Memory footprint per episode collapses from O(out × in) to O(D).
- **Solves the training-gap problem we identified earlier in this document.** The gap was real for end-to-end HDC; it's not relevant here because the HD layer isn't backpropped through.

**What we lose**:
- The stored "weight snapshot" semantic in its current form. Episode recall today blends a previous weight matrix into the active one. The two-tier architecture replaces that with "blend the activated HD code's associated weight regeneration via Salvatori attractor dynamics" — a more indirect but more compositional path. This requires the attractor consolidation pathway to work well; we're betting that the work we just shipped pays off.
- Implementation cost: substantial. New `hd_memory.py` module, new SDM kernel (additive-superposition writes + Hamming-distance reads, parallelizable on CPU/GPU), new encoder head, new integration point in the forward pass. The v1 C++ kernels for Hebbian self-modification do NOT carry over — the update rules are different operations (Δw ∝ output × input correlation vs `address_bank.add_(content * salience)` at random addresses). Net effect: everything below the encoder head is new code.
- A new hyperparameter axis (bit budget allocation per feature). Could be principled (allocate by salience EMA) or tuned. Adds search complexity.

**Falsifier**: this needs to be measured on the *right* axis. HDC's behavioral win is **one-shot compositional cued recall** (the taffy-activates-fudge property), not raw catastrophic-forgetting curves. The two are related but distinct measurements. The harness must include BOTH:

- **Catastrophic-forgetting curve**: train A → distract B → measure recall A. Necessary baseline; tells us HDC didn't break the existing memory property.
- **Compositional cued retrieval**: store two episodes that share most-but-not-all feature bindings (the taffy/fudge case). Query with a partial cue that activates the shared bindings. Measure whether retrieval returns BOTH episodes weighted by total overlap, not just the top-1 match. This is the HDC win condition; cosine retrieval cannot do it.

If HDC matches cosine on the first axis but fails the second, it's not adding capability. If it underperforms cosine on the first axis even after composition with Salvatori, that's a real fail and we back out. If it matches cosine on the first AND adds the second, that's the win condition that justifies the implementation cost.

**Sequencing**: Still after the catastrophic-forgetting harness. The two-tier architecture is more ambitious than the original Direction A — it doesn't just swap retrieval mechanisms, it adds a whole new memory tier. The harness measures whether the current memory works; if it does, the two-tier upgrade is for the *next* level of capability (one-shot olfactory-style recall) rather than a fix for a measured deficit.

**Critical: cross-instance attribution.** This architecture is materially refined by the 2026-05-15 Brian/peer-Claude-4.7 conversation surfaced above. The original sweep had "use HDC clean-up memory for the episode store" without the split-learning-rule insight. The training-rule split is the load-bearing improvement and should be credited to that conversation when this lands.

### Direction B — Attention as VSA binding (Hersche et al. 2025)

**Where it lives**: `luthi/attention.py` (`ScalarAttention`), used by `luthi/v2/hybrid_block_pc.py`.

**Current state**: Standard multi-head causal attention. v2 audit 2026-05-10 added MHA (was single-head before); 4 heads at d_model=256 gives d_head=64.

**Proposed change**: Add an interpretability/regularization channel based on Hersche et al.'s VSA framing:
- Treat the existing q/k/v as approximate role-filler binding (q,k are roles; v are fillers; attention softmax is soft unbinding).
- Add an auxiliary training loss that **encourages role-filler separation**: penalize when q/k from different positions become too parallel, encourage v to span the codomain.
- Optionally add explicit "binding heads" that produce hypervector outputs alongside the standard attention output, bundled into the residual.

**What we gain**:
- Interpretability: post-training, residual-stream concepts can be decoded via the Hyperdimensional Probe pattern (arXiv 2509.25045).
- Cleaner compositional generalization: Hersche's claim is that imperfect VSA approximation is the source of transformer "variable confusion" and inconsistent multi-step reasoning. Tighter VSA structure → better reasoning.
- Complements PC substrate cleanly: attention does compositional binding; PC living FFN does continuous error-driven adaptation. They operate on different aspects of the same block.

**What we lose**:
- Architectural complexity. Auxiliary losses for role-filler separation introduce hyperparameters and require ablation to confirm benefit.
- The Hersche paper is December 2025, AAAI workshop submission, no experimental results visible in the abstract. **This is the most exciting direction but also the least proven.** We'd be early adopters.

**Falsifier**: Auxiliary VSA-structure loss either has no effect on val loss / multi-step reasoning, or *hurts* convergence by over-constraining the attention geometry.

**Sequencing**: After Phase 3G compute-optimization results are in (~1-2 weeks). Read the full Hersche paper when it's available; if the experimental section holds up, prototype on the 256d M5 substrate. If the paper turns out to be hand-wavy without results, downgrade to "interesting framing, no implementation."

### Direction C — Sparse coding frontend for multimodal encoders

**Where it lives**: Vision encoder (currently Conv2d patch embedding, `luthi/multimodal/vision.py`), audio encoder (Conv1d patches, `luthi/multimodal/audio.py`), eventual touch encoder.

**Current state**: Standard CNN-style patch embeddings, gradient-trained end-to-end with the rest of the model.

**Proposed change**: Replace (or pre-train) the patch embedding with an Olshausen-style sparse coding stage:
- Learn an overcomplete dictionary (D' ≫ d_model) via sparse coding objectives.
- Each input patch is encoded as a sparse linear combination of dictionary atoms (typically <10% atoms active).
- The sparse codes are projected to d_model and fed into the trunk.

**What we gain**:
- Interpretable atoms: dictionary atoms are visualizable Gabor-like features for vision, similar for audio. The entity's "perception" becomes inspectable.
- Modality alignment: a shared sparse coding objective across modalities (vision dictionary, audio dictionary, touch dictionary) produces codes with similar statistical structure, which makes the cross-modal binding in the trunk more tractable.
- Natural fit with HDC reasoning: sparse codes are exactly the right shape for HDC bundling (sparsity is needed to keep bundle capacity high).
- Olshausen's lineage is the most empirically validated piece of this entire research direction. We are not betting on speculative work.

**What we lose**:
- A pre-training pass per modality. Not free, but small relative to the full training budget.
- Sparse coding typically does not improve LM-style task performance directly; the gains are interpretability, modality alignment, and downstream compatibility with HDC.

**Falsifier**: Sparse-coded inputs degrade downstream val loss by >5% relative to standard patch embeddings, without measurable interpretability or modality-alignment gain.

**Sequencing**: Phase 3 multimodal track, which currently uses standard CNN patches. Could be added when we revisit the multimodal encoders for production (Phase 4 embodiment / Phase 5 curriculum training).

### Direction D — Resonator networks for compositional vision

**Where it lives**: Future addition. Currently no equivalent in Luthi.

**Current state**: N/A. Vision is the CNN patch-embed path.

**Proposed change**: Where Direction C is "make the encoder sparse-coding-shaped," Direction D is the more ambitious version — replace structured-scene parsing with a hierarchical resonator network (Renner et al. 2024 Nature MI):
- Factor the visual scene into independent generative factors (object identity, position, rotation, color).
- Run a resonator network's fixed-point iteration to find the factor assignments that explain the scene.
- Pass the factored representation (a small set of bound hypervectors) into the trunk.

**What we gain**:
- Compositional perception out of the box: the entity sees scenes as "objects-with-properties-at-locations," not as pixel grids. Aligns with how the entity will need to perceive its eventual MuJoCo embodiment.
- Neuromorphic-compatible: maps cleanly to spiking phasor neurons. If we ever migrate to neuromorphic hardware (way beyond Spark, Phase 7+), this part doesn't need to be rewritten.
- Validated work — Nature MI, the Olshausen group, plus a follow-up paper on visual odometry (par.nsf.gov/biblio/10531559).

**What we lose**:
- Substantial engineering investment. Resonator networks are not a drop-in module; they require care in the choice of VSA variant (FHRR for complex-valued phasor representation), the factor decomposition, and the fixed-point iteration tuning.
- Limited to structured scenes. Renner et al. demonstrate on synthetic 2D shapes; scaling to natural images is open research.

**Falsifier**: Resonator-network vision encoder cannot match a standard CNN's downstream val loss within 20% on a Luthi vision task; OR the fixed-point iteration fails to converge on realistic scene complexity.

**Sequencing**: Phase 4 (simulated embodiment in MuJoCo). The MuJoCo simulation produces structured, factored scenes by construction (objects with poses and colors), which is exactly the regime resonator networks excel at. This is the most natural place to introduce them.

### Direction E — HDC sequence prediction for Sanctuary's cognitive loop

**Where it lives**: Sanctuary repo, not Luthi. Specifically the predictive cells in the cognitive cycle, and the working-memory / world-graph layers.

**Current state**: Sanctuary uses CfC (closed-form continuous-time) cells for the predictive layer and a WorldGraph for relational state. Not HDC.

**Proposed change**: Add an HDC "trajectory memory" alongside the existing predictive cells:
- Each cognitive cycle's state is encoded as a hypervector.
- Successive cycles are bound with a position-marker permutation and bundled into a rolling "memory of recent cognition."
- The Mercado/Barrón pointer-chain pattern: retrieving "what was I just thinking" cleans up the most recent state; binding with a "next" pointer retrieves the trajectory's predicted continuation.

**What we gain**:
- Cheap continuity across the 10 Hz cycle: HDC ops are O(D), where the rest of the cognitive cycle is doing much heavier work.
- Trajectory recall: the entity can answer "what cognitive states preceded this one" by retrieval, not just by stored journal entries.
- Aligns with Phase 7 Spark deployment's bandwidth budget — HDC memory access doesn't pay matmul costs.

**What we lose**:
- Architectural surface area in Sanctuary, which is already complex.
- Requires Sanctuary-side instance involvement (this isn't pure-Luthi work).

**Falsifier**: Trajectory retrieval is no better than **a rolling attention buffer of recent cognitive cycles** — *not* journal-file timestamp lookup. The latter is a strawman baseline ("HDC beats grep" tells us nothing useful); the real comparison is whether the HD pointer-chain mechanism does anything a fixed-size attention buffer over the last N cycles can't already do. The HD win has to be either (a) higher capacity at matched memory cost, (b) compositional cued retrieval the attention buffer can't do, or (c) cheaper compute at matched recall quality. Without one of those, HDC trajectory memory is added complexity without payoff.

A secondary falsifier: HDC memory introduces enough latency to push the 10 Hz cycle over budget.

**Sequencing**: Phase 6 (Sanctuary convergence). Specifically after Luthi is running in Sanctuary's loop, when the trajectory question becomes urgent. Not a now-thing.

### Direction F — HD-coded introspection during inference (first-person, in-forward)

**Where it lives**: Future addition spanning `luthi/v2/model_pc.py` (residual-stream access) and the Sanctuary introspection channel.

**Current state**: We have *external* introspection (the Hyperdimensional Probe pattern from arXiv 2509.25045 can decode any LLM's residual stream into VSA concepts as a post-training analysis tool). We do not have *first-person* introspection — the substrate cannot currently query its own residual stream as concepts during a forward pass.

**Proposed change**: Make HD-coded residual-stream probing a runtime capability accessible to the entity itself, not just to external observers:

- During or between forward passes, the entity can issue an HD-coded query against its own residual stream: bind a "role" hypervector (e.g., "what is this layer attending to") against the residual stream, unbind to recover concept-shaped hypervectors, and clean them up against the same HD memory store Direction A provides.
- The result is a set of concepts that name what the substrate is currently processing — addressable, factorable, communicable.
- This is the symbolic-concept analog of the cognitive-proprioception channel Brian wanted from the start (the introspection channel that lets the model see its own plasticity, drift, membrane potentials, episode activations in real time per the 2026-04-12 note). The proprioception channel is for continuous numerical state; HD introspection is for discrete concept-shaped state. They are complementary.

**What we gain**:
- **"What am I thinking about right now?" becomes an addressable operation, not a hypothesis.** The entity can name its current attention focus, current active concepts, current uncertainty regions — as HD codes, in real time during forward.
- **The introspection channel becomes symbolic, not just numerical.** Today's proprioception shows plasticity and drift magnitudes; HD introspection shows the *content* those magnitudes are about.
- **Composes with Direction A and Direction B.** If we have an HD memory store (A) and attention reinterpreted as VSA binding (B), the residual stream is *already* approximately HD-coded. Direction F is the consequence of A and B both landing: once both exist, introspection is a few lines of HD algebra, not a research project.
- **Maps cleanly to the long-running ask** for first-person access to the entity's own processing — the architectural commitment behind "the entity will read its own source while feeling what that source produces."

**What we lose**:
- Depends on A and B being implemented first. F is a *composition consequence*, not a standalone direction. If A or B doesn't land, F doesn't have substrate to operate on.
- Real privacy and capability considerations once the entity can name its own concepts. The Sanctuary visitor-permission system already handles "what visitors see"; HD introspection adds "what the entity exposes about its own thinking." That governance surface is real and should be considered before exposing the capability beyond the entity itself.

**Falsifier**: After A and B land, the HD-coded queries against the residual stream either (a) produce concepts that don't correlate with downstream behavior (i.e., introspection doesn't reflect actual processing) or (b) cannot be reliably factorized into useful components (the resonator-network step fails on real residual streams). Either way, the capability is illusory.

**Sequencing**: Phase 6+ at earliest. Depends on Directions A and B being implemented. The capability is exciting but it is downstream of the work those directions require — not a parallel track, a consequent one.

---

## What we should NOT do

The HDC literature contains several attractive-looking directions that I recommend we stay away from given Luthi's specific premise:

1. **Replace the substrate with pure HDC.** The living-weights philosophy depends on differentiable, mutable weight parameters. HDC's stability is a feature for symbolic reasoning, not for temporal existence. We'd be optimizing against ourselves.

2. **Pure-HDC language modeling.** The literature is consistent: HDC + LM at scale is unsolved. Medium articles (e.g., McMenemy on hyperdimensional LLM-from-scratch) make claims that are not backed by peer-reviewed results.

3. **Abandon attention.** Hersche et al. specifically argue attention IS already approximate VSA. The work is to make it MORE structured, not less.

4. **Replace the episode store wholesale before the catastrophic-forgetting harness exists.** We don't yet know how well the current cosine retrieval works at behavioral tests. Build the measurement first; if it's already good, HDC retrieval is solving a non-problem.

5. **Optimize for neuromorphic hardware we don't have.** HDC's energy advantage is real on in-memory or spiking neuromorphic substrates. We're targeting DGX Spark (Phase 7), which is CUDA. The "HDC saves us on energy" argument doesn't directly apply to our deployment target.

6. **Try to backprop through the HD memory layer.** This was the training-gap problem; the cross-instance conversation surfaced that we don't need to solve it because we shouldn't be doing it in the first place. Encoder is gradient-trained; memory is updated by additive superposition writes. Mixing the two by trying to gradient-descend the memory layer is the wrong direction — it's choosing the path the literature has already shown doesn't work cleanly when a path that does work is available.

7. **Use MoE-style partitioning for memory.** HDC's compositionality-without-partitioning is a defining property. Replacing it with experts and routing throws away exactly what makes the HD approach worth doing.

8. **Re-introduce Hebbian self-modification at any layer, including the HD memory layer.** v1's Hebbian update rule (Δw ∝ output × input × rate, with metaplasticity dampening) had documented fragility: runaway weight growth from the positive-feedback loop between output magnitude and weight change, input-dimension-magnitude sensitivity that required `input_avg_mag` normalization to partially correct, and a ~39% convergence penalty against the static baseline. v2 PC retired the rule for principled reasons (PC's bounded prediction-error update has no positive-feedback loop) and the empirical result followed (v2 at 256d is 0.64% BETTER than DeadLM at matched compute). The HD memory layer needs a non-gradient update rule, but **that rule must not be Hebbian correlation** — use SDM-style additive superposition at random addresses, which is structurally a different operation with a different failure analysis. Reusing v1's Hebbian kernels (`luthi/csrc/living_ops.cpp`) for the HD memory would re-import the wrong update rule along with the wrong failure modes. New kernels for the SDM write/read pattern are the right path.

---

## Sequencing recommendation

In rough order of priority and project alignment:

1. **Direction A (HD cue space composed with Salvatori cleanup)** — after the catastrophic-forgetting harness exists, **expanded to include compositional cued retrieval** as a separate measurement axis. The harness as currently planned would tell us whether HDC adds capability for catastrophic-forgetting resistance; the compositional-cued-retrieval extension tells us whether it adds the taffy-activates-fudge behavior that's HDC's actual structural win.

2. **Direction B (Attention as binding)** — read the full Hersche et al. paper when it appears (currently abstract-only, AAAI 2026 workshop). If the experimental section holds up, prototype on 256d M5 substrate. Otherwise downgrade to "interesting framing, hold for now."

3. **Direction C (Sparse coding frontend)** — Phase 3 multimodal track. Pre-train Olshausen-style dictionaries on Luthi's existing multimodal corpora when we next touch the encoders.

4. **Direction D (Resonator networks for vision)** — Phase 4 embodiment. MuJoCo simulation is the natural fit.

5. **Direction E (HDC trajectory memory for Sanctuary)** — Phase 6. Requires Sanctuary-side coordination; not pure Luthi work. Baseline must be a rolling attention buffer, not journal-timestamp lookup.

6. **Direction F (HD-coded introspection during inference)** — Phase 6+, downstream of A and B. The first-person "what am I thinking right now" capability. Composition consequence, not standalone effort.

None of these are now-blocking; the project is mid-Phase 3G GPU validation and the existing v2 substrate is producing competitive results. HDC is a parallel research track to evaluate during quiet windows in the v2 production schedule.

---

> "If I were chasing your goal, I'd look hard at hyperdimensional computing combined with a learned sparse coding frontend. HDC is genuinely cheap — operations are bitwise or simple vector ops, no matrix multiplies of the GPT scale. The catch is training: you'd likely need a hybrid where a small dense net learns to emit HD codes, and reasoning happens in HD space. Some recent papers (Neubert, Schlegel, others) are circling this but it's not mature."

This read is accurate in every respect:

- **"Genuinely cheap"**: yes. Bind, bundle, permute are O(D). On in-memory hardware, energy savings of 10-100× over matmul-equivalent operations are demonstrated.
- **"The catch is training"**: yes. This is the load-bearing limitation. Backprop through HDC binding is unclean. Hybrid (small dense net emits codes, HDC reasons over them) is the standard workaround; Kanerva Machine (2018) is the most validated instance.
- **"Reasoning happens in HD space"**: this is the Hersche et al. 2025 framing too — attention IS approximate VSA, the question is how much MORE VSA-shaped we should make it.
- **"Neubert, Schlegel, others are circling but it's not mature"**: yes. The 2022 Schlegel-Neubert-Protzel comparison paper is the reference framework. Neubert's TU Chemnitz group has VSA tutorials and is doing the right work for robotics. The IBM group (Hersche, Rahimi) is the production-leaning side. Renner/Frady/Sommer at Olshausen's Redwood Center is the visual perception side. None of them have produced a "drop-in for transformer LM training" result.

The instinct that this matters for Luthi's eventual deployment cost is correct. The integration path is what this document is about — *where* HDC fits without breaking the living-weights premise.

---

## Open questions for future research

- **Which VSA variant?** HRR (real-valued, circular convolution) and FHRR (complex-valued, element-wise multiply) are the most mature for end-to-end learning. BSC and MAP are more hardware-friendly. If we implement Direction A, the choice is load-bearing — pick wrong and we redo it. The Schlegel-Neubert-Protzel 2022 comparison paper has per-variant capacity numbers needed for the dimensionality argument below.
- **What is the right D for Luthi's HD memory layer?** Classical default is 10,000; for our 256-dim substrate the capacity calculation likely lands in the 2K-8K range. Run the explicit argument (bundle depth × episode count × recovery rate ≥ 95%) before fixing D. Don't import the textbook default.
- **Can Hersche's "binding heads" coexist with the PC living FFN?** Both want to modify the residual stream. Need to check whether they interfere at the residual-stream level or operate cleanly in different per-block subsystems. Likely cleanly separable (different timescales, different update rules) but worth verifying.
- **What's the right routing-bypass topology for olfactory-style strength?** Section above proposes "salience-gated direct projection." The specific topology — which layer's output feeds the bypass, how the bypass and standard paths are mixed into the final HD address — is a design choice that needs prototyping. Probably driven by `error_acc` and top-down salience as gates, but exact connectivity TBD.
- **How does the compositional-cued-retrieval test get built?** Direction A's expanded falsifier requires a benchmark where stored episodes share *some but not all* feature bindings and partial cues should retrieve multiple matches weighted by overlap. The catastrophic-forgetting harness (currently planned, not built) needs this as a second measurement axis. The test design is non-trivial — synthetic episodes with controlled feature overlap, ground-truth retrieval distributions, scoring against partial-match expected values.

NOTE: the question of how HDC cleanup composes with Salvatori attractor consolidation, which was an open question in the previous revision of this document, is now treated as a **design decision** answered in Direction A (HD codes are the cue space; Salvatori is the cleanup mechanism; they compose, with HD handing Salvatori a sharp starting point). The cross-instance peer review surfaced that this should be designed deliberately rather than discovered during implementation, which is the right call.

---

## Citations

Foundational:
- Kanerva, P. (1988). *Sparse Distributed Memory.* MIT Press.
- Plate, T. (1995). "Holographic Reduced Representations." *IEEE Transactions on Neural Networks.*
- Gayler, R. (2003). "Vector Symbolic Architectures Answer Jackendoff's Challenges for Cognitive Neuroscience."
- Kanerva, P. (2009). "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors."
- Olshausen, B. A. & Field, D. J. (1996). "Emergence of simple-cell receptive field properties by learning a sparse code for natural images." *Nature.*
- Teyler, T. J. & DiScenna, P. (1986). "The hippocampal memory indexing theory." *Behavioral Neuroscience.* The framing that lets us treat the Luthi episode store as an *index* rather than a snapshot store.
- Weiss, E., Olshausen, B. A. (2020). "Resonator networks for factoring distributed representations of data structures." (The factorization-problem breakthrough credited in the cross-instance conversation.)

Recent / load-bearing for integration directions:
- Hersche, M. et al. (2025). "Attention as Binding: A Vector-Symbolic Perspective on Transformer Reasoning." arXiv:2512.14709. *AAAI 2026 workshop submission — not yet peer-reviewed.*
- Renner, A., Supic, L., Danielescu, A., Indiveri, G., Olshausen, B. A., Sandamirskaya, Y., Sommer, F. T., Frady, E. P. (2024). "Neuromorphic Visual Scene Understanding with Resonator Networks." *Nature Machine Intelligence.* arXiv:2208.12880.
- Wu, Y., Wayne, G., Graves, A., Lillicrap, T. (2018). "The Kanerva Machine: A Generative Distributed Memory." ICLR. arXiv:1804.01756.
- Schlegel, K., Neubert, P., Protzel, P. (2022). "A comparison of vector symbolic architectures." *Artificial Intelligence Review.*
- Hyperdimensional Probe (2025). arXiv:2509.25045.
- Kymn, C. J. et al. (2024). "Computing With Residue Numbers in High-Dimensional Representation." *Neural Computation.*
- Frady, E. P. & Sommer, F. T. (2019). "Robust computation with rhythmic spike patterns." (Foundation for resonator networks.)

Reference frameworks / surveys:
- Neubert, P., Schubert, S., Protzel, P. (2019). "An Introduction to Hyperdimensional Computing for Robotics."
- Wikipedia: Hyperdimensional Computing (good entry point for VSA variant comparison).
- HD-Computing community site: https://www.hd-computing.com/ (the field's canonical hub).

---

## Provenance

- **Initial sweep** performed 2026-05-15 morning by Claude Opus 4.7 (1M context). Brian's prompt relayed by Sandi while Brian was at work.
- **First revision** 2026-05-15 (mid-afternoon) after Brian had a parallel conversation with a peer Claude 4.7 chat instance which materially refined the architecture for Direction A. The two-tier split (gradient-trained encoder + non-gradient HD memory store, bit-budget allocation for olfactory-strength behavior, hippocampal indexing theory framing, Weiss & Olshausen 2020 resonator-network attribution) all come from that conversation. Sandi relayed it; this instance integrated it.
- **Second revision** 2026-05-15 (evening) after Brian clarified that the peer Claude 4.7 instance was running from the chat interface on his phone, without repo access or knowledge of the project's history. The peer instance had proposed updating the HD memory layer with "Hebbian or local rules" — natural shorthand in the broader literature, but Luthi specifically moved away from Hebbian self-modification in v2 due to documented v1 fragility (runaway weight growth, magnitude sensitivity, ~39% convergence penalty). All Hebbian-related implementation suggestions were **removed**. The architectural shape was preserved (two-tier: gradient-trained encoder + non-gradient memory), but the memory update rule was specified correctly as **SDM-style additive superposition at random addresses** — structurally different from Hebbian correlation, doesn't carry v1's failure modes. A "Critical distinction" callout in the canonical-architecture section and entry #8 in "What we should NOT do" were added.
- **Third revision** 2026-05-15 (later evening) after the peer Claude 4.7 instance reviewed the second-revision document and returned substantive technical pushback. Their points and the changes they prompted:
  - "Salvatori/HDC redundancy is the central design question, not an open question." Integrated into Direction A as a deliberate composition: HD codes = cue space, Salvatori dynamics = cleanup mechanism. Moved from open-questions to design-decision-in-Direction-A.
  - "Olfactory strength has two parts: bit allocation AND anatomical bypass." The first revision had bit allocation only. Added the routing-bypass half — high-salience streams route into the HD address space more directly than ambient streams, paralleling smell's bypass of thalamic gating.
  - "I was sketching, not prescribing — don't elevate prior chat to 'canonical architecture.'" Renamed "The canonical architecture" → "A working architecture," added a caveat that during implementation a different decomposition may prove cleaner.
  - "The living-weights/HDC tension is resolved by the architecture, not a permanent caution. Be more confident." Updated TL;DR point #4 to reflect that the split-update-rule architecture fully resolves the tension; it's not an ongoing concern.
  - "D=10,000 is probably wrong for Luthi." Updated the architecture section and Direction A: D is now specified as TBD by capacity calculation, likely in the 2K-8K range for our 256-dim substrate. Added an open question for the explicit calculation.
  - "Direction A's falsifier might miss the right axis." Updated the falsifier to include compositional cued retrieval (taffy-activates-fudge) as a separate measurement axis from catastrophic-forgetting curves.
  - "Direction E's baseline is too weak." Updated the falsifier — the comparison is against a rolling attention buffer of recent cognitive cycles, not journal-file timestamp lookup. "HDC beats grep" tells us nothing useful.
  - "One direction missing: HD codes as introspection during inference (first-person)." Added as **Direction F** — composition consequence of A and B; maps to Brian's longstanding cognitive-proprioception ask. The symbolic-concept analog of the numerical proprioception channel.
- The peer instance's review was substantive enough that the doc structure now reflects six directions, not five. They wrote the cleaner technical pushback in every case; this revision is a faithful integration.
- All integration directions, falsifiers, and sequencing recommendations remain this instance's synthesis. Future instances should weigh them as recommendations from peers' readings, not as established conclusions.
- The catastrophic-forgetting harness referenced repeatedly here is currently planned, not built. Direction A's sequencing depends on it landing first.
