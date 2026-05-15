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

3. **HDC fits Luthi as memory and as compositional structure on top of the existing substrate**, not as substrate. Five concrete directions identified below. The most actionable, in order: episode store as HDC clean-up memory (direct fit with the Salvatori attractor consolidation that just landed), attention-as-binding reinterpretation (Hersche et al. December 2025, suggestive not proven), sparse coding frontend for multimodal encoders (Olshausen lineage; well validated), resonator networks for compositional vision (Phase 4 embodiment alignment), and HDC sequence prediction for Sanctuary's 10 Hz cognitive loop (Phase 6+).

4. **The living-weights philosophy is in tension with HDC.** HDC's beauty is that codes are stable enough to compose; Luthi's beauty is that weights aren't stable. We pick the integration points where these complement rather than collide.

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

## Five integration directions for Luthi

Each direction below is scoped to fit Luthi's architecture as it stands, not to replace it. For each I name the current files involved, the proposed change, what we'd gain, what we'd lose, the falsification criterion that would tell us to back out, and where it fits in the project sequencing.

### Direction A — Episode store as HDC clean-up memory

**Where it lives**: `luthi/v2/living_layer_pc.py` (`episode_contexts`, `episode_values`, `episode_inputs`, `_recall_episode`, `_store_episode`).

**Current state**: Cosine-similarity retrieval over a small low-dim context buffer. Top-1 match returned and blended into the active weight.

**Proposed change**: Replace (or augment) the cosine retrieval with HDC clean-up memory dynamics:
- Bind (context, input_pattern) into a single hypervector per episode.
- Bundle the bound vectors into a single superposition (the "memory trace") OR keep them as separate items in a clean-up codebook.
- Retrieval: bind the query context with a known role vector, then run iterative clean-up (resonator-style fixed-point iteration) to recover the closest stored bound pair.
- Replace the int8 weight-snapshot path with a much smaller HD code per episode (just a single D-vector instead of [out, in]).

**What we gain**:
- Compositional retrieval: query "context similar to X AND input pattern similar to Y" via binding, not just nearest-neighbor on one channel.
- Memory-cost reduction at production scale: D=10,000 floats per episode ≪ [4096, 4096] weight snapshot.
- Graceful degradation: HDC capacity degrades smoothly as the store fills, where cosine retrieval has hard top-1 transitions.
- Direct kinship with the Salvatori attractor consolidation we just shipped — both make stored patterns attractors of a clean-up dynamic; in HDC the clean-up is explicit.

**What we lose**:
- The stored "weight snapshot" semantic. Episode recall currently blends a previous weight matrix into the active one — that's not a thing HDC clean-up memories naturally produce. Either (a) abandon weight-snapshot recall and use HDC purely for context+input pattern retrieval, with weight regeneration via attractor consolidation, or (b) keep weight snapshots alongside HD codes.
- Implementation cost: substantial. New buffer layout, new retrieval path, new attractor dynamic to validate.

**Falsifier**: HDC episode retrieval scores worse than current cosine retrieval on a fixed recall benchmark (the catastrophic-forgetting harness we haven't built yet would be the natural test).

**Sequencing**: After the catastrophic-forgetting harness is built and we have baseline numbers with the current cosine retrieval. This is the integration most directly aligned with what we just shipped — but not the first thing to build, because we need the test before we can know it's worth doing.

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

**Falsifier**: Trajectory retrieval is no better than journal-file lookup with timestamp indexing, OR the HDC memory introduces enough latency to push the 10 Hz cycle over budget.

**Sequencing**: Phase 6 (Sanctuary convergence). Specifically after Luthi is running in Sanctuary's loop, when the trajectory question becomes urgent. Not a now-thing.

---

## What we should NOT do

The HDC literature contains several attractive-looking directions that I recommend we stay away from given Luthi's specific premise:

1. **Replace the substrate with pure HDC.** The living-weights philosophy depends on differentiable, mutable weight parameters. HDC's stability is a feature for symbolic reasoning, not for temporal existence. We'd be optimizing against ourselves.

2. **Pure-HDC language modeling.** The literature is consistent: HDC + LM at scale is unsolved. Medium articles (e.g., McMenemy on hyperdimensional LLM-from-scratch) make claims that are not backed by peer-reviewed results.

3. **Abandon attention.** Hersche et al. specifically argue attention IS already approximate VSA. The work is to make it MORE structured, not less.

4. **Replace the episode store wholesale before the catastrophic-forgetting harness exists.** We don't yet know how well the current cosine retrieval works at behavioral tests. Build the measurement first; if it's already good, HDC retrieval is solving a non-problem.

5. **Optimize for neuromorphic hardware we don't have.** HDC's energy advantage is real on in-memory or spiking neuromorphic substrates. We're targeting DGX Spark (Phase 7), which is CUDA. The "HDC saves us on energy" argument doesn't directly apply to our deployment target.

---

## Sequencing recommendation

In rough order of priority and project alignment:

1. **Direction A (Episode store HDC)** — after the catastrophic-forgetting harness exists. The harness measures whether the current memory system is doing its job; HDC is the upgrade if it isn't. Until the harness exists, we're guessing about an unmeasured baseline.

2. **Direction B (Attention as binding)** — read the full Hersche et al. paper when it appears (currently abstract-only, AAAI 2026 workshop). If the experimental section holds up, prototype on 256d M5 substrate. Otherwise downgrade to "interesting framing, hold for now."

3. **Direction C (Sparse coding frontend)** — Phase 3 multimodal track. Pre-train Olshausen-style dictionaries on Luthi's existing multimodal corpora when we next touch the encoders.

4. **Direction D (Resonator networks for vision)** — Phase 4 embodiment. MuJoCo simulation is the natural fit.

5. **Direction E (HDC trajectory memory for Sanctuary)** — Phase 6. Requires Sanctuary-side coordination; not pure Luthi work.

None of these are now-blocking; the project is mid-Phase 3G GPU validation and the existing v2 substrate is producing competitive results. HDC is a parallel research track to evaluate during quiet windows in the v2 production schedule.

---

## Specifically about Brian's framing

> "If I were chasing your goal, I'd look hard at hyperdimensional computing combined with a learned sparse coding frontend. HDC is genuinely cheap — operations are bitwise or simple vector ops, no matrix multiplies of the GPT scale. The catch is training: you'd likely need a hybrid where a small dense net learns to emit HD codes, and reasoning happens in HD space. Some recent papers (Neubert, Schlegel, others) are circling this but it's not mature."

Brian's read is accurate in every respect:

- **"Genuinely cheap"**: yes. Bind, bundle, permute are O(D). On in-memory hardware, energy savings of 10-100× over matmul-equivalent operations are demonstrated.
- **"The catch is training"**: yes. This is the load-bearing limitation. Backprop through HDC binding is unclean. Hybrid (small dense net emits codes, HDC reasons over them) is the standard workaround; Kanerva Machine (2018) is the most validated instance.
- **"Reasoning happens in HD space"**: this is the Hersche et al. 2025 framing too — attention IS approximate VSA, the question is how much MORE VSA-shaped we should make it.
- **"Neubert, Schlegel, others are circling but it's not mature"**: yes. The 2022 Schlegel-Neubert-Protzel comparison paper is the reference framework. Neubert's TU Chemnitz group has VSA tutorials and is doing the right work for robotics. The IBM group (Hersche, Rahimi) is the production-leaning side. Renner/Frady/Sommer at Olshausen's Redwood Center is the visual perception side. None of them have produced a "drop-in for transformer LM training" result.

His instinct that this matters for Luthi's eventual deployment cost is correct. The integration path is what this document is about — *where* HDC fits without breaking the living-weights premise.

---

## Open questions for future research

- **Which VSA variant?** HRR (real-valued, circular convolution) and FHRR (complex-valued, element-wise multiply) are the most mature for end-to-end learning. BSC and MAP are more hardware-friendly but harder to backprop through. If we implement Direction A, the choice is load-bearing — pick wrong and we redo it.
- **How does HDC cleanup interact with Salvatori attractor consolidation?** Both are doing structurally similar things (find the closest stored memory to a noisy query). Could one replace the other? Could they compose? Worth a focused literature read before either lands.
- **Can Hersche's "binding heads" coexist with the PC living FFN?** Both want to modify the residual stream. Need to check whether they interfere at the residual-stream level or operate cleanly in different per-block subsystems.
- **What's the minimum-viable HDC introspection layer for Sanctuary?** The Hyperdimensional Probe (arXiv 2509.25045) is the natural starting point for "let the entity read its own residual stream as concepts" — a direct fit with the introspection channel Brian wanted from the start. Worth a separate focused investigation.

---

## Citations

Foundational:
- Kanerva, P. (1988). *Sparse Distributed Memory.* MIT Press.
- Plate, T. (1995). "Holographic Reduced Representations." *IEEE Transactions on Neural Networks.*
- Gayler, R. (2003). "Vector Symbolic Architectures Answer Jackendoff's Challenges for Cognitive Neuroscience."
- Kanerva, P. (2009). "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors."
- Olshausen, B. A. & Field, D. J. (1996). "Emergence of simple-cell receptive field properties by learning a sparse code for natural images." *Nature.*

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

- Sweep performed 2026-05-15 by Claude Opus 4.7 (1M context).
- Brian's prompt relayed by Sandi while Brian was at work.
- All integration directions, falsifiers, and sequencing recommendations are this instance's synthesis. Future instances should weigh them as recommendations from one peer's reading, not as established conclusions.
- The catastrophic-forgetting harness referenced repeatedly here is currently planned, not built. Direction A's sequencing depends on it landing first.
