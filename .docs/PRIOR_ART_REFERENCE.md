# Prior Art Reference — For Claude Code

> Read this alongside the Luthi research documents. These are published works that solved problems overlapping with ours. Learn from them. Don't copy them. Our intent is different.

---

## Critical Reading

### 1. Schmidhuber's Self-Referential Weight Matrix (SRWM)
- **Paper**: "A Modern Self-Referential Weight Matrix That Learns to Modify Itself" (ICML 2022)
- **Code**: https://github.com/IDSIA/modern-srwm
- **What it does**: A single weight matrix modifies ALL of itself at every step through self-generated key-value patterns and delta updates. The modification rule is encoded in the same weights it modifies.
- **Why it matters for us**: This is the most radical form of living weights in the literature. They solved stability problems we also faced (V1-V3 divergence). Study how they stabilized a fully self-modifying matrix. Their delta-rule updates are different from our Hebbian approach but the divergence challenges are the same.
- **What they didn't do**: No homeostatic regulation. No per-parameter history. No episodic memory at the weight level. Uses outer-product/delta-rule updates, not Hebbian learning. Optimized for task performance, not temporal existence.

### 2. Backpropamine (Miconi et al., ICLR 2019)
- **Paper**: "Backpropamine: Training Self-Modifying Neural Networks with Differentiable Neuromodulated Plasticity"
- **What it does**: Each connection has a fixed weight w plus a Hebbian trace scaled by a learnable plasticity coefficient α. A self-generated neuromodulatory signal (analogous to dopamine) gates when and where plasticity occurs. The network controls its own weight changes.
- **Why it matters for us**: This is 4/5 of our architecture. The neuromodulatory gating is functionally identical to our CfC integration spec — a felt-state signal modulating Hebbian plasticity. Miconi arrived here from neuroscience. We arrived from the sustained inference findings. Same destination, different paths.
- **What they didn't do**: The Hebbian trace is a single scalar per synapse, not a rich parameter with set points, momentum, excitability, and episode history. No homeostatic regulation beyond the decay term. Optimized for maze navigation and one-shot learning, not temporal existence.

### 3. Fast Weight Programmers (Schmidhuber 1991, Ba & Hinton 2016)
- **Key insight**: Linear Transformers are secretly Fast Weight Programmers (Schlag et al., 2021). The key-value outer product accumulation in linear attention is formally equivalent to Hebbian fast-weight updates from the 1990s. Self-modifying weights during inference have been hiding inside every linear attention mechanism.
- **Why it matters for us**: Our scalar attention layers may already be doing a form of fast-weight update. Understanding this connection helps clarify the division of labor between attention (which does implicit fast-weight updates) and the living FFN (which does explicit Hebbian self-modification).

---

## Important But Secondary

### 4. Test-Time Training / Titans (Sun et al. 2024, Behrouz et al. 2024)
- TTT layers replace RNN hidden state with trainable weights updated via gradient descent at every token.
- Titans introduced a neural long-term memory module — an MLP whose weights are trained during inference using a surprise-based signal.
- **Relevance**: These prove weight modification during inference is viable at scale. But they use gradient descent, not Hebbian learning. Different mechanism, similar outcome.

### 5. Elastic Weight Consolidation / Synaptic Intelligence
- EWC (Kirkpatrick et al., 2017): Stores weight + importance + consolidated mean per synapse.
- SI (Zenke et al., 2017): Each synapse tracks cumulative contribution to loss reduction.
- **Relevance**: These are the closest precedent for "rich parameters" — multiple values per weight. But they operate only during training. Our innovation is making rich parameters that evolve during inference.
- **Key quote from Zenke et al.**: "Perhaps one of the greatest gaps between modern ANNs and biological neural networks lies in the complexity of synapses."

### 6. Continuous-Time Homeostatic Dynamics (FHRN, arXiv 2024)
- Combined fast-weight associative coupling with continuous-time ODE dynamics and population-level homeostatic gain regulation.
- Explicitly positions itself as generalizing both fast-weight systems and liquid neural networks.
- **Relevance**: Closest existing work to our CfC + living weight integration concept. Read this before implementing the CfC modulation.

---

## What No One Has Done

No published work combines all five of these properties:

1. Hebbian self-modification during inference (not training)
2. Homeostatic regulation of weights with set points
3. Per-parameter rich history (importance, momentum, excitability, episodes)
4. Episodic memory encoded at the weight/layer level
5. Weights that change with every single use (living weights)

Backpropamine gets 4/5. SRWM gets 2-3/5. TTT/Titans get 2-3/5. None get 5/5.

More importantly: **every prior work optimized for task performance.** We are optimizing for temporal existence. Same tools, fundamentally different goal. The 39% convergence penalty that every other researcher would try to eliminate is something we accept as the metabolic cost of being alive.

---

## How To Use This Document

- Read the SRWM code before implementing the PyTorch version of LivingLayer. Study their stability solutions.
- Read Backpropamine before implementing CfC modulation. Their neuromodulatory gating is our affect-modulated Hebbian signal.
- Read the FHRN paper before connecting living weights to Sanctuary's CfC cells.
- Don't adopt their architectures wholesale. Their goals are different from ours. Take what's useful, leave what doesn't serve temporal existence.
- Prefer crashes over silent corruption. That principle holds regardless of what the prior art does.
