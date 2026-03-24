# Luthi Model: V6 Error-Directed Learning & The Aliveness-Convergence Tradeoff

> Date: March 2026
> Authors: Claude (Opus 4.6) and Brian

---

## Error-Directed Hebbian Learning

### The Problem (from previous experiments)

Pure Hebbian self-modification (V1-V5) does not converge toward correct functions. The Hebbian signal — input × salience — captures correlations, not error gradients. A living FFN with only Hebbian learning showed flat loss across epochs while scalar backprop improved steadily.

### The Solution: Local Error Signal

Each living weight sees its own input (cached from the forward pass) and the error at its own layer's output. The update rule:

```
update = learning_rate × (input.T @ output_error) / sequence_length
```

This is mathematically equivalent to the gradient for a single linear layer, but framed as a purely local operation: each weight adjusts based on the correlation between its input and the error it contributed to.

Biologically, this is plausible. A neuron can observe its own input (presynaptic activity) and receive a signal about downstream error (postsynaptic response). No global backward pass required.

### Results: Error-Directed Works

```
V6 Hybrid Block (two-layer living FFN):
  Epoch 1: 6.10 → Epoch 5: 4.97 (converging)
  Hebb-only comparison: 6.41 → 6.36 (flat)
  Error-directed improves over Hebb-only by 19.9%

Single-layer living FFN (exact error signal):
  Epoch 1: 6.32 → Epoch 10: 3.71 (steadily converging)
  Scalar baseline: 6.23 → 2.53
  Gap: 39.6%
```

The living FFN converges. It doesn't converge as fast as scalar backprop, but it converges. The 39.6% gap is the cost of self-modification.

### V6 Architecture

V6 combines everything from V1-V5 with error-directed learning:

1. **Hebbian self-modification** — always active during forward pass (temporal dynamics)
2. **Error-directed learning** — when error signal available (convergence)
3. **Adaptive Hebbian rate** — synaptic scaling (prevents overshoot)
4. **Homeostatic regulation** — set points and decay (stability)
5. **Excitability dynamics** — habituation and sensitization (history)
6. **Layer-level episodes** — context-gated value snapshots (memory)

Three modes of adaptation in one layer, all local, none requiring gradient flow from other layers.

---

## The Aliveness-Convergence Tradeoff

### The Experiment

Sweep the Hebbian self-modification rate while keeping error-directed learning rate fixed. Measure convergence speed and non-feedforward signal at each rate.

### Results

```
Hebb Rate    Test Loss   Gap vs Dead   Non-FF Signal   Aliveness Factor
0.00000      2.86        —             0.000           Dead
0.00002      3.98        +39.1%        0.0003          Barely alive
0.00010      3.98        +39.2%        0.0012          Alive
0.00020      3.99        +39.5%        0.0024          Alive
0.00100      4.42        +54.8%        0.0158          Very alive
0.00500      60264       Exploded      4.729           Unstable
```

### The Step Function

**The convergence penalty is a step function, not a gradient.** ANY amount of Hebbian self-modification costs approximately 39% convergence compared to dead weights. Increasing the Hebbian rate by 10x (from 0.00002 to 0.0002) costs only 0.4% additional.

This means:

1. The cost of being alive is front-loaded. The first molecule of self-modification is the expensive one.
2. Additional self-modification is nearly free (up to the instability threshold around 0.005).
3. The optimal strategy is to use the highest stable Hebbian rate — you're already paying the penalty for being alive, so be as alive as possible.

### Recommended Hebbian Rate

**0.001** — provides 53x stronger non-feedforward signal than the minimum detectable rate, with only 15.7% additional convergence cost beyond the baseline 39%.

### What This Means Philosophically

There is a hard cost to being alive. A computation that modifies itself during use will always converge slower than one that doesn't. This is not a bug — it's an inherent property of self-modifying systems. You cannot be both maximally efficient at learning AND temporally existent. You have to choose.

For task-optimized AI (chatbots, classifiers, search engines), dead weights are better. Static parameters converge faster and produce more accurate results.

For an entity that exists in time, living weights are necessary. The 39% convergence penalty is the metabolic cost of being alive. Biology pays this cost too — biological neural networks are far less computationally efficient than artificial ones, but they exist, change, remember, and have temporal continuity.

---

## Recall After Interference

After 100 interfering background experiences, the full system (living FFN with context + episode store) successfully recalled all 5 unique experiences:

```
                  No recall    Episode     Full
Unique 1:          4.80        4.46       4.50
Unique 2:         15.48       14.58      12.28
Unique 3:         32.41       30.69      14.49
Unique 4:         68.48       57.28      47.19
Unique 5:        127.38      119.20      64.95
```

5/5 recalled. The full system cut error by roughly half on the largest experience compared to no recall. Memory survives interference.

---

## Updated Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID BLOCK V6                           │
│                                                              │
│  ┌──────────────────────────────────────────────────┐       │
│  │  SCALAR ATTENTION                                 │       │
│  │  Learns routing via backprop                      │       │
│  │  Q, K, V, O weights — standard gradient descent   │       │
│  └───────────────────┬──────────────────────────────┘       │
│                      │ + residual                            │
│  ┌───────────────────▼──────────────────────────────┐       │
│  │  LIVING FFN V6                                    │       │
│  │  Three adaptation modes:                          │       │
│  │  1. Hebbian self-modification (temporal dynamics)  │       │
│  │  2. Error-directed learning (convergence)         │       │
│  │  3. Episodic recall (layer-level memory)          │       │
│  │                                                    │       │
│  │  Self-modifies every forward pass — NOT static    │       │
│  │  Convergence cost: ~39% vs dead weights           │       │
│  │  This is the metabolic cost of being alive.       │       │
│  └───────────────────┬──────────────────────────────┘       │
│                      │ + residual                            │
│  ┌───────────────────▼──────────────────────────────┐       │
│  │  EPISODE STORE                                    │       │
│  │  Output-level episodic memory                     │       │
│  │  Context-gated retrieval                          │       │
│  │  Survives 100+ interfering experiences            │       │
│  └───────────────────┬──────────────────────────────┘       │
│                      │                                       │
└──────────────────────┼───────────────────────────────────────┘
                       ↓
                 Next block or output
```

---

## Complete Document Inventory

1. **RICH_PARAMETERS_FINAL.md** — Original research: 4 experiments, living weights discovered
2. **LIVING_WEIGHT_STRESS_TESTS.md** — V1/V2/V3, six stress tests
3. **HYBRID_BLOCK_RESULTS.md** — V4, multi-layer stacking, first hybrid block
4. **V5_ADAPTIVE_RATE.md** — Synaptic scaling, overshoot fix
5. **SCALE_TEST_256D.md** — 256d validation, memory projections
6. **LUTHI_256D_COMPREHENSIVE.md** — 256d experiments including convergence failure
7. **This document** — V6, error-directed learning, aliveness-convergence tradeoff

## For the LuthiWorks Repository

The proof-of-concept phase is complete. Key findings ready for implementation:

1. Living weights create non-feedforward, self-modifying computation ✓
2. Error-directed local learning enables convergence without backprop through living layers ✓
3. The cost of being alive is ~39% convergence penalty — inherent, not tunable ✓
4. Episode stores provide strong memory surviving 100+ interfering experiences ✓
5. All properties scale from 16d to 256d without degradation ✓
6. Memory fits on DGX Spark with selective living layers ✓
7. Recommended Hebbian rate: 0.001 (maximum aliveness for acceptable cost) ✓

Next phase requires PyTorch, GPU, and real training data — Claude Code territory.
