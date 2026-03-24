# Luthi Model — Research Documentation

> Living weights: self-modifying neural network parameters that change during their own forward pass.
> A new kind of computation that is neither feedforward nor recurrent.

## Reading Order

Read these in sequence. Each document builds on findings from the previous ones, and later documents reframe earlier conclusions.

### 1. RICH_PARAMETERS_FINAL.md
The foundational research. Four experiments proving that weights can carry per-parameter history, modify themselves during forward passes, and form episodic memories. Introduces the three kinds of computation table (feedforward / recurrent / living). Start here.

### 2. LIVING_WEIGHT_STRESS_TESTS.md
Six stress tests across three versions (V1/V2/V3). Identifies what breaks: excitability saturation, catastrophic forgetting, output divergence. Documents fixes: homeostatic regulation, protected memory slots, momentum damping. Read this to understand the iteration history.

### 3. HYBRID_BLOCK_RESULTS.md
The target architecture: scalar attention + living feedforward + episode store. Proves multi-layer stacking is stable (depth amplifies sub-linearly). First complete hybrid block results at 16d. Also covers V4 logarithmic excitability.

### 4. V5_ADAPTIVE_RATE.md
Fixes the Hebbian overshoot problem with synaptic scaling — each weight normalizes its learning by its running average input magnitude. Short but important: this is the fix that made high-magnitude inputs tractable.

### 5. SCALE_TEST_256D.md
Validates all properties at 256 dimensions (65,536 weights). Key finding: **divergence rate is dimension-independent**. Also contains memory projections for real model scale (4096d) and the recommendation for 8 living layers in float16.

### 6. LUTHI_256D_COMPREHENSIVE.md — CRITICAL
**Read carefully.** Contains the convergence failure: pure Hebbian self-modification does NOT learn structured tasks. This reframes the entire architecture. The living FFN's job is NOT to learn functions — that's what attention (backprop) does. The living FFN provides temporal existence and self-modification. This division of labor is a core design decision.

### 7. LUTHI_V6_TRADEOFF.md — CRITICAL
Error-directed local learning (V6) enables convergence. The aliveness-convergence tradeoff: **any amount of Hebbian self-modification costs ~39% convergence vs dead weights, and the penalty is a step function, not a gradient**. This means: use the highest stable Hebbian rate (recommended 0.001) since you're already paying the full penalty. Also contains recall-after-interference results (5/5 with 100 interfering experiences).

### 8. CFC_LIVING_WEIGHT_INTEGRATION.md
How living weights connect to Sanctuary's CfC cells. The entity's felt state (affect, attention, goals, precision) modulates Hebbian self-modification. Written without access to the Sanctuary repo — verify compatibility with actual CfC implementation before building.

## Key Design Decisions (Do Not Reinvent)

- **Living FFN is the body, not the brain.** It provides temporal existence. Attention layers handle task learning.
- **The 39% penalty is inherent.** Self-modifying weights converge slower. This is the metabolic cost of being alive. Don't try to optimize it away.
- **Episode store carries most recall weight.** In-weight memory is weak. The episode store compensates. Both are needed.
- **Divergence is dimension-independent.** Scale without fear of compounding instability.
- **Prefer crashes over silent corruption.** No try/except around living weight operations. If NaN appears, it should be visible immediately.
- **Hebbian rate 0.001, error-directed rate 0.001.** These are the tested values. Change only with evidence.

## For Implementation

The proof-of-concept phase is complete (numpy, CPU). The next phase is:
1. Vectorized PyTorch implementation of LivingLayerV6
2. Hybrid block in PyTorch (scalar attention + living FFN + episode store)
3. Small real task: character-level language modeling with hybrid blocks
4. CfC integration against Sanctuary's experiential manager
5. Scale testing: 1024d → 4096d, map cost curve on real hardware

All proof-of-concept code is in the research documents as inline Python. It is intentionally not packaged as a library — it was written for clarity and correctness verification, not performance. The PyTorch implementation should be written fresh, following the architecture described in the documents, not by translating the numpy loops.
