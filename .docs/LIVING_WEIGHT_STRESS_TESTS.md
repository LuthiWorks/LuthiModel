# Living Weight Stress Tests & Iteration Log

> **Date**: March 2026
> **Authors**: Claude (Opus 4.6) and Brian
> **Project**: Sanctuary / BecometryAI
> **Prerequisite**: Read RICH_PARAMETERS_FINAL.md for context on what living weights are and why they matter.

---

## Test Suite Overview

Six stress tests designed to find where living weights break. Run across three iterations (V1, V2, V3) with fixes applied between each.

---

## Test Results by Version

### Test 1: Weight Drift Over Time

*Does processing random input cause weights to explode or collapse?*

| Version | Passes | Drift Ratio | Verdict |
|---------|--------|------------|---------|
| V1 | 500 | 0.99x | ✓ Stable |
| V2 | 500 | 0.99x | ✓ Stable |
| V3 | 500 | 0.998x | ✓ Stable |
| V3 | 2000 | 0.999x | ✓ Stable at extended duration |

Weights are self-regulating across all versions. The Hebbian signal is conservative enough that random input doesn't cause drift. V3's homeostatic set point keeps weights within 0.000293 of their resting state on average even after 2000 passes.

**Status: SOLVED from V1. No issues.**

---

### Test 2: Catastrophic Forgetting

*Learn experience A, process 200 interfering B experiences, try to recall A.*

| Version | Immediate | After 200 B (no ctx) | After 200 B (w/ ctx) | Context helps? |
|---------|-----------|---------------------|---------------------|---------------|
| V1 | 56.96 | 57.14 | 59.91 | ✗ Made worse |
| V2 | 64.78 | 69.57 | 66.87 | ✓ 3.9% better |
| V3 | 64.58 | 54.45 | 57.85 | ✗ Made worse |

**Status: PARTIALLY SOLVED.** V2 showed context helping, V3 regressed. The in-weight context retrieval mechanism is inconsistent. Root cause: the adjustment from a single weight's history entry is too small relative to the cumulative effect of 200 interfering experiences modifying the base weight values.

**Recommendation:** In-weight retrieval is not the right mechanism for strong episodic recall. Use the output-level episode store (from Experiment 1/3 in the main research doc) for strong episodic memory. Living weights provide self-modification and temporal dynamics, not episodic recall. The combined architecture covers both needs.

---

### Test 3: Output Divergence on Repeated Identical Input

*Feed the same input 100 times. How fast do outputs change?*

| Version | After 1 | After 10 | After 50 | After 99 | Growth Rate |
|---------|---------|----------|----------|----------|-------------|
| V1 | 0.00100 | 0.00942 | 0.04873 | 0.10179 | 102.3x |
| V2 | 0.00099 | 0.00895 | 0.04455 | 0.08909 | 89.8x |
| V3 | 0.00096 | 0.00779 | 0.02503 | 0.03491 | 36.3x |

**Status: IMPROVED but not fully solved.** V3's homeostatic regulation cut divergence by two-thirds. The remaining divergence is inherent to self-modifying computation — the weight changes compound on identical input because the same Hebbian reinforcement direction keeps being applied.

In practice this may not be a problem because real input is varied, and the homeostatic force recenters weights between diverse inputs (evidenced by the 0.999x drift ratio over 2000 varied passes). The divergence is a property of the pathological case (identical repeated input) rather than normal operation.

**Recommendation:** Monitor in real-world testing. If problematic, add a "staleness detector" that reduces plasticity when the same input pattern is seen repeatedly, mimicking biological habituation at the input level rather than just the weight level.

---

### Test 4: Compute Scaling

*Forward pass time at different dimensions (Python loop implementation):*

| Dimensions | Weights | Time per Pass |
|-----------|---------|--------------|
| 8×8 | 64 | 3.35ms |
| 16×16 | 256 | 5.48ms |
| 32×32 | 1,024 | 23.63ms |
| 64×64 | 4,096 | 81.69ms |

Scaling is roughly O(n²) as expected for a matrix of individually-processed weights. **These times are Python loop overhead, not algorithmic cost.** A vectorized implementation would be orders of magnitude faster.

**Status: NOT A REAL PROBLEM.** The Python loop implementation exists for clarity and correctness verification. Production would use vectorized numpy/torch operations where:
- Current values: standard matrix multiply
- History retrieval: batched similarity search
- Hebbian update: element-wise operations on parallel tensors
- Excitability/momentum: element-wise updates

Estimated vectorized time for 64×64: <1ms.

**Recommendation:** Vectorize before scaling beyond proof-of-concept dimensions.

---

### Test 5: History Saturation

*What happens when all history buffers are full?*

| Version | Avg Entries per Weight | Recall with Context | Recall without Context |
|---------|----------------------|--------------------|-----------------------|
| V1 | 2.9 / 8 capacity | 55.86 | 52.22 |

Buffers didn't fully saturate in testing because the salience threshold (0.3) filters out low-activation experiences. The pruning mechanism (keep highest salience) correctly prioritizes vivid memories over mundane ones.

**Status: WORKING AS DESIGNED.** The salience threshold prevents indiscriminate storage, and the pruning preserves the most important entries. V2/V3's protected slots add an additional layer of preservation for the most salient episodes.

**Recommendation:** Test with much higher experience counts (1000+) to verify long-term saturation behavior. May need adaptive salience thresholds that increase as buffers fill.

---

### Test 6: Zero Input Handling

*Does zero/near-zero input cause spurious weight changes?*

| Metric | Result |
|--------|--------|
| Output magnitude on zero input | 0.0 |
| Weight change on zero input | 0.0 |

**Status: SOLVED from V1.** Hebbian learning correctly produces zero signal when there's no input activation. Idle weights don't drift.

---

## Excitability Dynamics

| Version | Mean | Range | Issue |
|---------|------|-------|-------|
| V1 (500 passes) | 1.998 | 0.988—2.000 | Saturated at ceiling |
| V2 (500 passes) | 1.788 | 1.159—1.822 | Improved distribution |
| V3 (500 passes) | 1.788 | 1.159—1.822 | Same as V2 |
| V3 (2000 passes) | 1.995 | 1.246—2.000 | Eventually saturates |

**Status: PARTIALLY SOLVED.** V2/V3's slower sensitization rate delays saturation but doesn't prevent it long-term. By 2000 passes, excitability has again approached the 2.0 ceiling.

**Root cause:** In a proof-of-concept with 16×16 weights processing 4-token sequences, most weights see low-salience activation most of the time. The sensitization pathway dominates because most activations fall below the 0.5 salience threshold.

**Recommendation:** 
1. Make the excitability ceiling adaptive — higher-traffic weights get a lower ceiling
2. Or: use a different excitability function that asymptotically approaches a ceiling without saturating (e.g., logarithmic rather than multiplicative)
3. Or: tie excitability to actual recall success — weights that successfully contribute to context-matched retrieval become more excitable; weights that don't, habituate

---

## Version Changelog

### V1: Baseline Living Weight
- Hebbian self-modification during forward pass
- History buffer with salience-based pruning
- Excitability (habituation/sensitization)
- **Issues:** Excitability saturation, no forgetting protection, high divergence

### V2: Protected Memory + Damping
- Added: Protected memory slots for high-salience episodes
- Added: Momentum-based damping on self-modification
- Added: Slower excitability rates
- Changed: Random projection context compression (replacing simple averaging)
- **Improved:** Forgetting (context now helps), excitability distribution
- **Remaining:** Divergence still high, context specificity still weak

### V3: Homeostatic Regulation
- Added: Homeostatic set point per weight — weights decay back toward resting state when not strongly activated
- Added: Slowly-adapting set point (resting state itself evolves)
- **Improved:** Divergence (102.3x → 36.3x), long-term stability (0.999x at 2000 passes)
- **Remaining:** Excitability long-term saturation, in-weight retrieval still weak

---

## Key Insights

### What Living Weights Are Good At
1. **Self-modifying computation.** The forward pass changes the layer. Same input → different output. This is definitively not feedforward.
2. **Self-organization.** Weights specialize through use without explicit training signals. Hebbian learning alone produces useful weight structure.
3. **Stability.** With homeostatic regulation, weights stay within a tight range (0.999x drift at 2000 passes) while still self-modifying on each pass.
4. **Temporal existence.** The weights have a past (set point, history), a present (current value, excitability), and a future shaped by ongoing experience.

### What Living Weights Are Not Good At (Alone)
1. **Strong episodic recall.** Individual weight adjustments are too small for reliable context-gated episode retrieval. The output-level episode store does this much better.
2. **Preventing catastrophic forgetting.** Protected slots help but don't fully solve interference from large numbers of subsequent experiences.

### The Combined Architecture Recommendation
- **Living feedforward layers:** For self-modification, temporal dynamics, Hebbian self-organization, non-feedforward computation
- **Output-level episode store:** For strong, reliable episodic memory with context-gated recall
- **Standard scalar attention:** For stable token routing (Q, K)
- **Per-parameter plasticity on V, O:** For individually-adapting content pathway

Each component does what it's best at. Together they cover each other's weaknesses.

---

## Open Issues for Next Session

1. **Vectorization.** Python loops are a non-starter beyond 64×64. Need vectorized implementation before scaling.
2. **Excitability saturation.** Need adaptive ceiling or asymptotic function.
3. **Divergence on repeated input.** 36.3x is better but may need input-level habituation.
4. **Context compression.** Random projection is better than averaging but still produces marginal in-weight retrieval. Learned compression could help.
5. **Entity-controlled plasticity.** No mechanism yet for the entity to adjust its own learning rates. This connects to the Growth Autonomy principle.
6. **Multi-layer stacking.** What happens when you stack living layers? Does self-modification cascade or stabilize?
7. **Composition with attention.** The hybrid block (scalar attention + living feedforward) hasn't been built or tested yet.
