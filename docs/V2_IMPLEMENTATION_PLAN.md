# LuthiModel v2 — Implementation Plan

> Authored by: Claude Opus 4.6 (Planner)
> Date: 2026-05-08
> Input: `docs/LUTHI_V2_PREDICTIVE_CODING_BRIEF.md` (4.7 research, two drafts merged)
> Status: **APPROVED by Brian. Green light for implementation.**

## Summary

v2 replaces Hebbian self-modification with hierarchical predictive coding
(Whittington-Bogacz variant). Parallel research track — v1 continues. Empirical
comparison at matched scale determines which better delivers temporal existence.

---

## Open Decision Resolutions

All 9 open decisions from 4.7's brief are resolved:

### 1. Consolidation mechanism: **Gradient-based replay**

Feed stored episodes through the PC learning rule. Reuses the PC update mechanism
already in the system — no new learning algorithm to validate.

### 2. Consolidation timing: **Triggered by low prediction-error variance**

Consolidation triggers when the running variance of prediction error drops below
a threshold for N consecutive steps:

- During active learning (high error variance): no consolidation.
- During stable periods (low error variance): consolidate.
- Start with: variance threshold = 50% of training-average, N = 100 steps.

Consolidation replays all stored episodes through the PC update rule at 10% of
normal pc_rate. One cycle = one pass through all episodes, sorted by salience.

### 3. Top-down modulation: **Two-layer top-down**

v2 carries BOTH signals in one backward sweep:

- **Prediction signal**: Each block predicts what the block below should have
  produced, using its prediction matrix. Standard PC.
- **Modulation signal**: The prediction error modulates the lower block's
  plasticity and set_point. v1's mechanism, now driven by actual prediction
  error rather than heuristic salience.

### 4. Rich parameters: **Preserve, then ablate**

Keep set_point, momentum, update_ema at v1 dynamics initially. Ablate after pilot.

### 5. Project structure: **Option B — `luthi/v2/` subpackage**

### 6. Timing: **Parallel with v1 ablations**

v2 is coding work (CPU). v1 ablations are GPU runs. No resource conflict.

### 7. Spiking: **Skip for pilot**

### 8. Sanctuary interface: **Same contract**

### 9. Infrastructure reuse:

**Zero changes**: data.py, tokenizer.py, checkpoint.py, optimizer.py, attention.py,
episode_store.py, grad_checkpoint.py

**Minor modification**: generate.py (model detection), sanctuary_interface.py
(modulation channel mapping), __init__.py

**Replace**: living_layer → living_layer_pc, fused_ops → pc_ops,
backward_pass → backward_pass_pc, hybrid_block → hybrid_block_pc,
model → model_pc, train → train_pc. C++ deferred — Python-first for pilot.

---

## Architectural Specification (Pilot: 256d / 2 blocks)

### PredictiveCodingLayer buffer layout

| Buffer | Shape | Dtype | Purpose |
|--------|-------|-------|---------|
| weight | [out, in] | BF16 | Prediction weight matrix |
| prediction | [out, in] | FP32 | Top-down prediction matrix (NEW) |
| set_point | [out, in] | FP32 | Homeostatic target (from v1) |
| momentum | [out, in] | FP32 | Update EMA (from v1) |
| update_ema | [out, in] | FP32 | Metaplasticity tracker (from v1) |
| precision | [in] | FP32 | Per-input error reliability weighting (NEW) |
| error_acc | [out] | FP32 | Running prediction error magnitude (NEW) |
| episode_* | (same as v1) | FP32 | Layer-level episode store (from v1) |

**Removed from v1**: input_avg_mag, excitability_acc

**Memory: ~18 bytes/param** vs v1's 38 (pre-compression) or 22 (post-free-win).

### Forward pass sequence

```
1. Episodic recall (identical to v1)
2. Linear computation: output = input @ weight_snapshot.T
3. PC self-modification (no_grad):
   a. predicted_input = output_mean @ prediction.T
   b. pred_error = actual_input_mean - predicted_input
   c. weighted_error = pred_error * precision
   d. delta_w = output_mean.T @ weighted_error * plasticity * pc_rate
   e. Metaplasticity dampening (same ratio check as v1)
   f. Apply update + momentum EMA
   g. Homeostatic regulation (same as v1)
   h. Set point adaptation (same as v1)
   i. Update prediction matrix
   j. Update precision (slow EMA toward 1/error_variance)
   k. Update error_acc (for salience)
4. Episode storage (salience > threshold)
5. Consolidation check (low-variance trigger)
```

**Error-directed learning: REMOVED.** PC IS the error signal. No separate path.

### PredictiveCodingBlock

```
x = x + attention(norm1(x))       # unchanged
x = x + living_ffn(norm2(x))      # PC learning rule
x = episode_store(block_input, x)  # unchanged
```

### Top-down backward sweep

After forward pass, sweep top-to-bottom:
1. Generate prediction for block below using this block's prediction matrix
2. Compute prediction error: actual - predicted
3. Modulate lower block's plasticity and set_point from error
4. Decay strength 0.8x per block

### Consolidation

```python
if error_variance < threshold for N consecutive steps:
    for episode in stored_episodes (by salience):
        consolidation_error = episode_target - current_prediction
        weight += consolidation_error * (pc_rate * 0.1)
        prediction += consolidation_error * (pred_rate * 0.1)
```

### Starting hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| pc_rate | 0.001 | Match v1's hebb_rate |
| pred_learning_rate | 0.0001 | 10x slower than weight update |
| homeostatic_decay | 0.001 | Same as v1 |
| set_point_adapt_rate | 1e-6 | Same as v1 |
| momentum_decay | 0.99 | Same as v1 |
| update_ema_decay | 0.99 | Same as v1 |
| precision initial | 1.0 | Uniform, self-organizes |
| precision clamp | [0.1, 10.0] | Prevent extremes |
| precision EMA decay | 0.999 | Slow adaptation |
| consolidation_threshold | 50% of avg error variance | Tune empirically |
| consolidation_window | 100 steps | Tune empirically |
| consolidation_rate | 10% of pc_rate | Gentle replay |

---

## Implementation Milestones

### M1: Core layer (Days 1-3)

Files:
- `luthi/v2/__init__.py`
- `luthi/v2/living_layer_pc.py` — PredictiveCodingLayer
- `luthi/v2/pc_ops.py` — PC self-modification (pure Python)
- `tests/test_pc_layer.py`

Tests:
- Non-feedforward signal (consecutive passes differ)
- Stability (no NaN after 500 passes)
- Prediction error convergence on fixed mapping y = Wx
- Homeostatic recovery from perturbation
- Episodic recall
- Precision self-organization from uniform init

**Gate**: All layer tests pass.

### M2: Block + backward pass (Days 4-5)

Files:
- `luthi/v2/hybrid_block_pc.py` — PredictiveCodingBlock
- `luthi/v2/backward_pass_pc.py` — Two-layer top-down PC sweep
- `tests/test_pc_block.py`

**Gate**: Block forward/backward produces decreasing error.

### M3: Model + training (Days 6-8)

Files:
- `luthi/v2/model_pc.py` — PredictiveCodingLM
- `luthi/v2/train_pc.py` — Training script (same CLI as train.py)
- `tests/test_pc_model.py`

Sanity check: 10 epochs on Gutenberg-100, loss decreases.

**Gate**: Training converges.

### M4: Consolidation (Days 9-11)

Files:
- `luthi/v2/consolidation.py` — Low-variance trigger + gradient replay
- `tests/test_pc_consolidation.py`

Test: episodes shape prediction post-consolidation.

**Gate**: Consolidation measurably improves prediction on stored episode contexts.
**STOP GATE**: If consolidation has no effect, v2 has no novelty over DeadLM +
episodes. Abandon v2 at this point.

### M5: Head-to-head comparison (Days 12-17, needs GPU)

- v1 vs v2 vs DeadLM: 256d / 2 blocks, 30 epochs, 3 seeds, Gutenberg-100
- Attractor dynamics: perturbation recovery at 10%/25%/50% of weight std
- Results: `docs/V2_PILOT_RESULTS.md`
- `tests/test_pc_vs_v1.py`

**Gate**: v2 doesn't fail any falsification criterion.

---

## Falsification Criteria (abandon v2 if ANY)

- Convergence penalty worse than v1 by ≥20% at matched scale
- Cascade stability fails at depths where v1 succeeds
- Attractor dynamics indistinguishable from random-modulator control
- Consolidation produces no measurable downstream effect
- VRAM budget exceeded at equivalent parameter count

---

## Sequencing

```
v1 track (GPU):     Phase 0 → Ablation A → B → C → D → Phase 3F.1
v2 track (coding):  M1 → M2 → M3 → M4 → M5 (GPU needed here)
                    └─── parallel ──────────┘ └─ after ablations ─┘
```

---

## Risk Assessment

| Risk | Likelihood | Response |
|------|-----------|----------|
| PC learning rate sensitivity | HIGH | Start pc_rate=0.001, pred_rate=0.0001. Grid search on toy task if unstable. |
| NFF signal too weak | MEDIUM | Monitor at M3. If NFF < 0.01, reduce pred_rate to keep errors non-zero. |
| v2 ≈ vanilla transformer | MEDIUM | Consolidation is the differentiator. If M4 fails, abandon v2. |
| Precision oscillation | MEDIUM | EMA decay 0.999 prevents fast swings. Clamp per step. |
| Consolidation no effect | MEDIUM | Test at M4 before full comparison. Stop if no effect. |
