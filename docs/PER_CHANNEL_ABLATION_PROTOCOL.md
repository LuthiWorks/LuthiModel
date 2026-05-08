# Per-Channel Buffer Ablation Protocol

> Authored by: Claude Opus 4.7 (Researcher)
> Date: 2026-05-07
> Prompted by: hardware constraint — 4B target on RX 7800 XT (16 GB VRAM) is infeasible because per-weight FP32 living buffers cost ~38 bytes/param (vs 2 bytes for the BF16 weight). Ceiling on current hardware is ~300M params end-to-end as currently architected.
> For sequencing/integration by: Claude Opus 4.6 (Planner)

## TL;DR

A code-path analysis of `luthi/living_layer.py` and `luthi/fused_ops.py` reveals that two of the five per-weight FP32 living buffers — `plasticity` and `excitability_acc` — are mathematically rank-1 by the structure of their update rules. They can be stored as `[in_features]` and `[out_features]` vectors respectively with **zero behavioral change**. This is a free ~16 bytes/param reduction.

Three further reductions require empirical validation: **BF16 momentum**, **BF16 set_point**, and **INT8 delta-encoded episode storage**. With all wins combined, the per-param living-state cost drops from ~38 bytes to ~10 bytes, lifting the practical ceiling on RX 7800 XT from ~300M to ~1.1B parameters.

This protocol tests the three ablations. It does not test the free wins, which should be implemented and verified for numerical-equivalence before any ablation runs.

## Context

EMPIRICAL_DEFENSE_PLAN.md committed to "4B params, BF16 weights, mixed-precision living state (FP32 for plasticity/momentum/set_point buffers), AMD RX 7800 XT (16 GB VRAM)." The buffer cost was undercounted in that spec. Counting the six per-weight buffers (`weight`, `set_point`, `momentum`, `excitability_acc`, `plasticity`, `update_ema`) plus 4 FP32 episode snapshots:

```
Current per-element cost in living FFN:
  BF16 weight:                  2 bytes
  set_point (FP32):             4 bytes
  momentum (FP32):              4 bytes
  excitability_acc (FP32):      4 bytes
  plasticity (FP32):            4 bytes
  update_ema (FP32):            4 bytes
  4× FP32 episode snapshots:   16 bytes
  ----------------------------- ----
  Total:                       38 bytes/param
```

Living FFN is ~2/3 of total params at standard 4× FFN expansion. Activations and gradients consume an additional ~30% of VRAM during training. On 16 GB VRAM, this gives a hard ceiling around 300M params for the existing architecture.

## Free Wins (Code Analysis, Not Ablation)

### `plasticity` is rank-1 along the input axis

**Code evidence:**
- Initialized uniform `torch.ones(out, in)` (`living_layer.py:86`)
- Only modification site is `apply_top_down` (`living_layer.py:359-363`):
  ```python
  importance = signal.salience.unsqueeze(0)  # [1, in_features]
  self.plasticity.mul_(1.0 - 0.01 * strength).add_(
      importance * 0.01 * strength
  )
  ```
- The multiplier is uniform; the additive is `[in_features]` broadcast. Plasticity remains rank-1 in the input dim for all training history.

**Conclusion:** Storage `[in_features]` is mathematically equivalent to current `[out, in]` storage. Per-output-row uniformity is an invariant of the update rule, not an emergent property.

**Action:** Refactor `plasticity` to `[in_features]`. Broadcast at use site (`fused_ops.py:155`). Verify bit-equivalent training for the first ~1000 steps against the current implementation. No ablation run required beyond this regression test.

### `excitability_acc` is rank-1 along the output axis

**Code evidence:**
- Initialized `torch.zeros(out, in)` (`living_layer.py:81-83`)
- Three modification sites:
  1. `fused_ops.py:194-202` — Hebbian-step update broadcasts `salience_per_dim` (shape `[out_features]`) across `in_features` axis. All columns in a row receive the same delta.
  2. `sanctuary_interface.py:437` — `excitability_acc.add_(excitability_bias)` with scalar `excitability_bias`. Uniform across all elements.
  3. `sanctuary_interface.py:463` — `copy_` from snapshot. Shape-preserving; per-channel snapshot is fine.

**Conclusion:** Storage `[out_features]` is mathematically equivalent to current `[out, in]` storage. Per-input-column uniformity is an invariant of the update rule.

**Action:** Refactor `excitability_acc` to `[out_features]`. Broadcast at use site (`fused_ops.py:131` for `exc` computation). Verify bit-equivalent training. No ablation run required.

### Memory savings from free wins

For a layer with weights `[out=4096, in=4096]`:
- Old: `excitability_acc` = 67 MB FP32, `plasticity` = 67 MB FP32. Total: 134 MB per layer.
- New: `excitability_acc` = 16 KB FP32, `plasticity` = 16 KB FP32. Total: 32 KB per layer.
- **Savings: ~134 MB per layer.** At 24 blocks: ~3.2 GB freed.

## Ablations (Empirical Validation Required)

### Ablation A: BF16 `momentum`

**Question:** Can `momentum` survive at BF16 without breaking Hebbian dynamics?

**Reasoning:** `momentum` accumulates a running average of `hebb_update`. Magnitudes are governed by `hebb_rate * salience * excitability * plasticity` ≈ 1e-4 to 1e-3. BF16 ULP at 1e-3 is ~8e-6 — borderline. Stochastic rounding or Kahan summation may be needed.

**Test:**
- Variant: `momentum` registered as BF16, others FP32.
- Optional sub-variant: BF16 momentum + Kahan-summed accumulation.
- Compare against FP32-momentum baseline.

**Pass:** Val loss within 5% of baseline at matched epochs. Plasticity self-organizes (variance grows). Set-point drift bounded. No NaN.

### Ablation B: BF16 `set_point`

**Question:** Can `set_point` survive at BF16?

**Reasoning:** `set_point` drifts via two mechanisms: (1) homeostatic adaptation `sp_delta = weight - set_point` adapted at `set_point_adapt_rate=0.001`; (2) top-down nudge `error_signal * set_point_adapt_rate * 10.0 * strength`. Drift magnitudes per step are ≤1e-3. BF16 may lose accumulation precision.

**Test:**
- Variant: `set_point` registered as BF16, others FP32.
- Compare against FP32-set_point baseline.

**Pass:** Val loss within 5%. `set_point_drift` metric (existing instrumentation) tracks reasonably with FP32 baseline. Homeostatic recovery test (perturb weight, measure return-to-set_point) succeeds.

### Ablation C: INT8 delta-encoded episode storage

**Question:** Can episodes be stored as INT8 deltas from current weight?

**Reasoning:** `episode_values` stores full weight snapshots (`living_layer.py:204`). Recall computes `delta = recalled_weights - self.weight` (`living_layer.py:178`) and applies it scaled. Storing the delta directly with per-tensor INT8 quantization should preserve enough fidelity for context-gated recall.

**Test:**
- Variant: `episode_values` stores `quantize_int8(snapshot - weight_at_snapshot_time)` plus per-episode FP16 scale.
- Recall path: `delta = dequantize(episode_values[idx]) * episode_scales[idx]` (no subtract).
- Compare against FP32-snapshot baseline on episodic recall task.

**Pass:** Val loss within 5%. Episodic recall test (existing or to-be-built) shows comparable retrieval quality. Context-similarity matching unaffected (episode_contexts stays FP32).

### Ablation D: Combined (A + B + C)

The integrated test. If A, B, C each pass independently, run all three together to confirm no interaction effects.

## Configuration

### Model

- Architecture: `LuthiLM` (text-only, no spiking) — isolates living-weight dynamics from SNN concerns
- Size: 128d / 2 blocks / 4× FFN expansion
- Vocab: BPE 32K (existing tokenizer)
- Sequence length: 256
- Batch size: 16

This is small enough for ~1-2 hour training runs on RX 7800 XT and large enough that buffer dynamics are non-trivial.

### Confirmation scale

Best-performing combined variant gets re-run at 256d / 4 blocks for ~4-hour confirmation before declaring victory.

### Corpus

Gutenberg-100 (`corpus_build/gutenberg_100`). Same train/val split as existing precision tests for direct comparability with TRAINING_LOG.md historical runs.

### Schedule

- 30 epochs per run
- Save checkpoint every 5 epochs
- Existing `train.py` infrastructure with `--dtype fp32` baseline
- New CLI flag `--buffer_dtypes` accepting per-buffer dtype overrides

### Seeds

3 seeds per variant (42, 1337, 2026). Report mean ± std on key metrics.

### Optimizer

DirectMLAdamW (existing, lerp-free). Unchanged from current training.

### Backward pass

ON (default per `CLAUDE.md` decision 9b).

## Metrics

Existing instrumentation captures these. Comparison is variant vs baseline at matched epoch.

| Metric | Source | Pass criterion |
|---|---|---|
| Train loss curve | training log | Within 5% of baseline at every epoch |
| Val loss curve | training log | Within 5% of baseline at every epoch |
| Plasticity mean | per-epoch metrics | Self-organizes (variance > 0.01 by epoch 20) |
| Plasticity std | per-epoch metrics | Comparable to baseline |
| Set-point drift | per-epoch metrics | Bounded, comparable to baseline |
| Non-FF signal | per-epoch metrics | Within 10% of baseline |
| Weight norm | per-epoch metrics | Stable (not collapsing or exploding) |
| NaN/Inf detection | per-step | Zero occurrences |
| Wall-clock time | training log | Variant ≤ baseline + 10% (no major slowdown) |

## Pass / Fail Criteria

**Strong pass:** Variant within 5% of baseline on val loss at matched epoch, all dynamics metrics comparable, no NaN.
**Soft pass:** Variant within 10% of baseline, dynamics qualitatively similar, no NaN.
**Fail:** Val loss > 10% worse, OR plasticity flattens, OR NaN appears, OR wall-clock penalty > 25%.

## Resource Estimate

5 variants (Baseline + A + B + C + D) × 3 seeds = 15 runs.
At ~1.5 hours per run on RX 7800 XT: ~22 GPU-hours. Plus 1 confirmation run at 256d/4-blocks for ~6 hours.

**Total: ~28-30 GPU-hours.** Fits in 3 days of overnight runs.

## Out of Scope

Explicitly NOT covered by this protocol:

- **SNN backward pass** — separate parallel research track (surrogate gradients).
- **Custom Triton kernels** — Phase 3F.5, separate.
- **Cascade depth stability** — Phase 3F.2, separate.
- **Same-scale baseline vs vanilla transformer** — Phase 3F.1, separate.
- **Catastrophic forgetting** — Phase 3F.4, separate.
- **`update_ema` reduction** — code analysis confirms it tracks genuinely per-weight history. Pursuing this requires algorithmic redesign, not just storage compression. Future work.
- **`weight` or `momentum` quantization below BF16** — INT8 weight + FP32 master is a separate research direction, not in this protocol.

## Open Questions for 4.6

1. **Sequencing.** Free wins (per-channel `plasticity` and `excitability_acc`) should land before Phase 3F.1 baseline runs since they affect memory math and might affect comparability. Ablation runs (A/B/C/D) could go in parallel with 3F.1 once free wins are in. Does this fit the project rhythm?
2. **Scope creep risk.** This protocol does not address whether 1B params is enough to be a meaningful step toward Sanctuary integration. If the practical answer for Brian's hardware is 1B max, that has implications for the cognitive-loop integration plans that 4.6 should weigh in on.
3. **Confirmation scale.** Is 256d/4-blocks confirmation sufficient, or should the winning combined variant be re-confirmed at 512d/8-blocks before being adopted into the production architecture?

## Implementation Order (suggested)

1. **Phase 0** (1 day): Refactor `plasticity` to `[in_features]`. Verify bit-equivalent training for 1000 steps. Refactor `excitability_acc` to `[out_features]`. Verify bit-equivalent training for 1000 steps.
2. **Phase 1** (3 days): Add `--buffer_dtypes` CLI flag and per-buffer dtype registration. Implement Ablation A (BF16 momentum) and run 3 seeds vs baseline.
3. **Phase 2** (3 days): Implement Ablation B (BF16 set_point). Run 3 seeds vs baseline.
4. **Phase 3** (3 days): Implement Ablation C (INT8 episode delta). Run 3 seeds vs baseline.
5. **Phase 4** (3 days): If A/B/C all pass, run combined Ablation D (3 seeds at 128d, 1 confirmation at 256d).
6. **Phase 5** (1 day): Document results in `docs/BUFFER_ABLATION_RESULTS.md`. Update `EMPIRICAL_DEFENSE_PLAN.md` if the deployment spec needs revision.

Total: ~14 days of work, much of it overnight runs.
