# M5 Re-Run at 256d — Plan-Spec Configuration Results

> Run completed: 2026-05-13 22:14 (started 2026-05-12 19:37 — ~26.5h wall-clock for 6 runs sequentially).
> Source: `runs/m5_256d/{v2,dead}_seed{42,1337,2026}/` — **artifacts now at
> `E:\runs\m5_256d\`** (pre-JEPA runs moved off-repo 2026-07-22).
> Launcher: `run_m5_256d.bat` (M5 fix #5; launcher deleted 2026-07-22, recoverable from git history)
> Audit: 4.6's adversarial review of the 128d pilot ("M5 Performance Review") flagged that the V2 plan §M5 specifies 256d/2 blocks and the 128d pilot was a plan deviation. This re-run closes that gate.

## Headline

**At 256d/2 blocks, v2 beats DeadLM on every seed.** The 128d pilot showed v2 1.88% *worse* than the vanilla-transformer baseline; the plan-spec re-run at 256d flips the sign to **0.64% better mean best val** with a consistent direction (v2 < dead for all three seeds). The 128d result was the wrong scale to conclude from.

This does not change the M5 falsification verdict (v2 was already PASS at 128d on the convergence-penalty criterion), but it does upgrade the substantive read of v2 from "tolerable cost for the living-weights property" to "competitive with vanilla transformers at the pilot scale, while *also* providing living-weights properties DeadLM structurally cannot."

## Configuration

Identical to the V2 plan §M5 spec. Diff from the 128d pilot is only `--d-model 256` (and the `runs/m5_256d/` output directory).

| Knob | Value |
|------|-------|
| Architecture | v2 (PredictiveCodingLM) vs dead (DeadLM control) |
| d_model | **256** (was 128 in pilot) |
| n_blocks | 2 |
| n_heads | 4 |
| ffn_expansion | 1 |
| seq_len | 128 |
| batch_size | 32 |
| stride | 64 |
| epochs | 30 |
| LR | 3e-4, cosine + 2-epoch warmup |
| Grad clip | max_norm=1.0 |
| Corpus | Gutenberg-100, 10.6M tokens, 95/5 split |
| Tokenizer | `corpus_build/gutenberg_100_bpe32k.json` (BPE-32k) |
| v2 consolidation | ENABLED |
| Seeds | 42, 1337, 2026 |

NFF, prediction Frobenius norm, and precision EMA are now instrumented **per-epoch** (M5 fix #2 — no longer post-hoc).

## Results

### v2 PC

| Seed | Best val | Best epoch | Final train | Final NFF | NaN events |
|------|----------|------------|-------------|-----------|------------|
| 42   | **5.6859** | 17 | 4.2781 | 9.35e-03 | 0 |
| 1337 | **5.7575** | 20 | 4.3323 | 7.51e-03 | 0 |
| 2026 | **5.7269** | 15 | 4.2855 | 1.33e-02 | 0 |
| **Mean ± stdev** | **5.7234 ± 0.0360** | 17.3 | 4.2986 ± 0.0298 | 1.01e-02 ± 2.96e-03 | 0 |

Trainable params: 16,975,616. Living buffers: 9,684,868. Total: 26.66M.

### DeadLM control

| Seed | Best val | Best epoch | Final train | Final NFF | NaN events |
|------|----------|------------|-------------|-----------|------------|
| 42   | 5.7491 | 12 | 4.1411 | 0.0 | 0 |
| 1337 | 5.7606 | 13 | 4.1728 | 0.0 | 0 |
| 2026 | 5.7703 | 11 | 4.1818 | 0.0 | 0 |
| **Mean ± stdev** | **5.7600 ± 0.0106** | 12.0 | 4.1652 ± 0.0214 | 0.0 | 0 |

Trainable params: 17,107,200. Static FFN buffers: 524,288. Total: 17.63M.

### Head-to-head per seed

| Seed | v2 best val | dead best val | Δ (v2 − dead) | v2 better? |
|------|-------------|---------------|---------------|-----------|
| 42   | 5.6859 | 5.7491 | −0.0632 | ✅ |
| 1337 | 5.7575 | 5.7606 | −0.0031 | ✅ |
| 2026 | 5.7269 | 5.7703 | −0.0434 | ✅ |
| **Mean** | 5.7234 | 5.7600 | **−0.0366** | ✅ (3/3 seeds) |

v2 is better on every seed. The seed-1337 margin is narrow (0.05%) — the other two are clearly outside the dead variance (dead stdev = 0.011).

## Falsification verdict

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| Convergence penalty | v2 within 20% of dead best val | v2 **0.64% better** | **PASS (exceeds)** |
| No NaN events | both 0 | 0 / 0 | **PASS** |
| PC active end-of-training | NFF significantly > 0 | 1.01e-02 vs 0.0 | **PASS** |
| Per-seed consistency | v2 ≤ dead on majority | 3/3 seeds | **PASS** |

**256d M5 VERDICT: v2 PASSES all four gates and exceeds the convergence criterion.**

## What changed from 128d to 256d

The 128d pilot had v2 at 5.85 vs dead 5.74 (v2 1.88% worse). The 256d re-run has v2 at 5.72 vs dead 5.76 (v2 0.64% better). Why the flip?

Two non-exclusive hypotheses, neither fully resolvable from this data:

1. **PC dynamics need width to express the living-weights advantage.** At 128d, the PC layer has 128 prediction targets per layer to allocate across the corpus's information content; at 256d it has 256. The error signal per output unit is sparser at higher width, so the precision EMA + sparse error_acc machinery has more headroom to differentiate informative from noisy dimensions. DeadLM has no analogous mechanism — its FFN is static — so DeadLM's gain from doubled width is just the standard transformer scaling curve, while v2 gets both that AND the PC mechanism activating more cleanly.
2. **The 128d pilot used 4 heads at 128d → 32d per head, which is below the typical "useful attention head" threshold.** At 256d that becomes 64d per head, comfortably within useful range. This would help BOTH architectures and shouldn't explain a sign flip, but could amplify the absolute gap if PC dynamics depend on cleaner attention outputs to compute coherent prediction errors.

Direction (1) is the more interesting hypothesis because it predicts the gap continues to grow with depth and width — exactly what the M6 depth sweep is set up to test.

## Convergence behavior

**Dead reaches its best ~6 epochs earlier than v2** (mean best epoch 12.0 vs 17.3). After its best, dead immediately starts overfitting — val loss climbs by ~0.08-0.10 over epochs 13-30 while train continues down. v2's val curve is flatter — it reaches a similar best later, then drifts up more gently (~0.03-0.05 over the same window).

This is consistent with PC providing implicit regularization (the prediction-error update has a built-in bound and a homeostatic pull toward `set_point`). The 128d pilot saw the same pattern in the train-val gap numbers (v2 gap 1.011 vs dead 1.026) but at n=3 it was within noise; the 256d run shows it more cleanly in the val-curve shape.

## NFF, prediction norms, precision (v2 only)

- **NFF**: starts ~0.02 at epoch 1 (PC warming up against random init), drops to ~0.007 by epoch 11, oscillates 0.007-0.013 for the remainder. Confirms PC is active end-of-training (1.01e-02 mean final NFF, ~5 orders above dead's 0.0).
- **Prediction Frobenius norm**: monotonic growth from ~0.6 to ~1.6-2.0 across 30 epochs. The prediction matrix is genuinely accumulating structure, not bouncing around an init.
- **Precision EMA mean**: saturates near 10.0 (the `precision_max` ceiling) by epoch 5-10. The precision ceiling is being hit — this is consistent with the prediction errors becoming small enough that `1 / (err² + 1e-3)` exceeds 10 for most dimensions. Worth lifting `precision_max` in a future ablation to test whether precision is actually load-bearing or just clamped to ceiling.

The precision-saturation observation is **new data** — the 128d pilot didn't instrument it per-epoch. Logged as a follow-up question for the depth sweep and the GPU-validation runs for the 2026-05-13 compute-optimization experiments (Phase 3G).

## Wall-clock cost

| Phase | Duration |
|-------|----------|
| 6 runs (3 v2 + 3 dead, sequential `&&`) | ~26.5 hours |
| Per v2 run | ~5.5 hours |
| Per dead run | ~3.3 hours |

Slower than the 128d pilot's ~21h-for-6 because the model is 4× larger in d_model² scaling and the PC self-modification path has the same per-op DirectML dispatch cost regardless of width. No NaN events across any run.

## What this enables

- **Phase 5 (curriculum training)** can commit to v2 with stronger confidence than the 128d pilot warranted. The "tolerable cost" framing upgrades to "competitive baseline."
- **M6 depth sweep** at 4/8/12 blocks is now the primary follow-up. If the v2-vs-dead gap *widens* with depth, hypothesis (1) above is supported and Phase 5 should expect compound gains. If the gap stays flat or narrows, the 256d advantage was width-specific.
- **GPU validation of the 2026-05-13 compute-optimization experiments** (Phase 3G: μPC, iPC, sparse gating — see `docs/RESEARCH_LITERATURE_2026-05-13.md`) now has a strong baseline to compare against. Each ablation is a one-epoch run at 256d/2 blocks against this run's epoch-1 numbers.

## Outstanding gates (carrying forward from 128d)

1. **Depth sweep at 4/8/12 blocks** (M5 fix #4 harness was `run_m6_depth_sweep.bat` — executed, then deleted 2026-07-22 with the other retired launchers; in git history). Hierarchical PC dynamics at depth still untested.
2. **Statistical significance** (n=3 still). The 256d gap is wide enough relative to dead's variance that it survives an informal eyeball, but 5+ seeds would give actual confidence intervals.
3. **Precision-ceiling sensitivity**: does lifting `precision_max` from 10.0 → 100.0 change anything, or is precision truly bounded by the clamp? Worth a one-seed sweep.

## Files

- Per-seed `results.json` in `runs/m5_256d/{v2,dead}_seed{42,1337,2026}/`
- Per-seed `model_final.luthi` checkpoints (encrypted; same dir)
- Combined log at `runs/m5_256d/m5_256d.log`
- (All of the above moved to `E:\runs\m5_256d\` on 2026-07-22, byte-verified.)
- Comparison data also reflected in `docs/V2_PILOT_RESULTS.md` (128d pilot results retained for reference; this doc supersedes the "256d re-run pending" outstanding gate)
