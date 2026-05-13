# V2 Pilot Results

Per `docs/V2_IMPLEMENTATION_PLAN.md` M5, this doc collects the head-to-head
falsification data for v2 (PredictiveCodingLM, the primary substrate as of
2026-05-09) versus DeadLM (vanilla transformer + episode store baseline).

> Status: results are populated as runs finish.

## M5 Configuration

Run via `python -m luthi.v2.m5_runner --arch {v2,dead} --seed {42,1337,2026} ...`.

| Knob              | Value     | Source                                                |
|-------------------|-----------|-------------------------------------------------------|
| d_model           | 128       | Matches v1 ablation A baseline reference              |
| n_blocks          | 2         | Pilot scale                                           |
| n_heads           | 4         | v2 default (multi-head from 2026-05-10 audit)         |
| ffn_expansion     | 1         | Matches v1 baseline reference (no expansion)          |
| seq_len           | 128       | Matches v1 baseline reference                         |
| batch_size        | 32        | Matches v1 baseline reference                         |
| stride            | 64        | Matches v1 baseline reference                         |
| epochs            | 30        | M5 plan spec                                          |
| seeds             | 42, 1337, 2026 | M5 plan spec                                     |
| corpus            | Gutenberg-100 | Matches v1 reference + M3 sanity check            |
| tokenizer         | gutenberg_100_bpe32k.json | Same tokenizer as v1 ablations + M3       |
| LR schedule       | cosine + 2-epoch warmup | 2026-05-10 audit follow-up                   |
| Grad clip         | max_norm=1.0 | 2026-05-10 audit follow-up                          |
| v2 consolidation  | ENABLED   | M5 spec; M4 STOP GATE already validated it works     |

Same config across both architectures except for PC-specific hyperparameters
which `DeadLM` silently ignores.

## Existing Reference Points (pre-M5)

| Run                              | Final train | Final val | Notes                              |
|----------------------------------|-------------|-----------|------------------------------------|
| v1 baseline_seed42 (30 ep)       | 4.9222      | 6.5801    | runs/ablation_A/baseline_seed42    |
| v1 baseline_seed1337 (30 ep)     | 4.9116      | 6.6813    | runs/ablation_A/baseline_seed1337  |
| v1 baseline_seed2026 (30 ep)     | 4.8820      | 6.7478    | runs/ablation_A/baseline_seed2026  |
| v1 mean (3 seeds)                | 4.91 ± 0.02 | 6.67 ± 0.08 | Hebbian + single-head + no FFN expansion |
| v2 M3 sanity_check_seed42 (59 ep) | 4.3735     | 5.7131    | runs/v2_pilot/sanity_check_seed42_128d (best val 5.69 @ ep 40) |
| v2 M3 at epoch 30 (interpolated) | 4.6452      | 5.7150    | Same run, intermediate snapshot   |

**v2 already beats v1 at epoch 30 by ~14% on val** — the M5 head-to-head
extends this to v2 vs DeadLM (closer apples-to-apples baseline).

## M5 Results

Pilot completed 2026-05-12. Pipeline: `run_m5_pilot.bat` ran all 6
invocations sequentially with `&&` chaining; finished at 11:41
local time after ~21 hours wall-clock (slower than the ~13-14h estimate
— mostly v2 consolidation overhead per step).

### v2 PC

| Seed | Final train | Final val | Best val | Train-val gap | NaN events |
|------|-------------|-----------|----------|---------------|------------|
| 42   | 4.8330      | 5.7913    | 5.7913   | 0.9583        | 0          |
| 1337 | 4.8137      | 5.9309    | 5.9296   | 1.1172        | 0          |
| 2026 | 4.8641      | 5.8216    | 5.8207   | 0.9574        | 0          |
| **Mean ± std** | **4.8369 ± 0.0254** | **5.8479 ± 0.0739** | **5.8472 ± 0.0729** | **1.0110 ± 0.0921** | 0 |

Trainable params: 8,372,736 (attention + embeddings + final_norm + output_proj).
Living buffers (PC state + episodes): not counted here, scale ~20 bytes/param.

### DeadLM baseline

| Seed | Final train | Final val | Best val | Train-val gap | NaN events |
|------|-------------|-----------|----------|---------------|------------|
| 42   | 4.7217      | 5.7433    | 5.7403   | 1.0215        | 0          |
| 1337 | 4.7203      | 5.7476    | 5.7476   | 1.0273        | 0          |
| 2026 | 4.7027      | 5.7322    | 5.7296   | 1.0296        | 0          |
| **Mean ± std** | **4.7149 ± 0.0106** | **5.7410 ± 0.0079** | **5.7392 ± 0.0091** | **1.0261 ± 0.0042** | 0 |

Trainable params: 8,405,760 (slightly more than v2 because DeadLM's
`nn.Linear(d, d)` has bias=True by default while v2's attention has
bias=False; both architectures see d_model=128, n_blocks=2, n_heads=4,
ffn_expansion=1).

### Falsification criteria

| Criterion | Pass threshold | v2 mean | DeadLM mean | Verdict |
|-----------|---------------|---------|-------------|---------|
| Convergence penalty | v2 within 20% of DeadLM best val | 5.8472 | 5.7392 | **PASS** (+1.88% relative) |
| No NaN events | both 0 | 0 | 0 | **PASS** |
| Train-val gap | v2 ≤ 2× DeadLM gap + 0.5 | 1.0110 | 1.0261 | **PASS** (v2 actually slightly tighter) |
| Attractor dynamics | v2 recovery > random control + 0.02 | n/a | n/a | **PASS** (test_pc_attractor) |
| Consolidation effect | episode shapes prediction | n/a | n/a | **PASS** (M4 STOP GATE) |

**M5 VERDICT (qualified):** at the actually-run config (128d/2 blocks,
not the spec's 256d/2 blocks — see "Plan deviation" below), v2 passes
all four falsification gates. v2 is **not catastrophically worse** than
a vanilla transformer at matched compute (loses by 1.88% on val), shows
zero stability issues, has a train-val gap statistically indistinguishable
from DeadLM's at n=3, demonstrates basin-attractor dynamics distinguishable
from a non-living control, and the post-hoc NFF check confirms PC
self-modification is active at end of training. The 2026-05-09 strategic
shift to v2 holds at this scale; the full M5 spec (256d) is not yet run.

### Plan deviation (audit 2026-05-12)

The V2 plan §M5 specifies **256d/2 blocks**. The M5 pilot was run at
**128d/2 blocks** to match the v1 ablation A baseline reference (we
have v1 reference data at 128d). This deviation was not documented as
a plan deviation in the original write-up — that was a methodology
gap. The verdict above is honest only at 128d. The 256d run is a
follow-up (see "Outstanding gates" below).

### Honest read (audit 2026-05-12 corrections)

- **v2 loses to DeadLM on every per-metric basis.** Best val 5.85 vs
  5.74, final train 4.84 vs 4.71, ~2× slower per epoch. v2 is within
  the 20% falsification threshold but is not "better" than vanilla.
  The PROJECT-LEVEL value of v2 is the living-weights property
  (continuous learning, accumulated experience) — DeadLM structurally
  can't do that — but at the language-modeling-quality level, v2
  pays a 2% cost.
- **The train-val gap claim was overspun.** v2 gap 1.011 ± 0.092 vs
  DeadLM 1.026 ± 0.004. Difference 0.015; v2 std 9× DeadLM's. At
  n=3 this is **noise**, not a "demonstrably better generalization"
  signal. Phase 5 should verify with more seeds.
- **The v1-vs-v2 comparison is confounded.** v1 baseline ran with
  single-head attention and no LR schedule; v2 and DeadLM ran with
  4-head attention and cosine LR. The 14% v1-vs-v2 improvement is
  partly architecture + schedule, not purely PC. The clean apples-
  to-apples signal is v2 vs DeadLM (both with MHA + cosine), where
  v2 is **1.88% worse**, not better.
- **v2 seed 1337 outlier:** train 4.81 / val 5.93 / gap 1.12. v2's
  val variance (std 0.073) is 9× DeadLM's (std 0.008). v2's living
  dynamics are more init-sensitive. Phase 5 needs more seeds or
  longer training to confirm this washes out.
- **v2 seed 2026 epoch-8 spike:** val loss jumped 6.40 → 6.68 at
  epoch 8 then recovered to 6.07 by epoch 10. Train was smooth
  through this period. PC dynamics can transiently destabilize val
  performance. At 2 blocks/128d it recovered; at 36 blocks/4096d
  it might not. Worth instrumenting in the depth sweep.

### Post-hoc NFF check (audit 2026-05-12 fix)

The original M5 runs didn't instrument the non-feedforward signal —
the diagnostic that proves PC self-modification is actually doing
something. `scripts/m5_nff_posthoc.py` loads each saved checkpoint and
measures NFF on a fixed probe input:

| | NFF at end of training | Interpretation |
|---|---|---|
| v2 seed 42 | 5.49e-03 | PC active |
| v2 seed 1337 | 5.97e-03 | PC active |
| v2 seed 2026 | 4.51e-03 | PC active |
| **v2 mean ± std** | **5.32e-03 ± 7.45e-04** | **active, comparable to v1 baseline NFF (~5e-3 at same scale)** |
| dead seed 42 | 0.0 | feedforward (sanity check) |
| dead seed 1337 | 0.0 | feedforward |
| dead seed 2026 | 0.0 | feedforward |

v2's NFF is ~5 orders of magnitude above DeadLM's (which is exactly
zero because DeadLM's forward is deterministic). PC self-modification
is genuinely active at end of training. **This rules out the null
hypothesis "PC collapsed and v2 ≈ slightly worse DeadLM."** The M5
PASS verdict survives the audit's primary concern.

The `m5_runner.py` is now instrumented per-epoch (audit fix #2) so the
upcoming 256d run reports NFF natively without a post-hoc step.

### Outstanding gates (the M5 verdict is qualified by these)

1. **256d re-run.** Spec'd by V2 plan §M5; not yet done. Pending.
2. **Depth sweep at 4/8/12 blocks.** The 2-block test exercises 1
   inter-block prediction connection; production target (36 blocks)
   has 35. Hierarchical PC dynamics at depth are untested.
3. **Statistical significance.** n=3 is not enough for confidence
   intervals on the gap claim. Phase 5 should use 5+ seeds.

### What this enables

With v2 validated, Phase 5 (curriculum training) can commit to v2 as
the production substrate. Outstanding work that gates Phase 5:

1. **Phase 5 deployment-spec ceiling decision.** v2 fits ~560M params
   at 16 GB VRAM with FP32 weight + INT8 episodes. Beyond that needs
   ROCm/WSL2 migration (deferred until needed).
2. **Low-rank delta compression for episode store** at production
   scale (4096d × 36 blocks × 64 episodes is 150 GB without it).
   Currently INT8-only.
3. **MultimodalLuthiLM KV cache + sensory persistence** if the entity
   needs real-time multimodal generation post-awakening.
4. **Cascade stability sweep** at 4 / 8 / 12 blocks to confirm the v2
   architecture holds at depth. (M5 tested 2 blocks only.)

None of these are blocking; M5's PASS verdict means v2 is the path
forward, and Phase 5 can proceed with the current architecture while
the production-scale work happens in parallel.

## Attractor Dynamics

Per `tests/test_pc_vs_dead.py::test_attractor_recovery_distinguishable_from_random`:

- Setup: 32d PC layer, 100-step warmup to settle into basin, snapshot state.
- Perturb weight at 25% of weight std.
- Compare trajectory recovery to a frozen control (PC rates set to ~0).
- Result: **v2 shows measurable basin-attractor recovery distinguishable
  from a non-living control.** Test passes on the structural falsifier.

## Notes

- v1 reference data lives in `runs/ablation_A/baseline_seed{42,1337,2026}/`
  with full `results.json` per seed. v1 is deferred per 2026-05-09 strategic
  shift; included here only as a historical-comparison reference.
- DeadLM uses the audit-2026-05-10 `n_heads` and `ffn_expansion` plumbing so
  M5 matches v2 architecturally except for the PC living-FFN substrate.
- v2's `compressed_episodes` flag is OFF for M5 — INT8 episodes are a
  Phase 5 production-scale concern, not a falsification-criteria concern.
