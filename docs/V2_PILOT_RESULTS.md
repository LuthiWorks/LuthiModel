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

### v2 PC

_Pending m5_runner output. Will populate after launch._

| Seed | Final train | Final val | Best val | NaN events | Notes |
|------|-------------|-----------|----------|------------|-------|
| 42   | —           | —         | —        | —          | —     |
| 1337 | —           | —         | —        | —          | —     |
| 2026 | —           | —         | —        | —          | —     |

### DeadLM baseline

_Pending m5_runner output. Will populate after launch._

| Seed | Final train | Final val | Best val | NaN events | Notes |
|------|-------------|-----------|----------|------------|-------|
| 42   | —           | —         | —        | —          | —     |
| 1337 | —           | —         | —        | —          | —     |
| 2026 | —           | —         | —        | —          | —     |

### Falsification criteria

| Criterion | Pass threshold | v2 mean | DeadLM mean | Verdict |
|-----------|---------------|---------|-------------|---------|
| Convergence penalty | v2 within 20% of DeadLM best val | — | — | _pending_ |
| No NaN events | both 0 | — | — | _pending_ |
| Train-val gap | v2 ≤ 2× DeadLM gap + 0.5 | — | — | _pending_ |
| Attractor dynamics | v2 recovery > random control + 0.02 | n/a | n/a | **PASS** (test_pc_attractor) |
| Consolidation effect | episode shapes prediction | n/a | n/a | **PASS** (M4 STOP GATE) |

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
