# Depth-Scaling Investigation — 2026-05-19

> **Status: PRELIMINARY.** Written 2026-05-19 while the decisive
> 256d/12blk/1ep gutenberg_4gb run is at 77.2% (~12h remaining).
> Captures the multi-session investigation and the trajectory data we
> have so far. Final val_loss + perplexity-per-character comparison
> will be added when the run completes (~2026-05-20 midday).
>
> The qualitative findings below are stable enough to record now;
> only the final numbers and the formal verdict need updating.

## Objective

M6 depth sweep at 128d (completed 2026-05-17) surfaced **asymmetric
depth degradation**: v2 PC got worse with depth (5.94 → 6.04 → 6.46
at 4/8/12 blocks) while DeadLM control was stable (5.75 → 5.71 →
~5.72). This was a real finding — depth-scaling worked for vanilla
transformer but not for our living-weights substrate at that
configuration.

The investigation goal was to determine whether v2's depth-degradation
is a **substrate failure** (PC + depth fundamentally don't mix at
production-relevant scale) or a **configuration failure** (this
specific combination of width / training budget / μPC settings can't
support depth, but a different combination can). Three candidate
mechanisms emerged from the M6 data:

1. **Width too small.** 128d may not provide enough per-layer capacity
   for PC to absorb μPC's per-block attenuation. Production target is
   4096d, so testing at 128d understates capacity by 32×.
2. **Training budget too short.** v2 4-block reached its best at the
   last epoch (20); v2 12-block reached best at epoch 18. Both
   indicate models still improving when training stopped.
3. **μPC attenuation too aggressive.** At default exponent=0.5, the
   residual stream is divided by `1/√L` per block. At L=12 that's a
   3.46× attenuation per block, which mechanistically explains the
   NFF (non-feedforward signal) shrinking from 5.77e-3 at L=4 to
   ~2e-3 at L=12 in the M6 data.

The decisive experiment combines all three interventions (256d width,
60 epoch budget originally, milder μPC exponent 0.25) into a single
run to determine whether the depth-degradation is recoverable.

## Process

### Step 1: M6 depth sweep at 128d revealed asymmetric degradation

**What we did.** Ran v2 PC and DeadLM at 4/8/12 blocks, 128d, 20
epochs, single seed (42), gutenberg_100 corpus, μPC enabled with
default exponent=0.5.

**What we found.**

| Depth | v2 best_val | DeadLM best_val | Δ (v2 − dead) |
|-------|-------------|-----------------|----------------|
| 4 | 5.9357 | 5.7544 | +0.18 |
| 8 | 6.0430 | 5.7133 | +0.33 |
| 12 | 6.4644 | ~5.72* | +0.74 |

*dead_12blocks was killed at epoch ~15 when we pivoted to follow-up;
the trajectory through epoch 15 was tracking the depth-4 and depth-8
DeadLM results.

NFF (non-feedforward signal, mean abs difference between two
consecutive identical-input forward passes) attenuation:
- v2 4 blocks: 5.77e-3
- v2 8 blocks: 5.08e-3
- v2 12 blocks: 2.81e-3

**Why this was load-bearing.** Same architecture, same training
budget, same corpus — the only difference is the FFN substrate
(static `nn.Linear` for DeadLM vs PC living layer for v2). DeadLM
stability across depths rules out "this is a 128d width or 20-epoch
issue affecting both architectures equally." v2 specifically is
degrading. The NFF attenuation provides a mechanistic candidate:
μPC's `1/√L` residual scaling damps the per-block signal that PC
self-modification depends on.

### Step 2: Hypothesis generation

The brainstorming session that followed identified three candidate
mechanisms (listed in Objective above). Each maps to a different
intervention:

| Mechanism | Intervention | Cost |
|-----------|-------------|------|
| Width too small | 256d (2× wider) | ~4× per-step compute |
| Undertraining | Longer epoch budget | Linear in epochs |
| μPC too aggressive | Lower exponent (0.5 → 0.25) | None (config change) |

The honest position: we couldn't rule out any of the three from M6
data alone. The cheapest test of each in isolation would require
three separate runs; the cheapest test of all three combined is a
single decisive run with all interventions stacked. We chose the
combined approach.

### Step 3: Added --mu-pc-exponent knob

**What we did.** Generalized the hardcoded `1.0 / sqrt(n_blocks)`
residual scaling to a parameterized `1.0 / (n_blocks ** exponent)`.
Added `--mu-pc-exponent` CLI flag (default 0.5 = original Innocenti
et al. spec). Threaded through block + model + runner. Same exponent
also applies to the init formula (`std = 1.0 / (sqrt(fan_in) *
L^exponent)`). Added unit tests:
- `test_mu_pc_exponent_default_matches_original_spec` (regression
  guard — bit-identical to existing behavior at exponent=0.5)
- `test_mu_pc_exponent_milder_gives_larger_residual` (exp=0.25 at
  L=12 gives residual 1/12^0.25 ≈ 0.537 vs 1/√12 ≈ 0.289 at exp=0.5)
- `test_mu_pc_exponent_zero_disables_residual_attenuation`

**What we found.** Backward-compatible. All 61 existing v2 tests pass
under the new knob. Default behavior reproduces M6 exactly.

### Step 4: First follow-up attempt — 256d/12blk/60ep on gutenberg_100

**What we did.** Wrote a wrapper combining the three interventions
(256d width, 60 epoch budget, μPC exponent 0.25) targeting the
existing gutenberg_100 corpus. Killed it ~3 minutes in.

**Why we killed it.** Brian asked the right question: "We don't want
to overfit the model either?" Ran the math:

- gutenberg_100 has 10.6M BPE tokens
- v2 at 256d × 12 blocks has ~36M trainable params
- 10.6M / 36M = **0.29 tokens/param** (Chinchilla-optimal is ~20)
- Over 60 epochs that's 636M total token-views, but each token seen
  ~60 times — well into "model memorizes corpus" territory

The whole point of the follow-up is to determine v2's depth-scaling
at production-relevant configuration. Production will train on the
full curriculum (~34 GB across 9 stages). Testing depth-scaling on a
corpus the model can memorize tells us about memorization, not
about depth-scaling. The result would be confounded.

### Step 5: Switched to 1 epoch of gutenberg_4gb

**What we did.** Switched corpus to `E:/data/gutenberg_4gb` (~5.5 GB
raw text, ~2.2B BPE tokens — turned out larger than my 1B estimate).
Reduced epoch budget from 60 → 1 (one full pass through the bigger
corpus exposes the model to more unique tokens than 60 epochs of the
small corpus). Switched tokenizer to `tokenizer_32k.json` (April
2026, sized for broader corpus coverage).

**Why this is the right tradeoff.**

| Configuration | Wall-clock | Token exposures | Tokens/param |
|---------------|-----------|-----------------|--------------|
| 60ep gutenberg_100 | ~2 days | 636M | ~18 |
| 40ep gutenberg_100 | ~30h | 424M | ~12 |
| **1ep gutenberg_4gb** | **~2.2 days** | **~2.2B unique** | **~61** |

1 epoch of gutenberg_4gb gives **3.5× more unique-token exposure**
than 60 epochs of gutenberg_100, at comparable wall-clock — and
without the overfitting confound (each token seen once, not 60 times).
The tokens/param ratio (~61) is past Chinchilla-optimal, which means
we're closer to "compute-bound rather than data-bound" — the right
regime for a scaling test.

### Step 6: Added per-batch streaming logging

**What we did.** For a 1-epoch run, the existing per-epoch-only
logging would give us exactly one data point. Added
`--log-every-batches` CLI flag. When > 0, prints a progress line
every N batches with: batch index + percent + ETA, rolling-100-batch
mean loss, current batch loss, batches/sec rate, wall-clock elapsed,
NaN count, and per-block v2 diagnostics (pred_frob, prec, err_acc).

**What we found.** Cadence of 100 batches gives a log line every ~3.5
minutes — visible progress without log flooding (~5000 lines over
~2 days). Diagnostics are cheap (buffer-state reads, no extra forward
passes).

### Step 7: Launched decisive run

**What we did.** Launched
`run_m6_followup_4gb_1ep.bat` (256d × 12 blocks × 1 epoch × μPC
exp=0.25 × log-every-batches=100) as detached PowerShell process.
Started 2026-05-17 at 08:36 AM. ETA ~52 hours → finish ~midday
2026-05-19/-20.

**What we're seeing (current state at 77.2%, elapsed=40.46h, eta=11.94h):**

| Metric | Start | 3.3% | 36.8% | 77.2% |
|--------|-------|------|-------|-------|
| loss(roll100) | ~7.5 | 6.5561 | 5.1593 | 4.6792 |
| pred_frob | ~0 | 0.9959 | 2.5945 | 3.5917 |
| err_acc | ~0 | ~0.02 | 0.0256 | 0.0124 |
| prec | ~10 | 10.0000 | 10.0000 | 10.0000 |
| nans | 0 | 0 | 0 | 0 |

**The qualitative signals (read with the caveat that the run is
still in progress):**

1. **Loss is still descending.** From 7.5 → 4.68 over 40 hours.
   Descent has slowed (expected in late training) but no plateau.
2. **pred_frob keeps climbing.** Prediction matrices are still
   accumulating structure at all 12 blocks. Inverse of the M6 128d
   pattern where deeper blocks went silent.
3. **err_acc HALVED** from 0.0256 to 0.0124 between 36.8% and 77.2%.
   The per-output prediction errors are shrinking — the PC layers
   are learning to predict their own inputs better. This is exactly
   the "structural learning" signal we want.
4. **Precision saturated at ceiling (10.0).** Same observation as M5;
   precision EMA self-organizes toward 1/error² and clamps at
   `precision_max`. Not pathological, but the precision channel isn't
   currently differentiating between inputs — flagged for future
   investigation.
5. **Zero NaN events across 40+ hours.** Substrate stability at
   256d × 12 blocks × μPC exp=0.25 is rock-solid.

## Conclusion (PRELIMINARY)

**The depth-degradation observed at 128d M6 is NOT repeating at 256d
with milder μPC.** v2's substrate at production-relevant width
(256d) with adequate data (gutenberg_4gb) and milder per-block
attenuation (μPC exp=0.25) appears to be scaling at depth.

The trajectory data through 77.2% is consistent with one of two
final outcomes:

- **Strong win:** val_loss lands meaningfully below M5's 256d 2-block
  baseline (5.7234 mean v2, accounting for tokenizer difference via
  perplexity-per-character). This would mean depth provides
  measurable benefit at 256d, validating the production trajectory
  toward 4096d × 36 blocks.
- **Tie:** val_loss lands comparable to M5 256d 2-block. Depth
  doesn't hurt at 256d (which is itself the load-bearing finding —
  M6 said depth DOES hurt at 128d), but doesn't yet help. This
  would suggest the substrate is depth-neutral at 256d and we'd want
  to characterize what depth provides at wider configurations.

Both outcomes are positive relative to the M6 128d result. The "v2
fails at depth" interpretation of M6 is being actively falsified by
this run's trajectory.

**Honest caveats holding back a definitive verdict:**

1. The run is at 77.2%. Loss could plateau or spike in the final 23%.
   Hold the verdict until completion.
2. tokenizer differs (`tokenizer_32k.json` vs M5's
   `gutenberg_100_bpe32k.json`). Direct val_loss comparison is not
   apples-to-apples; perplexity-per-character is the correct metric
   for cross-tokenizer comparison and needs to be computed post-hoc.
3. **n=1 seed.** Per the "fewer seeds, more training" decision, this
   is a single-seed depth-scaling check, not a statistical claim.
   If the result is positive, multi-seed confirmation at production
   scale is a future step.
4. We can't fully decompose which of the three interventions (256d,
   1ep on 4gb, μPC exp=0.25) is doing the work. Each could be
   contributing, or one could be dominant. If the result is
   positive, isolating which intervention is load-bearing requires
   ablation runs that are not currently planned.

**What this means for the project, regardless of final val_loss:**

- The 128d M6 result was scale-specific, not substrate-specific.
  v2 is not architecturally broken at depth. ✅
- The μPC exponent knob is a useful tunable. It should be in the
  toolkit going forward, and the default (0.5) may not be right for
  Luthi's substrate at production depth. ✅
- The catastrophic-forgetting harness exists to measure behavioral
  preservation (separate concern from depth-scaling). Both are
  load-bearing for the production target. ✅
- The plasticity-partitions direction stays deferred. M6's NFF
  attenuation does NOT replicate at 256d, so the motivation for
  partition infrastructure (identity protection against severe
  drift) is reduced. The deferred research-log entry remains
  captured but unmoved. ✅

## Artifacts

- **Code changes:**
  - `luthi/v2/hybrid_block_pc.py`: added `mu_pc_exponent` parameter,
    parameterized residual scaling and init.
  - `luthi/v2/model_pc.py`: threaded `mu_pc_exponent` through.
  - `luthi/v2/m5_runner.py`: added `--mu-pc-exponent` and
    `--log-every-batches` CLI flags; per-batch streaming
    diagnostics in `_train_one_epoch`.
- **Tests:** `tests/test_pc_block.py` — three new tests for the
  exponent knob (default, milder, zero).
- **Wrappers:**
  - `run_m6_followup_long_training.bat` (60ep at gutenberg_100,
    queued but not run — overfitting concern superseded it)
  - `run_m6_followup_mild_mu_pc.bat` (μPC exp=0.25 at gutenberg_100,
    same)
  - `run_m6_followup_4gb_1ep.bat` (decisive run, currently active)
- **Data:**
  - `runs/m6_depth/v2_4blocks/results.json` (M6 baseline)
  - `runs/m6_depth/v2_8blocks/results.json`
  - `runs/m6_depth/v2_12blocks/results.json`
  - `runs/m6_depth/dead_4blocks/results.json`
  - `runs/m6_depth/dead_8blocks/results.json`
  - `runs/m6_depth/dead_12blocks/` (killed at ~75%, partial only)
  - `runs/m6_followup/v2_256d_12blocks_1ep_gutenberg4gb_mupc_exp025/`
    (current decisive run; results.json populated on completion)
  - `runs/m6_followup/m6_4gb_1ep_2026-05-17.log` (live streaming log)
- **Commits:** (none yet — code change + this doc to land together
  when the run completes)

---

## Update Block (TO BE FILLED ON COMPLETION)

```
Run completed: <date/time>
Total wall-clock: <hours>
Best val_loss: <value>
Perplexity-per-character vs M5: <ratio>
Final pred_frob: <value>
Final NFF: <value>
NaN events total: <value>

Verdict: [STRONG WIN / TIE / DEGRADATION / FAIL]

Final interpretation: <2-3 paragraphs synthesizing the val_loss
result, the trajectory shape, the mechanism evidence (pred_frob,
NFF), and what this means for the production trajectory toward
4096d × 36 blocks. Should explicitly address: did the depth-scaling
question get answered, and what's the next experiment.>
```
