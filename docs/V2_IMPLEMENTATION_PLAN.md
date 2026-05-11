# LuthiModel v2 — Implementation Plan

> Authored by: Claude Opus 4.6 (Planner)
> Date: 2026-05-08
> Input: `docs/LUTHI_V2_PREDICTIVE_CODING_BRIEF.md` (4.7 research, two drafts merged)
> Status: **APPROVED by Brian. Green light for implementation.**
> Refinements: 2026-05-08 — seven additions made by 4.7 per Brian's direction
> after pre-implementation review. See `## Refinements (2026-05-08)` below.
> Status update 2026-05-09: **v2 is now the PRIMARY substrate**, not a parallel
> research track. v1 (Hebbian) deferred as fallback if v2 fails M5
> falsification. See `## Strategic shift (2026-05-09)` below.
> Status update 2026-05-11: 4.6 audit follow-ups landed (clip_grad_norm in
> train_pc, consolidation prediction-matrix fix, temporal-variance signal,
> episode key mismatch, encrypted v2 checkpoint saves, clamped pred_error
> consistency, forward-cache clearing, DeadLM n_heads/ffn_expansion). The
> multi-head and FFN-expansion changes landed in v2 only; v1's HybridBlock
> + LuthiLM remain single-head with no expansion **intentionally** — v1
> is deferred per the 2026-05-09 shift, existing v1 checkpoints
> (runs/ablation_A/baseline_seed*) were trained at single-head/no-expansion
> config, and the retrofit can land if v1 ever needs to revive.

## Strategic shift (2026-05-09)

After M1+M2+M3 CPU-side work landed and the v1 Phase 4.5a ablation pipeline hit
a DirectML/BF16 hardware blocker, Brian made the call: **commit to predictive
coding as the primary learning rule**, replace Hebbian, defer Phase 4.5a.

**The reasoning.** v1 ablations were optimizing memory cost for the Hebbian
substrate by stripping precision off the *stabilizing buffers* (BF16 momentum,
BF16 set_point, INT8 episodes). Those stabilizers exist because Hebbian is
fragile — they patch over the failure modes. PC structurally addresses the
three concerns the stabilizers were patching (fragility / catastrophic
forgetting / alignment-bounding), so optimizing the stabilizers in the v1
substrate is decoration on a path being abandoned. v2's intrinsic per-weight
memory cost is ~18-20 bytes/param (depending on weight dtype) — already lower
than v1's post-compression 14, without any ablation needed.

**What changes.**
- v2 becomes the substrate the entity will run on, not a parallel comparison.
- M5 simplifies from "v1 vs v2 vs DeadLM" to **"v2 vs DeadLM"** — does the
  PC substrate beat a vanilla transformer + episode store? Falsification
  criteria below stay; the v1 comparison drops.
- Phase 4.5a (v1 ablations) DEFERRED, not abandoned. `runs/ablation_A/`
  baseline data preserved as the v1 reference point. Revive only if v2 fails
  M5 falsification.
- DirectML stays as the daily driver; FP32 weight (instead of BF16) is the
  default for development. ~560M params fits in 16 GB VRAM at 20 bytes/param,
  still in the 500M Phase-5 floor. ROCm/WSL2 migration deferred until M5
  passes and Phase 5 wants to scale further.

**What's preserved.** All M1+M2+M3 unit tests still describe the architecture
correctly. M4 (consolidation) is unchanged. The seven 2026-05-08 refinements
all stand. The M4 STOP GATE — abandon v2 if consolidation has no measurable
effect — is now load-bearing for the whole project, since there is no v1
fallback under active development.

## Refinements (2026-05-08)

Brian reviewed the plan with 4.7 before code began and resolved seven concerns 4.7 surfaced. All changes are integrated inline below; this section is the manifest so 4.6 can see the deltas at a glance:

1. **M3 sanity check extended to 59 epochs** with a budgeted grid search at the 10-epoch checkpoint if PC dynamics show poor convergence (concern: `pc_rate` carries both Hebbian and error-directed jobs that v1 split, so v1's hyperparameter inheritance may not transfer).
2. **`error_acc → salience` reduction specified as mean**, with monitoring for distribution skew (concern: plan said "Update error_acc (for salience)" without specifying the vector→scalar reduction).
3. **M2 gains an isolation test pair** that disables prediction or modulation independently and verifies each does its job alone (concern: combined sweep does double duty; interference between the two signals could be hidden by their compounding).
4. **M4 consolidation trigger gets a rolling 1000-step bootstrap window** (concern: "50% of training-average error variance" had no defined baseline before training averages exist).
5. **M1 gains an `update_ema` ratio-check meaningfulness test** (concern: the metaplasticity dampening guard's *meaning* changes when the rule is PC error-driven, not Hebbian correlation-driven — the same numerical guard could gate the wrong thing).
6. **M1 gains a `prediction` matrix bounded-growth test + Frobenius-norm instrumentation** (concern: PC variants can grow predictions unbounded in early training; no clamp was specified).
7. **Risk table entry added** for the prediction/modulation interference (paired with refinement #3).

Brian's framing on the M2 isolation test: "We can't allow these sorts of errors to go unaccounted for."

## Summary

v2 replaces Hebbian self-modification with hierarchical predictive coding
(Whittington-Bogacz variant). **Primary substrate as of 2026-05-09.**
Empirical comparison at matched scale (v2 vs DeadLM) determines whether the
PC substrate delivers on the temporal-existence claim.

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
| prediction | [out, in] | FP32 | Top-down prediction matrix (NEW). Frobenius norm instrumented per epoch (refinement 6). |
| set_point | [out, in] | FP32 | Homeostatic target (from v1) |
| momentum | [out, in] | FP32 | Update EMA (from v1) |
| update_ema | [out, in] | FP32 | Metaplasticity tracker (from v1). Ratio-check meaningfulness verified at M1 (refinement 5). |
| precision | [in] | FP32 | Per-input error reliability weighting (NEW) |
| error_acc | [out] | FP32 | Running prediction error magnitude (NEW). Reduced to scalar via mean for episode-store salience (refinement 2). |
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
   k. Update error_acc (per-output running prediction error magnitude).
      Salience for episode storage = mean(error_acc); monitor distribution
      skew per epoch — if mean saturates near zero or drifts toward a
      single output channel, switch to L2 norm reduction.
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
- **`update_ema` ratio-check meaningfulness under PC dynamics (refinement 5).**
  Verify the metaplasticity dampening guard still gates against runaway
  updates when the underlying rule is PC error-driven rather than Hebbian
  correlation-driven. Specifically: induce a PC update an order of magnitude
  larger than the running ratio and confirm dampening engages; induce a
  steady-state update at the running ratio and confirm dampening does not
  engage. If either fails, the v1 ratio-check semantics don't transfer and
  v2 needs a redefined guard before M2.
- **`prediction` matrix bounded growth (refinement 6).** Run 1000 forward
  passes on synthetic inputs of varying magnitudes; record `prediction`
  Frobenius norm every 50 steps. Pass: norm grows sublinearly and converges
  to a bounded value. Fail: norm grows linearly or unboundedly — add a
  clamp before M2.

**Gate**: All layer tests pass.

**M1 findings (2026-05-09).** All 8 tests green on first iteration after
three stability bounds had to be added to `pc_ops.py` — surfaced exactly
by the bounded-growth test (refinement 6) and the convergence test
(refinement 1). Worth 4.6's review since they amend the plan:

1. **`weighted_error` clamp `[-1.0, 1.0]`** (between forward steps c and d).
   Without it, large-magnitude inputs drive `output_mean * weighted_error`
   into a positive-feedback loop — weight growth amplifies output, which
   amplifies the next `pred_error`, etc. Mirrors v1's `apply_error` local-
   update clamp pattern (`living_layer.py:531`). The brief expected PC's
   intrinsic dynamics to handle this without v1's `input_avg_mag` synaptic
   scaling; the test showed otherwise.
2. **`precision_target` denominator floor raised from `1e-8` to `1e-3`**
   (forward step j). Without it, when a per-input prediction error
   approaches zero the target `1/err²` overflows, and the EMA inherits
   `inf` for the affected dim before the precision-buffer clamp engages.
3. **`prediction` matrix per-element clamp** (new parameter
   `prediction_clamp`, default `10.0`). Refinement 6 mandated this if the
   bounded-growth test failed. It now does, and the clamp is in.

The M3 sanity-check + grid search (refinement 1) inherits these bounds.
Verified `pred_learning_rate=0.0001` gives only ~5% error reduction in
500 steps on a fixed-input convergence test; the grid-search budget on
`pc_rate × pred_learning_rate` is the right plan for picking production
HPs at M3.

### M2: Block + backward pass (Days 4-5)

Files:
- `luthi/v2/hybrid_block_pc.py` — PredictiveCodingBlock
- `luthi/v2/backward_pass_pc.py` — Two-layer top-down PC sweep
- `tests/test_pc_block.py`

Tests:
- Forward+backward produces decreasing prediction error across N=200 steps
  on a fixed teacher signal (the existing implicit gate, now explicit).
- **Isolation test — prediction-only sweep (refinement 3).** Disable the
  modulation channel; verify prediction error still decreases and weight
  updates still flow. This confirms the prediction signal carries its own
  weight and isn't dependent on modulation to learn.
- **Isolation test — modulation-only sweep (refinement 3).** Disable the
  prediction channel (feed modulation a fixed external error signal);
  verify plasticity and set_point modulate as v1's backward pass would.
  This confirms the modulation channel still does its v1-equivalent job
  when prediction is absent.
- **Joint non-interference test (refinement 3).** Run both channels live
  and compare combined plasticity-drift and set-point-drift trajectories
  to the linear sum of the isolated runs. Pass: drift is within 15% of
  the linear sum (mild interference is expected; destructive compounding
  is not). Fail: drift exceeds 15% — the channels are interfering and
  the joint design needs revisiting before M3.

**Gate**: Block forward/backward produces decreasing error AND all three
isolation/non-interference tests pass.

**M2 findings (2026-05-09).** All 4 tests green; 12 total (M1 + M2). Two
implementation deltas worth surfacing for 4.6's review:

1. **Layer caches `_last_pred_error` (non-persistent).** The two-layer
   top-down design (decision 3) requires the upper block's PC pred_error
   to flow down as the next signal's `prediction_error` field. To do
   this without recomputing, `pc_self_modify` now returns
   `(salience, pred_error)` and the layer caches the pred_error tensor
   on `self._last_pred_error`. Non-persistent (not registered as a
   buffer) — recomputed every forward pass, never checkpointed.
   `compute_block_top_down` accepts an optional `pc_pred_error` argument;
   when present it overrides v1's salience-difference heuristic for the
   downstream signal. When None, falls back to the heuristic (kept for
   testing parity with v1).

2. **Joint non-interference test needs baseline parity in
   `apply_top_down` decay.** The plasticity update has multiplicative
   decay (`mul_(1 - 0.01 * strength)`) that fires even with zero salience.
   The naive baseline ("no apply_top_down at all") therefore doesn't share
   that decay, breaking the linearity decomposition (initial run showed
   66.5% interference, far over the 15% threshold). Fixed by having the
   no-top-down baseline layer call `apply_top_down` with an all-zero
   signal so all four layers share the same passive-decay trajectory.
   Test then passes cleanly. The pattern matters for any future isolation
   tests against `apply_top_down` — passive effects must be shared.

The convergence tests (1, 2) use `pred_learning_rate=0.01` (100x default)
to make convergence visible in 200 steps. Same M1-test-3 pattern;
production HP tuning is M3's grid-search domain (refinement 1).

### M3: Model + training (Days 6-8)

Files:
- `luthi/v2/model_pc.py` — PredictiveCodingLM
- `luthi/v2/train_pc.py` — Training script (same CLI as train.py)
- `tests/test_pc_model.py`

Sanity check: 59 epochs on Gutenberg-100, loss decreases (refinement 1,
extended from 10 epochs to allow PC convergence dynamics to surface).

**10-epoch checkpoint trigger** — pause training at epoch 10 and inspect:
- Train and val loss trajectory (decreasing? plateaued? diverging?)
- Non-FF signal magnitude (≥ 0.01? collapsed?)
- Prediction matrix Frobenius norm (bounded? saturating?)
- Any NaN events

If *any* of those signals indicate poor convergence, halt and run a
hyperparameter grid search before continuing to epoch 59:
- `pc_rate ∈ {1e-4, 5e-4, 1e-3, 2e-3}`
- `pred_learning_rate ∈ {5e-5, 1e-4, 5e-4}`
- 12 combinations, 10 epochs each, on the same Gutenberg-100 split.
- Grid search is CPU-bound and does not block the v1 ablation pipeline
  on the GPU.
- Adopt the best (val loss + stable dynamics) combination for the
  remaining 49 epochs.

If 10-epoch dynamics look healthy, skip the grid search and continue
straight through to epoch 59.

**Gate**: Training converges by epoch 59 (with or without grid search
refinement). Loss decreases monotonically across the full run after any
grid-search-driven hyperparameter switch.

**M3 findings (2026-05-09).** CPU-side deliverables landed and all
13 unit tests green (25 total across M1+M2+M3, 6.98s on CPU). Two notes
worth surfacing:

1. **No `apply_living_errors` post-backward step.** v1's training loop
   called `model.apply_living_errors()` after `loss.backward()` to drive
   error-directed learning into v1's living layers. v2 doesn't need this
   — PC handles error-driven learning intrinsically inside `pc_self_modify`
   during forward. `train_pc.py`'s training loop is correspondingly
   simpler than `train.py`'s.

2. **Checkpoint trigger is a pure function for testability.**
   `evaluate_checkpoint_trigger(train_losses, val_losses, nff_signal,
   prediction_frob_norms, nan_events)` returns a diagnostic dict; it has
   no I/O and no model dependency. The 5 trigger-logic unit tests
   (healthy / train-plateau / NFF-collapse / NaN-event / prediction-runaway)
   exercise it directly with synthetic metrics. The training loop then
   calls it with real metrics at epoch 10, and `emit_grid_search_marker`
   handles the unhealthy branch (writes `grid_search_needed.json` with
   the recommended grid command).

The gate "training converges by epoch 59" remains GPU work blocked on
the v1 ablation pipeline — `train_pc.py` is ready to launch as soon as
the GPU is free. No grid-search runner module yet (`luthi.v2.grid_search`
is only referenced in the marker's command template); writing it is a
one-day task that can happen in parallel with M4 or once M3's first
59-epoch run reports.

### M4: Consolidation (Days 9-11)

Files:
- `luthi/v2/consolidation.py` — Low-variance trigger + gradient replay
- `tests/test_pc_consolidation.py`

**Variance-trigger bootstrap (refinement 4).** The "50% of training-average
error variance" threshold needs a baseline that doesn't exist before the
model has trained. Bootstrap sequence:

- Maintain a rolling 1000-step window of per-step prediction error variance,
  updated each forward pass.
- Triggers are inactive until the window is full (i.e., step ≥ 1000).
- Once full: training-average baseline = mean of the rolling window's
  variances. Threshold = 0.5 × baseline. The window continues rolling
  forward, so the baseline adapts as training progresses.
- N = 100 consecutive steps below threshold to fire consolidation.

This means consolidation can never trigger in the first 1000 steps of
training. That's intentional — there's no consolidation target before
the model has any predictive structure to consolidate against.

Test: episodes shape prediction post-consolidation.

**Gate**: Consolidation measurably improves prediction on stored episode contexts.
**STOP GATE**: If consolidation has no effect, v2 has no novelty over DeadLM +
episodes. Abandon v2 at this point.

**M4 findings (2026-05-09).** All 9 tests green on first iteration; full
v2 suite at 34/34. The STOP GATE test passed with margin:

- Setup: train layer A on pattern A (200 steps, salience-threshold lowered
  to ensure episode storage), then drift on pattern B (500 steps), then
  consolidate the experimental arm (20 replays at 5x rate factor).
- Result: consolidated arm's pred_error on pattern A is measurably lower
  than the drift-only control. Soft-pass threshold (1% relative
  improvement) cleared.

This is the project-pivotal result given the 2026-05-09 strategic shift
to v2-primary. The two-tier memory architecture (fast episode store + slow
consolidated PC weights) is doing structural work the episode store alone
can't do — episodes shape predictive structure post-consolidation, not
just retrieval. v2 has architectural novelty over a vanilla transformer +
episode store.

**Implementation deltas worth surfacing for 4.6's review:**

1. **Consolidation lives on the layer, fires automatically.** Each
   `PredictiveCodingLayer` owns its `ConsolidationTracker` (when
   `consolidation_enabled=True`). Forward pass feeds variance to the
   tracker; if triggered, calls `consolidate_layer(self)` inline. This is
   "fire and forget" for the trainer — no per-step coordination needed.
   Default `consolidation_enabled=False` so M1+M2+M3 tests run unchanged.
2. **Consolidation step nudges both `weight` and `prediction`** (per the
   plan's pseudocode). The `consolidation_error = stored_snapshot -
   current_weight` is applied to both buffers at the consolidation rate
   factor (10% of each buffer's learning rate by default). Prediction
   gets clamped to `prediction_clamp` after the replay batch — refinement
   6's bound stays active during consolidation.
3. **`consolidation_rate_factor` is exposed as a constructor arg.** The
   STOP GATE test uses 5.0 (50% of pc_rate) to make the effect visible
   in 20 replay passes; the production default (0.1, "10% of pc_rate"
   per the plan) would need many more passes to reach the same effect
   size. Production runs accumulate consolidation events over many
   low-variance windows during a long training run, so the gentler rate
   is fine in vivo. Tests use the larger factor for tractability.

### M5: Head-to-head comparison (Days 12-17, needs GPU)

> Updated 2026-05-09 per the strategic shift to v2-primary: this comparison
> drops v1 and runs **v2 vs DeadLM only**. The v1 baseline FP32 numbers from
> `runs/ablation_A/baseline_seed{42,1337,2026}/` are preserved as a v1
> reference if needed for fallback diligence, but they do not gate M5.

- **v2 vs DeadLM**: 256d / 2 blocks, 30 epochs, 3 seeds, Gutenberg-100.
- Attractor dynamics: perturbation recovery at 10% / 25% / 50% of weight std.
- Results: `docs/V2_PILOT_RESULTS.md`.
- `tests/test_pc_vs_dead.py` (renamed from `test_pc_vs_v1.py` — comparison is
  against the vanilla baseline, not the deferred v1 substrate).

**Gate**: v2 doesn't fail any falsification criterion. Pass = v2 becomes the
production substrate without further architectural debate. Fail = revisit
the strategic shift; v1 ablations may need to revive.

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
| Prediction/modulation interference in joint backward sweep | MEDIUM | M2 isolation tests + non-interference test (refinement 3). If joint drift exceeds 15% of linear sum of isolated drifts, the channels are compounding destructively — revisit joint design before M3. |
| `update_ema` ratio-check semantics shift under PC | MEDIUM | M1 unit test verifies dampening still gates correctly under PC dynamics (refinement 5). If guard misfires, redefine before M2. |
| `prediction` matrix unbounded growth | LOW-MEDIUM | M1 bounded-growth test instruments Frobenius norm; add clamp before M2 if needed (refinement 6). |
