# Luthi Model — Architecture Reference

*Technical companion to the README. **Brought current 2026-10-19.** The
previous version (current to 2026-07-29) described SIGReg as the regularizer
(VISReg replaced it in the current lineage on 2026-08-11). It said nothing
about the block-level output store, and it did not say that the living write
is computed from batch means. This version describes what the code computes,
says which settings the current lineage (`probe_768_visreg` and its
descendants) actually runs, and marks the opt-in options added on
2026-10-19. Constructor defaults are quoted where they matter; when an arm
overrides one and the override is load-bearing, it is named.*

## The block

Every block of the trunk (`luthi/v2/hybrid_block_pc.py`) computes:

```
x = x + s * attention(LN1(x))                  # backprop-trained
x = x + s * ffn(LN2(x))                        # see "the FFN path"
x = output_store(block_input, x)               # living arms only; see "Memory"
```

`s` is the depth-µP residual scale, 1.0 in the current lineage (muPC off
since the 768 family).

**The FFN path.** At `ffn_expansion=1`, which **every JEPA arm on the ladder
uses**, the FFN is a single `d × d` linear map with no nonlinearity:

- **living arm:** a `PredictiveCodingLayer` whose `weight` is a *buffer*. The
  optimizer never touches it, and no gradient from the objective reaches it.
  It changes only by its own local rule inside each forward pass.
- **dead arm** (`dead_ffn=True`): a plain `nn.Linear` of the same shape,
  trained by backprop, with no stores.

So each block is attention plus a linear residual branch on layer-normalized
input. At `ffn_expansion > 1` the path is `up → GELU → FFN → down`, and the
living layer works in the expanded space. In that case top-down modulation
does not reach it (its signal is d_model-shaped).

Attention is bidirectional inside the JEPA encoder (`causal=False`).

## Training objective — JEPA

Implementation: `luthi/v2/jepa_loss.py`, driven by `luthi/v2/jepa_runner.py`.

- **Encoder.** Online encoder only — no EMA target, no target network. The
  JEPA path is `encode()`, not the LM-style `forward()`; the trunk's
  `final_norm` is applied only in `forward()`.
- **Masking.** Disjoint 80/20 per-modality tail (`context_fraction=0.8`). Two
  forwards per step: context-only input first, then the full sequence for the
  targets. Both self-modify the living layers. The disjoint input-side slice
  is what prevents target leakage into the context encode.
- **L_pred.** MSE between predictor output and target-block latents, target
  **detached** (default since 2026-07-28/29). The predictor is a 2-layer
  transformer with a constant action-token stub kept for the m9 interface.
- **Anti-collapse, current lineage — VISReg** (arXiv 2606.02572;
  `luthi/v2/visreg.py`). It has scale, sliced-Wasserstein shape, and center
  terms, applied to **trunk latents** (`sigreg_projection="none"`, enforced
  by the constructor), and mixed convexly:
  `L = (1 − λ)·l_pred + λ·l_reg`, λ = 0.6, K = 1536 projections at 768.
  **Open dose question:** the shape term sums over N, so at batch 32 × 128
  the regularizer measured 98.6–99.99% of the objective for the whole 768
  family. The fix (`shape_normalize=True` with λ re-derived) is decided, not
  yet registered (`docs/DECISIONS.md`, 2026-08-14).
- **Anti-collapse, v1–v5 families and the code default — SIGReg** (LeJEPA,
  Balestriero & LeCun). An Epps-Pulley characteristic-function statistic over
  random 1-D projections against isotropic N(0, I), added to L_pred with
  weight λ (default 0.1; v4/v5 ran 0.2). Applied through a per-modality
  projection head (`"linear"` since 2026-07-28).

**The defect that shaped this section.** For the first five model families
the projection head was `Linear → BatchNorm1d`. BatchNorm subtracts the batch
mean and divides by the batch std — the two quantities SIGReg exists to
constrain — so the anti-collapse term stopped binding on the encoder. With a
non-detached, scale-sensitive MSE target, shrinking the representation reduced
the loss at no cost. Measured: under a 100× uniform shrink, SIGReg on raw
latents moves 0.86 → 706; post-BN it moves 0.566 → 0.545. Every verdict
dated before 2026-07-28 was measured in that regime (see KF7 in
`KEY_FINDINGS.md`). A learnable head between a regularizer and the trunk also
absorbs scale (singular values 0.42–0.55 measured), which is why VISReg runs
on trunk latents.

**Optional next-token term (LLM-JEPA).** `w_ntp > 0` adds a causal
next-token loss through `final_norm → output_proj`. When `w_ntp = 0`, as in
the VISReg lineage, `output_proj` receives no gradient and stays at
initialization (`KNOWN_INCOMPLETE.md`).

## The living write

The self-modification step is `pc_self_modify` in `luthi/v2/pc_ops.py`, called
inside every non-frozen forward of every living layer. It predicts the layer's
own input from its output through a learned `prediction` matrix, and writes
the error back into the weight.

### `write_rule="mean_field"` — the legacy rule (default; every run before 2026-10-19)

```
ō   = mean_t output            # over ALL B×L positions of the batch
x̄   = mean_t input
ē   = x̄ − ō @ P                # pred_error
w   = clamp(drive(ē) * trust, −1, 1)
ΔW  = outer(ō, w) * plasticity * pc_rate        # then metaplasticity, gain, homeostasis
ΔP  = outer(ō, clamp(ē, −1, 1)) * pred_learning_rate
```

This is an outer product of **batch means**, and that is the most important
fact about the substrate. Measured on 2026-10-19
(`docs/research/2026-10-19_the-living-write-is-mean-field.md`):

- **It cannot see content within a batch.** Two batches with the same
  per-dimension means produce the same write, whatever their tokens
  (`tests/test_living_write_rule.py`). After training, the per-token
  covariance it discards measured 0.6–4.3× the term it keeps.
- **Its accumulated change is ~rank-1.** Every write's output side is `ō`,
  and `ō` barely moves between batches (cosine ≥ 0.9996 at 4,096 positions
  per batch). So `weight − set_point` concentrates 96–99.5% of its energy in
  one direction, aligned with `ō` (|cos| 0.97–0.99). Its effect on a token is
  `Δoutput(x) = η·(w·x)·ō`: a fixed direction with per-token magnitude.
- **It explains why the drive goes quiet on all-novel data** (2026-07-29):
  the rule never sees documents, only an average that is stationary from the
  first batch.

Predictive-coding weight updates are usually formulated per sample, as
products of local error and activity. This rule takes the product of the
averages instead.

### `write_rule="covariance"` — opt-in, 2026-10-19

The same rule computed from per-token statistics, with the batch mean
removed:

```
e_t  = x_t − o_t @ P                            # per-token error
W    = mean_t (o_t − ō) ⊗ clamp(κ · (e_t − ē) · trust, −1, 1)
W   *= min(1, ‖outer(ō, w)‖_F / ‖W‖_F)          # SAME BUDGET as mean_field
ΔW   = W * plasticity * pc_rate
ΔP   = mean_t outer(o_t, clamp(e_t, −1, 1)) * pred_learning_rate
precision → 1 / mean_t(e_t²)
```

`κ` is the drive mode's scalar (1 for raw, 1/rms for rms, gain/forecast for
surprise). The **cap** makes each write's Frobenius norm at most the norm of
the mean-field write the same batch would have produced. The rule therefore
changes where plasticity goes, never how much. That is the attributable
comparison. Uncapped, the centered Hebbian term ran away (3.0 → 1.3e10 in
3,000 steps under stress). Capped, it grows less than the legacy rule at
production rates. The rule is Python-only: requesting it forces the Python
path, never a silent C++ run of the legacy rule. On CPU smoke it removes the
rank-1 signature (top share 0.10–0.13). Whether it produces
document-specific change at scale is registered as E3.

### Drive and trust (shared by both rules)

- **Trust.** `trust` derives from `precision`, a per-input reliability
  estimate that tracks 1/error². With absolute trust (default) the ±1 clamp is
  fully saturated at production precision scales (measured 100% on real
  checkpoints), so the legacy update degenerates to
  `outer(ō, sign(ē))·plasticity·pc_rate`. **Relative trust**
  (`relative_trust=True`, the v5 lineage) weights by precision/median,
  clamped to ratio bounds. That keeps trust O(1) and ~30% saturated.
  `precision.median()` has no DirectML kernel; it falls back to CPU there.
- **Drive** (`drive_mode`). `"raw"` (default; the current lineage) is the
  error itself, which self-extinguishes as the layer learns. `"rms"` divides
  by a running RMS; it is a **no-op under a saturated clamp**, kept for A/B.
  `"surprise"` is excess error over a Holt forecast of the error scale, gated
  at k deviations. It requires relative trust, enforced by a raise. Measured
  duty: 0 on stationary input, 0 on i.i.d. draws, ~0.035 after a shift.
  Under the mean-field rule, shuffled training batches *are* i.i.d. draws of
  the batch mean, which is why the surprise drive has nothing to respond to.

### Metaplasticity, gain, homeostasis

`update_ema` dampens updates far above the weight's typical size
(`adaptive_factor`). The inverted-U learning gain (`learning_gain_enabled`,
on in the v3+ lineage) amplifies directed, resolving change, floored at 1.0.
Homeostasis pulls `weight` toward `set_point` at `homeostatic_decay` (1e-3);
`set_point` itself adapts toward `weight` at 1e-6. So `weight − set_point`
is the living displacement from home, and the living-change spectrum gauge
reads exactly that. The plasticity taper (trainer-scheduled `rate_scale`)
scales the learning channels only.

## Memory

### Layer-level episode store (inside each living layer)

On a salient forward the layer snapshots itself: weight matrix, a 64-d
context, the mean input pattern, and a salience score. A later forward whose
context matches above `episode_recall_threshold` (0.7 in the v3+ lineage)
blends the stored weight delta into the active weight at `episode_blend`.

- **Admission.** Legacy (`adaptive_episodes=False`, the current lineage) uses
  a fixed salience threshold of 0.1 against per-block medians of 0.001–0.004.
  In practice the store is empty: in the 768 family's seed 97, blocks 0–4
  stored zero episodes in 54,000 steps. `adaptive_episodes=True` swaps in a
  Holt-forecast surprise test with a refractory gap and age-weighted
  stochastic eviction. That is decided for a future family, not yet run.
- **Context.** `context_mode="mean"` (default) is the normalized projection
  of the batch-mean input. Measured cosine across independent batches is
  ≥ 0.9995, which is why recall used to fire every step.
  `context_mode="deviation"` (opt-in, 2026-10-19) subtracts a slow running
  reference first, so the context encodes how this input differs from
  typical. The reference is persisted through living extra state, and frozen
  forwards never update it.

### Block-level output store (`luthi/episode_store.py`) — what it actually does

Every living block's output passes through an `EpisodeStore`. Its context is
the block input pooled over **batch and sequence**; its value is the output
pooled the same way. Salience (mean `|output|`) clears its 0.1 threshold on
every call. In a batched trunk the pooled contexts are near-identical, so
recall fires on every forward (duty 1.000 measured), and the block output
becomes

```
output ← 0.7 · output + 0.3 · (a stored batch-mean output)
```

This shrinks every token's deviation from the mean by 30% per block and
re-injects a constant. It happens in training and in frozen eval;
`freeze_plasticity` skips its write, not its recall. On CPU smoke it drives
offset dominance up through depth (0.78 → 0.93), the same shape as the 768
family's profile. **The dead arm has no such store**, so every living-vs-dead
comparison has carried it as an uncontrolled difference.
`block_episode_store="off"` (opt-in, 2026-10-19) removes it. Its counters
(`output_episode_recall_duty`, `_blend_share`, `_last_sim`) are emitted per
block at deep cadence.

### Consolidation

A 1000-step window of per-step prediction-error magnitude, with a baseline
frozen once at the end of warmup. 100 consecutive sub-threshold steps fire
the trigger. Two replay pathways:

- **gradient replay** pulls the weight toward stored snapshots;
- **attractor replay** (Salvatori et al. 2023) re-presents stored mean input
  patterns through the PC dynamics. It measured a 12.9–16.1% reduction in
  prediction error on stored patterns (2026-07-29). Since 2026-10-19 it runs
  with the layer's own trust regime; before, it clamped a relative-trust
  layer's precision ledger into [0.1, 10] on every replay.

Replay of an empty store does nothing. `consolidation_noop_fires` counts
those, and they were most fires in the 768 family.

## Homeostatic band and sparse gate

The sparse PC gate (`sparse_threshold > 0`; off by default) silences output
rows with low recent error. A collapsed row also has low error, so the
homeostatic band (`homeostatic_band_enabled`, off in the current lineage)
reopens rows quiet for the wrong reason. It does this by synaptic-scaling
style boosting and damping relative to the median, with a ~1000-step time
constant.

## Top-down sweep

After the bottom-up pass, in **training mode only**
(`self.training and backward_pass_enabled`), higher blocks send salience and
prediction-error signals down. These modulate lower blocks' plasticity and
set points. It is predictive processing, not backpropagation, and it is on
in the v3+ lineage.

## Evaluation regime

- **Registered evals are frozen.** `luthi/v2/eval_heldout.py` runs under
  `freeze_plasticity` + `no_grad` + `eval()`, with bitwise purity pinned by
  tests. Note that `eval()` alone does **not** stop a living layer from
  self-modifying; there is no `self.training` gate. Every forensic script in
  `scripts/` now freezes, enforced by `tests/test_scripts_freeze_plasticity.py`.
- **NMSE** is l_pred over the target block's variance. It is scale-free,
  but **it is flattered by degeneracy**: a rank-1.9 trunk posted the best
  NMSE of the August depth investigation. Read it together with effective
  rank and chorus rank, never alone.
- **The linear probe** reads latent at t → token t+1 from a bidirectional
  encode, which can see t+1. A shuffled-label floor calibrates chance;
  **the input-embedding baseline** (`stream="embedding"`, recorded in every
  `pilot_result.json` since 2026-10-19) calibrates what the residual carries
  for free. In the 768 family the embedding stream beat every block.
  `offset` and `causal` controls exist for audit item B7.
- **Inference-time adaptation** — the README's goal — is measured by
  `luthi/v2/adaptation_eval.py`: frozen, then read-own, then read-other.
  Its key number is document-specific change (B − C). A dead arm must be an
  exact null.

## Instruments for the living channel itself

| where | field / script | reads |
|---|---|---|
| deep cadence, per block | `living_change_top_share`, `_eff_rank`, `_rel_norm`, `_out_align` | the shape and size of `weight − set_point`; top share and align near 1 are the mean-field signature |
| deep cadence, per block | `output_episode_recall_duty`, `_blend_share`, `_last_sim` | what the block output store is doing |
| deep cadence, per block | `contrib_var_ratio`, `contrib_chorus` | what each block adds (h_out − h_in), not the residual stream it passes on |
| offline | `scripts/living_write_probe.py` | spectrum, batch-mean stationarity, write diversity, store activity, on any checkpoint |
| offline | `scripts/adaptation_eval.py` | A / B / C inference-time adaptation |
| offline | `scripts/per_block_probe.py` | per-block probe with the embedding baseline; `--offset`, `--causal` |

Older substrate fields (`drive_duty`, `episode_writes`, `recall_fires`,
`consolidation_noop_fires`, `nonfinite_forward_skips`, ...) are documented
at their emit site in `jepa_runner.py`.

## Rich parameters

| signal | tracks |
|---|---|
| **weight** | current value (a buffer; never a Parameter) |
| **prediction** | how the layer's output predicts its input; the PC error source |
| **set_point** | homeostatic home; adapts at ~1e-6/step |
| **momentum** | EMA of recent writes |
| **plasticity** | per-input rate multiplier, clamped 0.01–10.0, modulated top-down |
| **update_ema** | metaplasticity: typical update size per weight |
| **precision** | per-input reliability → trust |
| **error_acc** | per-output running error magnitude: salience, sparse gate |

Per-layer state (episode bank, admission forecast, drive forecast, band,
recall statistics, `context_ref` for deviation contexts) lives alongside.
State that is not a tensor buffer — consolidation history, slow traces, the
sparse-gate counter, the deviation reference — persists through
`luthi/living_extra_state.py`, because v2 checkpoint loading is
`strict=True`.

## Execution

- **Backends.** ROCm is the default declaration (`LUTHI_BACKEND`, since
  2026-08-21; 238 ms/step at 768×8 vs 836 on DirectML). DirectML is
  supported. CPU is allowed only deliberately (`LUTHI_ALLOW_CPU=1`), which
  is also how the test suite runs on a CPU box.
- **C++ `pc_ops`.** It is JIT-compiled when possible. It does **not** build
  on the ROCm wheel (unresolved `c10::ValueError`; `DECISIONS.md` 08-19). The
  Python reference is identical math for `mean_field` and costs 3–17% of a
  step. The driver records which path ran (`pc_ops_backend`) and no longer
  refuses to run without C++ (`LUTHI_REQUIRE_CPP=1` restores the refusal).
- **Triton.** A skeleton only (`KNOWN_INCOMPLETE.md`).

## Spiking dynamics (v1, reference baseline)

The v1 `SpikingLivingLayer` adds LIF membranes, thresholds, refractory
periods, and delay buffers, with self-modification gated by spiking. v2 is
non-spiking and recovers sparsity through the sparse PC gate (*which weights
update*, not *which neurons fire*). v1 is preserved as a reference, not
trained.

## Not on the training path

`luthi/v2/m9/` implements the action layer: expected free energy, MCTS
planning, habit net, preferences, activity bands, rest actions. It is
unit-tested and red-teamed, and **not** referenced by `jepa_runner.py` or the
trunk. The JEPA predictor's constant action-token stub keeps the interface
ready for it.
