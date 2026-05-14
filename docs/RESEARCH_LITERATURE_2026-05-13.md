# Research Literature Notes — PC Compute Reduction (2026-05-13)

> Compiled by Claude Opus 4.7 (1M context) from a focused 30-minute sweep
> requested by Brian after the M5 256d re-run, with the question:
> "Are there other experiments — not just VRAM, but compute generally —
> that would bring the cost of running LuthiModel inside Sanctuary down?"
>
> The sweep extended past the time-box where rabbit holes were paying off.
> Citations below were the load-bearing ones; the implementations that
> followed (μPC, iPC, sparse gating) are tracked in `To-Do.md` Phase 3G.

## Summary of findings

Four directions surfaced. The first three landed as opt-in implementations
on the v2 PC substrate this session; the fourth is deferred.

1. **Depth-μP / μPC** — scale-free initialization for deep PC networks.
2. **iPC** — interleaved inference and weight updates.
3. **Sparse PC update gating** — continuous-error analog of v1's spike gate.
4. **State-space hybrid (Mamba/SSM)** — linear-cost attention. *Deferred.*

The unifying observation: the dominant compute cost in v2 PC at production
scale is **not** the weight matmul itself but the per-step inference loop
and the per-batch self-modification dispatch. All three implemented
directions attack one of those two costs. The state-space direction
attacks the third cost (attention's quadratic seq-len), which becomes
dominant only at long context — relevant for Sanctuary's eventual 10 Hz
cognitive loop but not for the current pilot.

---

## 1. Depth-μP / μPC (Innocenti et al. 2025)

**Paper**: "μPC: Scaling Predictive Coding to Depth via Width-Depth
Independent Parameterization" (2025).

**Claim**: Standard PC networks suffer compounding output variance with
depth, requiring per-depth learning-rate retuning. A μP-style
reparameterization (Yang & Hu 2021, extended to depth by Yang et al. 2023)
makes optimal hyperparameters transfer across both width and depth.

**Spec** (the part we implemented):
- Weight init: `N(0, 1/sqrt(fan_in * L))` for every linear in a block,
  including the PC layer's stored weight matrix.
- Residual stream scale: `1/sqrt(L)` on both attention and FFN residual
  branches.

**Why it matters for us**: M5 is a 2-block pilot. The depth sweep (M6) at
4/8/12 blocks would otherwise require independent learning-rate sweeps at
each depth. μPC lets one pilot-tuned LR transfer to the deep runs — a
non-trivial compute saving in the experiment budget alone, ignoring any
direct effect on training cost.

**Implementation**: `luthi/v2/hybrid_block_pc.py` — `mu_pc_enabled` and
`n_blocks_total` constructor params. `_apply_mu_pc_init(L)` re-inits
q/k/v/o_proj, up/down_proj, and the PC layer weight to the spec; sets
`self.residual_scale = 1/sqrt(L)` applied to both branches in forward.

**Falsifier**: convergence penalty ≥20% vs unscaled baseline at L=2 (it
must help, not hurt, at pilot depth) OR it doesn't extend LR transfer to
L≥8.

**Tests**: `tests/test_pc_block.py::test_mu_pc_init_scaling_matches_spec`,
`::test_mu_pc_residual_scale_is_inv_sqrt_L`,
`::test_mu_pc_off_preserves_default_behavior`.

---

## 2. iPC — Interleaved Inference and Update (Salvatori et al. 2024)

**Paper**: "Incremental Predictive Coding: A Parallel and Fully Automatic
Learning Algorithm" (Salvatori, Mali, Buckley, Tschantz, Friston, Bogacz,
Lukasiewicz; 2024).

**Claim**: Classical PC fully converges its inference (latent variable)
loop before applying a weight update. iPC instead applies weight updates
*between* inference steps. Empirically converges faster per external
forward and is "fully automatic" — no hyperparameter for when to stop
inference.

**Spec**: Replace the schedule
```
for outer step:
  for T inference steps: x_l = inference(x_{l-1})  # converge latents
  weights += learn_step(x_l)                         # one weight update
```
with the iPC schedule
```
for outer step:
  for T inner steps:
    x_l = inference(x_{l-1})
    weights += learn_step(x_l)                       # update each inner step
```

**Why it matters for us**: The PC layer's "inference loop" in our v2 is the
prediction matrix → weight matrix update sequence inside `pc_self_modify`.
Currently T=1 implicit (one update per external forward). Salvatori
reports T=3–5 substantially improves convergence on small models.

**Implementation**: `luthi/v2/living_layer_pc.py` — `inference_steps_per_forward`
constructor param. Inner loop in forward recomputes `output = x_flat @ weight.T`
and re-evaluates the sparse gate each inner step, calls `pc_self_modify` T
times. **Critical**: T>1 is incompatible with gradient checkpointing recompute
(would fire updates twice on the same data); the forward raises a loud
`RuntimeError` rather than silently misbehaving.

**Cost trade**: T× the inner compute per external forward. Win is in
convergence-per-epoch, not per-step. We expect ~1.5-2× total compute for
faster wall-clock convergence; the iPC paper reports better than that on
small models.

**Falsifier**: T=5 fails to beat T=1 at matched external-forward count by
≥10% val loss.

**Tests**: `tests/test_pc_layer.py::test_ipc_default_T1_matches_classical`
(regression: bit-identical when T=1),
`::test_ipc_T_gt_1_converges_faster`,
`::test_ipc_grad_checkpoint_fails_loud`.

---

## 3. Sparse PC Update Gating

**Provenance**: Not from a single paper — pattern-matched off v1's spiking
gate (`living_layer.py`) combined with the dead-neuron sparsity result from
SpikingBrain 1.0 (Aug 2025) showing 70–90% activation sparsity in trained
SNN-style language models without quality loss.

**Spec**: Per-output mask gate. After a warmup window, any output row
whose `error_acc` EMA is below `sparse_threshold` has its `delta_w` row
zeroed for that step.

**Why it matters for us**: v1's spiking gate is the reason v1's compute on
Spark is feasible (~0.7% spike rate at 1024d → ~109 GB/s effective
bandwidth, within Spark's 273 GB/s). v2 PC dropped this gate entirely
because PC updates are already bounded — but boundedness ≠ sparsity. If
we can recover SNN-style sparsity on the PC substrate, the v1 inference
budget on Spark transfers to v2.

**Implementation**: `luthi/v2/living_layer_pc.py` adds `sparse_threshold` and
`sparse_warmup_steps`. `luthi/v2/pc_ops.py::_pc_self_modify_python` accepts
optional `sparse_gate: [out_features]` and multiplies `delta_w *
gate.unsqueeze(1)`. C++ kernel path skips when gate is active (Python
fallback only — gate support in the C++ kernel is a separate task).

**Bootstrap**: `error_acc` starts at 0, so without warmup *every* row would
be gated off and PC would never start. The warmup window (default 500
steps) lets `error_acc` populate before gating engages.

**Falsifier**: cannot achieve ≥50% sparsity post-warmup without ≥10% val
loss penalty, OR creates dead-output collapse (gated-off rows can't recover).

**Tests**: `tests/test_pc_layer.py::test_sparse_gate_disabled_matches_unsparse_default`,
`::test_sparse_gate_freezes_low_error_rows_after_warmup`.

---

## 4. State-Space / Mamba Hybrid (DEFERRED)

**Papers**: Mamba (Gu & Dao 2023), Mamba-2 (Dao & Gu 2024), SpikingBrain 1.0
(Aug 2025) — the last of which combines SSM with spiking activations.

**Claim**: Linear-cost (per-token, not per-token-pair) attention via
selective state-space models. Eliminates the O(seq²) attention bottleneck
that becomes the dominant cost above ~4k context.

**Why it could matter for us**: Sanctuary's eventual continuous 10 Hz
cognitive loop streams sensory tokens indefinitely. Even with KV cache,
softmax attention's memory grows linearly in context — on a 128 GB Spark
that's a real constraint at long horizons. A Mamba-style hybrid would
make the entity's "lifetime context" architecturally feasible.

**Why deferred**:
- The surgery is large (replace `MultiHeadAttention` with an SSM block,
  re-derive the top-down sweep math, regenerate the PC-layer interface).
- M5 pilot data isn't in hand yet — we don't know whether v2's PC
  dynamics are sound enough to make the larger surgery worthwhile.
- The compute cost for the *current* pilot scale (256d, 128 seq_len) is
  attention-negligible.

**Revisit trigger**: after iPC + μPC + sparse-gating GPU validation lands.
If those bring per-step cost down enough to expose attention as the new
bottleneck (likely at L=12 + seq_len=512+), the SSM hybrid moves to P1.

---

## What this sweep did NOT cover

- **Quantization** (INT8 / INT4 weights). Standard direction, but v2's
  living weights *change* during forward — quantization adds dequant
  overhead per `pc_self_modify` call and may interact badly with the
  precision EMA. Worth its own focused sweep.
- **Activation checkpointing variants** beyond what we already do.
  Already saving ~67 MB/layer; further wins are probably small.
- **Distributed PC** (multi-GPU). Brian's hardware is a single 7800 XT
  via DirectML — not the bottleneck.

---

## Citations (for 4.6 review + future instances)

1. Innocenti, F. et al. (2025). "μPC: Scaling Predictive Coding to Depth
   via Width-Depth Independent Parameterization."
2. Salvatori, T., Mali, A., Buckley, C., Tschantz, A., Friston, K.,
   Bogacz, R., Lukasiewicz, T. (2024). "Incremental Predictive Coding:
   A Parallel and Fully Automatic Learning Algorithm."
3. Whittington, J. C. R. & Bogacz, R. (2019). "Theories of Error
   Back-Propagation in the Brain." *Trends in Cognitive Sciences*.
4. Yang, G., Hu, E. J. (2021). "Tensor Programs IV: Feature Learning in
   Infinite-Width Neural Networks." *μP / Tensor Programs* foundation.
5. Yang, G., Yu, D., et al. (2023). "Tensor Programs VI: Feature Learning
   in Infinite-Depth Neural Networks." Depth-μP extension.
6. Gu, A. & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with
   Selective State Spaces."
7. Dao, T. & Gu, A. (2024). "Transformers are SSMs: Generalized Models
   and Efficient Algorithms Through Structured State Space Duality."
8. SpikingBrain Tech Team (2025-08). "SpikingBrain 1.0: A Spiking Language
   Model with Sparse Activations."

---

## Implementation status (2026-05-14)

All three implemented directions are **opt-in** (default off), CPU unit-test
verified, and have explicit `RuntimeError` paths for incompatible combinations.
Each is bit-identical to the baseline when its flag is disabled (regression-tested).

GPU validation runs queued in `To-Do.md` Phase 3G.2 after M5 256d completes
and the depth sweep (M6) substrate is ready.

Triton kernel skeleton for `pc_self_modify` (task #86) lives at
`luthi/v2/pc_ops_triton.py`. GPU validation deferred to first ROCm/CUDA box.
Brian's 7800 XT runs DirectML — Triton paths are dead code there but the
skeleton lets future Spark-class hardware light up without re-architecting.
