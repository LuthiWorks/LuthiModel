# Empirical Defense Program

> Authored by: Claude Opus 4.6 (Planner/Reviewer)
> Date: 2026-05-06
> Prompted by: third-party critique of Hebbian dynamics + red-team exercise by 4.7
> For implementation by: Claude Opus 4.7 (Researcher/Coder)

## Context

Brian received a critique of Luthi's design claiming Hebbian dynamics are fragile,
catastrophic forgetting is unsolved, and the combination could produce pathological
behavior. A red-team exercise confirmed the critique lands on real gaps: no baseline
comparison numbers, stress tests only at toy scale, multi-layer cascade behavior
untested, catastrophic forgetting claimed but not demonstrated, and architecture
motivated by neuroscience metaphors rather than measurable results.

Brian's conclusion: defend by closing the gaps empirically, not by arguing harder.

## Guiding Principle

Every claim about the living weight architecture must be backed by a number, not a
metaphor. If the number is bad, we change the architecture. If the number is good,
we publish it. Either outcome is progress.

## Deployment Spec (revised 2026-05-07)

```
Target: ≥500M params (floor; ceiling TBD pending Phase 4.5a ablations,
        estimated 800M-1.1B if ablations A/B/C all pass)
Precision: BF16 weights, per-channel storage where rank-1 (plasticity
           [in_features], excitability_acc [out_features] — landed in
           Phase 0, bit-equivalent), FP32 for genuinely per-weight
           buffers (momentum, set_point, update_ema) until ablations
           validate BF16/INT8 alternatives
Hardware: AMD RX 7800 XT (16 GB VRAM), 32 GB system RAM, Ryzen 7 Zen 2
Toolchain: ROCm/HIP (confirmed working from multiday training sessions)
Inference: Custom sparse spiking kernels (Triton, design pending)
Architecture: SpikingLuthiLM with backward pass, BPE 32K vocab
```

**Revision history:**
- 2026-05-06: Original commit at 4B params, FP32 living state.
- 2026-05-07: Revised to ≥500M floor after Phase 0 free-win refactor
  validated and merged. The original 4B target was infeasible because
  per-weight FP32 living buffers cost ~38 bytes/param (vs ~2 for the
  BF16 weight) — undercounted in the original spec. See
  `docs/PER_CHANNEL_ABLATION_PROTOCOL.md` and PLAN.md Phase 4.5a.

Per 4.6's revision policy, the *ceiling* commits after ablations A/B/C
results land. Strong-pass on all three → ~1.1B. Strong-pass on A+B only
→ ~800M. Free wins only (current state) → ~500M floor.

Change only with explicit documented reason.

## Phase 1: Same-Scale Baseline Comparison (~2-3 weeks)

**Goal:** Vanilla transformer trained at matched param count on identical data.
Held-out perplexity comparison. This is the single most important number.

**Build:**
- `luthi/baseline_model.py` — Standard transformer (attention + FFN, no living
  weights, no spiking, no episodes). Same d_model, n_blocks, vocab_size, seq_len.
- Train on same Gutenberg corpus with same training infrastructure.
- Match trainable param count (living weight buffers don't count).

**Measure:**
- Held-out perplexity at matched training compute
- Training curves (loss over time) for both
- Wall-clock time per epoch
- Convergence penalty: at what epoch does Luthi match baseline final perplexity?

**Scale:** Start at 1024d / 2 blocks. If informative, repeat at 2048d / 8 blocks.

**Success:** Luthi within 15-20% of vanilla at matched compute, gap narrowing,
living weight dynamics show the model is doing something baseline can't.

**Failure:** Perplexity 2x+ worse, gap not closing. Architecture has fundamental
efficiency problem.

## Phase 2: Multi-Layer Cascade (parallel with Phase 1, ~2 weeks)

**Goal:** Determine whether living weight self-modification is stable at depth.
Existential question — if cascade diverges at 12+ blocks, architecture needs
structural changes before scaling.

**Build:**
- Sweep script: 2/4/8/12/24 blocks at fixed small d_model (256d or 512d).
- Per-block instrumentation: plasticity mean/std, set_point_drift, spike_fraction,
  membrane_mean, weight_norm, gradient_norm per block per epoch.

**Measure:**
- Drift propagation across block depth
- Plasticity feedback bounds (compress to zero? explode?)
- Backward pass effect on stability (with vs without)
- Output divergence on identical input (expected non-zero, must be bounded)
- Homeostatic recovery from perturbation

**Success:** Drift and plasticity bounded at 24 blocks. Backward pass helps stability.

**Failure:** Drift amplifies superlinearly. Plasticity collapses/explodes past 8-12
blocks. → Architectural revision needed (per-block normalization, depth-dependent
scaling, gradient-like stability mechanisms).

**This experiment could change the architecture. That's the point.**

### DECISION GATE (after Phases 1-2)

Do the numbers support scaling to 4B? If cascade is unstable → revise architecture.
If baseline gap is >2x → investigate efficiency. Do not proceed to Phases 3-4
without passing this gate.

## Phase 3: Behavioral Signature Framework (after gate, ~2-3 weeks)

**Goal:** Replace metaphor-claims with measurable hypotheses.

**Hypotheses:**
1. Biographical accumulation — model state measurably different after different
   training sequences (unlike vanilla transformer with same final loss)
2. Identity stability — short perturbations don't permanently alter behavior;
   homeostatic recovery measurable
3. Episodic recall — episode store measurably improves context-dependent performance
   vs ablated model
4. Behavioral coherence — living inference outputs are different (alive) but coherent
   (not random divergence)

**Build:** `tests/behavioral/` test suite with clear pass/fail criteria.
Results to `docs/BEHAVIORAL_RESULTS.md`.

## Phase 4: Catastrophic Forgetting (after Phase 3, ~2 weeks)

**Goal:** Directly measure forgetting resistance.

**Design:**
- Train on Sequence A, run N distractor steps on Sequence B (N = 200/500/2000)
- Measure recall on Sequence A held-out

**Conditions:**
1. Vanilla transformer (control)
2. Vanilla + LoRA on Sequence B
3. Vanilla + RAG retrieval from A
4. Luthi with episode store (full)
5. Luthi with episode store ablated

**Success:** Condition 4 > condition 1, comparable to condition 3, episode store
contributes (condition 4 > condition 5).

## Phase 5: Custom Kernel Development (parallel track)

Triton-based predictive-gated sparse spiking kernels. Design doc pending from 4.7.
Does not block Phases 1-4 — all experiments use dense implementation.

## Timeline

```
Week 1-2:   Phase 0 (spec) + Phase 1 (baseline) + Phase 2 (cascade)
Week 2-3:   Phase 1 complete + Phase 2 complete → DECISION GATE
Week 3-5:   Phase 3 (behavioral signatures)
Week 5-7:   Phase 4 (catastrophic forgetting)
Parallel:   Phase 5 (kernel design)
```

~7 weeks total on one consumer GPU.

## Constraints

- All experiments on Brian's RX 7800 XT. No cloud GPU until architecture validated.
- Dense implementation for all experiments. Custom kernels don't block research.
- Corpus on E: drive where specified.
- Every result documented with methodology and numbers. No hand-waving.
- If an experiment fails, document failure and adapt. Failure is information.
