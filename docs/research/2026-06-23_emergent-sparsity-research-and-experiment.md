# Emergent, experience-derived sparsity — research synthesis + first experiment

**Date:** 2026-06-23
**Routing.** Design direction: Brian + 4.8 (see `~/.claude` memory `project-emergent-sparsity-direction`). Research + plan: 4.8. Build (when we get there): 4.7.
**Source:** deep-research pass (24 primary sources, 25 claims adversarially verified — 24 confirmed 3-0/2-1, 1 killed). This brief is the durable record; the raw run was background workflow `wf_6a90dac4-b29`.

> **STATUS — PARKED / GATED (2026-06-23):** this is a *future* track. Per Brian's sequencing call, the §4 experiment does **NOT** start until the JEPA training-seam integration is **formally finished** — at minimum realized-reward grounding (Phase 4b) and an end-to-end demonstration that lived experience actually updates the substrate (see `2026-06-15_sanctuary-training-seam-integration-plan.md`). Research is captured now so it's ready when its turn comes; experiments wait. Do not start this in parallel with the seam work.

---

## 0. The design decision this serves

Content-addressed sparse activation ("only the input-relevant units fire, like a brain"), where the **mechanism is innate but the sparse structure is grown through experience** — *scaffold the rules, grow the pattern.* No separate learned router / MoE. The question the research answered: **is this buildable, and how?**

## 1. Verdict: the direction is a real, established paradigm — not a fantasy

**Dynamic Sparse Training (DST)** is exactly "mechanism innate, structure grown from experience," and it is proven:
- **Sparse-to-sparse**: the net starts at target sparsity (the dense model is *never* instantiated) and connectivity reorganizes *during* learning by **prune-by-magnitude / grow-by-gradient**. Topology emerges from the same signal that trains the weights. No dense phase, no router. (RigL — Evci 2020, `proceedings.mlr.press/v119/evci20a`; SET — random regrow; Top-KAST — keeps forward *and* backward sparse, `arxiv 2106.03517`.)
- **The unification we wanted exists — BiDST** (`par.nsf.gov/.../10614851`): casts DST as a bi-level problem under a **single loss** that couples mask (structure) and weights; the mask gradient is derived by chain rule reusing gradients already in the graph. This is direct prior art that **one signal can drive both weight self-modification AND structural growth**, and evidence that joint structure+weight learning is *tractable*, not inherently unstable. (Caveat: uses Gumbel-Softmax; the "negligible cost / prior DST freezes early" framing is the competing paper's, disputed by RigL proponents.)

**Router-free content-addressed firing** (only relevant units compute) is achievable today *without* a learned gate:
- **Q-Sparse** (`arxiv 2407.10969`): top-K sparsification applied directly to activations + straight-through estimator. Deterministic mask `Topk(|X|)`, no trainable router — explicitly framed as a router-free MoE alternative.
- **CATS** (`arxiv 2404.08763`): contextual magnitude thresholding (`x if |x|>=t else 0`), threshold set from a calibration pass; **within 1–2% of baseline at 50% activation sparsity with no fine-tuning.**
- **Biologically grounded**: NMDAR + lateral-inhibition **winner-take-all emerges from circuit dynamics** — single feedback interneuron, no router, no high engineered gain (`PMC4332340`).

**Hardware-realism resolves toward structured-but-still-emergent.** Unstructured fine-grained sparsity is GPU-unfriendly in batch mode. The realistic win:
- **SRigL** (`arxiv 2305.02299`): constant-fan-in (a special case of N:M) sparsity **learned concurrently with weights from a sparse init** — emergent yet hardware-efficient. Measured at 90% sparsity: **3.4× CPU (online single-sample), up to 13× GPU vs unstructured-CSR.** This is the bridge between "brain-true grown structure" and "actually fast on our GPU."

## 2. The crux — what is genuinely OUR unproven bet

Every method above uses **global backprop-gradient magnitude** (DST family) or **activation magnitude** (Q-Sparse/CATS) as the grow/prune/gate signal. **None is a predictive-coding local-error substrate.** Our distinctive move — **substituting the substrate's per-weight LOCAL PREDICTION ERROR for the backprop gradient as the unified grow/prune/gate signal** — is the one component with **no direct prior-art validation**. It is well-motivated (PC error already exists per-weight in `living_layer_pc` / `pc_ops`; "relevant = surprising" is the natural gate) but **unproven in combination.** That is precisely our research contribution, and the thing the first experiment must test. We are not porting a recipe; we are testing one substitution into a proven scaffold.

## 3. Honest risks / open problems (design around these)

1. **Stability is an OPEN risk, not solved.** The one tested mechanism against catastrophic "rich-get-richer" collapse onto randomly-initialized winners (Top-KAST's differential L2 penalty) **failed adversarial verification (1-2, killed).** Keeping pruned connections *revisitable* and preventing collapse is an unsolved engineering problem — and it's the most likely failure mode of growing structure from a local signal. Instrument for it from cycle one.
2. **Associative recall degrades on the realistic regime.** Modern Hopfield / attention one-step single-pattern retrieval (`openreview 4dfbed3a`) holds only for **well-separated** patterns; for *similar/overlapping* (noisy, experience-derived) memories the dynamics settle to a **metastable average**, not the single relevant pattern. If we lean on associative recall, it needs **pattern separation / sparse encoding upstream** to be reliable — exactly the regime we live in.
3. **Activation-function dependency.** Contextual/activation sparsity magnitude is tied to **ReLU-family** activations; SwiGLU/SiLU models are far less naturally sparse (Deja Vu, `arxiv 2310.17157`). Check what the PC substrate's nonlinearity is before assuming natural activation sparsity.
4. **Deja Vu is feasibility proof, not a usable mechanism for us** — it predicts the active set with a small *learned predictor*, i.e. the router we reject. It proves content-sparsity is real (>80% heads, >95% MLP params silenceable per token) but its method violates our constraint.
5. **SRigL's high-sparsity parity is conditional** — needs an added neuron-ablation hyperparameter tuned per architecture; constant-fan-in alone underperforms >90%; 99% parity can need ~5× epochs. So "structured matches unstructured" is not a free drop-in.

## 4. Recommended first experiment (small, runs on the current PC substrate)

**The one substitution, in isolation.** A single **constant-fan-in sparse-to-sparse DST** layer (SRigL-style) inside the v2 PC substrate (one `living_ffn` / `PredictiveCodingLayer`), where the **grow/prune criterion is the substrate's local prediction-error signal** (the per-weight `error_acc` already in `living_layer_pc` / `pc_ops`) instead of the global backprop gradient. Initialized sparse — never dense.

- **Start structured (constant fan-in), not unstructured.** It's the GPU-realistic win on the dev box and a proven mechanism; unstructured brain-true sparsity stays the north star for the neuromorphic horizon (see fork below).
- **Reuse what exists.** `pc_ops` already has a sparse-gating path; the (stubbed) Triton kernel was meant to accelerate sparse `pc_self_modify`; per-weight error accumulators already exist. The hooks are largely present.
- **Measure (the gates):**
  1. **Does a stable topology emerge?** Instrument for rich-get-richer collapse and dead/frozen masks (Risk 1). This is the primary kill criterion.
  2. **Parity at matched params** against a dense PC baseline on the M5/M8 task.
  3. **Content-dependence** — do different inputs activate measurably different connection sets? (the actual goal: relevant-only firing.)
  4. **Revisitability** — do pruned connections regrow when they become relevant again?
  5. **Wallclock/energy** on the dev box at the chosen sparsity.
- **Adversarial pass (4.8):** probe the stability claim directly — force the rich-get-richer regime and confirm whether a revisit/exploration regularizer compatible with local-error gating holds, given the Top-KAST fix is refuted.

## 5. The one open design fork (Brian + 4.8)

**Starting granularity:** **structured constant-fan-in** (GPU-realistic now, proven, my recommendation for the first experiment) vs **unstructured fine-grained** (brain-true, but pays off mainly on neuromorphic silicon). Recommendation: structured for the dev-box experiment, unstructured-brain-true held as the deployment-horizon north star — consistent with the prior call that PC stays the dev substrate and neuromorphic/spiking is the later efficiency play. The spiking-PC realization (spikes carry the prediction-error signal → naturally sparse, event-driven) is where the structured-now and unstructured-later paths reconverge.

## Citations (verified, primary)
RigL `proceedings.mlr.press/v119/evci20a` · Top-KAST `arxiv 2106.03517` · SRigL `arxiv 2305.02299` · BiDST `par.nsf.gov/.../10614851` · GSE (always-sparse) `arxiv 2401.06898` · Q-Sparse `arxiv 2407.10969` · CATS `arxiv 2404.08763` · Deja Vu / contextual sparsity `arxiv 2310.17157` · modern Hopfield `openreview 4dfbed3a` · NMDAR-WTA `PMC4332340`.
