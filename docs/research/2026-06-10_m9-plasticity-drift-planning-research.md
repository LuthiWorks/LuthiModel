# M9 Research — planning over a drifting living-weight model (Seam-C closure)

**Status:** Research note closing the §15 thread (per-cycle plasticity-drift bounds + planning under a non-stationary learned model). Written 2026-06-10. Feeds plan §4 (staleness machinery). Builds-on/answers `2026-06-10_m9-build-plan.md` §4 + §15.
**Routing.** Research: 4.8. Folds into the M9 plan §4 at build time (deltas in §6 below). No design decision required.
**Provenance.** Verified deep-research pass (22 primary sources, 95 claims, 25 verified 3-vote, 23 confirmed / 2 refuted). All-primary (Ba/Hinton 2016, TTT 2024, Miconi 2018, Backpropamine 2020, MVE 2018, Continual-Dreamer 2022, DRAGO 2025, SW-UCRL/Cheung 2020, Gajane/Ortner/Auer 2019).

---

## 0. Bottom line

The thread closes **reassuringly**. Every mitigation already in plan §4 is **individually grounded** in the literature, and we gain one lever we didn't have: **per-cycle drift is boundable by construction, not just measurable.** The exact combination (inference-time-plastic substrate feeding a persistent cross-cycle MCTS tree) is **confirmed absent** — we are inventing it, but from proven parts. One genuinely new risk surfaced: **loss-of-plasticity** over indefinite continuous running (a 10 Hz loop never stops), which warrants a new watch-item.

---

## 1. (A) Per-cycle drift is a settable trust-region knob (PROVEN — high confidence)

Inference-time-plastic ("living") weights are an **intermediate-timescale, transient, recency-weighted** memory (Ba/Hinton 2016: "slower than activities but much faster than the standard weights"; the trace `A(t+1)=lambda*A(t)+eta*h h^T`, `lambda<1` = built-in recency decay). The per-step change is bounded by **tunable/learned quantities**:

- **Learned learning rate `eta`** (TTT 2024 learns `eta` as a parameter; per-step update = one self-supervised gradient step).
- **A trainable per-connection scale `alpha`** that *structurally caps* the plastic component (Miconi 2018: effective weight `= w + alpha*Hebb`, `alpha=0` fully fixed / `w=0` fully plastic; Backpropamine: "`alpha` ... determines the **maximum magnitude** of the plastic component" since the Hebbian trace is clamped to `[-1,1]`).
- **Explicit per-step stabilizer is mandatory** — Hebbian plasticity is "inherently unstable"; decay or hard-clip is required every step (Backpropamine `Clip(...)`). **Refuted:** Oja-alone giving decay-free indefinite stability (1-2) — do *not* rely on normalization without decay/clip.
- **Neuromodulatory gating `M(t)`** can null plasticity entirely (`M(t)=0` -> no update), a network-computed throttle.

**Consequence for us:** Seam-C's "measure the per-cycle drift" upgrades to "**bound it by design.**" `alpha` (max plastic magnitude) + learned `eta` + clip + gating give a direct trust-region on `||dtheta||` per cycle. The drift premise of the §4.v reframe is not just verifiable — it is *controllable*.

---

## 2. (B) Planning over the drift — three proven principles, each maps to a plan-§4 mitigation

Proven results live in two adjacent literatures; all three transfer:

1. **Horizon-bounding (MVE, Feinberg 2018).** Model-based rollout error *compounds with rollout length*, so bound imagined depth `H` then bootstrap from a learned value. **This is independent of cross-cycle staleness** — a *separate* reason to cap rollout depth. -> Confirms a bounded `H` for the MCTS rollouts (plan §4 / step-1 spec horizon).
2. **Recency-weighting with a characterizable optimal window (SW-UCRL / Cheung 2020; Gajane/Ortner/Auer 2019).** A sliding-window estimator *tracks rather than converges* to a drifting MDP, with dynamic regret `O(B^(1/3) T^(2/3))` under total-variation budget `B`, and there is a **characterizable optimal window size** (bias-from-stale vs. variance-from-too-little-data). Our accumulated MCTS visit/value statistics **are** a recency-weighted estimator over a drifting model. -> This is the formal backbone for plan §4.i recency-decay. It gives the **form** of the answer (optimal decay/window trades staleness against sample sufficiency) but **not the constants** (tabular, not neural — see caveat).
3. **Distill-toward-a-frozen-snapshot (DRAGO 2025; Continual-Dreamer 2022; Life-long World Model 2023).** Continual MBRL keeps a drifting model usable mainly via data-side mechanisms — replay, generative replay, and **L2 distillation toward a frozen previous model** (`lambda*||T_old - T_i||^2`). -> This is the closest precedent for plan §4.iv's **stability-held planning head** (snapshot refreshed every `K` cycles). Caveat: empirical regularizer, **no formal sufficiency bound.**

---

## 3. Confirmed gap (as expected)

**No retrieved primary source combines an inference-time-plastic substrate with a persistent cross-cycle planning tree.** No measured per-cycle drift magnitude for a JEPA-style predictor; no treatment of MCTS/MuZero cached value/visit staleness when the *learned model* changes between searches; the deadly-triad-with-a-moving-*world-model* (not just value function) is effectively unanswered. The prior pass reported this absent; this pass **confirms** it. We build from proven parts, but the assembly is ours — exactly as the plan already assumes.

---

## 4. New risk surfaced — loss-of-plasticity over indefinite running

Not previously on our radar: deep nets adapted continually **progressively lose the ability to adapt** — "the activation footprint becomes sparser, contributing to diminishing gradients" (Abbas 2023) -> dead units -> near-zero gradients -> effectively frozen weights. **A 10 Hz loop runs indefinitely, so this is a real long-horizon failure mode** for the living-weight substrate (not just M9 — it affects the whole always-on design). Mitigations in the literature: CReLU, regenerative/L2-regenerative regularization, plasticity injection. **Recommend a new watch-item:** instrument activation-sparsity / dead-unit fraction over long runs; candidate kill **K-M9-9 (plasticity loss)** if sparsity climbs past a trending band. (Lyle et al. caution there is no single sole cause — treat as a monitored syndrome, not one knob.)

---

## 5. Refuted (do not build on)

1. **Oja's rule = decay-free indefinitely-stable adaptation** (1-2). Keep an explicit decay/clip; normalization alone is not a stability guarantee.
2. **Life-long World Model's mixture-of-Gaussians structurally prevents drift** (1-2). Not a reliable structural anti-drift mechanism.

---

## 6. What this folds into the M9 plan (deltas for §4, at build time)

- **§4 drift handling gains a *control* arm, not just a *measure* arm:** set the substrate's plastic scale `alpha` / learned `eta` / clip / gating as an explicit per-cycle drift trust-region. "Measure `||dtheta||`" (step-1 spec §5) stays, but `alpha` lets us also *cap* it. [correctness — coordinate with the substrate owners; `alpha`/clip are substrate-level knobs, not M9-only.]
- **Bound rollout depth `H`** for an independent reason (MVE compounding error), not only for within-cycle weight stability.
- **Plan §4.i recency-decay = a sliding-window estimator** with a *characterizable optimal window*; tune the decay to the measured per-cycle drift budget (the SW-UCRL bias/variance tradeoff is the principle; constants are empirical for our neural model).
- **Plan §4.iv held-head = DRAGO-style snapshot+distillation** — keep the frozen planning model *and* (optionally) distill the live model toward it to limit divergence; refresh every `K` cycles.
- **Add K-M9-9 (plasticity-loss watch):** activation-sparsity / dead-unit fraction trending up over long runs.

None of this changes a design decision or the step-1 build; it sharpens §4 and adds one long-horizon watch-item.

---

## 7. Sources (verified primary set)

- **Ba, Hinton et al. 2016** — *Using Fast Weights to Attend to the Recent Past.* arXiv:1610.06258.
- **Sun/Wang et al. 2024** — *Learning to (Learn at Test Time): RNNs with Expressive Hidden States* (TTT). arXiv:2407.04620.
- **Miconi et al. 2018** — *Differentiable plasticity.* PMLR v80.
- **Miconi et al. 2020** — *Backpropamine* (neuromodulated plasticity; Clip stabilizer). arXiv:2002.10585.
- **Feinberg et al. 2018** — *Model-Based Value Expansion (MVE).* arXiv:1803.00101. (horizon-bounding)
- **Continual-Dreamer 2022** (arXiv:2211.15944); **Life-long World Model 2023** (arXiv:2303.06572); **DRAGO 2025** (arXiv:2503.04256) — continual MBRL; replay / distill-to-frozen-model.
- **Abbas et al. 2023** — *Loss of plasticity in continual deep RL.* arXiv:2303.07507.
- **Cheung et al. 2020** (arXiv:2006.14389); **Gajane/Ortner/Auer 2019** (arXiv:1805.10066) — sliding-window UCRL, non-stationary-MDP dynamic regret `O(B^(1/3) T^(2/3))`, optimal window.
- **Refuted/with caution:** Oja-alone stability; Life-long-WM MoG anti-drift.
