# M8 Brief — Collapse Research + 4.8 Cold-Eye Revisions (v0.4)

**Status:** Review artifact. Supersedes the relevant parts of M8 brief v0.3 (§1, §4, §5, §6, §9). Decisions still owned by Brian / 4.7 are marked **[DECISION]**; everything else marked **[FIX]** is a correctness change with no judgment required.

**Author:** Claude Opus 4.8 (1M context) — review & debugging line.

**Date:** 2026-06-05

**Provenance.** Consolidates (a) the 4.8 cold-eye correctness pass on brief v0.3, and (b) two adversarially-verified deep-research passes on representation collapse (each claim survived 3-vote verification; 48 confirmed, 2 refuted-and-excluded across both passes). Citations are primary sources (arXiv id / venue). This is the artifact the v0.3 routing line ("pending 4.8 cold-eye pass on the spec") was waiting on.

**Routing from here.** Brian settles the **[DECISION]** items in §9. 4.7 implements the corrected §1/§4/§5/§6 against a fresh instance. The 256d pilot resolves everything marked `[pilot-set]`.

---

## 0. Executive summary (read this if nothing else)

1. **The design is sound and literature-endorsed.** The EMA-target + VICReg variance/covariance recipe is exactly what C-JEPA (NeurIPS 2024) converged on for the JEPA family. The collapse worry is real but the chosen toolkit is the right one.

2. **There are two coherent published anti-collapse recipes, and our brief is a third mix.** V-JEPA / V-JEPA 2 use **L1 loss + EMA asymmetry only** (no VICReg). C-JEPA / VJ-VCR use **L2 + EMA + VICReg**. Brief v0.3 specifies **L2 + VICReg**, which nobody specifically validates. Recommend moving to **L1 + VICReg** (the conservative belt-and-suspenders option). The L1/L2 call is **[DECISION]** for 4.7 (optimization-dynamics judgment).

3. **The one genuine unknown is narrow and the pilot is built to probe it.** No published work studies collapse when weights self-modify *during inference* ("living weights"). But the nearest precedent — LPL (Halvagal & Zenke, *Nature Neuroscience* 2023), a local, backprop-free predictive rule — shows that under such dynamics a VICReg-style **variance term is *required*** to prevent collapse, not redundant. This (a) argues we should keep VICReg firmly, and (b) turns the pilot's EMA-vs-VICReg ablation into a hypothesis test, not a guess.

4. **Pilot at 256d first [DECISION — recommend yes].** It converts ~5 paper arguments (thresholds, memory fit, throughput, the EMA/VICReg ablation, collapse-on-this-substrate) into measurements in days rather than committing ~2 weeks of 1024d compute.

5. **The brief's collapse instrumentation was necessary but insufficient.** It watched complete collapse (per-dim std) and pairwise dimensional collapse (off-diagonal covariance) but would miss **rank collapse** and **local collapse**. §4 below adds spectrum/rank/LID metrics.

---

## 1. Research findings (cited, adversarially verified)

### 1.1 Taxonomy of collapse
- **Complete collapse:** encoder outputs a constant; signature = vanishing per-dimension variance. (Jing et al., ICLR 2022, arXiv:2110.09348)
- **Dimensional collapse:** embeddings span a low-rank subspace; mechanistically caused by strong inter-axis correlation. Detected via SVD of the embedding covariance, singular-value spectrum on log scale. (Jing et al., 2110.09348; Hua et al., ICCV 2021, arXiv:2105.00470)
- **Local collapse:** representations are high-rank *globally* but collapse within local neighborhoods — missed by global spectral metrics. BYOL exhibits this: global effective rank ~584 yet mean Local Intrinsic Dimensionality ~16. (LDReg, ICLR 2024)

### 1.2 Mechanism — the division of labor (load-bearing for §1)
- **Variance term prevents complete collapse; covariance/decorrelation prevents dimensional collapse — and variance alone is empirically insufficient.** Hua et al. (2105.00470): plain BatchNorm-style variance standardization still reaches inter-axis correlation **0.99** (dimensionally collapsed); full decorrelation/whitening reaches **0.00**. → The covariance term is *load-bearing*, not a minor add-on; do not zero it out or down-weight it to nothing.
- **VICReg** (Bardes, Ponce, LeCun, ICLR 2022, arXiv:2105.04906): explicit variance hinge (per-dim std) + covariance decorrelation; information-maximization, non-contrastive; deliberately avoids stop-grad/BN/EMA/negatives.
- **BYOL/SimSiam asymmetry:** predictor head **and** stop-gradient are *both* essential — removing either collapses the model (Tian, Chen, Ganguli, ICML 2021, arXiv:2102.06810). The predictor's eigenspace aligns with the input correlation matrix (basis of DirectPred).
- **Barlow Twins** (Zbontar et al., ICML 2021, arXiv:2103.03230): cross-correlation→identity; no predictor, stop-grad, EMA, or negatives.

### 1.3 EMA-teacher methods — alternatives compatible with our setup
- **DINO** (Caron et al., 2021, arXiv:2104.14294): EMA teacher (cosine 0.996→1) + **centering + sharpening**, which counter two *opposite* collapse modes (centering counters dominant-dimension but pushes toward uniform; sharpening the reverse). **Caveat:** centering/sharpening operate on a softmax over prototype dimensions — they assume a categorical/distribution head, so they are **not** a clean drop-in for a continuous-latent regression objective like ours.
- **DINOv2** (Oquab et al., TMLR 2023/2024, arXiv:2304.07193): replaces centering with **Sinkhorn-Knopp** (3 iters) and adds **KoLeo**, a feature-spreading term on L2-normalized features. **KoLeo is the directly-applicable alternative for continuous embeddings** if VICReg's covariance term proves finicky.

### 1.4 The JEPA lineage (closest to the target system)
- **I-JEPA** (Assran et al., CVPR 2023, arXiv:2301.08243): collapse avoided by EMA-target asymmetry alone; **L2** loss; no explicit variance/covariance term; momentum **0.996 → 1.0** (linear ramp).
- **V-JEPA** (Bardes et al., 2024, arXiv:2404.08471): EMA + stop-grad + predictor only — **no VICReg/whitening/centering** — with an **L1 latent-prediction loss**. Rationale is anti-collapse: the optimal L1 predictor tracks the conditional median (minimizes median absolute deviation), "forcing the encoder to capture as much information as possible." Momentum **0.998 → 1.0** (linear ramp). No dedicated collapse ablation reported.
- **V-JEPA 2** (Assran, Bardes, …, LeCun, Ballas, 2025, arXiv:2506.09985): same mechanism, L1; **simplified to a fixed EMA coefficient (no ramp)** and fixed weight decay, "minimal impact on downstream." → constant momentum is fine.
- **C-JEPA** (Mo & Tong, NeurIPS 2024, arXiv:2410.19560) and **VJ-VCR** (arXiv:2412.10925): *add* VICReg to the JEPA lineage, explicitly because EMA asymmetry alone is insufficient to guarantee against collapse. This is the recipe our brief most resembles.
- **Net:** the field genuinely splits — {L1 + EMA-only} (V-JEPA) vs {L2 + EMA + VICReg} (C-JEPA). The L1 choice and the VICReg term are partially *substitutable* anti-collapse devices.

### 1.5 The crux — does collapse theory transfer to non-gradient / living-weight dynamics?
- **Yes, for local plasticity.** **LPL** (Halvagal & Zenke, *Nature Neuroscience* 2023; bioRxiv 2022.03.17.484712): a layer-local, **backprop-free** predictive rule. The predictive term *alone* causes total collapse. Prevented by inversely scaling the Hebbian term by an online variance estimate — **the local-rule analogue of VICReg's variance term** (authors cite VICReg). Two dissociated modes: no variance term → full collapse (zero activity); no decorrelation term → dimensional collapse (dimensionality ~1); full rule → ~15.
- **Yes, for equilibrium models.** Neural Collapse provably occurs in Deep Equilibrium (implicit) models under balanced conditions (Sun & Shi, NeurIPS 2024, arXiv:2410.23391). *Caveat:* this is supervised-classification neural collapse, conceptually adjacent to (not identical with) SSL representation collapse.
- **Predictive-coding SSL exists** (MPC; Ororbia, Friston & Rao, 2025, arXiv:2503.21796): encoder-only, local Hebbian, no backprop — but its collapse-avoidance is by architectural design, and the claim that it needs *no* anti-collapse mechanism was **refuted** in verification. Do not cite MPC as proof that design-alone suffices.
- **Still unstudied:** weights that self-modify *during inference* (fast-weights / test-time plasticity) and their effect on collapse. LPL (training-time local plasticity) and DEQ (equilibrium fixed-points) are adjacent, not on-target. This is our last empirical mile.

### 1.6 Detection metrics + healthy ranges (feeds §4)
- Per-dimension std (complete collapse; VICReg targets std ≈ 1; collapse ≈ 0).
- Singular/eigenvalue spectrum of the embedding covariance, log scale (dimensional collapse).
- Off-diagonal covariance/correlation mass (≈ 0 healthy; 0.99 = collapsed, Hua et al.).
- Effective rank / stable rank (scalar dimensional-collapse trend).
- Local Intrinsic Dimensionality (local collapse; catches what global metrics miss).
- Online-vs-target cosine similarity (high but < ~0.99; ≈ 1.0 = predictor trivially solved).

### 1.7 Honest gaps & refuted claims
- **Gaps:** inference-time fast-weight collapse (unstudied); whitening comparison (W-MSE / Shuffled-DBN / Zero-CL) did not surface usable single-GPU compute/stability benchmarks — revisit only if VICReg misbehaves in the pilot.
- **Refuted (do not cite):** "MPC avoids collapse with no anti-collapse mechanism" (0–3); "DINOv2's teacher head is an EMA of the student" (0–3).
- **Coverage caveat:** the literature is vision-SSL-heavy. The collapse *theory* is about embedding statistics and is largely modality-agnostic, but transfer to text is a mild assumption.

---

## 2. Corrected specification (supersedes v0.3)

### §1 — Loss & architecture

- **[FIX] B1 — VICReg coefficients.** v0.3 lists "paper defaults λ=25, μ=25, L_pred=1," assigning **25 to covariance**. VICReg's actual defaults are **invariance/prediction = 25, variance = 25, covariance = 1** (Bardes et al. 2105.04906). Covariance = 25 is not a paper default. Pull the exact coefficients and any sensitivity ablation directly from the VICReg paper before locking. The covariance term is load-bearing (§1.2) — keep it, at its calibrated (small) weight; do not inflate it to 25 and do not drop it.
- **[DECISION] L1 vs L2 prediction loss.** v0.3 uses MSE (L2). The closest lineage (V-JEPA) deliberately uses **L1** for anti-collapse reasons (MAD property, §1.4). Recommend **L1 + VICReg** as the conservative default. This is 4.7's call on optimization-dynamics grounds (loss scale, gradient behavior on the PC substrate). If L1 is adopted, re-check the lr (loss-scale shift) and document.
- **[FIX] B2 — mask leakage.** v0.3: context = first 80%; target = 10–20% block from positions 60–100%. These overlap (a target at 60–80% sits inside the context), allowing trivial copying and a meaningless descending loss. Make context and target **disjoint**: either draw the target only from 80–100%, or remove sampled target positions from the context (I-JEPA style). [FIX]
- **[FIX] B6 — define "encoder" and EMA semantics.** Specify: (a) which part of the v2 PC stack is "the encoder," (b) which output (block, post-settling step, pooled vs per-position) feeds L_pred, and (c) **EMA applies to the slow/rich parameters only — not the living-weight transient state or prec/episode buffers.** Averaging fast dynamical state is a category error; both online and target encoders run their living-weight dynamics fresh per forward. This also bounds the EMA memory cost to ~parameters (resolves the §5.3/§10 "doubles encoder memory" ambiguity — it doubles *params*, not the 875M buffers). Confirm against substrate internals (4.7).
- **[DECISION→resolved] EMA momentum.** Use a **constant** momentum (V-JEPA 2 dropped the ramp with minimal impact, §1.4). Closes v0.3 §9.6. Suggested 0.996–0.998 constant.

### §4 — Instrumentation (collapse coverage)

All collapse metrics computed on a **fixed held-out probe batch** (M7 probe file list) for cross-checkpoint comparability, on **both** the online and EMA-target encoders.

**Light — every 100 batches:**
- Online per-dim std: 5th/50th/95th percentile; count of dims with std < 0.1 (collapse band) and < 0.5 (warning band; VICReg targets ≈ 1).
- Target-encoder per-dim std (same stats).
- Mean off-diagonal |correlation| of online embeddings.
- Online-vs-target cosine similarity (mean + std). Healthy: high, < ~0.99.
- Predictor-output std (catch predictor collapse independently).
- Loss broken out: L_pred, L_var, L_cov separately. **Note whether L_pred is L1 or L2** (per §1 decision) so a later comparison is auditable.

**Deep — every 1000 batches:**
- Singular-value spectrum of online embedding covariance, log scale (summary: top-k/bottom-k, index at 90%/99% cumulative variance).
- Effective rank (exp spectral entropy) and stable rank (‖C‖_F²/‖C‖₂²).
- Local Intrinsic Dimensionality on the probe batch (MLE/Fisher-Rao estimator).
- Existing M7 substrate-health metrics (non-FF, pred_frob, prec, err_acc), plasticity histogram, episode store utilization/eviction — **unchanged**.

### §5 — Acceptance gates (pre-flight)

Keep v0.3's five gates, with two corrections:
- **[FIX] B5 — checkpointing is time-based, not batch-based.** v0.3 promises "≤20 min lost" but computes it at M7's 0.85 b/s; M8 runs ~0.40–0.50 b/s (≈37 min per 1000 batches). Checkpoint **every ~15 minutes of wall-clock** so the guarantee holds regardless of throughput. Rolling 3 slots. Contents must include EMA-target state (and EMA schedule position if any).
- **[FIX] B6 memory accounting.** Gate 3 ("EMA roughly doubles encoder memory") resolves once §1-B6 defines EMA as params-only: the added cost is ~encoder parameters, not the 875M buffers. Verify the params-only duplication fits without reducing batch size below M7's.
- Add: kill-restart test (gate 1) must show resumed loss within a stated tolerance of the pre-kill smoothed value (an EMA target that isn't restored exactly will show a real step).

### §6 — Success & kill criteria

**Principle:** thresholds marked `[pilot-set]` are derived from the 256d pilot's healthy trajectory (margin beyond baseline), not guessed. The pilot runs metrics in observe-only mode until baselines exist. This also fills v0.3's literal blank for the covariance threshold.

**Kill if any trigger (sustained over the stated window):**
1. **Complete collapse —** online OR target per-dim std (5th pct) < `[pilot-set ≈0.1]` for 3 consecutive 1000-batch checkpoints.
2. **Dimensional collapse (spectrum) —** effective rank < `[pilot-set, e.g. 50% of healthy baseline]`, or > `[pilot-set]`% of singular values < `[ε]`, sustained 5 checkpoints.
3. **Dimensional collapse (correlation) —** mean off-diagonal |correlation| > `[pilot-set]` for 5 checkpoints. *(replaces the v0.3 unspecified threshold)*
4. **Local collapse —** probe-batch LID < `[pilot-set, fraction of early-run value]` over 5 checkpoints.
5. **Predictor-trivial / target collapse —** online-vs-target cosine > `[≈0.99]` sustained, or target-encoder std collapsing (criterion 1 on the target side).
6. **Substrate override (kept) —** non-FF signal degrades > 25% from M7-equivalent baseline over 5 checkpoints. *(depends on B4 — see §3.)*
7. **Objective unlearnable (kept, restated) —** no descent on the **smoothed/probe** loss curve over a 5,000-batch window after the first 5,000 batches. (Define the smoothing window; not raw per-batch loss.)

On any kill: halt, snapshot full run state, surface to Brian. Do not "give it another epoch."

**Pilot headline objective (the literature gap, made a deliverable).** LPL (§1.5) shows that under local, non-gradient plasticity the predictive term alone collapses and a VICReg-style variance term is *required*. The 256d pilot must run the ablation **{EMA + VICReg} vs {EMA-only} vs {VICReg-only}** on the collapse metrics, to determine on the *living-weight* substrate whether VICReg is load-bearing (LPL's prediction) or redundant given EMA (V-JEPA's claim for static backbones). Cheap at 256d; do not skip.

### §9 — Open decisions (updated status)

- **9.1 [DECISION] Pilot at 256d — recommend YES.** Resolves §6 thresholds, §5.3 memory, §7 throughput, and the §6 ablation empirically. **Scale-confound interpretation rule:** Li/Efros/Pathak (ECCV 2022, arXiv:2209.15007) show partial dimensional collapse appears when the model is small relative to the dataset — exactly the 256d-on-1.27B-token regime. **A 256d collapse may be a capacity artifact, not a verdict.** Confirm by (a) proportionally smaller pilot data, or (b) a larger-d confirmation run, before condemning the design. A clean 256d run is encouraging but not proof for 1024d.
- **9.2 Action-token stub — include, but relabel the check.** Keep the stub (cheap M9 interface continuity). But the gradient-magnitude "gate" is near-vacuous: a *constant* token receives nonzero embedding gradient regardless of whether the predictor conditions on it. Real conditioning is untestable until M9 has a varying action. Keep the stub; do not claim the check de-risks M9.
- **9.5 [resolved→keep] VICReg.** Keep it. LPL (§1.5) suggests it is likely *necessary* under local-plasticity-like dynamics, not redundant.
- **9.6 [resolved] EMA momentum.** Constant; skip the ramp (V-JEPA 2).
- **9.7 [clarify] Optimizer state.** Lean fresh optimizer (loss-scale differs CE→latent-prediction). Note: there is no surviving *M7* optimizer state (power loss); "M7 seed" in v0.3 refers to the M6→M7 expanded checkpoint's state. Fix the label.
- **9.x [DECISION — new] L1 vs L2 loss.** See §1. Recommend L1 + VICReg; 4.7's optimization call.

---

## 3. Things needing repo access / still open

- **B4 — confirm M7 baseline data exists.** Kill criterion 6 and several "vs-M7" comparisons assume `runs/m7_1024d/launch.log` saved the metric *time-series* (non-FF, prec, err_acc, descent shape) at usable cadence. M7 lost its checkpoints; confirm the log carries the series. If not, the 256d pilot becomes the substrate-health baseline too.
- **Whitening alternatives** (W-MSE / Shuffled-DBN / Zero-CL) — no single-GPU compute/stability benchmarks surfaced. Revisit only if VICReg misbehaves.
- **Inference-time fast-weight collapse** — genuinely unstudied in the literature; the pilot ablation is the only way to settle it for our substrate.

---

## 4. References (verified primary sources)

- Jing, Vincent, LeCun, Tian. "Understanding Dimensional Collapse in Contrastive Self-Supervised Learning." ICLR 2022. arXiv:2110.09348
- Hua et al. "On Feature Decorrelation in Self-Supervised Learning." ICCV 2021. arXiv:2105.00470
- LDReg. "Local Dimensionality Regularization for Self-Supervised Learning." ICLR 2024.
- Bardes, Ponce, LeCun. "VICReg: Variance-Invariance-Covariance Regularization." ICLR 2022. arXiv:2105.04906
- Zbontar et al. "Barlow Twins." ICML 2021. arXiv:2103.03230
- Tian, Chen, Ganguli. "Understanding Self-Supervised Learning Dynamics Without Contrastive Pairs." ICML 2021. arXiv:2102.06810
- Assran et al. "I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture." CVPR 2023. arXiv:2301.08243
- Bardes et al. "V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video." 2024. arXiv:2404.08471
- Assran, Bardes, …, LeCun, Ballas. "V-JEPA 2." 2025. arXiv:2506.09985
- Mo & Tong. "C-JEPA: Connecting Joint-Embedding Predictive Architecture with Contrastive Self-Supervised Learning." NeurIPS 2024. arXiv:2410.19560
- "VJ-VCR." arXiv:2412.10925
- Caron et al. "DINO: Emerging Properties in Self-Supervised Vision Transformers." 2021. arXiv:2104.14294
- Oquab et al. "DINOv2." TMLR 2023/2024. arXiv:2304.07193
- Halvagal & Zenke. "The combination of Hebbian and predictive plasticity learns invariant object representations in deep sensory networks" (LPL). Nature Neuroscience 2023. bioRxiv 2022.03.17.484712
- Sun & Shi. "Understanding Representation of Deep Equilibrium Models from Neural Collapse Perspective." NeurIPS 2024. arXiv:2410.23391
- Ororbia, Friston & Rao. "Meta-Representational Predictive Coding (MPC)." 2025. arXiv:2503.21796
- Li, Efros, Pathak. "Understanding Collapse in Non-Contrastive Siamese Representation Learning." ECCV 2022. arXiv:2209.15007
- Ermolov et al. "Whitening for Self-Supervised Representation Learning" (W-MSE). ICML 2021.
