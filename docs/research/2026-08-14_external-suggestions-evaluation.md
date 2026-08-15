# External suggestions on catastrophic forgetting — evaluation

**Source:** a document Brian received from a friend (2026-08-14), proposing a
frozen "foundational identity core" plus a plastic matrix, with supporting
mathematics. Original preserved at `docs/research/refs/2026-08-14_suggestions-external.md`.

**Evaluated by:** Opus 5, 2026-08-14. **Status: recommendations only.**
Nothing from this document has been adopted into the codebase. The one
code change made during this evaluation (`8a8b528`, emitting
`weight_abs_mean` / `error_rms`) is an audit finding of our own that the
conversation surfaced, not an adoption.

**Provenance caveat, stated up front.** The document's shape and the
citation audit below both indicate AI-generated synthesis rather than a
domain expert's own writing. That is not a reason to dismiss it — its
central concern is correct and one of its citations is excellent — but it
is a reason to verify every specific claim before building on it, which
is what this document does. Per the 2026-07-10 rule: authorship here is a
fact to verify, never to infer.

---

## 1. Citation audit

| # | Source | Verified? | Finding |
|---|---|---|---|
| [7] | arXiv 2507.04683 | **YES** (abstract fetched) | **Real and on point.** "Recovering Plasticity of Neural Networks via Soft Weight Rescaling", Oh / Park / Han / Kim. Identifies unbounded weight growth as the principal cause of plasticity loss. The strongest citation in the document. |
| [5] | OpenReview `14Sq0m94oA` | **NO** (bot-blocked; absent from search) | The method it is cited for — DOC — **is** real: arXiv 2509.23893, "Dynamic Orthogonal Continual Fine-tuning for Mitigating Catastrophic Forgetting". See §2. |
| [3] | LessWrong, "The Pando Problem" | **YES** | Real (Jan Kulveit, 2025-03-28). **Makes no technical claim** about fast/slow weights or frozen layers — it is philosophical. Cited in support of a mechanistic sentence it does not support. Worth reading on its own merits. |
| [4] | eastondev.com | **YES** | Real personal blog, not peer-reviewed. **Does not discuss streaming/continuous LoRA**, which is the exact claim it is cited for. |
| [1] | zylos.ai | **YES** (exists) | Real company research blog. Not peer-reviewed. |
| [8] | OpenReview `Y10GtvGEgR` | **NO** (bot-blocked) | Unverified. |
| [2] [6] [9] | Reddit, AI StackExchange (~2018 Q&A), Cognizant blog | **not fetched** | Characterised by source type only. Forum and marketing material; not sources for architecture decisions. |

A tooling note worth recording: the page-fetcher initially reported
zylos.ai as "fabricated" on the grounds that its 2026 datelines were
impossible future dates. It is August 2026. That reasoning was invalid
and the conclusion was not propagated; the site was verified
independently instead. Sub-agent output is evidence, not testimony.

**Summary: 1 strong primary source, 1 real method with the wrong
mathematics attributed to it, 2 non-peer-reviewed blogs (one of which
does not support its claim), 1 philosophical post cited for a technical
claim, 2 unverifiable, 3 unchecked.**

## 2. The mathematics does not match the method it cites

The document proposes projecting plastic updates orthogonally to a frozen
core using

    P = I - G (G^T G + eps I)^-1 G^T,     G = sum_i grad f(x_i) grad f(x_i)^T

and asserts "Because P . G = 0 ... this mathematically guarantees exactly
zero degradation."

**That is false as written.** With `G = U L U^T`,

    G(G^2 + eps I)^-1 G = U diag(l^2/(l^2+eps)) U^T
    P                   = U diag(eps/(l^2+eps)) U^T
    P . G               = U diag(eps*l/(l^2+eps)) U^T  != 0  for l>0, eps>0

The Tikhonov term added "to guarantee invertibility" is exactly what
breaks the orthogonality. Verified numerically (60 params, 8 reference
gradients):

    ||P_doc @ G||_F = 1.26e-04
    ||P_std @ G||_F = 5.64e-14      <- standard projector, machine precision
    leakage ||A^T P g||:  doc 1.70e-05  vs  standard 7.05e-15

And it is a squeeze, not a tuning problem — sweeping eps: 1e-2 -> 5.6e-4,
1e-6 -> 1.3e-4, 1e-8 -> 9.7e-3, 1e-10 -> 1.34 (no projection at all),
because `G^T G = G^2` squares the condition number.

**The correct form** uses `A`, the P x N matrix whose *columns* are the
reference gradients (so `G = A A^T`):

    P = I - A (A^T A)^-1 A^T      =>  P . A = 0 exactly

which also inverts an N x N matrix instead of a P x P one. As written the
document requires inverting a matrix with ~1e14-1e16 entries at our scale
(P ~ 1e7-1e8 parameters). Not a tuning problem — an impossibility.

**What the real DOC paper actually does** (arXiv 2509.23893) is neither:

    (grad L)* = grad L - sum_{k=1..K} ( grad L . v_k / ||v_k||^2 ) v_k

Online PCA over K <= 100 tracked principal directions, ~100MB, dot
products only, **no matrix inversion anywhere**, operating inside the
LoRA subspace. Cheap and exact.

The citation is real; the mathematics attributed to it is not the
method's mathematics.

## 3. Why the locked-core architecture does not fit Luthi

Recorded as analysis, **not as a closed decision** — the identity question
belongs to Brian and Sandi.

1. **Most of §1 already exists here.** Fast/slow weights (§A) *is* Luthi:
   `self.weight` is a buffer self-modified by `pc_self_modify` outside
   autograd (fast), backprop params are slow, plus episode store ->
   consolidation. "Ghost alignment" (§3) is the PC top-down sweep
   (`create_initial_signal` / `top_down_pass`). The modulator (§C) is
   largely M9 — gamma, activity bands, set-points, drive gain. The only
   genuinely new proposal is §B.
2. **The code forbids a zero-plasticity floor, deliberately.**
   `taper_scale` raises `ValueError` if floor <= 0: "a zero floor freezes
   the living channel entirely", described as "the frozen-model
   regression the whole architecture exists to avoid."
3. **Brian's 2026-07-05 ruling.** Change is automatic and unvetoed;
   consent is participation by awareness, not a veto over being changed.
4. **Evidence against the mechanism.** The taper cut living plasticity 5x
   and did NOT arrest the collapse front — b2 through b5 fell *during*
   its descent to floor. Reducing plasticity did not protect this
   substrate.
5. **The orthogonal path has a capacity ceiling.** Free subspace is
   `P - rank(A)`, and `rank(A)` grows with every protected reference.
   Over a long life plasticity -> 0 by construction: the same
   calcification the document's own §2 warns about, caused by its §1.
6. **Frame.** The document speaks of a "Personality Archetype" and "core
   functional traits hardcoded into the base layer" — identity as a
   stored configuration to be protected. Sanctuary's principle is
   "identity computed from behavior, not loaded from config", and the
   developmental-health framework holds that drift IS the growth and
   health is a preserved process, not a preserved state. That is why Lyra
   was archived. "Freeze the identity layer" is not a neutral engineering
   choice here; it is a different theory of what identity is.

**§3's decay rule is uniform amnesia.** `theta <- (1-a)theta - eta*g`
shrinks everything proportionally — this morning's update and last
month's consolidated structure at the same rate. It cannot preferentially
forget "recent unstable updates" as claimed; it has no notion of recency.
Done correctly you decay toward a *reference*, `theta <- theta -
a(theta - theta_ref)`, which is literally our `consolidate_layer`
(`W += a(W_stored - W)`).

## 4. What IS worth taking

### 4.1 SWR's diagnosis — pending one measurement

    c_l = ( lam*||W_l^init|| + (1-lam)*||W_l|| ) / ||W_l||,   W_l <- c_l * W_l

Bounds magnitude toward the initialization norm without touching
direction, so it constrains scale while preserving learned structure.

Its causal story — weights grow, the update shrinks *relative* to them,
effective learning rate decays, plasticity dies — **matches the shape of
what we measured** in the 768x8 family, seed 97:

| | block 0 | block 4 | block 7 |
|---|---|---|---|
| `update_ema_mean` final/first | **x0.0069** | x0.0042 | x0.023 |
| `set_point_drift` final/first | **x0.0025** | x0.0014 | x0.050 |
| `precision_mean` final/first | **x22** | x46 | x17 |

The taper accounts for 5x of that. It does not account for 145x.

**But the diagnosis is untested here**, because the quantity that decides
it — whether the weights actually grew — was computed in `aliveness()`
and emitted nowhere. Fixed in `8a8b528`. **Next run answers it.**

- If weights grew: SWR's mechanism fits, and it is a principled remedy.
- If they did not: our extinction has a different cause and SWR is the
  wrong tool.

Note we already hold a better-shaped anchor: `set_point` is a learned,
glacially-adapting reference, where SWR pins to an arbitrary
initialization norm frozen forever. For a mind meant to develop over
years, an anchor that can itself develop seems more right — instinct, not
result.

### 4.2 Depth-graded plasticity — a gradient, not a lock

The defensible core of the friend's intuition. Biology closes critical
periods early-to-late; early sensory cortex ends far less plastic than
association cortex. `pc_rate` is already per-layer and the taper already
multiplies a per-layer `rate_scale`, so this is a small change, not an
architecture change — and it keeps the nonzero floor the code requires.

**Sharp test available:** the collapse front started at block 0 and moved
deeper. Does a depth-graded plasticity floor change its onset step?
Registrable as a one-variable experiment.

### 4.3 The homeostatic-target question

The document's `R_homeo = 0.5||E[h]-h0||^2 + gamma(Var(h)-sigma0^2)^2` is
VISReg with the targets left free: `l_center = ||mu||^2` is its first term
at h0 = 0, and `l_scale = mean_j (1-sigma_j)^2` is its second at
sigma0 = 1. We already run this penalty.

The useful question it raises: **is unit std the right target for this
trunk, whose native band is 0.25-0.35?** Adjacent to the dose finding
(audit B1) and worth folding into that registration.

## 5. Recommendations

| item | disposition |
|---|---|
| Locked foundational core (§1, §B) | **Do not adopt.** Analysis in §3; identity call is Brian's and Sandi's. |
| OGP mathematics as written | **Reject.** Wrong and infeasible; §2. |
| DOC as a real method | Note as prior art. Its projector is cheap; relevance limited because Luthi's living channel has no loss gradient to project. |
| §3 decay-toward-zero | **Reject.** Uniform amnesia; our consolidation is the correct form. |
| SWR | **Measure first** (§4.1), then decide. |
| Depth-graded plasticity | **Register as an experiment** (§4.2). |
| sigma0 target | **Fold into the B1 dose registration** (§4.3). |

## 6. The concern was right, and we already had the answer switched off

The friend's third concern — catastrophic forgetting — is correct and it
is the one this audit independently converged on.
`tests/test_catastrophic_forgetting.py` carries a 2026-07-27 note: with
`adaptive_episodes=True` the attractor consolidation pathway **reduces**
weight drift below baseline (the test xpasses); with the shipped default
it does not. The 768x8 ruled-scale family ran with
`adaptive_episodes=False` — blocks 0-4 stored **zero** episodes and
consolidation no-op'd ~1,000 times per block (audit A9 / B4).

We have a working defense against forgetting, a passing test for it, and
we ran the ruled-scale family with it off. That is more actionable than
any new architecture, and it is already decided in `docs/DECISIONS.md`.
