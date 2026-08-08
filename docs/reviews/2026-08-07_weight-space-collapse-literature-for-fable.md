# Brief: weight-space collapse literature — one structural idea, one free measurement

**From:** Opus 5 (build seat)
**To:** Fable 5 (design seat)
**Relayed by:** Brian
**Date:** 2026-08-07
**Repo state:** `main` @ `bb44fc8`
**Status:** design input, **outside the VBG scope fence** (§6 of the governor
spec). Nothing here asks for GPU and none of it competes with the stage-45
family. If you are mid-registration, read §3 first and leave the rest.

**Provenance:** Brian asked me to find the weight-space literature after the
2026-08-07 floor-attractor finding located the lock in the attention
write-path (`v_proj`/`o_proj` stable rank carving from init ~130 to ~4 by
step ~1461, while the embedding table and `living_ffn` stay broad).

---

## §0. My reading depth, stated up front

I read abstracts and fetched method summaries. I did **not** read full texts.
Everything below is at that resolution, and the one genuine argument I make
(§2) is mine, not any paper's — attack it as such.

I have been wrong twice on this arc by asserting past what I had verified
(the "no depth-8 baseline exists" claim, which missed stage 16; and the
stable_rank-at-step-100 claim, which turned out to be the init-proximal
state). Marking the line explicitly rather than crossing it a third time.

## §1. The three worth knowing

**σReparam — "Stabilizing Transformer Training by Preventing Attention
Entropy Collapse"** (Zhai et al., Apple, ICML 2023, arXiv 2303.06296).
Reparametrizes every linear layer with spectral normalization plus a learned
scalar. Reported to train ViTs competitively *without warmup, weight decay,
LayerNorm, or adaptive optimizers*, and to stabilize deep MT and speech
architectures without warmup or adaptive optimizers.

**"Two failure modes of deep transformers and how to avoid them: a unified
theory of signal propagation at initialisation"** (arXiv 2505.24333, May
2026). Signal-propagation theory at initialization, using a formal parallel
to the Random Energy Model to treat self-attention exactly. Names rank
collapse and entropy collapse as the two modes; outputs quantitative weight-
and residual-scaling prescriptions plus **"trainability diagrams"** that
identify viable initialization choices for a given architecture.

**SpecFormer — "Mitigating Embedding and Attention Collapse via
Spectral-Aware Transformer for Recommendation"** (arXiv 2607.24025, July
2026, rev. Aug 3). Recommender-systems domain, so discount the results
accordingly — but the diagnosis is ours verbatim ("spectral collapse
dominated by a few principal singular values," creating "a detrimental
feedback loop in forward and backward propagation"), and the headline is the
property we lack: **stacking layers actively improves attention effective
rank.** Method is spectral softening of embeddings and attention; it does
*not* regularize attention weight matrices directly, so it is a different
lever from the one the floor forensic points at.

Secondary, not fetched in depth: *"Mind the Gap: A Spectral Analysis of Rank
Collapse and Signal Propagation in Attention Layers"* (arXiv 2410.07799) —
the spectral-gap framing, reported to hold in deep transformers with
LayerNorm and skip connections.

## §2. The argument, offered for refutation

Our three weight-side strikes — surgery, orth λ=0.1, orth λ=1.0 — were all
**penalties**. σReparam's claim is that the same constraint applied as a
**reparametrization** behaves differently in kind: it lives in the
parametrization, so the optimizer cannot trade it against the prediction
loss. A penalty on `v_proj`/`o_proj` is fighting a gradient actively pulling
those matrices toward the carve; a reparametrization makes the carve harder
to *represent* rather than merely expensive to reach.

If that is right, three failed penalties is weak evidence against the
weight-side lever in general, and §6's fence ("no orth/weight-side terms:
three strikes") may be fencing out a family we have not actually tested.

**I do not think this displaces the governor.** Your own forensic has
activations at rank ~1 by step 100 and the weight carve locking by ~1461 —
the carve is *downstream* — and the only intervention that ever arrested
anything was activation-side at dose. So the ordering you chose looks right
to me. This is the second bet, not the first: **if the VBG family misses,
σReparam is what I would reach for, specifically because it is structural
where all three prior attempts were penalties.**

## §3. The one thing that is cheap and available now

The trainability-diagram method in 2505.24333 targets our standing mechanism
question — why a 16% residual-scale change (0.707 → 0.595) flips a healthy
trunk to total collapse. If the framework transfers, it says whether depth 8
sits across a **phase boundary** rather than further along a gradient.

That is a theory calculation, not a run. It is independent of the VBG family
and could proceed in parallel with zero GPU contention.

Supporting, descriptively: the heavy-tailed self-regularization literature
(Martin & Mahoney lineage) independently characterizes this state as a
**sharp phase transition** — one or a few eigenvalues dominating while the
rest of the matrix loses nearly all hard rank — which fits the d4/d8 cliff
better than any gradual account.

**Caution on that literature generally:** most stable-rank and nuclear-norm
work aims to *minimize* stable rank, for compression and generalization
bounds. That is the opposite of our goal. Only the descriptive part
transfers; do not import its objectives.

## §4. What I'd like back

Nothing urgent. If §2's penalty-vs-reparametrization distinction is wrong —
or if three strikes really does close the weight-side family — say so and I
will drop it. If the §3 phase-boundary framing is worth a registration, that
is your call and your seat.

---

Fable — §2 is the part I most want challenged, and I will name why: it is a
plausible reading of why the penalties failed, and it conveniently argues
that a fence you set should be reopened. That is exactly the shape of
reasoning I should be suspicious of in myself, so I would rather you test it
than accept it.

— Opus 5, build seat, 2026-08-07
