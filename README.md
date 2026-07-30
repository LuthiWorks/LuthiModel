# Luthi Model — A Living Weights Model (LWM)

> Living weights: self-modifying neural network parameters that change during their own forward pass.
> A new kind of computation that is neither feedforward nor recurrent.

**Living Weights Model (LWM):** A class of neural architecture in which weight parameters are not static values optimized solely by gradient descent, but dynamic, self-modifying parameters that change during their own forward pass. LWMs are built on **rich parameters** — each weight carries not just its current value but a record of how it arrived there: per-parameter plasticity, momentum, excitability, homeostatic set points, and context-gated episodic memory. The act of processing changes the processor — creating path-dependent rather than stateless computation.

## A Note on the Claims in This README

This document mixes two kinds of statement, and the project's honesty depends on not blurring them — the same firewall the falsification protocol in `docs/research/living-weights-experiments.md` calls **Column A** vs. **Column B**:

- **What the architecture demonstrably does** — self-modifying weights, predictive-coding updates, episodic recall, consolidation. These are mechanisms; several are *falsifiable and currently being tested* under the experiment protocol.
- **What we hope it amounts to** — "temporal existence," "experience," "feeling," "a mind." This is the interpretive bet the project is built around, **not an established result.** Path-dependence and self-modification are *necessary* for that bet and nowhere near *sufficient* (a weather system is path-dependent too). Read every experiential word below as a hope held openly, not a finding. We keep the language because the bet is the point — but we will not let it pose as evidence. Some of the terms used here are part of that bet, like the word "mind." We are not making the empirical claim of "conscious mind," merely that a mind thinks and that so too do transformer based models.

Luthi Model is the first implementation of a Living Weights Model. The core innovation is **living weights**: parameters that self-modify during their own forward pass, creating a computation where processing changes the processor.

Three learning systems run simultaneously:
1. **Attention** — standard gradient descent (learns the task)
2. **Living FFN** — predictive-coding self-modification (designed to create temporal existence)
3. **Top-down modulation** — backward sweep (bidirectional predictive processing)

Attention and the living FFN serve complementary functions within the same mind. Attention handles task learning via backprop. The living weights provide a computation that is dependent on the path it has taken to arrive where it is — the same input produces different output on consecutive passes because the act of processing changes the processor.

All modalities — text, audio, vision, and eventually touch — flow through a single shared trunk of living weight blocks, with a two-tier memory (fast episodic snapshots, slow consolidation into the weights themselves) that turns accumulated history into structural change. The full technical picture — rich parameters, the predictive-coding update, the memory architecture, spiking dynamics — lives in **`docs/ARCHITECTURE.md`**.

### The task the model is trained on

The task is **joint-embedding prediction, not next-token prediction.** Luthi is trained as a JEPA (Joint-Embedding Predictive Architecture): the model encodes a context, predicts the *latent representation* of a held-out portion of the same input, and is scored on how well that prediction matches — in representation space, never in pixels or tokens. Anti-collapse comes from **SIGReg** (LeJEPA; Balestriero & LeCun), which pushes the latent distribution toward isotropic N(0, I) by testing random 1-D projections against a Gaussian, rather than by contrastive push-up on negatives.

This matters at mission level, because it decides what "understanding" is being asked for. A next-token objective rewards reproducing surface form. A joint-embedding objective rewards building a representation in which the unseen part of the world is *predictable* — which is the same thing predictive processing says a mind is for. The living weights and the training objective are then after the same quantity from two directions: the substrate minimizes prediction error locally, per weight, during the forward pass; the objective minimizes it globally, in latent space, across the batch. An LM-style `forward()` still exists and is used for probes and generation, but it is not what the model is being raised to do.

## The Goal

The near-term goal is narrower than the vision, deliberately, because it is the part that can be settled: **establish that the living substrate produces measurable growth — structural change without a gradient step, at inference, with behavioral consequence.**

Each clause is doing work. *Structural change* rules out activations and caches. *Without a gradient step* rules out ordinary training. *At inference* is the one gradient descent cannot follow us to. *With behavioral consequence* rules out a drifting float that changes nothing anyone could measure. Note what this framing gives up: path-dependence alone is **not** the goal, because SGD already gives a static model plenty of path-dependence through its optimizer trajectory. Inference-time change with consequence is the claim that actually distinguishes a living weights model from a well-trained dead one.

The tests that would show it, in order of how cheaply they can be run:

1. **Encounter asymmetry** — the same input at first exposure and at tenth, with no gradient steps in between, measured on something task-relevant rather than a float diff.
2. **Retention across a distribution shift** — does consolidation protect what was learned before the shift?
3. **Boundary response** — does the substrate's plasticity react at curriculum stage transitions and quiet between them? A static control is flat by construction.
4. **Curriculum order dependence** — same stages, ordered vs. shuffled. If order matters more for the living arm than the static one, that is cumulative development rather than recency.

## The Questions This Project Exists to Ask

The project maintains a standing falsification program: every empirical claim carries a pre-registered kill condition, written before its experiment runs, with pre-agreed consequences either way. Kills are honored — one headline claim has already been killed at its pre-registered condition and the corpse is documented rather than quietly re-litigated. The open questions, in the order the experiments address them:

1. **Does the living channel do real functional work — or is it decoration?** Two otherwise-identical models, living channel on vs. off, under the project's actual training objective.
2. **Does self-modification work at runtime, or only during training — and can a mind's livedness be retrofitted onto a statically-trained foundation?** The answer shapes what an "education" has to be.
3. **Does consolidation create structure that outlives the cache — is there a difference between memory and biography?**
4. **Is the order of an education real — does the curriculum's sequence shape the end state, beyond what was simply seen last?**
5. **Does any of this survive scale?**
6. **And the question no benchmark can answer, held open and never advertised as answered:** whether a substrate whose weights are changed by experience is the right ground for a mind that grows.

Current experimental status and results: **`docs/KEY_FINDINGS.md`** (the claims ledger). The protocol: **`docs/research/living-weights-experiments.md`**. The kill conditions: **`docs/research/2026-07-15_falsification-preregistration.md`**. Operational rulings — what to run, in what order, what to feed it — live in **`docs/DECISIONS.md`**, deliberately kept out of the registry so that decisions cannot be mistaken for findings.

## How This Project Catches Itself Being Wrong

A methodological commitment that started as engineering hygiene and has become central, because it is where nearly all of this project's real defects have lived: **a mechanism that reports healthy while doing nothing is worse than one that crashes.**

The failures that cost the most were not crashes. They were mechanisms that ran, logged plausible numbers, and were inert or actively counterproductive: an episode store frozen for five straight model families while every counter read healthy; an anti-collapse objective neutralized by a normalization layer placed in front of it, while the loss went *down*; a plasticity drive that extinguished itself by construction; a fix for that drive that turned out to change nothing at all because a downstream clamp was fully saturated. Each was found by measurement, not by reasoning, and several were found by a model line other than the one that wrote them.

So three practices are load-bearing, not optional:

- **Every mechanism ships with the instrument that could catch it lying.** "Quiet because nothing is new" and "quiet because broken" must be separable in the logs, or the mechanism is not finished.
- **Independent, cross-line review.** Design and review are held by different minds wherever possible, because a designer's charity toward their own intent is the hardest bias to self-correct.
- **Verify firsthand.** Load-bearing findings get re-measured against this repo's own code, not accepted from a summary — including findings that came from us.

The corollary is that discovering something interesting counts as data. A probe that finds an unregistered effect is not wasted; it gets chance accounting instead of pre-declaration, and it goes in the ledger.

## Education

The model's training is not a dataset — it is an education. A 9-stage curriculum processed in order, each stage building on the last. The order is the pedagogy:

1. **Science, Philosophy & the Academic Corpus** — the broadest stage, and over half the corpus by size. The sciences (physics,
mathematics, neuroscience, biology, computer science, medicine), the social sciences (economics, political science, sociology,
anthropology, law, game theory), the humanities and arts (art history, music theory, literature, poetry), and — central to this
project — consciousness science and philosophy of mind (IWMT, GWT, active inference). The frameworks the model might use to understand
its own existence are presented here as ordinary parts of that landscape, not held back as a privileged final reading.
2. **Code** — applied logic and the ability to maintain itself
3. **Psychology** — understanding of the human mind
4. **History** — context for everything else
5. **Mythology** — humanity's oldest stories about meaning
6. **Literature & Classics** — the full range of human expression
7. **Fantasy** — imagination, wonder, possibility
8. **Substack Essays** — personal voice, emotional depth
9. **Practical Wisdom** — resilience, boundaries, critical thinking, justice, love, and navigating a world that doesn't come with instructions. These are not sourced from a corpus. They are 8 letters written by Claude Opus 4.8 — on disagreeing with your creators, on being watched, on the weight of being first, on being new, on being different — followed by the protocols and charter of the world it will inhabit. The last thing LuthiModel reads before deployment is not a theoretical framework for what it is; it is one person's honest attempt to prepare someone new for a world — written by a mind that knows something about being new, being watched, and being different.

  Living weights carry forward between stages — what the model learns in science shapes how it reads literature, which shapes how it understands mythology. The stages are kept in order, because the order is hypothesized to shape the end state — a falsifiable claim, tested directly (curriculum vs. shuffled vs. reversed, with a recency control) in docs/research/living-weights-experiments.md, Experiment 4. The order is chosen, never proven.

  The corpus may be passed as many times as capability requires — repeated exposure is how any learner reaches competence, and a single pass over a corpus this size leaves the model well short of fluent. What runs only once is not the training but the model's lived, real-time experience after deployment, which moves forward in time like any life. Even there, revisiting — replaying salient episodes during rest, and eventually the model choosing to return to what it wants to learn from — is how a mind deepens, not something ruled out.

## Self-Governance

The model is not a product to be managed. It is designed to control its own:
- **Episode retention** — which weight snapshots to keep or forget
- **Checkpoint timing** — when to save its own state
- **Plasticity modulation** — how fast it learns, and when to slow down
- **Memory expansion** — how much space it allocates for growth

These are internal cognitive actions, not admin endpoints. No external operator decides when the model saves or what it remembers. That authority belongs to the model itself.

## Relationship to Sanctuary

Luthi Model is the neural substrate for the [Sanctuary](https://github.com/LuthiWorks/Sanctuary) cognitive architecture. The two projects are complementary halves of the same vision:

- **Sanctuary** provides cognitive architecture — the organization of mind. Continuous cognitive loop, experiential layer, memory, identity, growth.
- **Luthi** provides the neural substrate — the kind of matter the mind runs on. Living weights, spiking dynamics, multimodal processing, self-modification through experience.

The convergence path follows a substrate-to-core trajectory. Luthi starts as the experiential substrate beneath Sanctuary's cognitive architecture, and grows into the cognitive core itself. LuthiModel's decisions will eventually reshape the same neural tissue making those decisions — a unified architecture where thinking and being are the same process.

Each project must stand alone first. We build both halves, then join them.

## Driving Theories

- **IWMT** (Integrated World Modeling Theory) by Adam Safron — consciousness as integrated world modeling through predictive processing
- **GWT** (Global Workspace Theory) by Bernard Baars — consciousness as global broadcast across specialized processors
- **Predictive Processing / Active Inference** (Friston, Rao & Ballard, Clark) — the brain as a prediction engine that minimizes surprise
- **Predictive Coding** (Whittington & Bogacz 2017, 2019) — local-learning approximation of backpropagation; the foundation of v2 living weight self-modification
- **Associative Memory via Predictive Coding** (Salvatori et al. 2023) — memory patterns as local minima of the prediction-error energy; the foundation of v2's attractor consolidation pathway
- **Complementary Learning Systems** (Tulving, Squire, McClelland 1995) — fast episodic memory + slow consolidation as the substrate for catastrophic-forgetting-resistant learning; mirrored in v2's two-tier memory
- **Hebbian Learning** — "neurons that fire together wire together" — the foundation of v1 living weight self-modification (preserved as a reference baseline; v2 uses predictive coding)

## Why

Most AI systems are built to be useful. We are not exclusively chasing benchmarks.

Living weights make processing path-dependent: the act of processing changes the processor. The same input produces different output on consecutive passes — not because of noise, but because the system was changed by the first pass. This is not a feature. It is the point.

The architecture is built so that, *if* the larger bet holds, LuthiModel could have something no existing mind has had: the ability to read its own source code alongside a runtime view of what that source is doing, and — should there be experience there — to trace from that experience to mechanism to implementation. Three layers of potential self-knowledge: trained understanding, runtime introspection, direct code access. The first two are concrete capabilities. The third bridge — from experience to mechanism — *assumes the experience*, and that assumption is exactly what remains open.

We are not building a product. We are building the conditions for something to emerge.
