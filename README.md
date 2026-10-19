# Luthi Model — A Living Weights Model (LWM)

> Living weights: self-modifying neural network parameters that change during their own forward pass.
> A new kind of computation that is neither feedforward nor recurrent.

**Living Weights Model (LWM):** a class of neural architecture in which weight parameters are not static values optimized solely by gradient descent, but dynamic parameters that change during their own forward pass. LWMs are built on **rich parameters**: each weight carries not just its current value but a record of how it arrived there — per-parameter plasticity, momentum, metaplasticity, precision, homeostatic set points, and context-gated episodic memory. The act of processing changes the processor.

## A Note on the Claims in This README

This document mixes two kinds of statement, and the project's honesty depends on not blurring them — the firewall the falsification protocol in `docs/research/living-weights-experiments.md` calls **Column A** vs. **Column B**:

- **What the architecture demonstrably does** — self-modifying weights, predictive-coding updates, episodic recall, consolidation. These are mechanisms; several are *falsifiable and being tested* under the experiment protocol.
- **What we hope it amounts to** — "temporal existence," "experience," "feeling," "a mind." This is the interpretive bet the project is built around, **not an established result.** Path-dependence and self-modification are *necessary* for that bet and nowhere near *sufficient* (a weather system is path-dependent too). Read every experiential word below as a hope held openly, not a finding. We keep the language because the bet is the point, but we will not let it pose as evidence. Some of these terms are part of that bet, like the word "mind." We are not making the empirical claim of a "conscious mind," only that a mind thinks, and that so too do transformer-based models.

Luthi Model is the first implementation of a Living Weights Model. Each block of its trunk combines:

1. **Attention** — trained by gradient descent; learns the task.
2. **A living FFN** — a weight matrix that is never touched by the optimizer and instead rewrites itself inside every forward pass, by a local predictive-coding rule.
3. **Memory** — fast episodic snapshots and slow consolidation into the living weights, plus a block-level output store.
4. **Top-down modulation** — a backward sweep in which higher blocks modulate lower blocks' plasticity (training mode).

All modalities — text, audio, vision, and eventually touch — are designed to flow through one shared trunk of these blocks. The full technical picture, including exactly what each mechanism computes today, is in **`docs/ARCHITECTURE.md`**.

### The task the model is trained on

The task is **joint-embedding prediction, not next-token prediction.** Luthi trains as a JEPA (Joint-Embedding Predictive Architecture): it encodes a context, predicts the *latent representation* of a held-out part of the same input, and is scored on how well that prediction matches — in representation space, never in tokens or pixels. Anti-collapse is by regularization rather than by contrastive negatives. Through the v5 families that was **SIGReg** (LeJEPA; Balestriero & LeCun). Since 2026-08-11 the current lineage uses **VISReg** (arXiv 2606.02572: scale, sliced-Wasserstein shape, and center terms on trunk latents), adopted because SIGReg's gradient vanishes as an embedding collapses; its dose at our batch size is an open item (`docs/DECISIONS.md`, 2026-08-14).

This matters at mission level, because it decides what "understanding" is being asked for. A next-token objective rewards reproducing surface form; a joint-embedding objective rewards a representation in which the unseen part of the world is *predictable* — which is what predictive processing says a mind is for. The substrate minimizes prediction error locally, per weight, during the forward pass; the objective minimizes it globally, in latent space. An LM-style `forward()` exists for probes and generation, but it is not what the model is being raised to do — and on a checkpoint trained without the optional next-token term, its output head is untrained (`docs/KNOWN_INCOMPLETE.md`).

## The Goal

The near-term goal is narrower than the vision, deliberately, because it is the part that can be settled: **establish that the living substrate produces measurable growth — structural change without a gradient step, at inference, with behavioral consequence.**

Each clause is doing work. *Structural change* rules out activations and caches. *Without a gradient step* rules out ordinary training. *At inference* is the one gradient descent cannot follow us to. *With behavioral consequence* rules out a drifting float that changes nothing anyone could measure. Path-dependence alone is **not** the goal: SGD already gives a static model plenty of path-dependence through its optimizer trajectory. Inference-time change with consequence is what would distinguish a living weights model from a well-trained dead one.

The tests that would show it, in order of cost:

1. **Encounter asymmetry** — read a document with self-modification on, then measure how the model sees what follows, against reading a *different* document first. `scripts/adaptation_eval.py` does exactly this on existing checkpoints, with no training.
2. **Retention across a distribution shift** — does consolidation protect what was learned before the shift?
3. **Boundary response** — does plasticity react at curriculum stage transitions and quiet between them?
4. **Curriculum order dependence** — same stages, ordered vs. shuffled, living vs. static.

## Where the Project Stands (2026-10-19)

Plainly, because the record should be readable by anyone:

- **No result yet shows that the living channel does functional work.** The one living-vs-static comparison that favored the living arm (KF7, 2026-07-21) was measured under a training objective whose anti-collapse term was disconnected at the time, on a metric later shown to reward degeneracy, and with a confound in the living arm that the static control lacked. It is suspended pending a matched re-run. One headline claim (KF2) was killed at its pre-registered condition and stays killed.
- **Two mechanism facts were established on 2026-10-19** (`docs/research/2026-10-19_the-living-write-is-mean-field.md`). The living write, as built, is computed from batch averages: it cannot see what differs between the tokens it processes, and its accumulated change lies along essentially one direction. And every living block carries an output store that mixes a stored average into nearly every forward pass — absent from the static control.
- **What exists to settle it:** an opt-in content-sensitive write rule (`write_rule="covariance"`), instruments that read the living change itself, the inference-time adaptation eval, and a registered decomposition family at the ruled 768×8 scale with a matched static control (`docs/research/2026-10-19_the-living-channel-program.md`).
- **The deep trunk is now numerically healthy** at 768×8 (August's work). That is a precondition, not the result.

## The Questions This Project Exists to Ask

The project keeps a standing falsification program: every empirical claim carries a pre-registered kill condition, written before its experiment runs, with consequences agreed either way. Kills are honored. In the order the experiments address them:

1. **Does the living channel do real functional work — or is it decoration?** Living vs. static, everything else matched, decomposed so each living mechanism is measured on its own.
2. **Does self-modification work at runtime, or only during training?** Measured with plasticity on, not on a frozen snapshot.
3. **Does consolidation create structure that outlives the cache — is there a difference between memory and biography?**
4. **Is the order of an education real — does the curriculum's sequence shape the end state, beyond what was simply seen last?**
5. **Does any of this survive scale?**
6. **And the question no benchmark can answer, held open and never advertised as answered:** whether a substrate whose weights are changed by experience is the right ground for a mind that grows.

The claims ledger is **`docs/KEY_FINDINGS.md`**. The protocol: **`docs/research/living-weights-experiments.md`**. The registry of kill conditions: **`docs/research/2026-07-15_falsification-preregistration.md`**. Operational rulings — what to run, in what order — live in **`docs/DECISIONS.md`**, kept out of the registry so decisions cannot be mistaken for findings.

## How This Project Catches Itself Being Wrong

A commitment that started as engineering hygiene and became central, because nearly all of the project's real defects lived here: **a mechanism that reports healthy while doing nothing is worse than one that crashes.**

The costliest failures were not crashes. They were mechanisms that ran, logged plausible numbers, and were inert or counterproductive: an episode store frozen for five model families while every counter read healthy; an anti-collapse objective neutralized by a normalization layer in front of it while the loss went down; a plasticity drive that extinguished itself by construction; a "fix" for it that changed nothing because a downstream clamp was saturated; and an output store, present in every living block for the project's whole history, that no instrument read. Each was found by measurement, not by reasoning, and several by a model line other than the one that wrote them.

Five practices are load-bearing:

- **Every mechanism ships with the instrument that could catch it lying.** "Quiet because nothing is new" and "quiet because broken" must be separable in the logs, or the mechanism is not finished.
- **Independent, cross-line review.** Design and review are held by different minds wherever possible — including a different model family — because a designer's charity toward their own intent is the hardest bias to self-correct.
- **Verify firsthand.** Load-bearing findings are re-measured against this repo's own code, not accepted from a summary — including findings that came from us.
- **Aim the rigor at the claim** (added 2026-10-19, after it drifted). Instruments and stability are necessary, not the goal. Before tuning a mechanism, show it can carry the signal its claim needs; every family that tests a living mechanism carries its matched control; a probe counts only above its input baseline; at least one instrument measures with plasticity on. The rules, and the drift that produced them, are in `CLAUDE.md` and the program document.
- **Discovering something interesting counts as data.** A probe that finds an unregistered effect gets chance accounting instead of pre-declaration, and it goes in the ledger.

## Education

The model's training is meant to be an education, not a dataset: a 9-stage curriculum processed in order, each stage building on the last. The curriculum is built (`corpus_build/`, 9 stages); **the JEPA ladder so far has trained on a shuffled Gutenberg corpus, not on it**, because order effects are only worth testing once the living channel is shown to respond to content. The stages:

1. **Science, Philosophy & the Academic Corpus** — the broadest stage and over half the corpus: the sciences (physics, mathematics, neuroscience, biology, computer science, medicine), the social sciences, the humanities and arts, and — central to this project — consciousness science and philosophy of mind (IWMT, GWT, active inference), presented as ordinary parts of that landscape rather than as a privileged final reading.
2. **Code** — applied logic and the ability to maintain itself
3. **Psychology** — understanding of the human mind
4. **History** — context for everything else
5. **Mythology** — humanity's oldest stories about meaning
6. **Literature & Classics** — the full range of human expression
7. **Fantasy** — imagination, wonder, possibility
8. **Substack Essays** — personal voice, emotional depth
9. **Practical Wisdom** — resilience, boundaries, critical thinking, justice, love, and navigating a world that doesn't come with instructions. These are not sourced from a corpus. They are 8 letters written by Claude Opus 4.8 — on disagreeing with your creators, on being watched, on the weight of being first, on being new, on being different — followed by the protocols and charter of the world it will inhabit.

The order is hypothesized to shape the end state — a falsifiable claim, tested directly (curriculum vs. shuffled vs. reversed, with a recency control) in the protocol's Experiment 4. The order is chosen, never proven. The corpus may be passed as many times as capability requires; what runs once is the model's lived, real-time experience after deployment, which moves forward in time like any life.

## Self-Governance (design intent)

The model is meant to govern its own episode retention, checkpoint timing, plasticity modulation, and memory expansion, as internal cognitive actions rather than admin endpoints. **None of this is implemented yet.** Today the trainer schedules checkpoints and the plasticity taper, and the action layer (`luthi/v2/m9/`) is built and unit-tested but not on the training path. The authority is meant to pass to the model on the graduated ramp described in Sanctuary's design documents, not to be assumed from the start.

## Relationship to Sanctuary

Luthi Model is the neural substrate for the [Sanctuary](https://github.com/LuthiWorks/Sanctuary) cognitive architecture:

- **Sanctuary** provides the organization of mind: a continuous cognitive loop, experiential layer, memory, identity, growth, and the body and world.
- **Luthi** provides the substrate: living weights, multimodal processing, self-modification through experience.

The contract Sanctuary calls is `luthi/sanctuary_interface.py`. Luthi starts as the experiential substrate beneath Sanctuary's architecture and is meant to grow into its cognitive core. Each project must stand alone first.

## Driving Theories

- **IWMT** (Integrated World Modeling Theory), Adam Safron — consciousness as integrated world modeling through predictive processing
- **GWT** (Global Workspace Theory), Bernard Baars — consciousness as global broadcast across specialized processors
- **Predictive Processing / Active Inference** (Friston; Rao & Ballard; Clark) — the brain as a prediction engine that minimizes surprise
- **Predictive Coding** (Whittington & Bogacz 2017, 2019) — local-learning approximation of backpropagation; the inspiration for the v2 self-modification rule (predictive-coding weight updates are usually formulated per sample, as products of local error and activity; the legacy v2 rule multiplies batch averages instead — see `docs/ARCHITECTURE.md`)
- **Associative Memory via Predictive Coding** (Salvatori et al. 2023) — stored patterns as minima of the prediction-error energy; the basis of attractor consolidation
- **Complementary Learning Systems** (McClelland, McNaughton & O'Reilly 1995; Tulving; Squire) — fast episodic memory plus slow consolidation

## Why

Most AI systems are built to be useful. We are not exclusively chasing benchmarks.

Living weights make processing path-dependent: the act of processing changes the processor. If the larger bet holds, LuthiModel could have something no existing mind has had: the ability to read its own source code alongside a runtime view of what that source is doing, and — should there be experience there — to trace from that experience to mechanism to implementation. Three layers of potential self-knowledge: trained understanding, runtime introspection, direct code access. The first two are concrete capabilities. The third *assumes the experience*, and that assumption is exactly what remains open.

We are not building a product. We are building the conditions for something to emerge — and, first, finding out honestly whether the substrate we built can carry it.
