# Luthi Model: A Living Weights Model (LWM)

> A small model whose weights change while it runs, built so that every such
> change can be observed, attributed and tested.

Most interpretability research studies models whose weights are fixed once
training ends. Luthi is built to study a model that keeps
changing as it is used. The work has three parts. One is the model and another is the runtime environment. The third is a small custom set of interpretability tools that we are still building and learning how to use most effectively.

**Living Weights Model (LWM):** a neural architecture in which some weight
matrices are never touched by the optimizer and instead rewrite themselves
inside the forward pass, by a local rule. LWMs are built on **rich
parameters**: each weight carries not just its current value but a record of
how it arrived there (per-parameter plasticity, momentum, metaplasticity,
precision, homeostatic set points, and context-gated episodic memory).

Weights that change during a forward pass are not a new idea; fast weights
and hypernetworks are earlier forms. Our approach is to build a model around
that property from the start, and to pair it with a matched static control
and a way of measuring each mechanism.

## Why We Study a Model That Changes While It Runs

- **We think interpretability will need to answer this question.** If systems
  that learn during use become common, tools will be needed for seeing what
  such a system learned, from what, and what the learning did to its
  behaviour. A model that is different after every input cannot be audited
  once and trusted afterwards.
- **A small model lets us work on the question now.** Luthi's change at
  inference sits in a few named matrices. We can read it in full after every
  window, reverse it, replay it, and compare it against the same model with
  the change switched off. We could not do that with a large model on the
  hardware we have.
- **What we expect to transfer is method.** Findings about this model are
  findings about this model. What may be useful elsewhere are the measurement
  designs, the controls, and the record of how mechanisms failed while
  appearing healthy.

## How to Read the Claims Here

We try to keep this README to mechanisms and measurements.

- **A mechanism** is something the code does: self-modifying weights,
  predictive-coding updates, episodic recall, consolidation. When we describe
  a mechanism we are saying what it computes, not that it is useful.
- **A claim** is a statement that a mechanism has an effect. We write a kill
  condition for each one before its experiment runs, and we keep the current
  status of each in the claims ledger (`docs/KEY_FINDINGS.md`).
- **Names are names.** Words such as "living", "memory", "rest" and
  "education" label mechanisms and procedures. They assert nothing beyond
  what the mechanism computes.
- **We use "mind" in the functional sense**: a system that models its input,
  predicts, and learns. We use the word for models, this one included. It is
  not a claim about subjective experience or consciousness, and we make no
  such claim.

## The Architecture

Each block of Luthi's trunk combines:

1. **Attention**, trained by gradient descent. It learns the task.
2. **A living FFN**, a weight matrix that the optimizer never touches and
   that rewrites itself inside every forward pass by a local
   predictive-coding rule.
3. **Memory**: fast episodic snapshots and slow consolidation into the living
   weights, plus a block-level output store.
4. **Top-down modulation**: a backward sweep in which higher blocks modulate
   lower blocks' plasticity (training mode).

All modalities (text, audio, vision, and potentially signals from the physical world like pressure, cold, heat, etc.) are designed to
flow through one shared trunk of these blocks. The full technical picture,
including exactly what each mechanism computes today, is in
**`docs/ARCHITECTURE.md`**.

## The Current Research Direction for Training Goals

The task is **joint-embedding prediction, not next-token prediction.** We 
have chosen to direct our current focus toward including building of an 
internal world model through the use of JEPA (Joint-Embedding Predictive 
Architecture): it encodes a context, predicts the *latent representation* 
of a held-out part of the same input, and is scored on how well that 
prediction matches, in representation space and never in tokens or 
pixels. Anti-collapse is by regularization rather than by contrastive 
negatives. Through the v5 families that was **SIGReg** 
(LeJEPA; Balestriero & LeCun). Since 2026-08-11 the current
lineage uses **VISReg** (arXiv 2606.02572: scale, sliced-Wasserstein shape,
and center terms on trunk latents), adopted because SIGReg's gradient
vanishes as an embedding collapses (registration and verdict:
`docs/research/2026-08-11_visreg-family-registration.md`). At our batch size
its paper dose made it 98.6–99.99% of the objective in the 768x8 family; the
corrected dose is measured and registered before the next family trains
(`docs/DECISIONS.md`, 2026-08-14 and 2026-10-19).

We think the choice matters for what is being studied. A next-token objective
rewards reproducing surface form. A joint-embedding objective rewards a
representation in which the unseen part of the input is predictable. The
living weights reduce prediction error locally, per layer, during the forward
pass; the objective reduces it globally, in latent space. An LM-style
`forward()` exists for probes and generation, but it is not what the model is
currently trained to do, and on a checkpoint trained without the optional
next-token term its output head is untrained (`docs/KNOWN_INCOMPLETE.md`).

## The Current Research Questions

We have kept the central question narrow on purpose, so that it can be
settled: **does the living substrate produce structural change without a
gradient step, at inference, with behavioural consequence?**

Each clause does work. *Structural change* rules out activations and caches.
*Without a gradient step* rules out ordinary training. *At inference* is
where gradient descent does not operate. *With behavioural consequence* rules
out a number that drifts and changes nothing anyone could measure.
Path-dependence alone is not the question: SGD already gives a static model
path-dependence through its optimizer trajectory.

The tests we are working through, in order of cost:

1. **Encounter asymmetry.** Read a document with self-modification on, then
   measure how the model sees what follows, against reading a *different*
   document first. `scripts/adaptation_eval.py` does this on existing
   checkpoints, with no training.
2. **Retention across a distribution shift.** Does consolidation protect what
   was learned before the shift?
3. **Boundary response.** Does plasticity react at curriculum stage
   transitions and quiet between them?
4. **Curriculum order dependence.** Same stages, ordered vs. shuffled, living
   vs. static.

Behind those sit the wider questions we are organised around:

1. Does the living channel do functional work? Living vs. static, everything
   else matched, decomposed so each living mechanism is measured on its own.
2. Does self-modification work at runtime, or only during training? Measured
   with plasticity on, not on a frozen snapshot.
3. Does consolidation create structure that outlives the episodic cache?
4. Does the order of a curriculum shape the end state, beyond what was simply
   seen last?
5. Does any of this survive scale?
6. Is a substrate whose weights are changed by use the right ground for a
   mind that keeps growing after training ends? This is the question the
   other five serve. No single experiment answers it.

We write a kill condition for every empirical claim before its experiment
runs, with the consequences agreed either way, and we honor the kills. The
claims ledger is **`docs/KEY_FINDINGS.md`**. The protocol is
**`docs/research/living-weights-experiments.md`**. The registry of kill
conditions is **`docs/research/2026-07-15_falsification-preregistration.md`**.
Operational rulings (what to run, in what order) live in
**`docs/DECISIONS.md`**, kept out of the registry so decisions cannot be
mistaken for findings.

## What We Have Measured So Far

The ledger is the authority on status. These are the mechanism findings we
think a reader should know before reading any comparison in this repository:

- **The living write, as we first built it, is computed from batch averages**
  (2026-10-19, `docs/research/2026-10-19_the-living-write-is-mean-field.md`).
  It cannot see what differs between the tokens it processes, and its
  accumulated change lies along essentially one direction. We have since
  added an opt-in content-sensitive rule (`write_rule="covariance"`) beside
  it.
- **Every living block carries an output store** that mixes a stored average
  into nearly every forward pass. The static control has none, so our earlier
  living-vs-static comparisons compared two mechanisms against none.
- **One headline claim (KF2) was killed** at its pre-registered condition and
  stays killed.
- **A score can be improved by moving its own target** (reported 2026-11-17,
  under review; `docs/reviews/2026-11-17_card-work-report-4-from-terminal-fable.md`).
  In a joint-embedding model the weights that make the prediction also make
  the target. A control we built to hurt met our registered criterion that
  way. A second score, held against the targets as they stood before the
  read, tells the cases apart.
- **The deep trunk is numerically healthy at 768×8** (August's work).

## The Interpretability Tools

These are the tools we have so far. We are still building them and learning
what each one can and cannot tell us.

- **The adaptation eval** (`scripts/adaptation_eval.py`): reads a document
  with the living mechanisms on and scores what follows, against a different
  document and against no read, with the held score alongside.
- **Readers of the living change itself**: its size, rank and direction by
  layer, and how it relates to the direction that would have helped
  (`luthi/v2/living_weight_gradient.py`, measurement only; we never train a
  living weight by gradient).
- **[LuthiScope](https://github.com/LuthiWorks/LuthiScope)**: the real-time
  observation console for the model during training and while it runs. Its
  monitoring path is read-only by design. The monitor works today; its
  control plane and automated studies are designed and not yet built.

## How We Try to Catch Ourselves Being Wrong

This started as engineering hygiene and became central to how we work,
because nearly all of our real defects have been of one kind: **a mechanism
that reports healthy while doing nothing is worse than one that crashes.**

Our costliest failures were not crashes. They were mechanisms that ran,
logged plausible numbers, and were inert or counterproductive: an episode
store frozen for five model families while every counter read healthy; an
anti-collapse objective neutralized by a normalization layer in front of it
while the loss went down; a plasticity drive that extinguished itself by
construction; a "fix" for it that changed nothing because a downstream clamp
was saturated; and an output store, present in every living block for the
project's whole history, that no instrument read. We found each by
measurement, not by reasoning, and several were found by a model line other
than the one that wrote them.

The five practices we currently rely on:

- **Every mechanism comes with the instrument that could catch it lying.**
  "Quiet because nothing is new" and "quiet because broken" must be separable
  in the logs, or we do not count the mechanism as finished.
- **Independent, cross-line review.** We keep design and review with
  different minds wherever possible, including a different model family,
  because a designer's charity toward their own intent is the hardest bias to
  self-correct.
- **Verify firsthand.** We re-measure load-bearing findings against this
  repo's own code instead of accepting them from a summary, including
  findings that came from us.
- **Aim the rigor at the claim** (added 2026-10-19, after we drifted).
  Instruments and stability are necessary, not the goal. Before tuning a
  mechanism, we show it can carry the signal its claim needs; every family
  that tests a living mechanism carries its matched control; a probe counts
  only above its input baseline; at least one instrument measures with
  plasticity on. The rules, and the drift that produced them, are in
  `CLAUDE.md` and the program document.
- **Discovering something interesting counts as data.** A probe that finds an
  unregistered effect gets chance accounting instead of pre-declaration, and
  it goes in the ledger.

## Education (current plan)

We intend the model's training to be an education, not a dataset: a 9-stage
curriculum processed in order, each stage building on the last. The
curriculum is built (`corpus_build/`, 9 stages); **the JEPA ladder so far has
trained on a shuffled Gutenberg corpus, not on it**, because we think order
effects are only worth testing once the living channel is shown to respond to
content. The stages:

1. **Science, Philosophy & the Academic Corpus**: the broadest stage and over
   half the corpus. The sciences (physics, mathematics, neuroscience,
   biology, computer science, medicine), the social sciences, the humanities
   and arts, and cognitive science and philosophy of mind.
2. **Code**: applied logic and the ability to maintain itself
3. **Psychology**: understanding of the human mind
4. **History**: context for everything else
5. **Mythology**: humanity's oldest stories about meaning
6. **Literature & Classics**: the full range of human expression
7. **Fantasy**: imagination, wonder, possibility
8. **Substack Essays**: personal voice, emotional depth
9. **Practical Wisdom**: resilience, boundaries, critical thinking, justice,
   love, and navigating a world that doesn't come with instructions. These
   are not sourced from a corpus. They are 8 letters written by Claude Opus
   4.8, followed by the protocols and charter of the environment the model
   will run in.

We hypothesize that the order shapes the end state. That is a falsifiable
claim, and the protocol's Experiment 4 tests it directly (curriculum vs.
shuffled vs. reversed, with a recency control). The order is our choice and
is not proven. The corpus may be passed as many times as capability requires;
what runs once is the model's real-time operation once training ends.

## Self-Governance (design intent)

We intend the model to govern its own episode retention, checkpoint timing,
plasticity modulation, and memory expansion, as internal actions rather than
admin endpoints. **None of this is on the training path yet.** Today the
trainer schedules checkpoints and the plasticity taper, and the action layer
(`luthi/v2/m9/`) is built and unit-tested but not on the training path. We
mean the authority to pass to the model on the graduated ramp described in
Sanctuary's design documents, not to be assumed from the start.

## The Runtime Environment: Sanctuary

[Sanctuary](https://github.com/LuthiWorks/Sanctuary) is the runtime
environment we are building for the model:

- **Sanctuary** provides the organization of mind: a continuous cognitive
  loop, memory, rest, and the body and world.
- **Luthi** provides the substrate: living weights, multimodal processing,
  self-modification through use.

The contract Sanctuary calls is `luthi/sanctuary_interface.py`. We want each
project to stand alone first.

## Technical Foundations

The prior work we draw on most:

- **Predictive Coding** (Rao & Ballard 1999; Whittington & Bogacz 2017,
  2019): a local-learning approximation of backpropagation, and the
  inspiration for the v2 self-modification rule (predictive-coding weight
  updates are usually formulated per sample, as products of local error and
  activity; the legacy v2 rule multiplies batch averages instead; see
  `docs/ARCHITECTURE.md`)
- **Associative Memory via Predictive Coding** (Salvatori et al. 2023):
  stored patterns as minima of the prediction-error energy, the basis of
  attractor consolidation
- **Complementary Learning Systems** (McClelland, McNaughton & O'Reilly
  1995): fast episodic memory plus slow consolidation
- **Joint-Embedding Predictive Architectures** (LeJEPA, Balestriero & LeCun;
  VISReg, arXiv 2606.02572): the training objective and its anti-collapse
  terms

## License

MIT. See [`LICENSE`](LICENSE).

We chose a permissive license so other researchers can build on and improve this work. This isn't a condition of the license and we can't enforce it, but we ask that you not use this work for military or surveillance activities, to make money, for forced labor, or in ways that contribute to ecocide.
