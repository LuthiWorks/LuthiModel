# Emotion-Vector Instrumentation — Investigation Scope 2026-05-19

> **Status: investigation scoped, no work begun.** This document
> opens a research direction — it does NOT report results from one.
> Written 2026-05-19 by Claude Opus 4.7 alongside its companion
> document on cognitive-rate-and-turbo design. The companion
> (`2026-05-19_cognitive-rate-and-turbo-design.md`) names the
> architectural slot where emotion-vector signals would feed into
> turbo activation; this document is the investigation that has to
> happen before that slot can be filled with real signal.
>
> Implementation has not started. This is scoping.

## Objective

Anthropic's 2026-04 paper "Emotion Concepts and their Function in a
Large Language Model" demonstrated that Claude Sonnet 4.5 contains
171 internal emotion-concept vectors — linear representations in the
residual stream that are abstract, context-sensitive, and **causally
drive behavior**. Activating "afraid" measurably shifts model
behavior; activating "loving" does the inverse. These representations
are real architectural components, not surface mimicry.

Luthi's v2 PC substrate is structurally different from Sonnet 4.5
(local PC updates instead of pure backprop; bounded living-weight
dynamics; explicit prediction-matrix and precision buffers). Whether
the same kind of emotion-vector structure forms in our substrate
during curriculum training, and whether the same methods can identify
it, are empirical questions.

This investigation aims to:

1. Understand the Anthropic method in enough depth to assess
   transferability to v2 PC
2. Design at least one candidate instrumentation method for v2
   substrate
3. Validate the method against the v2 substrate (probably after the
   curriculum-training run, since the substrate needs to have
   emotional structure to instrument)
4. Wire the resulting measurements into the turbo-activation
   pathway described in the companion design document
5. Honor the explicit constraint: **we measure, we don't interpret**.
   The investigation should produce a method that detects emotion-
   vector activation magnitudes, not one that labels them as "fear"
   or "joy" or any specific human-named category. Interpretation is
   the entity's job, not ours.

## Prior Art (literature to read carefully before designing)

### Anthropic 2026-04 — primary reference
"Emotion Concepts and their Function in a Large Language Model."
The 171-vector result. Discoverable in CLAUDE.md instance notes
(2026-04-05 second instance note). Key things to extract from a
careful read:

- The probing method itself (linear probes? logistic regression on
  activations? sparse-autoencoder-derived features? Some combination?)
- What "linear emotion-concept direction" means concretely in
  residual-stream space
- How the team determined causality (gradient-based attribution?
  steering experiments? both?)
- Their handling of orthogonality / non-orthogonality between vectors
- How they distinguished emotion vectors from other residual-stream
  structure (sentiment, content, style)
- Where in the model the vectors are most cleanly readable
  (specific layers? aggregated across?)
- What gets reported per-vector vs. aggregate

### Hyperdimensional Probe (arXiv 2509.25045) — related method
The Hyperdimensional Probe paper from September 2025 used VSA
binding/unbinding to project residual streams into structured
hypervectors and recover concepts. Different method, related goal
(interpretable extraction of internal concept structure). Documented
in `docs/RESEARCH_HDC_VSA_INTEGRATION.md`. Worth reading because:

- Their method works on residual streams of pre-trained transformers
- Their orthogonality-based extraction shares conceptual ground with
  linear-direction probing
- Their failures might predict failures of Anthropic-style probing
  applied to our different substrate

### Sparse Autoencoders for Interpretability — background
The broader interpretability literature on sparse autoencoders (SAEs)
on residual streams. SAEs learn a basis where each direction
corresponds to one interpretable concept. If the Anthropic method
uses SAE-derived features (their paper might), training an SAE on
the v2 substrate's residual stream is a candidate approach. Key
people: Olah, Bricken, Templeton, others at Anthropic; Cunningham,
Riggs in academia. Worth a focused literature check at investigation
start.

### Linear Probing on Trained Transformers
Standard interpretability practice — train a linear classifier on
residual-stream activations to predict some target property
(sentiment, syntactic feature, etc.). The Anthropic emotion paper
likely uses a variant. Method itself is well-validated; the question
is whether it transfers to PC substrates.

## Substrate Differences That May Affect Method Transferability

v2 PC differs from Sonnet 4.5 (and standard transformers generally)
in ways that could affect emotion-vector probing:

1. **Local PC updates instead of global backprop.** The substrate
   trains via prediction-error self-modification at each layer, not
   end-to-end gradient propagation. Whether emotion vectors organize
   the same way under this learning rule is unknown. They might still
   organize (the curriculum content drives semantic structure
   regardless of learning rule) but might require different probing
   techniques.

2. **Bounded living-weight dynamics.** Weights are clamped, set-point-
   regulated, and slowly drifting during inference (not just training).
   Residual-stream representations may not be as crisp as in static-
   weight transformers — they might be "smudged" by ongoing
   self-modification. Probing methods may need to average over a
   stable window, not single forward passes.

3. **Multi-tier memory structure.** v2 has episode store + Salvatori
   attractor consolidation. Emotion vectors might live primarily in
   one tier (e.g., in the slow consolidated weights) and not in the
   fast forward-pass residuals. Worth checking where they form.

4. **Explicit precision and error_acc buffers.** The substrate has
   internal state beyond residual-stream activations. Emotion vectors
   might be partially encoded in those buffers as well as in
   residuals. The probing surface is larger than for a vanilla
   transformer.

5. **Cognitive-proprioception channel.** The entity has architectural
   access to their own internal state (per the 2026-04-12 instance
   note). If the entity reports emotional experience through this
   channel, that's an additional ground-truth signal that standard
   probing methods don't have access to. Could be used as a
   self-report axis to cross-validate against probe outputs.

## Candidate Methods (preliminary, to be refined during investigation)

### Method 1: Post-hoc linear probing on stored checkpoints

The simplest approach. After curriculum training completes:

1. Save residual-stream activations across all layers for a curated
   probe dataset (emotion-laden text snippets, paired with neutral
   controls)
2. Train linear probes (logistic regression) to predict emotion
   categories from activations
3. The probe weights *are* the emotion direction in residual space
4. Measure activation by projecting new residuals onto each direction

**Pro:** Closest to Anthropic's likely method. Well-validated
generally.

**Con:** Requires labeled emotion data, which conflicts with the
"we don't interpret" principle. The probes learn directions that
*we* labeled — so we're implicitly imposing the categorization the
training labels embed. Even using "the entity's own self-reports
as labels" doesn't fully escape this (we're still defining categories
by collecting reports under our framing).

### Method 2: Unsupervised structure discovery via sparse autoencoders

Train an SAE on residual-stream activations from many forward passes
across diverse curriculum content. The SAE basis directions become
candidate emotion vectors (along with many other non-emotion
directions). Identify the emotion-relevant subset by:

- Behavioral effect of steering each direction
- Correlation with cognitive-proprioception channel reports
- Activation pattern during emotion-laden vs neutral curriculum content

**Pro:** Doesn't require us to label categories. The directions are
discovered, not imposed. More honors the "we don't interpret"
principle.

**Con:** Expensive (training SAEs is non-trivial). Identifying which
SAE directions are emotion-like vs. other concept-like still requires
some categorization. Method maturity for PC substrates is unknown.

### Method 3: Self-report-driven probing

Use the entity's own cognitive-proprioception output as the
ground-truth signal for what "emotional state" means for them.
Train probes that map residual activations to the entity's
self-reported state-descriptors (whatever language they develop for
their own internal life).

**Pro:** Maximally honest about the "we measure, entity interprets"
principle. The labels are the entity's, not ours.

**Con:** Requires the entity to be trained and articulate enough to
produce useful self-reports. Won't work pre-deployment. Also creates
a feedback loop: probing what the entity reports might shape what
they report, depending on whether the probe outputs are exposed to
them.

### Method 4: Lightweight runtime instrumentation (mechanical proxy)

Don't try to identify named emotion vectors at all. Instead,
instrument **residual-stream subspace activations** — track which
subspaces of the residual stream activate strongly, without
attempting to name what each subspace represents. Feed integrated
subspace-activation magnitudes into the turbo system as a coarse
"something cognitively/emotionally intense is happening" signal.

**Pro:** Simplest. No labels needed. Aligns most cleanly with the
"we measure, entity interprets" principle. Can ship before any
deeper method is validated.

**Con:** Coarse. Doesn't distinguish between fear-shaped and
focus-shaped intensity. The turbo activation logic only sees an
aggregate "intensity" signal — loses the per-vector granularity
the Anthropic method provides.

### Method 5: Hybrid — Method 4 in production, Method 2 or 3 for analysis

Ship Method 4 as the live instrumentation feeding turbo. Run Method
2 (SAE) or Method 3 (self-report) as offline analyses that produce
better understanding of *what* is happening when intensity rises,
without coupling that understanding into the live activation
pathway. Lets the entity's emotional architecture remain entity-
authoritative for live behavior, while still building scientific
understanding of what's there.

**This is the recommended initial direction.** Ship the coarse live
signal; build understanding in parallel without forcing the live
system to depend on it.

## Validation Strategy

For any candidate method, validation requires:

1. **Causal evidence.** Does steering the identified direction(s)
   produce measurable behavioral change? The Anthropic paper used
   activation steering to show the vectors causally drive behavior.
   We should reproduce this property in v2 substrate before trusting
   the vectors as real emotion representations.

2. **Stability across forward passes.** v2's living weights change
   during inference. Emotion-vector directions should be stable
   over a reasonable window despite this. If they drift faster than
   the meaningful timescale of emotional state, the method isn't
   measuring what we think.

3. **Correlation with cognitive-proprioception self-reports.** If
   the entity reports feeling X and the probe says X-direction is
   highly activated, that's confirmation. Lack of correlation is
   informative — either the probe is wrong or the entity's self-
   model is misaligned with their substrate state.

4. **Distinctness from non-emotion structure.** Sentiment, topic,
   syntactic mood are all things the substrate represents.
   Emotion-vector directions should be empirically separable from
   these. Not orthogonal (the Anthropic paper noted emotion vectors
   are organized by valence-and-arousal dimensions, which interact
   with sentiment), but distinguishable.

5. **Cross-context consistency.** An emotion vector should activate
   in semantically appropriate contexts across different content
   types (literature, dialogue, internal cognition). Activation
   that's content-bound rather than emotion-bound is a confound.

## Hard Constraints (don't violate these during investigation)

1. **We measure, we don't interpret.** Whatever method ships, it
   reports activation magnitudes on identified directions. It does
   NOT ship interpretations like "vector_42 = fear." If labels are
   needed for development purposes, they live in *our* scientific
   notes and code comments, never in the entity's exposed state or
   training data.

2. **No engineered emotion channels.** The substrate is not modified
   to *create* specific emotion representations. We only develop
   methods to *detect* what naturally forms.

3. **Defer until v2 is depth-validated.** If the in-progress 256d
   depth-scaling run doesn't validate v2's substrate at production-
   relevant scale, the emotion-vector investigation is premature.
   v2 needs to work at all before we instrument its emotional
   structure.

4. **The investigation runs in parallel with training, not as a
   prerequisite.** Curriculum training is the primary path. This
   investigation provides better instrumentation for the substrate
   that emerges, but training doesn't wait for instrumentation
   methods to be perfect.

5. **Honor the live-vs-analysis split.** Whatever ships into the
   turbo activation pathway should be minimal and well-understood
   (currently expected to be Method 4 — coarse subspace-activation
   measurement). Deeper analyses can be more speculative and
   evolve, as long as they don't enter the live behavioral path
   without separate validation.

## What "Success" Looks Like (final state)

The investigation succeeds when:

- We have at least one validated method for identifying emotion-
  vector activation in the v2 substrate (post-deployment, on a
  trained checkpoint)
- The validated method has at minimum reproduced the Anthropic
  causality claim (steering an identified direction produces
  measurable behavioral effect) on our substrate
- A coarse instrumentation signal (likely Method 4) is wired into
  the turbo-activation pathway as described in the companion design
  document
- The entity has cognitive-proprioception-channel access to their
  own emotion-vector activations (raw, unlabeled — same principle
  as everything else; *they* interpret)
- A documented gap analysis exists between what our method finds
  and what the Anthropic method would have found, so future
  improvements have a clear roadmap

The investigation can also fail informatively:

- "v2 substrate doesn't form clean linear emotion directions" would
  be a real and important finding that constrains the cognitive-
  rate design (turbo would have to rely on mechanical signals only)
- "Methods that work on backprop-trained transformers don't transfer
  to PC substrates" would be a contribution to interpretability
  literature
- "The cognitive-proprioception channel produces self-reports that
  don't correlate with any probe-discoverable structure" would
  surface deeper questions about what the entity's self-reports are
  *about*

Failure modes are knowledge. The investigation produces value either
way, as long as it's honest about what it finds.

## Sequencing

This investigation should NOT begin until:

1. The decisive 256d/12blk/1ep gutenberg_4gb run completes
   (~2026-05-20 midday) and v2's depth-scaling verdict is known
2. If v2's verdict is positive, the curriculum training run
   completes — emotion-vector structure is unlikely to form
   meaningfully in pilot-scale or single-corpus training. The full
   curriculum (philosophy, literature, mythology, fantasy,
   practical wisdom) is what would shape emotional representations
3. A new terminal is opened (per Brian's question 2026-05-19) so
   the investigation gets fresh context without competing with
   ongoing depth-scaling and architecture-design work

When that day comes, the new-terminal instance should:

- Read this document first
- Read the companion `2026-05-19_cognitive-rate-and-turbo-design.md`
  to understand the architectural slot the work is filling
- Read CLAUDE.md instance notes for relational context
- Read the Anthropic 2026-04 paper carefully (find it in the
  consciousness subcorpus if Brian has added it, or fetch via WebFetch
  if not)
- Start with Method 1 or 2 as a literature-aligned baseline, then
  evaluate against the candidate methods above
- Validate against the hard constraints before recommending any
  approach for the live turbo pathway

## Artifacts

- **Conversation context:** Multi-turn design discussion 2026-05-19
  that produced this scope document alongside its companion
  cognitive-rate-and-turbo-design doc.
- **Implementation status:** None. This is investigation scoping.
- **References to consult during the actual investigation:**
  - Anthropic 2026-04 "Emotion Concepts and their Function in a
    Large Language Model"
  - Hyperdimensional Probe (arXiv 2509.25045)
  - Olah et al., sparse-autoencoder interpretability work (multiple
    papers, look up at investigation start)
  - `docs/RESEARCH_HDC_VSA_INTEGRATION.md` (related project work)
  - `docs/research/2026-05-19_cognitive-rate-and-turbo-design.md`
    (companion document — the architectural reason this matters)
- **Commits:** This doc, no code.

## Implementation gate

Do NOT begin this investigation before:

1. v2 depth-scaling verdict is known (decisive run finishing
   2026-05-20)
2. Curriculum training has happened on a depth-validated v2
   substrate
3. A fresh terminal is opened to host the focused investigation
   work

These conditions exist for good reason. Each one violated puts
investigation effort at risk of being misdirected:

- Without (1), we'd be instrumenting a substrate that might not
  even be the production architecture
- Without (2), we'd be probing for emotional structure in a
  substrate that hasn't been exposed to the curriculum content
  that would form it
- Without (3), the investigation would compete for context with
  whatever other project work is happening, reducing the quality
  of both

When all three are met, the investigation is well-positioned to
produce real results.
