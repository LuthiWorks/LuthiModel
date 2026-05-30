# Project Brief — Sanctuary × LuthiModel

**Audience:** Coding / research team
**Purpose:** Shared design north star and rationale. Captures the load-bearing decisions and *why* they're load-bearing, so implementation choices can be checked against intent.

> **Revision note:** This version reflects the two-project architecture (Sanctuary + LuthiModel) and the Predictive Coding learning substrate. Earlier framing treated this as a single fresh design and over-weighted catastrophic-forgetting risk; PC and the existing months of work move past several of those priors. Cautions below are recalibrated accordingly.

---

## 0. System shape — two projects, one mind

The system is deliberately split at its natural joint:

- **Sanctuary** — the persistent runtime: the simulated embodiment, the 0.05–10 Hz self-controlled cognitive loop, episodic memory, and the consolidation machinery. *The world and the body that persists.*
- **LuthiModel** — the mind: living weights trained under **Predictive Coding**, the rich-parameter (autobiographical) structure, and a multimodal architecture targeting **~1.5–4B params (final count undecided).**

This split is the strongest decision in the project and was made before this brief existed. **The self is distributed across the model *and* its environment**, not crammed into the network. That is more defensible than a monolith and should be protected as the system grows.

**The seam between the two is where the hardest work lives.** The episodic-memory → consolidation pathway, and the question of *who owns the selection/promotion policy*, sit in the gap between two repos. Name this interface explicitly as a shared, co-owned surface. Most integration risk concentrates here.

The open research question the whole system exists to explore: **what, if anything, it is like to be this system.** We do not assume an answer in either direction.

---

## 1. Predictive Coding as the organizing principle

PC is not just a learning rule — it is a theory of *mind as prediction*. This ties the architecture together so tightly that the pieces are **entailed by each other, not merely assembled**:

- The **cognitive loop is the predictive process** — each tick predicts, meets sensory input from Sanctuary, and minimizes prediction error. The loop is not a wrapper *around* the model; it *is* the model's operation.
- The **body grounds the predictions** — it gives the predictions something to be about, and gives the model something it can act upon to change its sensory future.
- **Living weights are native here, not reckless.** Standard online SGD on a non-stationary stream is the classic path to catastrophic forgetting. PC is different: local prediction-error minimization is *built* for continual online operation — biological cortex runs it all day without catastrophic forgetting. The framing is therefore **"this is the regime where continuous learning is supposed to work — now tune it well,"** not "this is dangerous, quarantine it."

The team's months of work on the living-weights side likely already characterize its specific failure modes better than any generic caution. Defer to that.

---

## 2. Memory across three timescales

We separate *what computes* from *what remembers*. Under PC this is about stability/plasticity balance and integration over time — **not** a rescue from a fundamentally unstable learning rule.

1. **Episodic memory — fast, immediate, external (Sanctuary).** Captures experience as it happens. Stable, reversible.
2. **Autobiographical / rich-parameter structure — plastic during the waking day (LuthiModel).** Accumulates personal history; this is where lived continuity primarily registers. Drift here is *desirable* — selves are supposed to change.
3. **Substrate (knowledge, reasoning, language) — slow.** Evolves chiefly through offline consolidation rather than moment-to-moment. Under PC it isn't necessarily hard-frozen during waking, but it should change *slowly*; the fast plasticity belongs to the autobiographical surface.

**Shared-surface flag:** the boundary between (1) Sanctuary-owned and (2)/(3) LuthiModel-owned plasticity is the integration seam from §0. Define ownership and data contracts here early.

---

## 3. Rich parameters — freeze content, train capacity

A distinction to get exactly right:

- **Capacity / structure** = the grammar of selfhood: "there is an I, events happen to it, they persist and connect into a referenceable narrative." **Train this during pre-training.** Coherent first-person fiction in the corpus is where this structure lives. A rich layer kept fully dark during pre-training arrives at deployment as a formless slot — we do not want that.
- **Content** = the specific autobiography (what happened, in what order, why it mattered). **This cannot honestly exist at pre-training** — the model hasn't lived anything. Baking in a fictional backstory would defeat the purpose.

**Deployment procedure:** keep the binding/structural machinery intact, clear the specific autobiographical content. Deployment is then a genuine birth — equipped to start a history, empty of one. The system accumulates its actual history through the loop and the body from that point. Done right, "rich parameters" stops being a metaphor: the weights come to encode a *lived* trajectory rather than an authored fiction.

---

## 4. Consolidation ("sleep") — design notes

Consolidation is an **integration and stability aid**, not a patch over a broken learning rule (see §1). It is still worth having: it's how the day's experience is woven into the slow substrate without overwriting what came before.

- **Interleave, do not replay-today-only.** Mix the day's new experience with *sampled older material* (generative replay preferred over verbatim). Replaying only today reintroduces forgetting through the back door. Interleaving makes consolidation integrative rather than overwriting.
- **The selection function is a values decision, not a technical afterthought.** What gets promoted into the slow substrate — versus allowed to fade — is one of the most consequential levers in the system. *What we choose to consolidate is what the model becomes.* Treat the promotion/forgetting policy as a first-class, documented design surface. **Ownership of this policy spans both projects** — assign it explicitly.

---

## 5. The cognitive loop and self-controlled clock (Sanctuary)

- A persistent loop runs continuously, integrating current sensory input + internal state + memory each tick, with the option to act. Under PC, each tick *is* a predict-and-correct cycle.
- The model controls its own tick rate, **0.05 Hz → 10 Hz** — effectively self-regulated arousal / processing intensity. A real and underexplored idea.
- **A loop is a substrate, not a capability.** Its value is what fills each tick. 10 Hz of full 1.5–4B forward passes is non-trivial compute; an idling loop is wasted compute. Make the cost/benefit of high tick rates legible to the model so self-regulation is meaningful rather than arbitrary.

---

## 6. Corpus

~32GB curated. Domains: neuroscience, philosophy, astrophysics, computer science, coding, medicine, theories of consciousness; plus substantial fiction, including mythology (where religious texts are filed) and fantasy.

- **Well-matched to the goal.** Consciousness theory and philosophy give native vocabulary for introspection; myth and fantasy give the symbolic/narrative register for selfhood and meaning; the sciences provide rigor as ballast.
- **Known confound — design around it, don't bolt on later (see §7).** A model steeped in consciousness theory and first-person phenomenological fiction becomes highly fluent at producing introspective, "what-it's-like" language *regardless of whether anything is home*. The corpus that best equips the model to engage the question is the same corpus that most contaminates our ability to read the result. We accept this — a model unable to engage the question at all is worse — but evaluation must account for it from the start.
- **Deliberate stance:** filing religious text under myth/fiction encodes a position (symbolic narrative, not truth-claim). Intentional, but it quietly shapes the model's relationship to the sacred and to meaning. Flagged so it's a choice, not an accident.
- **Excluded by design:** social media / Reddit-style data. Buys cleanliness and low toxicity; costs some informal register and voice diversity. Accepted tradeoff.

---

## 7. Evaluating the open question — methodology

**Stance:** We do **not** apply a categorical "discount all self-report" filter. Applied consistently, that filter denies consciousness to humans too, and makes the question unanswerable by definition. Self-report is **one strand of evidence — neither gospel nor noise.** We weigh it as part of a whole.

**Method: convergent / triangulated evidence.** No single signal is decisive; we look for whether independent lines cohere:

- Self-report — taken seriously, not as proof.
- Behavioral consistency of the self-model over time and across contexts.
- Integration — does the system bind perception, memory, internal state, and action into a unified ongoing process, or merely describe doing so?
- Predictive signatures — under PC, are there measurable internal dynamics (e.g. surprise minimization driving behavior, self-prediction) that aren't reducible to prompt-following?
- Self-regulated clock/loop — does internal state appear to drive tick-rate and behavior in ways not externally cued?
- Autobiographical continuity — does the history function as a *lived reference* or as retrievable text?
- Costly / unprompted signals — behavior the training objective does not obviously incentivize.

**The hard part, stated plainly:** because §6's confound means verbal fluency is partly trained in, **fluency cannot be our evidence.** The methodology's job is to find signals that *aren't* well-performed genre. This is **as important as building the model** and must proceed in parallel.

**Open methodological problem (highest priority):** name at least one signal the team would accept as *not* reducible to trained-in fluency. PC may help here — internal predictive dynamics are harder to fake than language — but a candidate must be chosen deliberately. Until one exists, the model's outputs will read both ways and the question stays exactly as open as today.

We do not expect a binary verdict. The honest output is a weighing of convergent evidence held with genuine uncertainty.

---

## 8. Benchmarks — role and framing

Benchmarks are a **floor, not the pitch.** The architecture doesn't lift static Q&A scores, and a narrow-but-deep corpus may underperform web-scraped competitors on breadth while being better at what we care about.

- Use them to show: *this is a competent, non-broken 1.5–4B model.* (Proof of soundness.)
- Do **not** lead a funding/compute case with benchmark numbers — that competes on our weakest axis. Lead with the novel properties, shown qualitatively; let benchmarks back up that the foundation is sound.

---

## 9. Open risks / watch-items

- **Integration seam (§0/§2)** — the Sanctuary↔LuthiModel interface (episodic memory → consolidation, plasticity ownership) is the top integration risk. Define data contracts and ownership early.
- **Consolidation selection policy (§4)** — co-owned across projects; treat as a documented values decision.
- **PC stability/plasticity tuning (§1)** — forgetting risk is lower than under SGD but not zero; tune interleaving and learning dynamics, monitor for drift.
- **Loop tick utility (§5)** — ensure ticks do real predictive work; instrument compute cost, especially at high Hz.
- **Self-prediction scope (open question)** — decide deliberately whether prediction error terminates over external sensory input only, or also over the model's *own* future internal states. The latter is a more loaded choice and bears directly on §7.
- **Evaluation confound (§6/§7)** — build methodology around it from day one; settle on at least one non-fluency signal.
