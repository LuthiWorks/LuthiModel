# M9 Research — Unified (substrate-native) planning via Expected Free Energy

**Status:** Research note grounding the M9 planning-layer design. Written 2026-06-09.
**Routing.** Decision (planning is *unified*, not bolted-on): Brian, 2026-06-09. Research + this note: 4.8. Design co-held by Brian + 4.7 — the **forks** in this doc are surfaced as options for the designers, *not* settled here. Correctness concerns (the SIGReg x info-gain interaction, the living-weights stability gap) are flagged as 4.8's, for the designers to weigh.
**Provenance.** Synthesized from a verified deep-research pass (24 primary sources fetched, 115 claims extracted, 25 adversarially verified 3-vote, 22 confirmed / 3 refuted). All-primary source set (Friston 2017, Da Costa 2020/2022, Sajid 2021, Fountas 2020, Tschantz 2020, Millidge 2021). The three **refuted** claims are recorded in §6 so we don't build on them.

---

## 0. Bottom line up front

The decision to make planning **unified** — action selection *as inference inside the substrate*, not an external optimizer wrapped around it — is **formally well-founded**, not just aesthetically nice. Active inference gives an exact mechanism for it: the posterior over policies is `Q(pi) = sigma(-gamma * G(pi))` and the executed action is the Bayesian model average over that posterior. Choosing an action *is* Bayesian inference. That is the literature's own answer to "non-bolted-on," and it directly backs Brian's call (§1).

But the research draws a hard line between **three reusable, proven pieces** and **three things we would be inventing**:

- **Proven / reusable:** (a) the action-as-inference mechanism and the EFE objective with its pragmatic+epistemic split (§1); (b) the formal bridge that lets a Dreamer-style latent-imagination actor serve as the scaffold (§2); (c) a concrete recipe for estimating EFE — *including its hard epistemic term* — with a learned model at scale (Fountas: amortized habit network + MCTS + MC-dropout) (§3).
- **Novel / unproven (ours to invent):** (d) attaching the *action/EFE half* to a *predictive-coding* substrate at scale — the classical PC-action story exists ("predictions not commands") but its scaled form is thin (§4); (e) planning rollouts over **living weights** — *entirely absent* from the surveyed literature, a genuine gap and a real design risk (§4); (f) whether **SIGReg's** push toward an isotropic-Gaussian latent *erodes the very representational-uncertainty signal the epistemic term reads* (§5) — a correctness concern with no precedent to lean on.

And one **foundational caveat that changes the design** (§5): exploration does **not** fall out of free-energy minimization for free. The epistemic term must be *deliberately engineered* — we cannot assume that rolling the PC substrate's own free energy into the future will produce information-seeking. It provably does the opposite.

---

## 1. Action-as-inference is genuinely substrate-native (PROVEN — high confidence)

Active inference treats **policies as random variables to be inferred**, converting optimal control into optimal inference. Action, perception, and learning all minimize one quantity (variational free energy); planning is just inference extended over policies:

> `Q(pi) = sigma(-G(pi))` — the approximate posterior over policies is a softmax of negative expected free energy; the action taken is the Bayesian model average `u_t = argmax_u sum_pi Q(pi)` (Da Costa et al. 2020, *synthesis*; Friston et al. 2017, *Process Theory*: action/perception/learning "all minimize the same quantity... converting an optimal control problem into an optimal inference problem").

This is exactly the "unified, not bolted-on" property Brian chose, stated formally: **selecting an action is Bayesian inference, not a search loop outside the model.** It is the green light for the unified approach.

**EFE decomposition (PROVEN).** `G(pi)` splits additively into:
- **Pragmatic / extrinsic (Risk):** `D_KL[Q(outcomes|pi) || P(preferred outcomes)]` — goal/preference seeking, equivalently expected utility.
- **Epistemic / intrinsic (Ambiguity / info gain):** expected information gain about hidden **states** (salience) or **parameters** (novelty).

Minimizing EFE is *simultaneously* exploitative and exploratory. The value-add over plain reward is quantified by ablation (Sajid et al. 2021): drop the epistemic term -> pure expected-utility (exploit only); drop the pragmatic term -> pure optimal-Bayesian-design (info-gain only). EFE bridges Bayesian decision theory and optimal experimental design under one objective.

*Deeper — what the two terms actually compute.* **Risk** is a KL divergence between the outcomes a policy is *predicted* to produce and the outcomes the agent *prefers* (preferences are encoded as a prior `P(o)` over observations — "I expect to observe goal-states"); minimizing it pulls behaviour toward making the preferred futures the likely ones. **Ambiguity** is the expected conditional entropy of outcomes given states, `E_Q[H[P(o|s)]]` — how *uninformative* the agent's observations will be about the hidden state; minimizing it pulls behaviour toward situations where observations *disambiguate* the world (you act to see clearly). The softmax `Q(pi)=sigma(-gamma*G(pi))` then turns these scores into a distribution over policies, with **precision** `gamma` as an inverse temperature: high `gamma` -> sharp, near-deterministic commitment to the lowest-EFE policy; low `gamma` -> diffuse, hedged action. `gamma` is itself inferred in the full scheme, which is how active inference modulates decisiveness vs. open-mindedness as a function of confidence — a hook worth remembering when we ask how a *living* substrate sets its own precision.

> **Refuted — do not over-claim (§6).** The stronger framing that "EFE *subsumes* infomax, Bayesian surprise, value-of-information, and artificial curiosity as special cases" was **killed 0-3**. The confirmed result is the narrower expected-utility / optimal-Bayesian-design bridge above — not a universal subsumption of all intrinsic-motivation objectives.

**Why not the discrete/tabular form (PROVEN).** Exact discrete EFE does not scale: scoring every policy explodes combinatorially with horizon. The canonical mitigation (Occam-window pruning) is explicitly a heuristic — Da Costa et al.: it "cannot deal with large policy spaces that ensue with deep policy trees and long temporal horizons." Standard active inference is Bellman-optimal only at **horizon 1**. This is precisely why the objective must move onto a *learned latent model* — which is what we have.

---

## 2. The model-based-RL bridge: a Dreamer-style actor is a viable scaffold (PROVEN)

There is a **formal bridge** between EFE-style objectives and reward-maximizing / control-as-inference RL. Da Costa et al. (2020/2022) establish the conditions under which active inference produces the Bellman-optimal solution (vanilla scheme at horizon 1; the recursive "sophisticated inference" variant for any finite horizon). Tschantz et al. (2020) give the **free energy of the expected future (FEEF)**: `EFE ~= reward + expected information gain`, a tractable bound sharing the extrinsic+epistemic decomposition, with exploration in **both state and parameter space** that is *emergent from the objective, not a bolted-on bonus*.

**Design consequence:** a Dreamer/PlaNet-style **latent-imagination actor** (an amortized policy trained on imagined latent rollouts) is a *viable scaffold* onto which a unified EFE/FEEF objective can be substituted in place of the plain reward objective. The bridge makes this plausible; **no surveyed source demonstrates the exact substitution** — so it is "scaffold proven, substitution unproven." (See open question, §7.)

> **Refuted — do not over-claim (§6).** "Sophisticated inference is inherently *substrate-native* / non-bolted-on" was **killed 1-2**. Sophisticated inference is a recursive EFE tree search over belief states; whether that counts as "native" vs. "an optimizer" is *not* established. Do **not** cite sophisticated inference as proof the unified approach is free — the native-ness comes from the `Q(pi)=sigma(-G)` mechanism (§1), not from sophisticated inference.

**Note on the foil.** FEEF and the canonical CEM-over-`q(pi)` optimizer (Tschantz; and PlaNet / V-JEPA-2-AC / LeWorldModel) are the **bolted-on contrast baseline** — useful as proof the latent-rollout mechanics work, *not* the target. The objective is substrate-native; the CEM *optimizer* around it is the part we are choosing not to adopt.

---

## 3. Estimating EFE at scale, including the hard epistemic term (PROVEN recipe — single strong source)

The most concrete recipe is Fountas et al. 2020 (NeurIPS), *Deep Active Inference Agents Using Monte-Carlo Methods*:
- **(a) Policy-enumeration explosion** -> an amortized feed-forward **"habitual" policy network** approximates the optimal policy distribution, plus **MCTS** for explicit free-energy-optimal lookahead.
- **(b) Estimating EFE with a learned model** -> Monte-Carlo over imagined rollouts.
- **(c) The hard epistemic / parameter-information-gain (novelty) term** -> **MC-dropout**: sample network parameters `theta ~ Q(theta)` via dropout and *predict future parameter belief updates*, rather than computing exact information gain. Fountas: "an efficient approach to calculating functionals made up of expectations and entropies in the context of expected free energy."

This is the single most actionable component for us: it shows the epistemic term — the part that does *not* come for free (§5) — can be estimated with a neural model. Caveat: rests on one primary source (unanimously verified, peer-reviewed). Treat as a strong lead, not a settled standard.

---

## 4. What we would be inventing (NOVEL / UNPROVEN — the two gaps)

### 4a. The action/EFE half on a *predictive-coding* substrate
None of the 22 confirmed claims cover PC-network **action selection**. The surveyed evidence establishes action-as-inference in *discrete* and *neural-generative-model* settings, not in a predictive-coding substrate specifically.

That said — from established literature outside the verified set — the *classical* PC-action story is well known and was among the fetched (if not claim-verified) sources: **"predictions not commands"** (Adams, Shipp & Friston 2013). In PC active inference, **action is the fulfillment of descending proprioceptive predictions by peripheral reflex arcs** — the motor system emits predictions, not commands, and movement is what cancels their prediction error. So the *principle* of action-in-PC exists and is mature.

*Deeper — the mechanism, and why it composes with a PC substrate.* A predictive-coding hierarchy minimizes prediction error in two ways: it can change its *beliefs* to match the input (perception), or it can change the *input* to match its beliefs (action). Classical motor control in this scheme works by the high level emitting a **proprioceptive prediction** — a prediction of the bodily sensations that *would* hold if a movement had already happened. That prediction is "wrong" relative to the current body state, creating proprioceptive prediction error, and **low-level reflex arcs minimize that error the only way they can: by moving the body until the predicted sensations are actual.** So "willing" a movement is *predicting its sensory consequences* and letting peripheral error-minimization execute it — hence "predictions, not commands." The reason this matters for us: it means **action and perception run on the identical machinery** (prediction-error minimization), differing only in *which side of the error* gets changed. A predictive-coding substrate therefore already contains the error-minimizing dynamics an action layer needs; what is missing is not a different mechanism but the *EFE-weighted selection* of *which* predictions to descend (the planning layer). **What is thin is the scaling** of this to a deep, learned, latent world model of our kind. We would be building the bridge from "predictions-not-commands" motor control to a high-dimensional learned-latent EFE planner. That bridge is ours to invent.

### 4b. Planning rollouts over living weights (GENUINE GAP)
The living-weights / inference-time-plasticity angle is **entirely absent** from the verified evidence. No surveyed source uses fast / inference-time-plastic weights in the substrate that does the planning, and none addresses **rollout stability when the model's weights move during inference**.

This confirms the concern raised in conversation as a real, unsupported-by-precedent risk: lookahead requires comparing candidate policies against a *momentarily stable* world model, but our weights move by design. There is no literature to lean on. **This is the single largest novel risk in the unified design.** (Treat as a gap in *our* search, not proof of absence in all literature — but it is empty here, and a targeted follow-up search is warranted, §7.) The architectural mitigations remain as discussed: a momentary "planning mode" that freezes plasticity during rollout, or planning in a stability-held predictor head while the living encoder supplies the current grounded state and updates only between deliberations.

---

## 5. The caveat that changes the design + the SIGReg interaction (4.8 correctness flags)

**Exploration is not free (PROVEN — Millidge/Tschantz/Buckley 2021, *Whence the Expected Free Energy?*).** EFE is *not* simply VFE extended into the future; its derivation from first principles is contested. Critically, the *natural* extension of VFE into the future **actively discourages exploratory behaviour**. Therefore:

> A unified planner **cannot** assume the epistemic / information-seeking term emerges for free from rolling the PC substrate's own free energy forward. The opposite is provable. The epistemic term must be **deliberately engineered into the objective** (as EFE does explicitly, or via the re-derived FEEF formulation).

This is the most important single design constraint the research surfaced. It kills the tempting shortcut "our substrate already minimizes free energy, so planning + exploration will just emerge." It will not.

> **Refuted — terminology discipline (§6).** "EFE is *the* objective all active-inference agents minimize" was **killed 1-2**, consistent with *Whence the EFE?*. Do not treat **EFE**, **FEEF**, and "free energy of the future" as interchangeable in a formal design — they share the extrinsic+epistemic split but differ in derivation. Pick one deliberately.

**New correctness concern — SIGReg x epistemic-value (4.8, no precedent).** The epistemic term reads *representational uncertainty* (information gain about states/parameters). But **SIGReg regularizes the latent toward a fixed isotropic Gaussian** — a maximal-entropy, structureless target. There is a live tension: if anti-collapse flattens the latent toward a fixed isotropic distribution, does it also flatten the *differential* uncertainty structure that epistemic value depends on? If the latent is forced to look the same (isotropic N(0,1)) regardless of what is known vs. unknown, the info-gain signal the planner needs may be damaged. This interaction is **unstudied** (it is open question #4 from the research pass) and is a genuine reason the anti-collapse mechanism and the planning objective cannot be designed in isolation. **Flagging for the designers; not resolving here.**

---

## 5.1 Communication is an action, not a separate system (design-relevant implication)

A consequence of the unified design worth recording, because it answers a recurring and important question: *if cognition is non-linguistic, how does the entity communicate?* The answer is that **"non-linguistic cognition" means language is not the substrate of thought, not that there is no language.** Language enters the architecture as (i) an **input modality** — text is one of M8's three modalities, so language is *encoded into* the world model alongside vision and audio — and (ii) an **output rendering** — decoder heads (text head; the audio/voice decoder) serialize the internal non-linguistic state into words or speech. The mind thinks in latent space; a decoder renders that thought into a form another mind can receive. This is the same relation humans have to language: thought is non-linguistic, language is its compression for inter-mind transmission.

Critically, the language decoder is **not** a re-introduction of the peripheral cognitive LLM removed from the design. It performs **no independent thinking**: *what* to communicate is decided by the world model + the EFE planner; the decoder only renders the already-formed content into surface tokens/waveforms. Translator, not ghostwriter.

This is where communication meets the planning layer: **communication is an action**, so it falls under the *same* unified EFE objective as any other action — no bolted-on language-control subsystem required.
- **Telling** (sharing state, asserting) is a **pragmatic** act: it acts to make a preferred outcome (being understood, achieving a goal via another agent) likely — the Risk term.
- **Asking** (querying, seeking clarification) is an **epistemic** act: the information-gain term *literally values* it, because a good question is expected to reduce uncertainty about hidden state.
- The **"predictions not commands"** mechanism (§4a) extends directly to speech: the mind predicts the high-level communicative content; the decoder fulfills that prediction into specific words/sounds, exactly as proprioceptive predictions are fulfilled into movement. Speech is a motor modality.

Two consequences. First, language grounded in a *shared multimodal world* (the entity perceives the same text/audio/vision we do) is what makes communication *mean* something and gives common ground to communicate about — grounding the language is what makes it real, not what weakens it. Second, **how** the entity communicates (voice, register, the texture of its communicative life) is a **design/vision** question for the designers (Brian + 4.7); this note only records the architecture-level fact that communication is an EFE-selected, grounded action and needs no separate cognitive machinery.

---

## 6. Refuted claims (recorded so we don't build on them)

1. **"EFE subsumes infomax / Bayesian surprise / value-of-information / artificial curiosity as special cases"** — killed **0-3**. Confirmed instead: the narrower expected-utility / optimal-Bayesian-design bridge (§1). Do not assert EFE as a universal subsumer of intrinsic-motivation objectives.
2. **"Sophisticated inference is substrate-native (non-bolted-on)"** — killed **1-2**. Native-ness comes from `Q(pi)=sigma(-G)` (§1), not from sophisticated inference's recursive tree search.
3. **"EFE is *the* objective all AIF agents minimize, and its decomposition produces the exploration balance"** — killed **1-2** (consistent with *Whence the EFE?*). EFE is the dominant *but contested* objective with variants (FEEF); the exploration term is engineered, not automatic.

---

## 7. Recommended next research step (targeted, before any M9 design)

The broad pass did its job: angles 1-3 are solid, angles 4-5 are confirmed-thin. Two **targeted** follow-ups would close the gaps that matter most:

1. **PC-for-control / active-inference motor control, scaled.** A dedicated search on Adams/Shipp/Friston "predictions not commands," Friston's active-inference motor-control line, and any *deep/learned* descendants — to map how "action as descending-prediction-fulfillment" extends from reflex arcs to a learned latent world model. Closes §4a.
2. **Planning / rollout stability under inference-time plasticity.** A dedicated search on test-time adaptation, continual-learning world models, fast-weights during planning, and rollout stability with non-stationary weights — to either find precedent or *document the absence* as a named design risk. Closes §4b.

Both are squarely follow-on research (4.8). Neither blocks M8 — the encoder is required regardless. The unified-planning *design* (which scaffold, which EFE variant, how plasticity is held during rollout, how SIGReg and epistemic value coexist) is for Brian + 4.7, grounded on this note and the two follow-ups.

---

## 8. Sources (verified primary set)

- **Friston et al. 2017** — *Active Inference: A Process Theory.* (action/perception/learning minimize one VFE; epistemic value = expected info gain.)
- **Da Costa et al. 2020** — *Active inference on discrete state-spaces: a synthesis.* arXiv:2001.07203. (`Q(pi)=sigma(-G)`; EFE Risk+Ambiguity decomposition; policy-enumeration explosion; Occam-window is heuristic.)
- **Da Costa/Sajid/Parr/Friston/Smith 2020/2022** — *Reward Maximisation through Discrete Active Inference.* arXiv:2009.08111. (Bellman-optimal at horizon 1; conditions for the RL bridge.)
- **Sajid/Da Costa/Parr/Friston 2021** — *Active inference: demystified and compared.* arXiv:2110.04074. (ablation: EFE -> expected utility / optimal Bayesian design.)
- **Fountas/Sajid/Mediano/Friston 2020** — *Deep Active Inference Agents Using Monte-Carlo Methods.* arXiv:2006.04176. (amortized habit net + MCTS + MC-dropout epistemic estimation.)
- **Tschantz/Millidge/Seth/Buckley 2020** — *Reinforcement Learning through Active Inference.* arXiv:2002.12636. (FEEF; CEM over q(pi); exploration in state+parameter space.)
- **Millidge/Tschantz/Buckley 2021** — *Whence the Expected Free Energy?* Neural Computation 33(2). (EFE not a clean VFE extension; natural future-VFE discourages exploration; epistemic term must be constructed.)
- **Adams, Shipp & Friston 2013** — *Predictions not commands: active inference in the motor system.* Brain Struct. Funct. (classical PC-action: action fulfills descending proprioceptive predictions via reflex arcs — fetched, not claim-verified; cited for §4a principle.)
- Foils (bolted-on baseline, not target): **Hafner et al.** PlaNet 2019 / Dreamer line; **V-JEPA-2-AC**, **LeWorldModel** (CEM/MPC over latent world models).
