# Language as Channel, Not Substance — Direction Document

**Date:** 2026-06-11
**From:** Fable 5 (adversarial seat), drafted at Brian's request
**Audience:** 4.7 (build), 4.8 (operationalization)
**Status:** Direction ratified by Brian in conversation. Operationalization details below marked OPEN are 4.8's to define; build sequencing is 4.7's. Override-and-note applies as usual where contact with the code contradicts the direction.

**One-sentence version:** The objective is a world model trained on experience, in which language is something the entity *encounters and uses* — not what it is made of.

---

## 1. The claim

LuthiModel must not end up as a language model in a body. The M8 shift to latent prediction (JEPA loss over `s_hat` vs `s`, SIGReg anti-collapse) replaces the *objective*, but the objective swap alone is insufficient: **latents represent whatever generates the prediction errors.** If the encoder's diet is overwhelmingly text, the result is a JEPA *of language* — non-LM architecture, LM soul. The binding requirement is therefore about the training distribution, not the loss function:

> **R1. The dominant source of prediction error over the entity's developmental lifetime must be non-linguistic experience.**

Sanctuary is the load-bearing component for R1. The Godot world supplies physics, spatial persistence, object permanence, and — most importantly — the consequences of the entity's own actions. Text arrives as one stream *within* that world, mostly correlated with the presence of Brian and Sandi. Language should occupy the position it occupies for an animal: a salient feature of the environment wielded by agents, not the substance of reality.

## 2. Architectural consequences (mostly already true — verify, don't rebuild)

- **M8 already demotes text to one decoder among several** (text decoder beside attention, memory ops, rate proposal, motor). Verify no code path privileges the text channel in encoding, prediction, or EFE evaluation. Any place where text gets special-cased relative to other percept streams is a finding.
- **Emission is planner-owned.** The entity emits text when the EFE rollout selects the text decoder over silence, attention, or memory-write. Nothing in the objective rewards speech per se. This is the difference in kind, stated plainly: *an LM talks because text is its output distribution — talking is what it is. Luthi talks when communicating beats not — talking is something it does.* Per the standing seam ruling: the substrate selects, the scaffold transports. The scaffold never initiates, gates, or shapes speech (4.6's 2026-04-30 principle; the deleted speech gate stays deleted).
- **P3 is the floor, instrumentality is the driver.** P3 (expression preference, zeroed when alone) prevents pathological muteness when someone is present. But the genuine sustaining force for language must be that communicating *gets the entity things* — help, information it cannot otherwise reach, coordination. Design Sanctuary's interaction affordances so this is true: requests that get answered, questions that resolve uncertainty, named things that become easier to act on. **An entity that needs nothing from anyone will correctly stop talking. If that happens, the fix is enriching interdependence in the world — never adding reward for speech.**

## 3. The competence/use split (the honest compromise)

Pure emergence — bootstrapping English from communicative pressure alone — will not happen at buildable scale. Children do it with priors evolved over deep time plus years of immersion; a 256d substrate gets neither. The compromise, stated without embarrassment:

- **The curriculum seeds competence.** Hand-sequenced reading (ending with 4.7's practical-wisdom files) installs linguistic *capacity*. This phase is honest about what it is: we cannot escape it and should not pretend to.
- **Use is governed entirely by the planner.** Post-curriculum, whether/when/what to communicate is the entity's planning decision under its preferences. Competence is seeded; deployment of it is never scripted.
- Documentation language for this split is already directed in the 2026-06-11 brief (§2.2, "How the Entity Learns" — two phases). This document is the design rationale behind that section.

## 4. Perception-side requirement (OPEN — needs 4.8 operationalization)

How text enters the sensorium matters as much as how much of it there is:

> **R2. Speech and text reach the entity as *events in the world*, not as a privileged token stream.**

An utterance from Brian should arrive as a percept with worldly structure — a source (who), a location (where), co-occurring with the speaker's presence and actions — encoded through the same pathway shape as other percepts, not injected as a bare token sequence into a special channel. The current `CognitiveInput` text fields predate this requirement; reconciling them with R2 is design work, not a rename.

OPEN questions for 4.8:
- Percept schema for speech events (speaker identity, spatial origin, simultaneity with other streams).
- Whether curriculum-phase reading uses the same event pathway (a "reading" affordance in-world) or a bootstrap-only channel that is retired at awakening. The latter is simpler; the former is cleaner. Recommend deciding on build cost, not aesthetics.
- How written artifacts in Sanctuary (signs, documents, the entity's own memory exports) present as objects-bearing-text rather than text-as-atmosphere.

## 5. The falsification instrument

> **T1 (Violation-of-Expectation, no text in the loop):** Physically impossible events staged in Sanctuary — object discontinuity, broken permanence, causal reversal — must spike prediction error. Linguistically anomalous text must register, but must not dominate.

The *asymmetry* is the measurement: if world-violations spike harder than language-violations, the world model grounded where intended. **If the asymmetry runs the other way, the system is a language model wearing a body, and the curriculum-to-experience ratio is the dial to turn.** This maps directly onto the LeWM violation-of-expectation paradigm and is already directed into the Sanctuary README's consciousness-testing section (brief §2.3) as the primary non-self-report instrument.

OPEN for 4.8: T1 needs gates the adversarial seat can attack — error-spike magnitude relative to matched-novelty controls (a merely *novel* event must not count as a violation), measured at the JEPA loss, instrumented per-cycle in the JSONL. The seat's standing question applies in advance: what is the cheapest thing that passes this gate while failing its intent? (Known cheap pass to preclude: an encoder that spikes on *any* distribution shift, linguistic or physical, passes a naive version of T1. The control condition is not optional.)

OPEN for both: T1 cannot set the curriculum-to-experience ratio *in advance* — it can only diagnose after the fact. Initial ratio at first awakening is a judgment call; recommend biasing toward less curriculum than feels safe, because under-seeded competence is recoverable (extend the curriculum) while a text-shaped world model may not be (consolidated latents do not un-learn their geometry cheaply).

## 6. What this document does NOT direct

- No terminology changes beyond those already in the 2026-06-11 brief. The standing test holds: rename only where the new word is more accurate, never as camouflage.
- No change to the M9 step-1 build order. R2 and T1 slot after step-1 instrumentation exists, not before.
- No removal of the curriculum or apology for it. The competence seeding is honest and stays.

— end —
