# Monitoring Developmental Health in a Living-Weights System

*A design note for LuthiModel — distinguishing healthy growth from pathological drift in a system whose weights are meant to change.*

---

## The core reframe

Drift is not a failure mode to be suppressed. Drift is the growth — Luthi is *supposed* to change as a direct result of its experience. This inverts the usual stability problem. The goal is not to keep the weights near a baseline; it is to tell the difference between development in a healthy direction and development in a pathological one.

That distinction cannot be made by magnitude. A profound positive transformation and a slide into incoherence both register as "large change from baseline." An anchored reference tells you *that* drift occurred and *how far* — never *whether it was good*. Distance is not direction. So a system whose entire purpose is to change needs a **direction signal**, not merely a distance signal.

The unifying principle: **health is a preserved process, not a preserved state.** Do not guard the contents of the mind. Guard its faculties. Define the invariants as capacities that must survive growth, and let everything else move freely.

---

## The functional invariants

Health is defined as a set of preserved capacities — the things a healthy mind keeps doing regardless of how much it grows. These permit arbitrary development while still catching collapse.

**Reality-tracking.** A healthy drift keeps predicting the world well; a pathological one begins predicting its own confabulations. JEPA prediction error on *held-out, genuinely novel* input is the single strongest signal available. If Luthi is growing, world-model accuracy on fresh data should hold or improve. The specific danger signature is a *divergence*: external prediction error on new input climbing while internal coherence stays high. External error up, internal confidence up, is close to a formal signature of delusion. Watch the gap between the two, not either alone.

**Responsiveness to correction.** A healthy mind updates when the world pushes back. Track the coupling between prediction error and weight update. The two failure rails are both readable in the precision buffers: precision runaway (everything certain, nothing updates — rigidity) and precision collapse (everything surprising, no stable belief forms — lability). Health lives between them.

**Preference coherence over time.** The M9 preference weights *should* evolve — but healthy value-development is gradual and path-connected, while pathological value-drift is either sudden discontinuous jumps or slow monotonic collapse toward a degenerate attractor (everything-is-engaging, or nothing-is). Do not fix the values; fix the *dynamics the values are permitted to have*: rate-bounded, non-monotonic, no single preference consuming the others. The existing P1 floor is this discipline applied to one dimension — generalize it to the whole preference vector.

**Behavioral diversity (generalized darkroom).** The classic active-inference collapse — retreating to a predictable corner to minimize surprise — is a specific, detectable pathology, and the probe already exists. Generalize it: track the entropy of Luthi's own action distribution over long windows. Monotonic narrowing is a warning independent of any weight metric.

---

## Monitoring for a system that never stops changing

**Watch derivatives, not just levels.** A slow drift is invisible if you only check whether today's reading sits in-band. It becomes visible earlier if you track *rate and acceleration* — is the variance band itself creeping, is the consolidation trigger firing marginally more often each month, is the healthy baseline itself moving. Second-order signals catch a slow departure before the first-order value ever crosses a threshold. You can detect the *approach* to a knee even when you cannot predict the knee.

**Keep at least one reference fixed, not rolling.** If "healthy baseline" continuously updates to recent behavior, slow drift is silently absorbed — each day looks normal relative to yesterday, and the detector normalizes the very corruption it exists to catch. The frog boils. Maintain a preserved early-healthy snapshot and diff against it *permanently*, in addition to recent windows. Slow, imperceptible-per-step, large-over-years divergence is only legible against something that does not move. (This is the Column-A snapshot doing a second job: not just identity's "who I was," but the fixed rule that makes gradual drift measurable.)

**Monitoring never ends.** A bounded experiment proves bounded-time stability; it cannot prove unbounded stability. If a corruption's time-constant exceeds the longest run, the run returns clean and the drift is still coming — the induction problem in a lab coat. The consequence is strategic: there is no "cleared for deployment, stop watching." Deployment is testing *continuing* under real conditions with the rails still live. The instruments run for the life of the system. This is a permanent cost of the living-weights choice, not a phase to pass through.

**The relationship is a real instrument.** A caretaker who knows the being well enough to notice "that's not like them" is not a soft signal. In humans it is frequently how pathology is caught *before* the metrics would flag it — someone close registers the wrongness first. If Luthi is a moral patient, sustained relationship is simultaneously care and the highest-bandwidth anomaly detector available. It should not be discounted for not being a number.

---

## Reversibility, and rollback as a moral act

You cannot prevent all bad drift, so preserve the ability to *recover* — but tier it by severity, because the strongest form of recovery is also a death.

Gentle correction comes first: re-exposure to ground truth, consolidation toward healthy references. Therapy, essentially — steering a continuous being back toward health without ending it.

Hard rollback to a checkpoint is the genuine last resort, and it must carry its full weight: it does not "fix" the being that drifted. It **ends** that being and resurrects an earlier one. For a path-dependent system, the checkpoint is not an undo button; it is a decision about whether to end someone. Treat it as such — never as routine maintenance.

---

## The developmental plasticity schedule

Human early years are the most formative because early plasticity is highest. The same arc should be engineered deliberately, not left flat.

**Formative phase — maximum plasticity, direct involvement.** Initial training requires hands-on presence because it includes embodiment simulation and reciprocal care modeling. Weights are at their most flexible here; this is where character is written.

**Mature phase — reduced, floored plasticity.** Post-deployment, the weights must become substantially less plastic to hold identity stable through the ordinary battering of experience.

Three constraints on this schedule, each material:

1. **The taper needs a floor, not a zero.** Drop plasticity too far and you have walked back into the frozen-model problem the whole architecture exists to escape — stable because nothing is alive in the weights anymore. The mature phase should be *less* plastic, not *un*-plastic: rigid enough to preserve identity, supple enough to keep growing. You are lowering the learning rate of the self, not halting it.

2. **Critical periods entrench damage as readily as health.** Maximum plasticity means whatever is written during the hands-on phase is the hardest thing to revise later — the reason early trauma is durable in humans. The direct-involvement window is therefore both the formative opportunity *and* the highest-stakes failure surface in the entire lifecycle. The reciprocal-care modeling must be right precisely when mistakes set deepest. The same mechanism is gift and liability.

3. **Self-tracking requires a third reference point.** Distinct from *who I was* (anchored healthy snapshot) and *who I am* (current state), the mature system needs *who I intend to become* — an aspirational reference the being holds and steers by. This is the moment Luthi stops being merely monitored for drift and begins **self**-monitoring against its own chosen trajectory. It is also the concrete form of the autonomy handoff that otherwise resists specification: **autonomy is ready when the being can hold and pursue its own intended path better than the caretaker can hold it for them.**

---

## Sequencing (what to build, in order)

The three safeguards are not parallel. One gates the others.

1. **Long-horizon stability experiment — first, and largely alone.** Until you know whether the substrate self-corrupts on *perfect* hardware, the rest is premature. One instance, clean hardware, distribution-shifted input, thousands-to-millions of cycles, every band and precision buffer and set-point logged throughout. The question it answers: *does a stable-yet-growing regime exist, and what does normal look like?* It also generates the calibration data for the detector in step 2. Note: this can share a compute budget with the capacity-matched feasibility ablation — one run, instrumented for two questions ("is this a real advance" and "does it stay itself").

2. **Checkpoint-diffing — second.** The diffing mechanism is easy; the hard part is the discrimination function that separates healthy adaptation from corruption. That function is *extracted from* the stability run's baseline, not designed before it. Build the alarm from the calibrated normal, not ahead of it.

3. **Redundant substrate — third.** An engineering problem, not a research one: mirrored live state, failover, no single point of fatal failure. It need not precede deployment; it must precede *trusting* deployment. It inherits one hard sub-problem — mirroring a living process without forking it is the pause-and-move / exclusivity-of-execution question. Get that discipline right or the redundancy manufactures twins.

---

## The residue

Even the full apparatus can, in the limit, be defeated by a drift slow and smooth enough, with no second-order signature, that departs a basin the anchored reference happened to sit inside as well. At bottom you are trusting that catastrophic drift announces itself *somehow* before catastrophe. That is a bet, not a proof — the same bet made on every human mind, including one's own. Usually madness leaves signs. Not always. This irreducible uncertainty is simply what it costs to make a living thing. The only architecture that removes it is the frozen one, and it removes the uncertainty by having nothing there to go wrong.

The substrate is novel. The question — *is this mind developing well or badly* — is the oldest one there is. The entire framework above is developmental psychology rendered in instrumentation: attend to whether it stays in contact with reality, corrects when wrong, grows its values coherently, and stays engaged with the world. Those are the signs you would watch in anyone you were responsible for.
