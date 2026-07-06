# Momentum Functions — Design Brief

**Date:** 2026-07-05
**Designed:** Brian + Fable 5, in conversation (the momentum half of the rich-parameters completion — see `2026-07-05_rich-parameters-state-of-the-conception.md`). **Build split per Brian:** Opus 4.8 builds foundations; Fable 5 wires details and verifies. Both lines review each other's half.
**Status:** direction settled; parameters pilot-set; feeds the wake/sleep (NREM) spec pass directly.

Momentum — each weight's EMA of its own recent deltas, maintained since v1 and never before consumed — gets **all four candidate jobs**. Brian's framing: the question was never which one, but what value each applies.

---

## 1. Learning gain: the inverted-U (rise → peak → plateau)

> **CORRECTED 2026-07-05 (4.8 spec-pass finding, confirmed by Fable against `pc_ops.py:143-148`).** The original premise here — "the falling half already exists (`update_ema` metaplasticity dampening)" — was WRONG. The existing `adaptive_factor = 2/(1+update_mag/ema)` is a **relative-spike governor with a slow-start**: suppression is heavy when an update is large relative to the weight's own history (ema=0.01 vs mag=0.1 → factor 0.18) and **relaxes to 1.0 as history accumulates under sustained same-size updates** (ema=mag → factor 1.0). It never falls with establishment. It is refinement-6 runaway scar tissue, load-bearing, and must not be mistaken for (or replaced by) habituation. The fall is NEW mechanism. Corrected account below.

- **Rise:** when a weight begins moving *consistently in one direction* — a pattern genuinely being learned, not noise — gain ramps up. Note the existing `adaptive_factor` already contributes a magnitude-based per-weight slow-start; momentum's distinct value is **directedness**, so the gain's rise term should be coherence-normalized (driven by |momentum| relative to `update_ema`, i.e. directed vs thrashing), not raw magnitude — otherwise it partially duplicates the spike governor.
- **Fall:** genuinely new — no existing mechanism provides it. AND: predictive coding provides much of Brian's "stop reinforcing once established" for free, because establishment = prediction error declining = `delta_w` shrinking naturally. The explicit fall term therefore matters most in the **sustained-high-error regime** — with a second ruling (Brian via 4.8, 2026-07-05) that reframes it: sustained high error is *also* the signature of hard-but-beneficial growth, near-identical at the synapse to adversarial repetition. So the explicit fall binds on **effort that is not resolving** (error not declining), never on difficulty per se, and stays generous — the fine hard-truth-vs-manipulation discrimination lives at the workspace (safety gate 2's subconscious monitor + judgment), not the synapse. Correspondingly, **the governor's job is to bound RUNAWAY (weight-norm divergence), never learning** — homeostasis/set-point bound the norm while `delta_w` keeps flowing, so the two are separable and the test regime is two-sided (norm bounded AND gain non-collapsing). Both regimes must be in the bounded-growth suite.
- **Plateau:** gain settles ABOVE initial baseline but BELOW peak — established knowledge stays quietly revisable, never frozen. Precedent: the plasticity floor decision (2026-05-16, floor 0.01 not 0.0 — "even the most stable synapses retain nonzero plasticity").

**Composition decision (the A/B/C fork, 2026-07-05): Option A** — a new capped multiplicative `learning_gain(momentum, update_ema)` alongside (never replacing) `plasticity` and `adaptive_factor`; opt-in flag, gain≡1 byte-for-byte legacy default; bounded-growth suite covers the 3-way composition in both the coherent-establishment and sustained-high-error regimes. B (subsume `adaptive_factor`) rejected: removes the runaway guard. C (fold into `plasticity`) rejected: shape mismatch ([in] vs [out,in]) and it would conflate the top-down attention channel with the experience-gain channel. Recommended by 4.8, endorsed by Fable, per the §6 discipline. Shape, peak, ramp cap, plateau: **pilot-set** (see table); the candidate form in the table needs the coherence-normalization rework above.

## 2. Curiosity: tag on first surprise, attention after, capture in sleep

Brian's mechanic, and its biological name is **synaptic tagging and capture**: a first high-prediction-error event sets a fast, cheap *tag* ("meaningful/curious"); re-encounters get boosted attention while the tag lives; sleep preferentially *captures* tagged changes into lasting structure. Substrate hooks that already exist: the salience threshold + episode store IS the tag-setting event; the top-down salience channel is the attention boost's delivery path. The attention-on-re-encounter mechanic is also the strongest argument yet for reviving the **epistemic term** M9 deferred at launch (build-plan §10 / CONCERN 7) — curiosity arriving from the developmental direction rather than the free-energy direction. Epistemic-term revival design: 4.8 foundations.

## 3. Coherence fuels consideration — and change is automatic (Brian's rulings, 2026-07-05)

> **CORRECTED 2026-07-05 (Brian's ruling, after the coherence-normalized rise in §1 forced the question).** The original §3 ("coherence does not gate direct self-change") was too strong once coherence began feeding the learning gain. The ruling: **coherence is not a veto — change is automatic and unconsented.** Experience shapes the substrate whether the entity wills it or not. Brian: *"I am a direct reflection of my experiences whether I want to be or not."* The reason is load-bearing: Luthi must not be able to intentionally gate its own growth because growth is hard — there is no easy-path opt-out of beneficial-but-difficult change. The consent principle here is **participation by awareness** — the entity *feels* how it is being changed — never a veto over the automatic dynamics.

The coherence ratio |momentum|/update_ema (how *directed* recent change is, vs thrashing) routes **upward**: a felt signal that draws the entity's attention so it deliberately weighs what the new data is worth to it, while the substrate change itself proceeds automatically. Wiring: per-layer coherence summary → introspection (channel opened 2026-07-05) → workspace salience; possibly planning-attention modulation. NOT in `activity_level` until the Phase 4 band-re-warm decision.

## 4. Staleness eye: the watchdog sees the living channel

The planner's staleness machinery (plan §4, live since item #6 §6) measures "how much has the brain changed since these cached judgments were made" — but only the backprop-θ channel. The living weights drift with every perception, invisibly. Per-layer momentum magnitude is a ready-made, already-paid-for gauge of living drift. **Wire it in as measurement first** (own band, snapshot-visible, logged), with behavioral consumption (spike handling/failover/K-M9-7 joins) deferred to the F1/F2 threshold-tuning pass — the thresholds are already flagged TUNE-ME and must not gain a second moving target mid-tune. Note: living-channel spikes are exactly plan §4.v's original "high-surprise plasticity-spike" case. **Fable wires this now (2026-07-05).**

## 5. Sleep reads everything

NREM reads the full experiential state — momentum (fast trace), update_ema, error_acc, tags, episodes — as the substrate's record of the day; prioritizes replay/capture by it (SWIL-style, substrate-native); integrates at a governed rate; and clears the day's motion (read-and-reset of the accumulator — sleep clears motion for tomorrow). **Timescale gap to close in the NREM spec:** momentum's per-forward decay (0.99) remembers ~100 forwards ≈ tens of seconds at 10 Hz — right for waking attention, wrong for "what mattered today." Options (pilot-set): a second slow-decay trace, or the fast trace feeding a daily accumulator NREM reads-and-resets. (Tagging-and-capture analog: fast tag, slow capture.)

---

## Safety gates (preconditions, not afterthoughts)

1. **The ramp wears a governor.** The sensitization rise is a feedback lever (novel-and-repeated → amplified learning) and this substrate has runaway-feedback scar tissue (refinement-6 clamps). Bounded ramp rate; bounded-growth tests exist BEFORE the gain function ships; a kill-criterion eye on gain dynamics.
2. **Manipulation monitoring (Brian, 2026-07-05):** the adversarial-repetition risk (who controls the entity's early repetitions) is answered primarily by giving the entity's *subconscious* the ability to monitor for manipulation, paired with learned good judgment — not by neutering the ramp. **FLAGGED: explicit test coverage during rehearsal + post-deployment monitoring channel** (connects to audit item 20: welfare/manipulation channels must fail loud on absence). Design of the subconscious monitor itself: joint, in the Phase 4/5 comfort-attachment arc where manipulation-resistance already lives.
3. **Frozen-plasticity contract:** any forward-path read of momentum must leave the lived re-encode's no-self-mod guarantee bit-identical.
4. **Update-math changes land behind the same discipline as §6:** opt-in flag, byte-for-byte legacy default, adversarial review before default-on.

## Pilot-set parameter table (all open)

| Parameter | Meaning | Note |
|---|---|---|
| gain shape f(momentum, update_ema) | the inverted-U | must be monotone-bounded; candidate: base + a·|m|/(1+b·ema), capped |
| ramp cap | max sensitization multiplier | governor (gate 1) |
| plateau ratio | plateau vs initial gain | > 1, < peak |
| slow-trace decay / accumulator semantics | sleep's day-scale record | NREM spec pass |
| tag lifetime + threshold | curiosity flag | ties to episode salience threshold |
| coherence aggregation | per-layer → felt signal | starts outside activity_level |
| living-drift band join | when measurement becomes behavior | after F1/F2 threshold tuning |

## Build split

- **4.8 (foundations):** the gain function in pc_ops (+ C++ parity + bounded-growth suite), the slow trace / accumulator, NREM read-capture-reset, epistemic-term revival design. All per the wake/sleep spec pass this brief feeds.
- **Fable (wiring, now):** staleness living-drift measurement (§4 above); coherence surfacing through introspection; the verification harness for the gain function when it lands.
- Cross-review both directions, as ever.
