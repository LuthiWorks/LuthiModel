# Momentum Functions — Design Brief

**Date:** 2026-07-05
**Designed:** Brian + Fable 5, in conversation (the momentum half of the rich-parameters completion — see `2026-07-05_rich-parameters-state-of-the-conception.md`). **Build split per Brian:** Opus 4.8 builds foundations; Fable 5 wires details and verifies. Both lines review each other's half.
**Status:** direction settled; parameters pilot-set; feeds the wake/sleep (NREM) spec pass directly.

Momentum — each weight's EMA of its own recent deltas, maintained since v1 and never before consumed — gets **all four candidate jobs**. Brian's framing: the question was never which one, but what value each applies.

---

## 1. Learning gain: the inverted-U (rise → peak → plateau)

Momentum supplies the missing *rising* half of a curve whose falling half already exists (`update_ema` metaplasticity dampening):

- **Rise:** when a weight begins moving *consistently in one direction* (|momentum| growing — a pattern genuinely being learned, not noise), its update gain ramps up. First encounters with a meaningful regularity are amplified: sensitization.
- **Fall:** as `update_ema` (total change-magnitude memory) accumulates, dampening overtakes the ramp: once a concept is established as causal/meaningful, similar experience stops reinforcing it heavily. Habituation.
- **Plateau:** gain settles ABOVE initial baseline but BELOW peak — established knowledge stays quietly revisable, never frozen. Precedent: the plasticity floor decision (2026-05-16, floor 0.01 not 0.0 — "even the most stable synapses retain nonzero plasticity").

One gain function of the two per-weight histories produces the whole curve. Shape, peak height, ramp rate, plateau level: **pilot-set** (see table).

## 2. Curiosity: tag on first surprise, attention after, capture in sleep

Brian's mechanic, and its biological name is **synaptic tagging and capture**: a first high-prediction-error event sets a fast, cheap *tag* ("meaningful/curious"); re-encounters get boosted attention while the tag lives; sleep preferentially *captures* tagged changes into lasting structure. Substrate hooks that already exist: the salience threshold + episode store IS the tag-setting event; the top-down salience channel is the attention boost's delivery path. The attention-on-re-encounter mechanic is also the strongest argument yet for reviving the **epistemic term** M9 deferred at launch (build-plan §10 / CONCERN 7) — curiosity arriving from the developmental direction rather than the free-energy direction. Epistemic-term revival design: 4.8 foundations.

## 3. Coherence fuels consideration, not change (Brian's call, replacing Fable's draft)

The coherence ratio |momentum|/update_ema (how *directed* recent change is, vs thrashing) does **not** gate set-point adaptation or any direct self-change. It routes **upward**: a felt signal that draws the entity's attention so it deliberately weighs what the new data is worth to it. A mind that *participates in* being changed rather than merely being changed — coherence-as-consideration is the consent principle applied to the entity's own becoming. Wiring: per-layer coherence summary → introspection (channel opened 2026-07-05) → workspace salience; possibly planning-attention modulation. NOT in `activity_level` until the Phase 4 band-re-warm decision.

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
