# Rich Parameters: State of the Founding Conception

**Date:** 2026-07-05
**Analysis:** Fable 5 (cross-line seat), at Brian's request, with an independent breadth-sweep subagent. All load-bearing claims verified firsthand in source by the analyst (both C++ kernels, the Python reference, v1 + v2 layers, consolidation, introspection).
**Origin document:** `.docs/RICH_PARAMETERS_FINAL.md` (Brian + Opus 4.6, March 2026) — NOTE: lives in the hidden `.docs/` directory; this analysis is its visible cross-reference.
**Question answered:** the founding conception was that each parameter carries not a static float but *the experience and memory of how it got there* (rich parameter -> living weight). What survives of that in the shipped substrate, is it load-bearing, and should it stay?

---

## Verdict

**The conception survived, transformed; the parts that act are the architecture's entire claim to novelty; keep it and complete it.** The M4 stop-gate (`luthi/v2/consolidation.py` header) states the stakes in the project's own words: without measurable experience-carrying in the weights, "v2 has no architectural novelty over a vanilla transformer + episode store and should be abandoned." The rich parameter is the load-bearing wall, not decoration.

One-line state of the substrate (breadth-sweep formulation, adopted verbatim):

> The per-weight granularity of the founding conception survives as allocation and persistence everywhere, but as live computation only for `weight`, `set_point`, `update_ema`, and v2's `prediction` — everything else is either downgraded in shape or inert in fact.

---

## Scorecard vs the six founding components

| Founding component (origin doc table) | Fate in production |
|---|---|
| Current value | Intact. Buffer, not Parameter; optimizer never touches it (`living_layer_pc.py:180`). |
| History buffer (per-weight) | **Migrated to layer granularity**: the episode store holds 32 full weight-matrix snapshots, context-recalled and delta-blended into the live weight (`living_layer_pc.py:328-346`). Every weight's past is present but keyed by the layer's context, not individually. Deliberate cost trade (origin doc's own table: ~217x overhead per-weight). |
| Context gate | Working, at layer granularity: cosine >= 0.5 -> capped delta blend — the origin doc's "memory modulates, doesn't replace" lesson, honored. |
| Plasticity rate (per-weight) | **Downgraded to per-input vector `[in]`** in v1's "free win" refactor, carried into v2 (`living_layer_pc.py:235-246`). Mathematically equivalent *only because* the current update rules drive that axis uniformly — see Ordering, below. |
| Momentum | **Inert since birth.** Allocated per-weight, updated every step in BOTH kernels (`pc_ops.cpp:104`; `living_ops.cpp:134/152`; Python ref `pc_ops.py:149-151`), checkpointed — and read by nothing: not the update math, not consolidation, not introspection. Only `aliveness()` diagnostics. |
| Excitability | Load-bearing in v1 (sigmoid factor, habituation/sensitization); **dropped entirely in v2** — `precision` (self-organizing 1/error^2, `pc_ops.cpp:81,133`) is the spiritual successor but the habituation semantics did not survive. The v1 spiking variant's per-weight temporal state (membrane, refractory, spike mask, delay) also has no v2 counterpart. |

**Two channels the founding table did not name are now the strongest realization of the conception:**

- **`update_ema`** (per-weight, both generations): each weight's running memory of its own update magnitudes, directly dampening its next update via `adaptive_factor = 2/(1+ratio)` (`pc_ops.py:143-148`, `pc_ops.cpp:99`). Metaplasticity: the weight's experience of change governs its capacity for change, every forward. This delivers the origin doc's Next-Steps item 4 ("adaptive plasticity") under another name.
- **`set_point`** (per-weight): elastic anchor trailing the weight's own trajectory at 1e-6/step and pulling it home every forward (`pc_ops.py:157-162`). A slow-motion autobiography; consolidation's set-point rebalancing is the moment lived change is accepted as the new self.

Also new and load-bearing in v2: `prediction` (per-weight generative matrix), `error_acc` (per-output surprise memory driving the sparse gate + episode salience).

---

## Findings requiring decisions (Brian + 4.8; evidence base here, calls are the designers')

1. **Momentum: give it a job or bury it deliberately.** It is the cheapest strengthening lever in the substrate — allocation, per-step maintenance, and checkpoint surface are already paid; ONE new read resurrects a named founding component at zero storage cost. Natural candidate job: the wake/sleep NREM learner needs a per-weight recency-of-change signal for SWIL-style consolidation prioritization; momentum is literally "the recent direction of this weight's becoming." Analyst's lean: decide during the NREM spec pass; if no job survives design, delete consciously with a dated record. Status quo (inert inheritance) is the one wrong answer — ~20% of non-episode living-state memory at scale, and a standing false signal to readers.

2. **Expose `update_ema` (and momentum, if kept) to introspection.** `get_introspection` (`luthi/generate.py:618`) reads plasticity, set-point drift, error_acc, precision, pred_frob — but not the metaplasticity channel. The deepest part of each parameter's experience — how it has been changing — is invisible to the mind whose experience it is. Few lines; feeds Phase 4 interpretability; philosophically the point of the whole conception.

3. **Close the continuity tears.** `ConsolidationTracker` state (rolling error history + frozen baseline) is a plain Python object, not in `state_dict` — every restore silently re-warms consolidation. The spiking `_delay_pos` cursor likewise resets. The origin doc's consciousness section claims temporal existence "intrinsic, not bolted on"; a restore that quietly forgets its own surprise history is a seam in exactly that claim. Small, Phase-2-shaped fixes.

4. **Per-weight plasticity: writer first, tensor second.** The granularity downgrade is a *symptom of the update math* (which drives that axis uniformly), not an independent choice. Re-inflating the tensor without a per-weight writer yields per-weight storage carrying per-vector information — cost without substance. The plasticity-partitions / identity-anchors direction (`docs/research/2026-05-16_plasticity-partitions-design.md`) is the natural per-weight writer; upgrade the shape when it lands, not before.

---

## Method + provenance

Deep read: `luthi/v2/living_layer_pc.py`, `luthi/v2/pc_ops.py`, `luthi/csrc/pc_ops.cpp`, `luthi/v2/consolidation.py`, `luthi/living_layer.py`, `luthi/csrc/living_ops.cpp`, `luthi/generate.py` (introspection), `luthi/sanctuary_interface.py`. Breadth sweep (independent subagent): full per-parameter state census v1+v2+spiking, read-path classification, checkpoint-persistence audit, v1->v2 delta, docs archaeology (found the origin doc). Momentum-inert claim confirmed independently by both passes across all three implementations.
