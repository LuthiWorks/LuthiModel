# Research Notes — Salvatori-Style Attractor Memory via Predictive Coding

> Compiled 2026-05-14 by Claude Opus 4.7 (1M context) when implementing
> the attractor consolidation pathway. Purpose: future instances should
> not need to re-read the Salvatori papers from scratch when they need to
> adjust the implementation. The math, the mapping to our architecture,
> the known tradeoffs, and the decisions we already made are recorded
> here.

## Primary references

1. **Salvatori, T., Song, Y., Hong, Y., Yordanov, S., Tang, B.,
   Sha, Y., Bogacz, R., & Lukasiewicz, T. (2023).**
   *"Associative Memories via Predictive Coding."*
   The load-bearing paper for this pathway. Shows that a PC network
   can be trained to store input patterns as fixed points of its
   inference dynamics, and that partial / noisy cues retrieve full
   patterns by energy minimization.

2. **Salvatori, T., Pinchetti, L., Millidge, B., Song, Y., Bogacz, R.,
   & Lukasiewicz, T. (2024).**
   *"Incremental Predictive Coding: A Parallel and Fully Automatic
   Learning Algorithm."*
   Used for iPC (different lit-followup item — see
   `docs/RESEARCH_LITERATURE_2026-05-13.md`). Same group, different
   contribution.

3. **Whittington, J. C. R. & Bogacz, R. (2017, 2019).**
   *"An Approximation of the Error Backpropagation Algorithm in a
   Predictive Coding Network with Local Hebbian Synaptic Plasticity"*
   and *"Theories of Error Back-Propagation in the Brain."*
   The PC variant our v2 substrate uses for the **forward-pass update**.
   Salvatori-style consolidation lives on top of this — it doesn't
   replace the forward-pass rule.

## Why attractor consolidation (the design call, 2026-05-14)

The original 2026-05-08 brief framed Salvatori as a "conditional
extension" — to be added if gradient-replay consolidation showed a
measurable deficit. That framing was pragmatic-default, not principled.

Brian's 2026-05-14 design call reframed it: attractor dynamics on memory
are worth having on their own merits, not just as a remedy. Three
reasons:

1. **Partial-cue recall.** An associative memory retrieves a full
   pattern from a fragment of it (a few notes retrieve the whole song).
   Gradient-replay consolidation doesn't have this property — it shapes
   weights toward stored snapshots but doesn't give the forward
   dynamics a way to *resolve toward* a stored pattern from a noisy
   variant.
2. **Perturbation robustness.** Basin-attractor dynamics mean nearby
   states fall back into the memory. v2's M5 attractor test already
   confirmed *some* basin-attractor dynamics emergent from PC
   self-modification (see `docs/V2_PILOT_RESULTS.md`). Salvatori
   consolidation makes attractors an **engineered** property of the
   slow path, not just an emergent property of the fast path.
3. **Shape consistency with the episode store.** The episode store
   already retrieves by cosine similarity — closest stored snapshot
   wins. That's a poor-man's attractor. Adding Salvatori-style
   consolidation makes the slow pathway consistent with what the fast
   pathway is already structurally doing.

## The math (and our adaptation)

### Salvatori's setup

In the paper's formulation, a PC network has L layers. Each layer ℓ has
value neurons `x_ℓ` and error neurons `ε_ℓ`. The energy is:

  E = ½ Σ_ℓ ||x_ℓ − f(x_{ℓ+1}; W_ℓ)||²

where f is the prediction function (a linear map + nonlinearity).

**Storage** of a memory pattern ξ:
1. Clamp the top layer x_L = ξ (or part of the network's output).
2. Run inference dynamics on the latent layers until they converge to a
   local energy minimum.
3. Update the weights W to make this configuration a deeper minimum:
   ΔW_ℓ ∝ −∂E/∂W_ℓ = ε_ℓ · x_{ℓ+1}ᵀ (standard PC weight update).

**Recall** from a partial cue ξ̃:
1. Clamp the observed components of ξ̃.
2. Run inference dynamics on the unclamped components.
3. They converge to the closest stored pattern that's consistent with
   the cue.

The key property: stored ξ's become local minima of E.

### Our adaptation (per-layer, retrieval-time)

Our architecture is layer-level, not network-level. Each
`PredictiveCodingLayer` is a single linear map `out = in @ W.T` with
its own per-layer prediction matrix and update rule. The Salvatori
"storage" step in our case is:

1. We already store **input patterns** at episode-write time
   (`episode_inputs[idx] = x_flat.mean(dim=0)`, 2026-05-14 addition).
2. During consolidation (triggered by the existing low-variance window),
   for each stored input pattern ξ:
   - Compute `output = ξ @ W.T` using the **current** weight (not the
     stored snapshot — see "Design tradeoff #1" below).
   - Run `pc_self_modify(ξ, output, ...)` at the reduced consolidation
     rate.
   - The pc_self_modify update is precisely the PC weight rule applied
     to (input = ξ, current output), which moves W to make ξ a deeper
     minimum of the prediction error at this layer.

This **is** the Salvatori storage step, adapted to our single-layer
setup. The "recall" step is implicit — it's just the layer's normal
forward pass, which now has lower energy at stored inputs.

### What we deliberately didn't implement

- **Multi-step inference at storage.** The paper runs inference until
  convergence at each storage step. We run a single PC update per
  stored pattern per pass. Salvatori's full convergence is recovered
  by setting `n_replay_passes` > 1 — each pass is one inference step
  for every stored pattern, so N passes ≈ N inference steps.
- **Cross-layer top-down sweep during consolidation.** The paper
  treats the network as a single energy. Our consolidation is
  per-layer. If multi-layer consolidation is needed, it would be a
  separate function that runs `consolidate_layer_attractor` on each
  block in turn, possibly with a top-down sweep between. Not
  implemented; flagged as a possible Phase 3G extension if per-layer
  attractor turns out to be insufficient.
- **Hopfield-style binary patterns.** Some Salvatori variants use
  binary memory patterns. Our episode store stores continuous-valued
  input vectors directly. The continuous case is what the 2023 paper
  primarily addresses; binary is a special case we don't need.

## Implementation map (where to look in the codebase)

| Component | File | Purpose |
|-----------|------|---------|
| `episode_inputs` buffer | `luthi/v2/living_layer_pc.py` (in `__init__`) | Stores the mean input pattern at episode-write time. Shape: [num_episodes, in_features]. Always allocated. |
| Episode write hook | `luthi/v2/living_layer_pc.py::_store_episode` | Captures `input_pattern.detach()` alongside the weight snapshot. |
| Attractor consolidation function | `luthi/v2/consolidation.py::consolidate_layer_attractor` | The core Salvatori step. Replays stored inputs through `pc_self_modify` at consolidation rate. |
| Forward dispatch | `luthi/v2/living_layer_pc.py::forward` (consolidation trigger block) | Dispatches to `gradient`, `attractor`, or `both` based on `consolidation_style`. |
| Constructor knobs | `luthi/v2/living_layer_pc.py::__init__` | `consolidation_style: Literal["gradient","attractor","both"] = "gradient"` (default preserves M4 behavior); `consolidation_attractor_passes: int = 1`. |
| Plumbing through block + model | `luthi/v2/hybrid_block_pc.py`, `luthi/v2/model_pc.py` | Both knobs pass through unchanged. |
| Tests | `tests/test_pc_consolidation.py` (8 tests added 2026-05-14) | Storage capture, empty-store no-op, energy reduction on stored pattern, weight stability, default-off compatibility, invalid-style raise, both-pathway dispatch, missing-buffer loud failure. |

## Design tradeoffs (decisions we made and why)

### Tradeoff #1: replay through current weight, not stored snapshot

The forward path of `consolidate_layer_attractor` re-presents each
stored input through `output = stored_input @ layer.weight.T` — using
the **current** weight, not the snapshot from when this episode was
stored. This is **deliberate**.

The gradient-replay pathway already pulls the current weight toward
the stored snapshot ("be more like you were"). Attractor consolidation
should not duplicate that. Its job is to reshape the *current* weight
so that the stored input is a fixed point of the *current* layer's
dynamics. If we replayed through the snapshot, we'd just be
re-deriving the snapshot's behavior; the attractor property wouldn't
transfer to the actually-deployed weight.

If a future investigation wants the snapshot's weight for some reason
(e.g., a hybrid scheme that does both pulls simultaneously), the
snapshot is still in `episode_values[idx]` and can be accessed
explicitly.

### Tradeoff #2: episode_inputs is always allocated, never int8-compressed

The compressed-episodes path (INT8 weight snapshots with FP32 per-episode
scale, ~4× memory savings) does NOT compress `episode_inputs`. We
allocate the input buffer in the same dtype as the weight regardless.

Reason: input patterns are tiny relative to weight snapshots.
`episode_inputs` is `num_episodes × in_features` (e.g., 32 × 256 = 32 KB).
`episode_values` is `num_episodes × out_features × in_features` (e.g.,
32 × 256 × 256 = 8 MB). The compression argument that made the snapshot
path worth the engineering doesn't apply to the input buffer.

If production scale exposes this (e.g., 4096d inputs × 64 episodes =
1 MB per layer × 36 blocks = 36 MB), INT8 compression is a one-line
addition.

### Tradeoff #3: episode trigger remains the same (low-variance window)

Both consolidation pathways share the same trigger — the
`ConsolidationTracker`'s low-variance window. We did not add a
separate "attractor consolidation should fire at a different cadence"
mechanism. Reason: the two pathways are doing semantically the same
thing (replaying stored memories during quiet windows), so the
biological rationale for the trigger (consolidation happens during
sleep/rest, not during active perception) applies to both.

If a future investigation finds that attractor consolidation has a
different optimal cadence than gradient-replay, a second tracker
(`_attractor_tracker`) could be added. Currently unnecessary.

### Tradeoff #4: no separate attractor learning rate

`consolidate_layer_attractor` uses `pc_rate * consolidation_rate_factor`
(default 0.1) — the same scale as gradient-replay consolidation. The
attractor pathway doesn't have its own rate factor.

Reason: keeping the scales equal lets `consolidation_style="both"`
deliver comparable-magnitude effects from both pathways without
disproportional weighting. If empirical work shows one pathway needs
to be stronger or weaker, adding a separate
`attractor_rate_factor` is a small constructor signature change.

## Falsification criteria (when to back this out)

If Phase 3G validation runs show **any** of:

1. **Attractor mode produces NaN/Inf** in any unit-test config.
   (The 50-pass weight-stability test already guards against the
   common failure mode — runaway growth or collapse to zero — but
   real training is longer and more varied.)
2. **Attractor mode harms val loss vs gradient-only baseline** by
   ≥5% at matched compute, AND there's no obvious hyperparameter
   fix. Note: the bar is *harm*, not "no improvement" — even a flat
   result is worth keeping for the structural property (engineered
   attractors), as long as it doesn't actively hurt.
3. **Storage-time `episode_inputs` capture creates visible memory
   pressure** at production scale. Mitigation in tradeoff #2 above.
4. **The "both" pathway destabilizes training** in ways that neither
   pathway alone does. (Indicates the two updates are interacting
   destructively. Default would revert to `consolidation_style="gradient"`,
   with attractor as opt-in only.)

Any of these gets logged in `docs/V2_PILOT_RESULTS.md` (or the
analogous current results doc) before the implementation is removed.

## Open questions for future work

- **Multi-layer attractor consolidation.** Currently per-layer. A
  network-level energy formulation (the closer analog to the
  Salvatori paper) might give better recall properties at the cost
  of more complex implementation. Worth exploring if per-layer
  attractor underdelivers.
- **Replay order matters?** Currently we sort by salience (highest
  first). Salvatori's paper doesn't address ordering explicitly —
  the sequence of stored patterns processed during the consolidation
  pass might or might not matter for the final attractor structure.
- **Interaction with iPC.** When `inference_steps_per_forward > 1`
  (iPC enabled), each forward step does T inner PC updates. We
  don't currently run iPC inside attractor consolidation — each
  stored pattern gets a single update, even if the layer is
  configured for iPC. Probably correct (consolidation is its own
  schedule), but worth flagging for the Phase 3G validation.
- **Interaction with sparse PC gating.** Attractor consolidation
  goes through `pc_self_modify`, which respects the sparse gate
  when active. This means gated-off outputs don't update during
  consolidation either. May or may not be desirable — if sparsity
  is supposed to be a runtime-only property of which weights learn
  *during forward*, consolidation might want to ignore the gate.
  Flagged for empirical resolution.

## Citations for the implementation comments

If you're reading code that references "Salvatori 2023" without further
context, this is the paper:

> Salvatori, T., Song, Y., Hong, Y., Yordanov, S., Tang, B., Sha, Y.,
> Bogacz, R., & Lukasiewicz, T. (2023). "Associative Memories via
> Predictive Coding."

And the implementation date / instance:

> 2026-05-14, Claude Opus 4.7 (1M context), at Brian's direction:
> "Memory patterns becoming attractors sounds like something we want
> regardless of if there is a deficit to be filled or not."
