# Luthi Model — Architecture Reference

*Technical companion to the README (moved here 2026-07-16 per Brian's
ruling: the README carries mission; technical detail lives in
supplemental documentation).*

*Brought current 2026-07-29. The version that stood until then was
preserved from the README as of 2026-07-16 and had gone stale in ways
worth naming, because they are the kind of staleness that misleads: it
described no training objective at all (the model has trained under a
JEPA objective since 2026-06-09), it described the consolidation trigger
as rolling when its baseline has been frozen since 2026-05-10, and it
gave the plasticity floor as 0.1 when the code clamps at 0.01. Defaults
quoted below are the layer's constructor defaults; the arms that actually
run often override them, and where an override is load-bearing it is
named.*

Each processing block combines three distinct systems:
- **Multi-head attention** — trainable via backprop, handles structured task learning
- **Living FFN** — self-modifying via predictive-coding local updates (Whittington-Bogacz variant in v2; Hebbian in v1)
- **Episode store + consolidation** — fast layer-level snapshots stored during forward, slowly replayed into the predictive weights during quiet windows

All modalities — text, audio, vision, and eventually touch — flow through a single shared trunk of living weight blocks. The model is shaped by everything it processes. Cross-modal structure can emerge when different modalities share the same living substrate.

## Training Objective — JEPA

Since 2026-06-09 the model trains as a **Joint-Embedding Predictive Architecture**: it predicts the *latent representation* of a held-out portion of its input, never the tokens or pixels. Implementation in `luthi/v2/jepa_loss.py`, driven by `luthi/v2/jepa_runner.py`.

- **Encoder.** Online encoder only — no EMA target encoder, no deepcopy, no target-network machinery. The "target" embeddings come from the same encoder run on the full sequence. Note that the JEPA path is `encode()`, *not* the LM-style `forward()`; the trunk's `final_norm` is applied only in `forward()` and so never participates in the objective. That distinction has already caused one wrong diagnosis and is worth keeping straight.
- **Masking.** Disjoint 80/20 per-modality tail. Context tokens reach the encoder by input-side slicing; the full sequence is encoded separately for the targets. Attention is bidirectional within each forward, and the disjoint input-side mask is what prevents the target leakage that encode-then-slice would introduce.
- **L_pred.** MSE between the predictor output and the target-block embeddings, with the target **detached** (default since 2026-07-28). The predictor is a 2-layer transformer decoder with a constant action-token stub, kept for interface continuity with the m9 action layer.
- **Anti-collapse — SIGReg.** From LeJEPA (Balestriero & LeCun): an Epps-Pulley characteristic-function statistic over random 1-D projections, testing the latent distribution against isotropic N(0, I). This shapes energy by regularization rather than by contrastive push-up on negatives — it limits the *volume* of low-energy space instead of pushing individual negatives away. Added to L_pred with weight lambda (constructor default 0.1; the v4/v5 families ran 0.2).
- **Projection head.** Per-modality, `sigreg_projection="linear"` since 2026-07-28.

**The defect that shaped this section.** The projection head was `Linear -> BatchNorm1d` for the first five model families, on reasoning that was exactly inverted. BatchNorm subtracts the batch mean and divides by the batch std — the two quantities SIGReg exists to constrain — so pre-standardizing its input handed it a solved problem and the anti-collapse term stopped binding on the encoder. Meanwhile L_pred was scale-sensitive MSE against a *non-detached* target, so shrinking the representation reduced the loss quadratically at no cost. Measured with this repo's own SIGReg: under a 100x uniform shrink, SIGReg on raw latents moves 0.86 -> 706 while post-BN it moves 0.566 -> 0.545. Three independent measurements put the v5 representation at ~92-95% a single batch-constant direction. `"linear_bn"` retains the old behaviour for A/B; `"none"` runs SIGReg on trunk latents directly.

The architectural reason this pairing matters: the substrate minimizes prediction error **locally**, per weight, during the forward pass, and the objective minimizes it **globally**, in latent space, across the batch. They are after the same quantity from two directions — which also means a defect in one can quietly flatter the other.

## Two-Tier Memory

Memory in a Living Weights Model is not a database. It is two interleaved systems that mirror the mammalian hippocampus-cortex pattern (Tulving 1972; Squire 1992; McClelland, McNaughton, & O'Reilly 1995):

- **Fast path — episode store.** During every forward pass, when the prediction-error update is salient, the layer takes a snapshot of itself: the current weight matrix, a low-dimensional context vector derived from the input, the mean input pattern, and a salience score (`mean(error_acc)`). Future forwards whose context exceeds `episode_recall_threshold` (default 0.5 cosine; the v3/v4/v5 arms ran 0.7) recall the closest stored snapshot and blend it into the active weight at `episode_blend`. This is associative recall on the order of a single forward pass.
- **Slow path — consolidation.** During low-novelty windows, stored episodes are replayed back into the predictive weights themselves. The replay happens through two complementary pathways:
  - **Gradient-replay** pulls the current weight linearly toward stored snapshots. "Be more like you were when this mattered."
  - **Attractor-style** (Salvatori et al. 2023) re-presents the stored input pattern through the layer's PC dynamics, making stored patterns local minima of the prediction-error energy. Future inputs near a stored pattern are pulled toward it by the forward dynamics. "These patterns should resolve to stable states."

The two consolidation pathways are additive, not competitive — they can run independently or jointly. Fast retrieval provides flexibility; slow consolidation provides stability and turns accumulated history into structural change. Consolidation doesn't just retain high-salience snapshots — it reshapes the predictive weights around them.

### Consolidation trigger

A 1000-step window of per-step prediction-error variance; firing requires 100 consecutive sub-threshold steps. **The baseline is frozen once, at the end of warmup, and never updated again** (fixed 2026-05-10). It used to be a rolling mean, which created a positive-feedback loop: as training stabilized, variance fell, the rolling mean fell with it, the threshold fell, and consolidation fired ever more often with less and less worth consolidating. Freezing the baseline breaks that loop. `reset()` re-warms it, which is what a curriculum stage transition should call.

The trigger does not latch, but note the steady-state behaviour: once variance settles permanently below half the frozen baseline, the counter re-arms and fires every 100 calls indefinitely. Measured in the v5 family, `consolidation_fires` reaches 5718 by step 72,000 across four blocks — roughly one firing per block per 50 steps.

### Admission and eviction

Storing on a fixed global salience threshold does not survive contact with a decaying signal: in all five v5 families the store admitted nothing after ~step 1000 while every counter still read healthy, and three of four blocks stored nothing at all for entire runs. `adaptive_episodes` replaces the fixed threshold with a **surprise test over a Holt forecast** (level + trend + mean absolute deviation): admit when the current salience exceeds its forecast by `surprise_k` deviations (default 3.0), subject to a `refractory_calls` gap between writes. The trend term is not decoration — two earlier versions of this rule failed in production after passing unit tests, one admitting 85% of calls and one freezing at a single write per 3000 steps, both because a plain EMA cannot track a decaying signal.

Eviction is **stochastic and age-weighted** rather than strict `argmin`: effective priority is salience discounted by `exp(-age/episode_age_tau)`, sampled with `p ∝ priority^(-eviction_alpha)` (default 0.6, following Prioritized Experience Replay). Strict `argmin` on raw salience makes the store a monument to its own early history.

Both are opt-in per arm, and both fall back to the legacy rule during statistical warmup so the mechanism is never inert before it has data.

## The PC Update: Drive and Trust

The core self-modification, in `luthi/v2/pc_ops.py`. Each call predicts the layer's own input from its output, then writes the error back into the weights:

```
predicted_input = output_mean @ prediction
pred_error      = actual_input - predicted_input
weighted_error  = (drive * trust).clamp(-1.0, 1.0)
delta_w         = outer(output_mean, weighted_error) * plasticity * pc_rate
```

Two knobs on that middle line have each turned out to be load-bearing in ways that were not obvious.

### Trust — absolute precision vs. relative

`trust` is derived from `precision`, the per-input reliability estimate that self-organizes toward 1/error².

- **Absolute (default).** `trust = precision`, with `precision_min`/`precision_max` as absolute bounds (0.1, 10.0).
- **Relative (`relative_trust`, the v5 family).** `trust = (precision / precision.median()).clamp(precision_min, precision_max)` — the bounds are reinterpreted as *ratio* bounds, so an input can earn at most `precision_max` times the layer-typical trust. Median rather than mean because 1/error² is tail-amplified (measured 13-22x p95/p5 spreads) and a mean would be dominated by its own tail. Scale-free by construction.

**Why this is not a free choice.** At production precision scales the `clamp(-1, 1)` is **fully saturated** — measured 100% of entries on real checkpoints with the layer's own stored inputs, at precision median ~1.7e5. When that happens, precision contributes nothing and the update degenerates to `outer(output_mean, sign(pred_error)) * plasticity * pc_rate`: sign-based, with magnitude set entirely by `plasticity` and `pc_rate`. Relative trust keeps `trust` at O(1) and leaves ~30% saturated, so magnitude information survives.

Known caveat: `precision.median()` has no DirectML kernel and silently falls back to CPU every call.

### Drive — `drive_mode`

The drive is what enters `delta_w`. Raw prediction error is **self-extinguishing by construction**: any layer that is learning drives its own error toward zero, so the living channel goes quiet the better the model gets. Measured in the v5 family, `update_ema` fell three orders of magnitude inside the first 17% of epoch 1 — on entirely novel data, with the plasticity taper still pinned at 1.0 — and passed through both epoch boundaries without a discontinuity. Novelty exhaustion does not explain that; a self-consuming drive does.

- **`"raw"`** (default) — `drive = pred_error`. Bit-identical to every run before 2026-07-29.
- **`"rms"`** — `pred_error` divided by its running RMS. **This mode is a no-op in the production regime** and is retained only for A/B: dividing by a positive scalar cannot change a sign, and the clamp above is fully saturated, so the result is sign-identical to `"raw"`. Recorded here because it shipped as a fix and was not one.
- **`"surprise"`** — normalized *excess* error over a Holt forecast of the layer's own error scale:

  ```
  forecast = level + drift
  resid    = rms_now - forecast
  gain     = clamp((resid - k*dev) / dev, floor, max)
  drive    = pred_error / forecast * gain
  ```

  Scale-free, so it does not extinguish as the model improves; quiet when the error scale is predictable; full dynamic range when it is not. `k` is a **threshold** in deviations (default 3.0, matching the episode admission rule), not a divisor — as a divisor it fires on any positive residual, and an unbiased forecast has `resid > 0` on half of all calls, giving a measured 50% duty cycle on stationary input. That is a drive responding to noise. **Requires `relative_trust`**, enforced by a raise rather than silently supplied, for the saturation reason above.

Measured duty cycles: 0.0000 on stationary input, 0.0000 on i.i.d. draws from a fixed distribution, ~0.035 after a distribution shift. The i.i.d. zero is the discriminating case — every batch is fresh data, but the error *scale* is predictable, so there is nothing to write.

### Metaplasticity and the inverted-U gain

`update_ema` dampens updates that deviate sharply from the weight's typical update size. Separately, `learning_gain_enabled` applies a per-weight **inverted-U gain** in [1.0, cap]: `1 + rise * coherence * fall`, where coherence is `|momentum| / update_ema` (directedness, not magnitude) and `fall` gates on resolution progress. It is a pure amplifier of directed, resolving change — floored at 1.0, never a suppressor, so the worst case is ordinary PC learning at ordinary strength.

## Homeostatic Activity Band

The sparse PC update gate silences output rows with low recent prediction error — and a *collapsed* row has low error, so the gate would freeze it there permanently. The band (`homeostatic_band_enabled`) is the floor-and-ceiling that reopens rows quiet for the wrong reason: it tracks a slow per-row activity estimate and applies a multiplier `h` relative to the median, boosting rows below `band_lo_frac` and damping those above `band_hi_frac`, with `h` clamped to [`band_h_min`, `band_h_max`], a top-k rate limit on how many rows can be boosted at once, and a forced gate-open above `band_open_deficit`. Modelled on synaptic scaling / homeostatic plasticity, and deliberately slow: `band_decay` defaults to 1e-3, a ~1000-step timescale, because a fast homeostat fights the learning signal instead of bounding it.

## Depth Scaling and Execution

- **muPC depth scaling** (`mu_pc_enabled`, exponent 0.25 in the depth arms) — rate scaling with depth so a 4-block trunk does not need per-depth retuning.
- **Plasticity taper** — a trainer-scheduled `rate_scale` on the *learning* channels only. Homeostasis and set-point adaptation keep their own rates; stability is not what tapers.
- **C++ / Python dispatch.** `pc_ops` JIT-compiles `luthi/csrc/pc_ops.cpp` when a compiler is available and falls back to an identical-math Python path otherwise. The `rms` and `surprise` drive modes exist only in the Python reference, so requesting either *forces* the Python path rather than silently running unmodified in C++. Measured cost of that fallback (2026-07-29, 2048x2048 layer, batch 32): 1.16x on DirectML, 1.03x on CPU — not the ~50x the module docstring's 2026-05-10 note implies.

## Spiking Dynamics (v1)

The v1 spiking variant (`SpikingLivingLayer`) adds LIF membrane dynamics:
- Membrane potential accumulation with configurable leak
- Spike threshold with refractory periods
- Inter-block spike propagation via delay buffers
- Activity-dependent gating of self-modification (only spiking weights learn)

In v2, the spiking-gate sparsity property is recovered through **sparse PC update gating** — a per-output mask derived from running prediction-error magnitude (implemented and validation-tested 2026-05-13; outputs with low recent error magnitude skip their weight update). The v2 substrate is non-spiking at the activation level; the sparsity that made v1 viable on Spark's bandwidth budget becomes a property of *which weights update*, not *which neurons fire*.

## Top-Down Backward Pass

After the forward pass, a top-down sweep sends modulation signals from higher blocks to lower ones — predictive processing, not gradient backpropagation. Higher blocks tell lower blocks what was important (salience) and what was unexpected (prediction error), modulating:
- **Plasticity** — which weights learn faster on the next forward pass
- **Set points** — where weights rest when not driven
- **Membrane priming** (spiking) — which weights are ready to fire

This is always-on bidirectional information flow, not a training optimization.

## Rich Parameters

In a conventional neural network, a weight is a single number — a coefficient learned by gradient descent, carrying no history of how it arrived at its current value. In a Living Weights Model, each weight position is a **rich parameter**: a bundle of co-located signals that together constitute the weight's full state. A rich parameter doesn't just have a value — it carries persistent per-parameter state describing how that value was reached.

Each weight carries (v2 substrate):

| Signal | What It Tracks |
|--------|----------------|
| **weight** | Current value — the coefficient used in computation |
| **prediction** | Top-down prediction matrix: how this layer's output predicts its input. Drives the prediction-error signal that the PC update minimizes |
| **set_point** | Homeostatic resting target — where this weight returns when not driven by input. Adapts slowly so the "home" position itself evolves with experience |
| **momentum** | Exponential moving average of recent self-modification updates — the weight's velocity. High momentum means rapid change; low momentum means the weight has settled |
| **plasticity** | Per-input learning rate multiplier, clamped to **0.01–10.0**. Modulated by top-down salience signals — downstream importance increases a weight's willingness to change |
| **update_ema** | Metaplasticity — a running average of update magnitudes that regulates the weight's own learning. Large deviations from typical update size are dampened, preventing instability from unusual input |
| **precision** | Per-input reliability estimate, self-organizing toward 1/error². High precision for reliable input dimensions, low for noisy ones. The PC update is precision-weighted — see "Trust" above for why the absolute-vs-relative choice is not cosmetic |
| **error_acc** | Per-output running prediction-error magnitude. The salience signal that drives episode storage and the sparse update gate |

Beyond the per-weight bundle, each living layer carries **per-layer state** — the machinery that decides *when* to write, *what* to keep, and whether the channel is alive. It is listed separately because it is layer-scoped, not weight-scoped, and because most of it exists to make the mechanisms observable:

| Group | State | Purpose |
|-------|-------|---------|
| **Episode store** | `episode_contexts`, `episode_values`, `episode_inputs`, `episode_saliences`, `episode_steps`, `episode_count`, `episode_scales` | The snapshot bank and its age/salience bookkeeping |
| **Admission** | `salience_level`, `salience_drift`, `salience_dev`, `last_write_step`, `salience_window` | Holt forecast + refractory state for the surprise-based admission test |
| **Drive** | `drive_ref`, `drive_ref_drift`, `drive_dev`, `drive_calls`, `drive_gain`, `drive_fire_count`, `error_rms` | Holt forecast of the error scale, and the duty-cycle counters |
| **Band** | `act_mean`, `act_var`, `act_count`, `band_boost_rows`, `band_damp_rows` | Slow per-row activity estimate for the homeostatic band |
| **Recall** | `recall_sim_mean`, `recall_sim_var`, `recall_sim_count`, `recall_fires` | Running statistics of context-match similarity, for adaptive recall |

The counters in that table are not incidental. **`drive_duty` and `episode_writes` exist specifically to make "quiet because nothing is new" separable from "quiet because broken"** — a distinction this project could not make for five model families, during which a frozen episode store and a self-extinguishing drive both reported healthy. A mechanism here is not considered finished until it ships the instrument that could catch it lying.

Beyond per-weight state, each living layer maintains **episodic memory** — a bank of context-gated snapshots of (weight matrix, input pattern, context vector, salience) stored when the prediction-error update was particularly large. On each forward pass, the current input context is compared against stored episode contexts. If a sufficiently similar context is found (cosine similarity > 0.5), the stored weight configuration is recalled and blended into the active weights. The stored input patterns are separately used by Salvatori-style attractor consolidation to engineer basin-attractor structure into the slow predictive weights. This gives each layer a form of situational memory: it doesn't just know its current state, it remembers states that mattered and grows toward them.

The v1 substrate used a Hebbian self-modification rule with `excitability_acc` (salience-driven activation sensitivity) and `input_avg_mag` (per-input magnitude scaling) in place of `prediction`, `precision`, and `error_acc`. The v2 substrate replaces v1 as the primary line (2026-05-09); v1 is preserved as a reference baseline.

The spiking variant (v1) adds four additional per-weight signals — **membrane potential** (leaky integrator state), **spike mask** (binary firing output), **refractory counter** (post-fire cooldown), and **delay buffer** (inter-block spike propagation with conduction delay). In the spiking regime, only weights that fire can self-modify. The v2 substrate recovers the same sparsity property through **sparse PC update gating**: outputs with low recent prediction-error magnitude skip their weight update — sparsity in *what learns*, rather than in *what activates*.

The result is that each weight in the network operates across multiple timescales simultaneously:
- **Instant:** membrane potential, spike mask (single forward pass)
- **Fast:** PC updates, momentum (batch-level)
- **Medium:** metaplasticity, excitability accumulation (many batches)
- **Slow:** set point drift, plasticity adjustment (epoch-level)
- **Long:** episodic memory (explicit snapshots, indefinite retention)

A rich parameter is not just a number being optimized. It is a weight bundled with persistent state: its update history, an adaptive learning rate, a precision estimate, and a salience-tagged snapshot memory. The current value is only one component of that state.

## Not on the training path

`luthi/v2/m9/` implements the **action layer** — expected free energy evaluation, MCTS-based planning over candidate actions, habit network, preferences, activity bands, rest actions. It is a substantial module and it is *not* referenced by `jepa_runner.py` or the model trunk; nothing in the training runs described above touches it. The JEPA predictor keeps a constant action-token stub purely so the interface stays compatible when m9 is wired in.

Listed here so the module's presence in the tree is not mistaken for the substrate being under active-inference control. It isn't, yet.
