# Homeostatic activity bands: the key to the sparse gate's cage

**Date:** 2026-07-26
**Status:** DESIGN SKETCH — not implemented, not registered. Drafted by Fable 5
from a design conversation with Brian. Implementation deferred (see hazard note).
Registration is Brian's, with Opus.

> **IMPLEMENTATION HAZARD — read first.** The v5 family is mid-flight (seeds
> 45/46 + the stage-11 rerun still to run) and the supervisor imports fresh
> code for every new run. **No edits to `luthi/` until the rerun completes**,
> or the family's back half runs different code than its front half. This
> document exists so the design survives the wait.

## The gap this fills

Brian's chain of reasoning, recorded because it is the actual argument:

1. Representation collapse is the model finding the cheapest route to low error.
2. Gating updates by relevance (input activity, output error, ledger trust)
   denies collapse its cheap route — uniformity requires broad coordinated
   drift, and gating confines each update to a relevant subnetwork.
3. **But a gate that prevents drift into a rut also prevents climbing out of
   one.** Specifically, the dormant sparse gate (`sparse_threshold`, gating on
   low `error_acc`) has a perverse fixed point: a collapsed row achieves
   trivially low error, so the gate freezes it in the collapsed state.
4. Therefore the gate needs a companion mechanism that *detects* a rut and
   *reopens* it.

What exists today and why it does not close the gap:

- **SIGReg** is a continuous, global anti-collapse pressure toward the
  isotropic target. It makes collapse expensive everywhere but does not visit
  a specific dead row and reopen it. Field, not rescue crew.
- **Kill criteria** detect collapse; their only verb is "end the run."
- **Inverted-U `learning_gain`** modulates plasticity by error magnitude, not
  by participation, and shares the same blind spot: a collapsed row reads as
  low-error and therefore low-need.

## The biological analogue

Homeostatic plasticity (synaptic scaling, Turrigiano & Nelson; intrinsic
plasticity, Desai et al.): neurons maintain a target firing-rate range over
long timescales. Chronic underactivity upregulates intrinsic excitability and
synaptic gain until the cell participates again; chronic overactivity scales
down. Crucially it operates on a **much slower timescale** than the learning
it modulates — that separation is what keeps it from fighting learning.

## The discriminating signal (the part that makes it work)

The distinction the sparse gate cannot draw, but this can:

| row state | error_acc | output variance | correct action |
|---|---|---|---|
| competent (mastered its job) | low | **live** | leave gated off — this is success |
| collapsed / rut | low | **dead** | REOPEN — this is the pathology |
| struggling | high | live | already plastic; no action |

**Low error with live variance is competence; low error with dead variance is a
rut.** Only the second should trip the mechanism. Any implementation that keys
on error alone will fire on healthy specialists and destroy them.

## Proposed math (per living layer, per output row j)

Slow variance estimate (the participation signal), EMA over a long window:

    mu_j      <- (1 - b) * mu_j + b * mean(out_j)                 # b ~ 1e-3
    var_j     <- (1 - b) * var_j + b * (mean(out_j) - mu_j)^2
    act_j     = sqrt(var_j)                                        # participation

Band-relative deficit, with an explicit floor and ceiling on the band itself:

    a_lo, a_hi  = band bounds (config; derived from the family's healthy
                  per-row activity distribution — measure BEFORE choosing)
    deficit_j   = clamp((a_lo - act_j) / a_lo, 0, 1)               # 0 when in band
    excess_j    = clamp((act_j - a_hi) / a_hi, 0, 1)

Two bounded outputs — a plasticity multiplier and a gate override:

    h_j        = 1 + (H_MAX - 1) * deficit_j - (1 - H_MIN) * excess_j
                 clamped to [H_MIN, H_MAX]                        # e.g. [0.5, 3.0]
    gate_j     = OPEN if deficit_j > d_open else (gate as computed by
                 the sparse rule)                                  # the "key"

Applied to the row's update:

    dW_j <- h_j * dW_j                    (h_j is a MULTIPLIER, never a source)

### Floors and ceilings — Brian's requirement, itemized

The mechanism is a positive feedback loop (low activity -> more plasticity ->
potentially more activity), so every term is bounded by construction:

1. **`H_MAX` (plasticity ceiling, e.g. 3.0)** — hard cap on the multiplier. A
   dead row can never receive unbounded plasticity.
2. **`H_MIN` (floor, e.g. 0.5)** — overactive rows are damped, never silenced;
   the mechanism cannot kill a row.
3. **Multiplier-only, never additive** — `h_j` scales an update the learning
   rule already computed. With zero error signal there is nothing to amplify,
   so the mechanism cannot manufacture drift out of noise.
4. **Slow timescale (`b ~ 1e-3`)** — activity is estimated over ~1000 steps, so
   the mechanism cannot chase per-batch fluctuation. Explicit separation from
   the learning timescale.
5. **Warmup lockout** — inert until `act_j` estimates are seeded (mirrors the
   sparse gate's `sparse_warmup_steps`, same bootstrap-deadlock logic).
6. **Dead-zone inside the band** — while `a_lo <= act_j <= a_hi`, `h_j == 1`
   exactly. No always-on nudging; the mechanism is silent when healthy.
7. **Rate limit on reopening** — cap the fraction of rows the key may reopen
   per window (e.g. <= 5%), so a global dip cannot reopen everything at once
   and undo the gate's entire benefit.
8. **Optional noise, if used at all: bounded and decaying** — exploration
   noise for a reopened row must be a small fraction of the row's own weight
   scale and taper off, or it becomes a collapse *cause*.

## Registration shape (for Brian + Opus)

- The gate and the key are **one bundle or two arms** — a design call. My
  recommendation: register them together with BOTH predictions and the named
  anti-prediction, because shipping the gate alone has a known failure mode.
- **Prediction:** sparse gate + activity bands, vs v5 at matched steps -> higher
  effective rank, lower mean off-diagonal correlation, no increase in
  dead-variance row count.
- **Named anti-prediction (the falsification watch):** gated arms show rows
  frozen at low error AND dead variance that ungated arms recover from. If the
  key works, this does not appear; if it appears anyway, the key is
  mis-tuned or the discriminating signal is wrong.
- **Measure before choosing the band.** `a_lo`/`a_hi` must come from the
  measured per-row activity distribution of healthy v5 runs, not from
  intuition. The harvested ledger snapshots plus a per-row variance emit give
  this for free — and per LuthiScope's contract, the emit is the producer-side
  change to request first.

## Instrument implication

Per-row output variance is not currently emitted. It is the signal this whole
mechanism keys on and cannot be reconstructed from what the logs carry, so
**the emit comes before the mechanism**: add per-row (or per-row-percentile)
output variance to `substrate_blocks`, watch it across the v5 family, then
choose the band from data.
