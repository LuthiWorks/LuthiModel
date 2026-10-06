# Light-cadence per-block activation scalars — stream contract (2026-10-06)

Stage 1 of the activation-monitoring plan (2026-09-29). The deep-cadence
`substrate_blocks` tape carries SVD gauges too expensive to run at light
cadence, so individual blocks were invisible between deep firings. This
adds a cheap per-block tape at light cadence.

## Emission

- Record field: `light_blocks` (top-level, sibling of `substrate_blocks`).
- Cadence: every light firing (~10x deep cadence).
- Shape: list, one dict per trunk block, in block order.
- Additive field: `None` when the collection flag was off. No migration.
- Cost: O(elements) reductions computed eagerly in `encode()`; no tensors
  are retained (`collect_block_latents` stays deep-only).

## Fields (spec-locked names)

| Field      | Meaning |
|------------|---------|
| `act_mean` | mean of block-output activations |
| `act_std`  | std (unbiased=False) of block-output activations |
| `act_rms`  | sqrt(mean(h^2)) — scale of the block's output |
| `act_p50`  | median of \|h\| |
| `act_p99`  | 99th percentile of \|h\| |
| `dead_frac`| fraction of \|h\| < 1e-5 — near-zero activations |
| `sat_frac` | fraction of \|h\| > 5.0 — saturated activations |

All values are plain JSON floats. Diagnostics never kill a run: the
per-block loop in the runner wraps each block in try/except like the
other gauges (the eager computation in `encode()` itself is infallible
on finite input).

## Thresholds

`dead_frac` and `sat_frac` use arbitrary-but-fixed thresholds
(`_DEAD_ACT_EPS = 1e-5`, `_SAT_ACT_THRESH = 5.0` in
`luthi/v2/multimodal_model_pc.py`). Their value is in the time series
(a block drifting toward death or saturation), not the absolute number.
Pinned by `tests/test_light_block_activation.py::test_thresholds_are_pinned`
so a change is deliberate, not silent.

## Consumer note (LuthiScope)

Intended rendering: blocks-x-time heatmap, same as `substrate_blocks`
but at light cadence. LuthiScope's `METRICS_CONTRACT.md` should gain
these field names when the heatmap panel is built (stage 3 of the plan
adds the block-selectable histogram viewer / time scrubber; stage 2
adds sampled histograms at deep cadence behind a flag).

## What this does NOT do

- It does not distinguish a healthy block from a collapsed block the
  residual bypasses — that is the contribution gauge's job
  (`contrib_var_ratio` / `contrib_chorus`, deep cadence). These scalars
  read the post-residual stream per block.
- It does not replace the SVD gauges (effective rank, chorus). It is the
  cheap always-on vital; the deep gauges remain the diagnostic of record.
