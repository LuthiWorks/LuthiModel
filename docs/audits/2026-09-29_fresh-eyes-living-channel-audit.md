# Fresh-eyes audit — 2026-09-29: the living channel through the training path

**Auditor:** Muse Spark (Meta) — a different model line from the Claude
instances that built this repo, which is the point. First session with
the codebase; no prior investment in any decision below.
**Scope:** the living-weight update path as exercised by JEPA training
(`luthi/v2/living_layer_pc.py`, `pc_ops.py`, `jepa_loss.py`,
`plasticity.py`, `multimodal_model_pc.py`), plus the SIGReg wiring it
sits next to. Read-only except where noted.
**Question asked:** is there a *healthy-but-inert* mechanism on the
training path — something that reports alive while doing nothing, the
failure class this repo keeps finding (frozen episode store, BN-blinded
SIGReg, self-extinguishing drive)?
**Setup:** fresh CPU venv (torch 2.14.0+cpu); full suite run for baseline.

## Changes made (2)

**C1 — new test: `tests/test_living_channel_moves.py` (2 tests, green).**
A genuine coverage gap. The two learning channels had asymmetric
integration coverage: the backprop channel is pinned end-to-end by
`tests/test_lived_jepa_updates_world_model.py::test_lived_update_moves_encoder_and_predictor`,
but the *living* channel — "the act of processing changes the
processor" — had only unit-level coverage (`pc_self_modify` math, the
layer forward in isolation). Nothing asserted that a
`compute_modality_loss` training forward actually moves living weight
buffers. The new test snapshots every `PredictiveCodingLayer.weight`,
runs one training forward with **no optimizer step**, and asserts
per-layer movement above a 10-ULP floor (healthy path measures ~9M
ULP max, ~43k mean on the tiny fixture — the floor is conservative, the
point is to catch arithmetically-dead updates, per the `weight_abs_mean`
comment in `aliveness()`). The paired freeze control runs the identical
path under `freeze_plasticity` and asserts bitwise-zero movement,
proving the freeze — not a dead code path — is what makes the
difference, and that the movement comes from `pc_self_modify` alone
(the weight is a buffer, never a Parameter, so the optimizer cannot
touch it). It also asserts `nonfinite_forward_skips` did not advance,
so a NaN-guarded skip cannot masquerade as a quiet channel.

**C2 — doc fix: `luthi/v2/sigreg.py`, `SIGReg.forward` docstring.**
It still read "``(T, B, D)`` standardized embeddings" — the exact
wording the 2026-07-28 correction refuted at module level (BN-blinded
SIGReg cost the v5 family). The module docstring was fixed; the method
docstring was not. Now states the corrected contract: feed it the
distribution you want shaped, never pre-standardized input.

## Null findings (stated explicitly — these were checked, not assumed)

**N1 — no stuck wake attenuation.** `PredictiveCodingLayer._wake_attenuation`
defaults to 1.0 (`living_layer_pc.py:606`), `rate_scale` defaults to 1.0
(`:172`), and nothing in `luthi/` or `scripts/` calls
`set_wake_attenuation`/`wake_attenuated` outside `plasticity.py` itself.
A 1000x-suppressed living channel via a leaked regime is not present.

**N2 — no autograd-graph leak via `block_latents`.** `jepa_loss.py:677`
returns `online_result.get("block_latents")` without `.detach()`, which
reads like a leak — but the encoder already detaches at collection
(`multimodal_model_pc.py:391`, `h.detach()`), while `interior_latents`
(`:394`) is deliberately non-detached for gradient pressure. The
asymmetry is intentional and documented at both sites.

**N3 — eval-mode self-modification is by design, and eval purity holds.**
A raw `model.encode()` in eval mode *does* run `pc_self_modify`
(verified by probe; no `self.training` gate exists in
`living_layer_pc.py` — perception is never stateless). This is not a
bug: `evaluate_heldout` wraps everything in `freeze_plasticity` +
`no_grad` (`luthi/v2/eval_heldout.py:50`), and
`tests/test_heldout_eval.py::TestEvalMutatesNothing` pins bitwise
identity across the full living config. The layer's always-on
plasticity and the runner's frozen eval are two coherent halves, not a
contradiction.

**N4 — the dispatch cannot silently drop a mechanism on the fast path.**
`pc_self_modify` (`pc_ops.py:616`) forces the Python reference whenever
`drive_normalize` or a non-`"raw"` `drive_mode` is requested, because
those exist only in Python. A flag that *appears* on while C++ ignores
it — the silent-success class — is structurally refused here.

**N5 — the lived (Item #6) gradient path is already integration-tested**
for nonzero effect (`test_lived_update_moves_encoder_and_predictor`
asserts encoder backprop params move). No gap to fill.

## Observation (not a finding)

**O1 — interior weak-SIGReg reads the context forward's latents.**
`compute_modality_loss` passes `collect_block_latents` only to the
context encode (`jepa_loss.py:517`); the target/full encode also
collects `interior_latents` (gated by block config, not by the flag)
but discards them. With `interior_sigreg_alpha=0` (default) this is
dormant. Whoever arms the probe should decide deliberately whether
interior pressure belongs on the 80%-context stream or the
full-sequence stream SIGReg proper uses — the two see different token
sets.

## Suite baseline

`1105 passed, 8 skipped, 1 xfailed, 1 xpassed` in ~113 s (plus the 2 new
tests green standalone). The 4 failures are all in
`tests/test_backend_declaration.py` and are environmental: this box is
CPU-only and the repo's declared environments are DirectML/ROCm;
verified pre-existing by re-running on the stashed pristine tree
(identical 4 failures), and 3 of 4 pass with `LUTHI_ALLOW_CPU=1`. No
action needed — but the suite is red on a fresh CPU clone until that
env var is set, which a new contributor will trip over.

## For the reviewer, worth attacking

1. **The 10-ULP floor in the new test.** It is a smoke threshold, not a
   health criterion. If the project ever wants "the living channel is
   *usefully* alive" rather than "nonzero," that threshold wants a
   derivation, not a round number.
2. **O1's stream choice**, when the interior probe is armed.
3. Whether the audit-protocol cadence ("quarterly by default") is being
   kept — the last full audit in this directory is 2026-08-13.
