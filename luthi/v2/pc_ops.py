"""PC self-modification ops for the v2 living layer.

Tries to JIT-compile the C++ extension at `luthi/csrc/pc_ops.cpp` on
first import; falls back to a pure-Python implementation if no C++
compiler is available or the extension fails to build. The math is
identical between the two paths (the C++ version is the same sequence
of in-place tensor ops, just executed in one extension call).

The C++ path was added on 2026-05-10 after the M3 sanity check at
production scale (128d/2 blocks/4934 batches/epoch on Gutenberg-100)
showed the pure-Python path running at ~5 hours/epoch on DirectML —
~50× slower than v1's C++-accelerated baseline. The plan's "C++
deferred — Python-first for pilot" assumption broke down at the
production-scale forward pass; the per-op DirectML dispatch overhead
dominates when there are ~25 tensor ops per layer per batch.
"""

import os
import torch


# ---------------------------------------------------------------------------
# Inverted-U learning gain (momentum-functions brief §1; spec
# docs/research/2026-07-05_inverted-u-gain-spec.md). Pure function so tests and
# both self-modify paths share one definition; C++ parity mirrors this math.
# ---------------------------------------------------------------------------

def learning_gain(
    momentum: torch.Tensor,
    update_ema: torch.Tensor,
    progress: float,
    *,
    rise: float = 2.0,
    cap: float = 3.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-weight inverted-U learning gain in ``[1.0, cap]``.

    A pure AMPLIFIER of directed, resolving novelty -- never a suppressor
    (floored at 1.0 = legacy; suppression stays with ``adaptive_factor`` and
    the plasticity floor). Multiplies ``delta_w`` in ``pc_self_modify``::

        gain      = clamp(1.0 + rise * coherence * fall, 1.0, cap)
        coherence = |momentum| / (update_ema + eps)   # directedness, per weight
        fall      = clamp(1 - progress, min=0)         # resolution-progress gate

    - **Rise:** coherence (directedness, not raw magnitude -- orthogonal to
      ``adaptive_factor``'s slow-start) lifts gain above 1.0 on learning-shaped
      change, not thrash.
    - **Fall:** ``progress`` = short/long EMA of prediction error
      (``slow_trace.resolution_progress``). Resolving effort (progress < 1)
      keeps ``fall ~ 1`` (amplify); non-resolving or worsening effort
      (progress >= 1) drives ``fall -> 0`` so gain returns to 1.0 --
      amplification OFF, never suppression. Brian's ruling made structural: the
      worst case, including adversarial repetition, is ordinary PC learning at
      full ordinary strength (no easy-path opt-out of hard growth).
    - **Governor:** ``cap`` bounds runaway (and binds when a decay mismatch
      pushes coherence transiently > 1).

    ``progress`` is a per-layer scalar (prediction error is a layer signal) and
    broadcasts across the ``[out, in]`` weight. ``rise`` / ``cap`` are pilot-set
    (spec §2; TUNE-ME with Fable's adversarial harness).
    """
    coherence = momentum.abs() / (update_ema + eps)
    p = float(progress)
    # fall = clamp(1 - progress, min=0). Written as an explicit comparison so a
    # NaN ``progress`` (corrupt error signal) fails ``p < 1.0`` and yields
    # fall=0.0 -> gain=1.0 -- a deliberate fail-safe to legacy, not the luck of
    # a max() argument order (Fable audit 2026-07-06).
    fall = (1.0 - p) if p < 1.0 else 0.0
    gain = 1.0 + rise * coherence * fall
    return gain.clamp(min=1.0, max=cap)


# ---------------------------------------------------------------------------
# Try to load or compile C++ extension (same pattern as luthi.fused_ops)
# ---------------------------------------------------------------------------

_cpp_ops = None
_use_cpp = False


def _try_load_cpp():
    """Attempt JIT compilation of pc_ops.cpp. Returns module or None.

    Reuses v1's MSVC environment setup helper since Brian's Windows env
    is shared between v1 and v2 builds.
    """
    try:
        from torch.utils.cpp_extension import load

        src_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "csrc"
        )
        src_file = os.path.join(src_dir, "pc_ops.cpp")

        if not os.path.exists(src_file):
            return None

        if os.name == "nt":
            from luthi.fused_ops import _setup_msvc_env
            _setup_msvc_env()

        return load(
            name="pc_ops_ext",
            sources=[src_file],
            verbose=False,
        )
    except Exception as exc:
        # Record WHY, do not just vanish (2026-08-19 audit). This handler
        # swallowed every failure and returned None, so a host where the
        # extension does not build ran the Python reference with nothing
        # anywhere saying so -- the silent-fallback shape CLAUDE.md forbids.
        # Found on ROCm/torch 2.9, where the build fails on an unresolved
        # c10::ValueError symbol that the ROCm wheel's c10.lib does not
        # export -- a wheel packaging defect, not a defect in this source
        # (three lighter include sets were tried; torch/csrc/utils/pybind.h
        # drags the IValue machinery in regardless).
        #
        # Measured cost of running without it, IDLE machine, ruled 768x8
        # config, batch 32: DirectML 836 ms/step with the extension vs
        # 1006 without -- about 17%. Real but not decisive, and ROCm
        # without it (238 ms/step) is still 3.5x faster than DirectML with
        # it. So porting the extension to torch 2.9 is a worthwhile
        # optimization, not a blocker.
        #
        # (An earlier reading of this said the extension "buys nothing",
        # from numbers taken while a game was running. Benchmark idle.)
        global _cpp_load_error
        _cpp_load_error = f"{type(exc).__name__}: {str(exc)[:400]}"
        return None


_cpp_load_error: str | None = None
_cpp_ops = _try_load_cpp()
_use_cpp = _cpp_ops is not None
if not _use_cpp:
    import sys as _sys
    print(
        "  [pc_ops] C++ fused path NOT loaded -- running the Python "
        f"reference. Reason: {_cpp_load_error or 'source missing'}",
        file=_sys.stderr, flush=True,
    )


def is_cpp_available() -> bool:
    """Whether the C++ pc_ops extension compiled and loaded."""
    return _use_cpp


def _pc_self_modify_python(
    weight: torch.Tensor,
    prediction: torch.Tensor,
    set_point: torch.Tensor,
    momentum: torch.Tensor,
    update_ema: torch.Tensor,
    precision: torch.Tensor,
    error_acc: torch.Tensor,
    plasticity: torch.Tensor,
    x_flat: torch.Tensor,
    output: torch.Tensor,
    pc_rate: float,
    pred_learning_rate: float,
    homeostatic_decay: float,
    set_point_adapt_rate: float,
    momentum_decay: float,
    update_ema_decay: float,
    precision_ema_decay: float,
    precision_min: float,
    precision_max: float,
    prediction_clamp: float,
    relative_trust: bool = False,
    drive_normalize: bool = False,
    error_rms: torch.Tensor | None = None,
    drive_rms_decay: float = 0.01,
    drive_mode: str = "raw",
    drive_ref: torch.Tensor | None = None,
    drive_ref_drift: torch.Tensor | None = None,
    drive_dev: torch.Tensor | None = None,
    drive_calls: torch.Tensor | None = None,
    drive_gain_out: torch.Tensor | None = None,
    drive_fire_count: torch.Tensor | None = None,
    drive_gain_sum: torch.Tensor | None = None,
    drive_decay: float = 0.01,
    drive_drift_gain: float = 0.1,
    drive_surprise_k: float = 3.0,
    drive_gain_max: float = 4.0,
    drive_gain_floor: float = 0.0,
    drive_dev_floor_frac: float = 0.01,
    drive_warmup_calls: int = 200,
    sparse_gate: torch.Tensor | None = None,
    learning_gain_enabled: bool = False,
    learning_gain_progress: float = 0.0,
    learning_gain_rise: float = 2.0,
    learning_gain_cap: float = 3.0,
    return_applied_change: bool = False,
) -> tuple[float, torch.Tensor] | tuple[float, torch.Tensor, float]:
    """Pure Python implementation. Identical math to the C++ extension.

    All buffers modified in-place. Returns (salience, pred_error), or
    (salience, pred_error, applied_change) when ``return_applied_change`` is
    set -- ``applied_change`` = mean|delta_w * adaptive_factor * gain|, the
    per-layer reduction of the ACTUAL applied weight change (spec §4/§8 step 5).
    Unlike ``momentum`` / ``update_ema`` (which stay pre-gain by measured
    decision), this is the truthful "becoming" the observation-only sinks want.

    `sparse_gate` (optional, [out_features]): per-output mask in {0, 1}
    or [0, 1]. When provided, multiplies delta_w by `gate.unsqueeze(1)`
    so gated-off output rows skip their weight / momentum / update_ema
    updates. Lit-followup 2026-05-13: analog of v1's spiking gate in
    continuous error space. Caller (the layer) is responsible for
    computing the gate from error_acc + threshold and for any warmup
    bootstrap. None = no gating (current behavior, bit-identical to
    pre-sparse code path).
    """
    # a. Predict input from output via prediction matrix.
    #    prediction has shape [out, in] per the buffer table; the mapping is
    #    prediction[j, i] = contribution of output j to predicted input i.
    #    The plan's `output_mean @ prediction.T` was a notation slip — the
    #    correct expression matching the [out, in] shape is `output_mean
    #    @ prediction` (no transpose), which produces the [in] target.
    output_mean = output.mean(dim=0)              # [out]
    actual_input = x_flat.mean(dim=0)             # [in]
    predicted_input = output_mean @ prediction    # [in]
    pred_error = actual_input - predicted_input   # [in]

    # Drive normalization (external review 2026-07-28, item 1.1). delta_w is
    # driven by raw reconstruction error, which any layer that is learning
    # drives toward zero -- so the living channel is self-extinguishing by
    # construction: the better the model gets, the less alive the substrate.
    # Measured: update_ema fell 9.5e-5 -> 5.3e-9 monotonically and was still
    # falling at step 72,000; err_acc fell 45x.
    #
    # Dividing by a running RMS makes the drive respond to RELATIVE surprise,
    # so the channel stays alive at any error scale. Same lesson as the
    # 2026-07-21 precision finding one level up: absolute magnitude was the
    # destroyer there too.
    #
    # Applied ONLY to the delta_w path. Precision still EMAs toward 1/err^2 on
    # the raw error (it is an estimate of actual noise), the prediction matrix
    # still learns from raw error (it is a generative model of the real input),
    # and the returned pred_error stays raw so the top-down sweep is unchanged.
    # ---------------------------------------------------------------------
    # 2026-07-29 CORRECTION to the paragraph above. The `rms` mode below is a
    # NO-OP in the production regime, and the reasoning that introduced it was
    # incomplete. Step (b/c) computes
    #     weighted_error = (drive_error * precision).clamp(-1.0, 1.0)
    # and at production precision scales (~1.7e5 median, measured on
    # probe_storefix_512d_seed45) that clamp is 100% saturated -- every single
    # entry. Measured on the real code path with the layer's own stored episode
    # inputs: raw pred_error (rms 0.23) gives frac|w|>=1 = 1.0000, and
    # rms-normalized pred_error gives frac|w|>=1 = 1.0000 with a SIGN-IDENTICAL
    # result. Dividing by a positive scalar cannot change a sign, so under
    # absolute precision weighting the rms mode changes literally nothing.
    #
    # Two consequences worth stating plainly:
    #   * The update in that regime is sign-based:
    #     delta_w = outer(output_mean, sign(pred_error)) * plasticity * pc_rate.
    #     Precision contributes nothing; the clamp has eaten it.
    #   * Any bounded drive is incompatible with absolute-precision weighting.
    #     Making the drive scale-free is necessary but NOT sufficient -- the
    #     trust term has to be bounded too, or the clamp destroys the magnitude
    #     structure the drive was fixed to preserve.
    #
    # Hence `surprise` mode requires `relative_trust`, enforced below rather
    # than silently supplied. See docs/research/2026-07-29_the-drive-fix.md.
    # ---------------------------------------------------------------------
    #
    # `surprise` mode: the drive is normalized EXCESS error -- how much worse
    # than expected this input was, in units of the layer's own recent
    # deviation. Holt's linear method (level + drift + mean abs deviation),
    # the same estimator the episode-store admission rule v3 uses, and for the
    # same reason: a plain EMA without a trend term froze on a decaying signal.
    #
    #     forecast = level + drift          expected error scale
    #     resid    = rms_now - forecast     excess over expectation
    #     gain     = clamp((resid - k*dev) / dev, floor, max)
    #     drive    = pred_error / forecast * gain
    #
    # `k` is a THRESHOLD in deviations, not a divisor. That distinction is the
    # whole gate: an unbiased forecast has resid > 0 on half of all calls by
    # symmetry, so `gain = resid / (k*dev)` fires at ~50% duty on pure noise --
    # measured exactly 0.500 on stationary input, which is a drive responding to
    # noise rather than to novelty. Requiring resid to clear k deviations before
    # any gain applies is the same test the episode-store admission rule v3
    # uses, with the same default (k=3), and for the same reason.
    #
    # Properties, which are the point:
    #   * Uniform improvement: forecast tracks rms_now down together, so the
    #     ratio stays O(1). The channel does not self-extinguish with scale.
    #   * Familiar input: rms_now ~ forecast, resid ~ 0, gain ~ 0. Quiet.
    #   * Novel input: resid large, gain up to gain_max on an O(1)-normalized
    #     error. Full dynamic range available at ANY absolute error scale --
    #     which is exactly what raw error cannot do once it has shrunk.
    #
    # "Quiet when familiar" is intended, not a regression to the dead regime.
    # The difference from dead: a dead drive cannot respond to novelty because
    # its magnitude is gone; this one is quiet but retains full range. That
    # distinction is only credible if it is observable, so `drive_gain_out` and
    # `drive_fire_count` are emitted per layer -- the discriminator the
    # 2026-07-29 review asked for.
    drive_error = pred_error
    mode = drive_mode
    if mode == "raw" and drive_normalize:
        mode = "rms"  # back-compat for the 2026-07-28 flag
    if mode == "rms" and error_rms is not None:
        rms_now = pred_error.detach().pow(2).mean().sqrt()
        error_rms.mul_(1.0 - drive_rms_decay).add_(rms_now, alpha=drive_rms_decay)
        drive_error = pred_error / error_rms.clamp(min=1e-12)
    elif mode == "surprise" and drive_ref is not None:
        if not relative_trust:
            raise ValueError(
                "drive_mode='surprise' requires relative_trust=True. Absolute "
                "precision weighting saturates the +/-1 clamp at 100% in the "
                "production regime (measured), which discards the drive's "
                "magnitude and reduces the update to sign(pred_error) -- "
                "defeating the entire point of a surprise drive. This is "
                "raised rather than silently enabling relative trust, because "
                "a mechanism dependency belongs in the arm config where it can "
                "be attributed, not hidden in a default."
            )
        rms_now = pred_error.detach().pow(2).mean().sqrt()
        forecast = (drive_ref + drive_ref_drift).clamp(min=0.0)
        resid = rms_now - forecast
        b = drive_decay
        # Holt update. Order matters: forecast is built from the OLD level and
        # drift before either is advanced.
        drive_ref.copy_(forecast + b * resid)
        drive_ref_drift.copy_(drive_ref_drift + b * drive_drift_gain * resid)
        drive_dev.copy_(drive_dev + b * (resid.abs() - drive_dev))
        warm = int(drive_calls.item()) < drive_warmup_calls
        drive_calls.add_(1)
        if warm:
            # Statistical warmup: behave EXACTLY like raw while the estimator
            # converges. Same discipline as the episode-store fix -- a new
            # mechanism must never be inert (or wild) before it has data.
            gain = torch.ones_like(drive_ref)
            drive_error = pred_error
        else:
            # Floor dev relative to the level: on very stationary data dev can
            # collapse and make trivial residuals look infinitely surprising.
            dev_eff = torch.maximum(
                drive_dev, drive_dev_floor_frac * forecast
            ).clamp(min=1e-12)
            gain = (
                (resid - drive_surprise_k * dev_eff) / dev_eff
            ).clamp(drive_gain_floor, drive_gain_max)
            drive_error = (
                pred_error / forecast.clamp(min=1e-12).to(pred_error.dtype)
            ) * gain.to(pred_error.dtype)
        if drive_gain_out is not None:
            drive_gain_out.copy_(gain.detach().reshape(()))
        # Count fires only POST-warmup. Warmup holds gain at 1.0 to stay
        # bit-identical to raw, so counting those calls would report a duty
        # cycle of exactly warmup/total and hide the real signal underneath --
        # measured 0.0833 = 50/600 in all three test regimes, including a
        # regime with 19 genuine fires, before this guard.
        # Threshold is the FLOOR, not zero: with drive_gain_floor > 0 the gain
        # never reaches zero, so `gain > 0` would report duty 1.0 always and the
        # instrument would go blind exactly when a run chooses to keep a
        # baseline trickle. Measured 1.0000 on stationary input at floor=0.05
        # before this.
        if drive_fire_count is not None and not warm:
            fired = (gain > drive_gain_floor).reshape(())
            drive_fire_count.add_(fired.to(drive_fire_count.dtype))
            # Accumulate gain magnitude on firing calls only, so
            # mean-gain-when-firing is recoverable independently of how often
            # it fires. Extinction by "fires rarely" and extinction by "fires
            # feebly" are different diagnoses with different fixes.
            if drive_gain_sum is not None:
                drive_gain_sum.add_(
                    torch.where(fired, gain.reshape(()),
                                torch.zeros_like(gain.reshape(())))
                )

    # b/c. Precision-weighted error, clamped per-input.
    #    The clamp mirrors v1's `apply_error` clamp on the local update
    #    (living_layer.py:531) and prevents the runaway weight growth
    #    that the bounded-growth test (refinement 6) surfaced under
    #    large-magnitude input — without input normalization (v1's
    #    `input_avg_mag`, removed in v2), large pred_error can drive
    #    delta_w into a positive-feedback loop where weight growth
    #    amplifies output, which amplifies the next pred_error.
    if relative_trust:
        # Relative trust (v5, 2026-07-21): weight by the RATIO of each
        # input's reliability to the layer's MEDIAN reliability, with
        # precision_min/max reinterpreted as RATIO bounds (an input can
        # earn at most precision_max times the layer-typical trust).
        # Median, not mean: 1/err^2 is tail-amplified (measured 13-22x
        # p95/p5 spreads), so a mean would be dominated by its own tail.
        # Scale-free by construction -- survives lambda/regime changes
        # without re-derivation. Design + measurements: pre-registration
        # doc, 2026-07-21 precision entries.
        trust = (precision / precision.median().clamp(min=1e-12)).clamp(
            precision_min, precision_max
        )
        weighted_error = (drive_error * trust).clamp(-1.0, 1.0)  # [in]
    else:
        weighted_error = (drive_error * precision).clamp(-1.0, 1.0)  # [in]

    # d. Weight delta — outer product of output_mean and weighted error,
    #    scaled by per-input plasticity and the global PC rate.
    delta_w = (
        torch.outer(output_mean, weighted_error)  # [out, in]
        * plasticity                              # broadcasts as [1, in]
        * pc_rate
    )

    # Sparse PC gate: zero out delta_w rows for gated-off outputs.
    # Per-output mask shape [out]; unsqueeze to [out, 1] so it broadcasts
    # against [out, in]. When sparse_gate is None this branch is skipped
    # and the math is bit-identical to the pre-sparse implementation.
    if sparse_gate is not None:
        delta_w = delta_w * sparse_gate.unsqueeze(1)

    # e. Metaplasticity dampening — v1's ratio check, retained for v2.
    #    Refinement 5 verifies this still gates correctly under PC dynamics.
    update_mag = delta_w.abs()
    ratio = update_mag / (update_ema + 1e-8)
    adaptive_factor = (2.0 / (1.0 + ratio)).clamp(max=1.0)

    # f. Apply update; update momentum and update_ema.
    #    Inverted-U learning gain (opt-in, 2026-07-06): multiplies the APPLIED
    #    delta only. The histories below stay PRE-gain (delta_w / update_mag) --
    #    measured decision (gain spec §4): post-gain would inflate update_ema
    #    and weaken the adaptive_factor spike guard by ~2.3x. Reading momentum /
    #    update_ema here (pre-step values) keeps the gain's inputs and
    #    adaptive_factor's inputs untouched, so the guard is bit-identical
    #    gain-on vs off.
    if learning_gain_enabled:
        _gain = learning_gain(
            momentum, update_ema, learning_gain_progress,
            rise=learning_gain_rise, cap=learning_gain_cap,
        )
        applied = delta_w * adaptive_factor * _gain
    else:
        applied = delta_w * adaptive_factor
    weight.add_(applied)
    # Applied-change reduction for the observation-only sinks (spec §8 step 5).
    # Computed only on request so gain-off callers pay no extra host sync and
    # stay byte-identical to legacy. ``applied`` is the exact tensor added to
    # the weight above -- one source of truth, no recomputation.
    applied_change = applied.abs().mean().item() if return_applied_change else None
    momentum.mul_(momentum_decay).add_(
        delta_w, alpha=1.0 - momentum_decay
    )
    update_ema.mul_(update_ema_decay).add_(
        update_mag, alpha=1.0 - update_ema_decay
    )

    # g. Homeostatic regulation.
    homeostatic_force = set_point - weight
    weight.add_(homeostatic_force, alpha=homeostatic_decay)

    # h. Set point adaptation.
    sp_delta = weight - set_point
    set_point.add_(sp_delta, alpha=set_point_adapt_rate)

    # i. Update prediction matrix toward gradient of squared prediction error.
    #    L = 0.5 * ||x_mean - output_mean @ prediction||^2
    #    dL/d(prediction[j, i]) = -output_mean[j] * pred_error[i]
    #    Gradient descent: delta_prediction = +outer(output_mean, pred_error)
    #
    #    Audit 2026-05-11 fix: use the clamped pred_error here too. The
    #    weight update already clamps via weighted_error.clamp(-1, 1).
    #    Using raw pred_error for the prediction update created
    #    inconsistent per-step bounding — the prediction matrix could take
    #    large jumps while the weight matrix couldn't. The per-element
    #    prediction.clamp_ below is a hard backstop but doesn't bound
    #    per-step velocity. Same clamped error for both updates.
    pred_error_clamped = pred_error.clamp(-1.0, 1.0)
    delta_pred = (
        torch.outer(output_mean, pred_error_clamped) * pred_learning_rate
    )
    prediction.add_(delta_pred)
    # Refinement 6: per-element clamp prevents the runaway prediction growth
    # that the bounded-growth test surfaced at large input magnitudes
    # (norm went 0.65 -> 5e3 -> NaN around step 600 without this clamp).
    prediction.clamp_(-prediction_clamp, prediction_clamp)

    # j. Precision EMA toward 1/error² — high precision for reliable inputs,
    #    low for noisy ones. Clamped to [precision_min, precision_max].
    #    The target's denominator floor (1e-3) prevents 1/eps -> inf when
    #    error is small for some input dims; without it, the EMA can
    #    inherit inf even though the running buffer is clamped.
    err_sq = pred_error.pow(2)
    if relative_trust:
        # The eps was the first-stage differentiation destroyer (measured
        # 2026-07-21: err^2 runs 40-1000x SMALLER than the legacy 1e-3
        # floor, flattening a real 13-22x reliability spread to ~1.05x).
        # In relative mode eps is a numerics guard only, and the LEDGER
        # is freed: wide numerics bounds instead of the ratio bounds
        # (the ratio clamp lives at use time, step b/c above). Raw
        # magnitudes stay recorded so absolute instruments keep their
        # calibration even though weighting no longer uses them.
        precision_target = 1.0 / (err_sq + 1e-8)
        precision.mul_(precision_ema_decay).add_(
            precision_target, alpha=1.0 - precision_ema_decay
        )
        precision.clamp_(1e-6, 1e12)
    else:
        precision_target = 1.0 / (err_sq + 1e-3)
        precision.mul_(precision_ema_decay).add_(
            precision_target, alpha=1.0 - precision_ema_decay
        )
        precision.clamp_(precision_min, precision_max)

    # k. error_acc — per-output running prediction-error contribution.
    #    error_acc[j] tracks |output_mean[j]| * mean|pred_error|, so
    #    mean(error_acc) ≈ mean|output| * mean|pred_error| — a salience
    #    signal that is high when both output activation and prediction
    #    error are large. This places the salience scale near v1's
    #    output-magnitude scale (~0.1-1.0), comparable enough that the
    #    episode-store threshold transfers without rescaling.
    #    Refinement 2: salience = mean(error_acc) for episode storage,
    #    monitored for distribution skew (fall back to L2 norm if mean
    #    saturates near zero or drifts toward a single output channel).
    output_mag = output_mean.abs()                # [out]
    pred_err_mag = pred_error.abs().mean()        # scalar
    per_output_contrib = output_mag * pred_err_mag  # [out]
    error_acc.mul_(update_ema_decay).add_(
        per_output_contrib, alpha=1.0 - update_ema_decay
    )

    if return_applied_change:
        return error_acc.mean().item(), pred_error.detach().clone(), applied_change
    return error_acc.mean().item(), pred_error.detach().clone()


# ---------------------------------------------------------------------------
# Public dispatch — C++ if available, else Python fallback
# ---------------------------------------------------------------------------

def pc_self_modify(
    weight: torch.Tensor,
    prediction: torch.Tensor,
    set_point: torch.Tensor,
    momentum: torch.Tensor,
    update_ema: torch.Tensor,
    precision: torch.Tensor,
    error_acc: torch.Tensor,
    plasticity: torch.Tensor,
    x_flat: torch.Tensor,
    output: torch.Tensor,
    pc_rate: float,
    pred_learning_rate: float,
    homeostatic_decay: float,
    set_point_adapt_rate: float,
    momentum_decay: float,
    update_ema_decay: float,
    precision_ema_decay: float,
    precision_min: float,
    precision_max: float,
    prediction_clamp: float,
    relative_trust: bool = False,
    drive_normalize: bool = False,
    error_rms: torch.Tensor | None = None,
    drive_rms_decay: float = 0.01,
    drive_mode: str = "raw",
    drive_ref: torch.Tensor | None = None,
    drive_ref_drift: torch.Tensor | None = None,
    drive_dev: torch.Tensor | None = None,
    drive_calls: torch.Tensor | None = None,
    drive_gain_out: torch.Tensor | None = None,
    drive_fire_count: torch.Tensor | None = None,
    drive_gain_sum: torch.Tensor | None = None,
    drive_decay: float = 0.01,
    drive_drift_gain: float = 0.1,
    drive_surprise_k: float = 3.0,
    drive_gain_max: float = 4.0,
    drive_gain_floor: float = 0.0,
    drive_dev_floor_frac: float = 0.01,
    drive_warmup_calls: int = 200,
    sparse_gate: torch.Tensor | None = None,
    learning_gain_enabled: bool = False,
    learning_gain_progress: float = 0.0,
    learning_gain_rise: float = 2.0,
    learning_gain_cap: float = 3.0,
    return_applied_change: bool = False,
) -> tuple[float, torch.Tensor] | tuple[float, torch.Tensor, float]:
    """One PC self-modification step.

    Dispatches to the C++ extension when it's loaded; otherwise routes to
    the pure-Python fallback. Both paths now handle `sparse_gate` —
    when present, per-output mask rows where the gate is 0 do not get a
    weight update this step. All buffers modified in-place. Returns:
      - salience scalar = mean(error_acc) per refinement 2.
      - pred_error tensor [in_features], the per-input prediction error
        for this step. Caller (the layer's forward) stores this on
        `_last_pred_error` for the inter-block top-down sweep.
      - applied_change float (ONLY when `return_applied_change=True`):
        mean|delta_w * adaptive_factor * gain|, the truthful applied-change
        reduction for the observation-only sinks (spec §8 step 5).

    `return_applied_change=True` forces the Python path (the C++ extension
    cannot surface `delta_w`). In practice it is only requested together with
    `learning_gain_enabled=True`, which already forces Python, so this costs
    nothing extra: the applied-change signal diverges from pre-gain momentum
    only when the gain is on.
    """
    # The C++ extension now implements the inverted-U learning gain and the
    # applied-change reduction at parity with the Python reference (spec §8
    # step 7, verified by tests/test_pc_ops_gain_parity.py), so the gain runs
    # on the fast path too. When C++ is loaded it handles every case; the
    # Python fallback covers hosts without a compiler.
    # Drive normalization (item 1.1) and the surprise drive (2026-07-29) exist
    # only in the Python reference for now, so requesting either forces the
    # Python path rather than silently running unmodified in C++ -- a flag that
    # appears to be on while the fast path ignores it is exactly the
    # silent-success failure this project keeps finding.
    #
    # Measured cost of that fallback, 2026-07-29, 2048x2048 layer, batch 32:
    # DirectML 4.34 ms/call (C++) vs 5.04 ms/call (Python surprise) = 1.16x;
    # CPU 24.87 vs 25.68 ms = 1.03x. The module docstring's "~50x slower"
    # dates from 2026-05-10 and does NOT apply to this path -- it is not a
    # reason to defer a run. Porting to csrc/pc_ops.cpp is still worth doing,
    # but as an optimization, not a prerequisite.
    #
    # Note for whoever ports it: `precision.median()` in the relative-trust
    # branch has no DML kernel and silently falls back to CPU every call. That
    # applies to the whole v5 family too, not just this path.
    if _use_cpp and not drive_normalize and drive_mode == "raw":
        salience_tensor, pred_error, applied_change = _cpp_ops.pc_self_modify(
            weight, prediction, set_point, momentum, update_ema,
            precision, error_acc, plasticity, x_flat, output,
            pc_rate, pred_learning_rate, homeostatic_decay,
            set_point_adapt_rate, momentum_decay, update_ema_decay,
            precision_ema_decay, precision_min, precision_max,
            prediction_clamp,
            relative_trust=relative_trust,
            sparse_gate=sparse_gate,
            learning_gain_enabled=learning_gain_enabled,
            learning_gain_progress=learning_gain_progress,
            learning_gain_rise=learning_gain_rise,
            learning_gain_cap=learning_gain_cap,
            return_applied_change=return_applied_change,
        )
        if return_applied_change:
            return salience_tensor.item(), pred_error, applied_change.item()
        return salience_tensor.item(), pred_error
    return _pc_self_modify_python(
        weight, prediction, set_point, momentum, update_ema,
        precision, error_acc, plasticity, x_flat, output,
        pc_rate, pred_learning_rate, homeostatic_decay,
        set_point_adapt_rate, momentum_decay, update_ema_decay,
        precision_ema_decay, precision_min, precision_max,
        prediction_clamp, relative_trust=relative_trust,
        drive_normalize=drive_normalize,
        error_rms=error_rms,
        drive_rms_decay=drive_rms_decay,
        drive_mode=drive_mode,
        drive_ref=drive_ref,
        drive_ref_drift=drive_ref_drift,
        drive_dev=drive_dev,
        drive_calls=drive_calls,
        drive_gain_out=drive_gain_out,
        drive_fire_count=drive_fire_count,
        drive_gain_sum=drive_gain_sum,
        drive_decay=drive_decay,
        drive_drift_gain=drive_drift_gain,
        drive_surprise_k=drive_surprise_k,
        drive_gain_max=drive_gain_max,
        drive_gain_floor=drive_gain_floor,
        drive_dev_floor_frac=drive_dev_floor_frac,
        drive_warmup_calls=drive_warmup_calls,
        sparse_gate=sparse_gate,
        learning_gain_enabled=learning_gain_enabled,
        learning_gain_progress=learning_gain_progress,
        learning_gain_rise=learning_gain_rise,
        learning_gain_cap=learning_gain_cap,
        return_applied_change=return_applied_change,
    )
