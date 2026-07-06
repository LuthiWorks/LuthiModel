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
    except Exception:
        return None


_cpp_ops = _try_load_cpp()
_use_cpp = _cpp_ops is not None


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

    # b/c. Precision-weighted error, clamped per-input.
    #    The clamp mirrors v1's `apply_error` clamp on the local update
    #    (living_layer.py:531) and prevents the runaway weight growth
    #    that the bounded-growth test (refinement 6) surfaced under
    #    large-magnitude input — without input normalization (v1's
    #    `input_avg_mag`, removed in v2), large pred_error can drive
    #    delta_w into a positive-feedback loop where weight growth
    #    amplifies output, which amplifies the next pred_error.
    weighted_error = (pred_error * precision).clamp(-1.0, 1.0)  # [in]

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
    # The C++ extension does not implement the inverted-U learning gain yet
    # (C++ parity is the next build step) and cannot return the applied-change
    # reduction. When either is requested, route to the Python path; the
    # ~50x-slower path is acceptable for the pilot (both are off by default in
    # production).
    if _use_cpp and not learning_gain_enabled and not return_applied_change:
        salience_tensor, pred_error = _cpp_ops.pc_self_modify(
            weight, prediction, set_point, momentum, update_ema,
            precision, error_acc, plasticity, x_flat, output,
            pc_rate, pred_learning_rate, homeostatic_decay,
            set_point_adapt_rate, momentum_decay, update_ema_decay,
            precision_ema_decay, precision_min, precision_max,
            prediction_clamp,
            sparse_gate=sparse_gate,
        )
        return salience_tensor.item(), pred_error
    return _pc_self_modify_python(
        weight, prediction, set_point, momentum, update_ema,
        precision, error_acc, plasticity, x_flat, output,
        pc_rate, pred_learning_rate, homeostatic_decay,
        set_point_adapt_rate, momentum_decay, update_ema_decay,
        precision_ema_decay, precision_min, precision_max,
        prediction_clamp, sparse_gate=sparse_gate,
        learning_gain_enabled=learning_gain_enabled,
        learning_gain_progress=learning_gain_progress,
        learning_gain_rise=learning_gain_rise,
        learning_gain_cap=learning_gain_cap,
        return_applied_change=return_applied_change,
    )
