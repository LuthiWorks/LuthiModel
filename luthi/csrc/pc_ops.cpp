/**
 * Fused predictive-coding self-modification in C++ for v2.
 *
 * Replaces ~25 separate Python-level tensor operations with a single C++
 * function call. Mirrors the v1 living_ops.cpp pattern but implements PC
 * dynamics: predict input from output, weight error by precision, update
 * weight + prediction matrix from outer-product of output_mean and pred_error.
 *
 * Uses the torch::Tensor API exclusively — no CUDA, no vendor-specific code.
 * Operations dispatch to whatever backend the tensors live on (DirectML,
 * ROCm, CUDA, MPS, CPU). Per Luthi CLAUDE.md design decision 15, no .item()
 * inside the C++ extension when running on DirectML — return tensors and
 * let Python do the host sync.
 *
 * Build via torch.utils.cpp_extension.load() from luthi/v2/pc_ops.py.
 */

#include <torch/extension.h>

using torch::Tensor;

/**
 * Fused PC self-modification for PredictiveCodingLayer.
 *
 * Performs all per-forward-pass self-modification in a single call, matching
 * the steps in `_pc_self_modify_python` exactly:
 *   a. Predict input from output via prediction matrix
 *   b/c. Precision-weighted error, clamped per-input
 *   d. Weight delta — outer(output_mean, weighted_error) * plasticity * pc_rate
 *      d2. Sparse PC gate — when `sparse_gate` is present, zero out delta_w
 *          rows for gated-off outputs. Same broadcast multiply as the Python
 *          reference's `delta_w = delta_w * sparse_gate.unsqueeze(1)`.
 *   e. Metaplasticity dampening (v1 ratio check, retained)
 *   f. Apply update; update momentum and update_ema
 *   g. Homeostatic regulation (decay toward set point)
 *   h. Set point adaptation (slow track of weight)
 *   i. Update prediction matrix toward gradient of squared prediction error
 *      (clamped to ±prediction_clamp per refinement 6)
 *   j. Precision EMA toward 1/error² (clamped to [min, max])
 *   k. error_acc — per-output running prediction-error contribution
 *
 * All living-state buffers modified in-place. Returns:
 *   - salience: 0-dim tensor (mean(error_acc)) — caller does .item()
 *   - pred_error: [in_features] tensor — caller stores on layer for the
 *     inter-block top-down sweep (see hybrid_block_pc.top_down_pass)
 *   - applied_change: 0-dim tensor mean|delta_w * adaptive_factor * gain| when
 *     return_applied_change is set (else a zero placeholder the caller ignores)
 *     — the truthful applied-change reduction for the observation-only sinks
 *     (spec §8 step 5). No .item() here (design decision 15); caller syncs.
 *
 * Inverted-U learning gain (opt-in, spec §8 step 7 — C++ parity): mirrors
 * luthi.v2.pc_ops.learning_gain exactly. A pure per-weight AMPLIFIER in
 * [1.0, cap]; multiplies the APPLIED delta only. momentum / update_ema stay
 * PRE-gain (record delta_w / |delta_w|), so adaptive_factor's inputs are
 * bit-identical gain-on vs off (the measured spike-guard guarantee, spec §4).
 */
std::tuple<Tensor, Tensor, Tensor> pc_self_modify(
    Tensor weight,
    Tensor prediction,
    Tensor set_point,
    Tensor momentum,
    Tensor update_ema,
    Tensor precision,
    Tensor error_acc,
    Tensor plasticity,
    Tensor x_flat,
    Tensor output,
    double pc_rate,
    double pred_learning_rate,
    double homeostatic_decay,
    double set_point_adapt_rate,
    double momentum_decay,
    double update_ema_decay,
    double precision_ema_decay,
    double precision_min,
    double precision_max,
    double prediction_clamp,
    c10::optional<Tensor> sparse_gate,
    bool learning_gain_enabled,
    double learning_gain_progress,
    double learning_gain_rise,
    double learning_gain_cap,
    bool return_applied_change
) {
    // a. Predict input from output via prediction matrix.
    //    prediction is [out, in]; output_mean @ prediction = [in].
    auto output_mean = output.mean(0);              // [out]
    auto actual_input = x_flat.mean(0);             // [in]
    auto predicted_input = output_mean.matmul(prediction);  // [in]
    auto pred_error = actual_input - predicted_input;       // [in]

    // b/c. Precision-weighted error, clamped per-input dim.
    //    Mirrors v1 apply_error's local-update clamp; prevents the runaway
    //    weight growth that the bounded-growth test (refinement 6) found
    //    when input_avg_mag normalization (v1 only) is absent.
    auto weighted_error = (pred_error * precision).clamp(-1.0, 1.0);  // [in]

    // d. Weight delta = outer(output_mean, weighted_error) * plasticity * pc_rate.
    //    plasticity is [in], broadcasts as [1, in] against [out, in].
    auto delta_w = torch::outer(output_mean, weighted_error) * plasticity * pc_rate;

    // d2. Sparse PC gate — per-output mask in {0, 1}; rows where the gate is
    //     0 do not get a weight update this step. The mask is [out_features];
    //     unsqueeze to [out, 1] so it broadcasts against [out, in]. This
    //     mirrors the Python reference at pc_ops.py:138-139 and produces
    //     bit-identical results under the same float-op ordering.
    if (sparse_gate.has_value()) {
        delta_w = delta_w * sparse_gate.value().unsqueeze(1);
    }

    // e. Metaplasticity dampening — v1's ratio check, retained for v2 (M1
    //    refinement 5 verified the semantics still hold under PC dynamics).
    auto update_mag = delta_w.abs();
    auto ratio = update_mag / (update_ema + 1e-8);
    auto adaptive_factor = (2.0 / (1.0 + ratio)).clamp_max(1.0);

    // f. Apply update; update momentum and update_ema (standard EMA on both).
    //    Inverted-U gain (opt-in): multiplies the APPLIED delta only. The
    //    histories below stay PRE-gain (delta_w / update_mag), mirroring the
    //    Python reference and keeping adaptive_factor bit-identical gain-on vs
    //    off. `applied` is the exact tensor added to the weight — one source
    //    of truth for both the update and the applied-change reduction.
    Tensor applied;
    if (learning_gain_enabled) {
        // learning_gain: coherence = |momentum| / (update_ema + eps);
        // fall = clamp(1 - progress, min=0) as an explicit comparison so a NaN
        // progress fails `< 1.0` and yields fall=0 (fail-safe to legacy);
        // gain = clamp(1 + rise * coherence * fall, 1.0, cap).
        auto coherence = momentum.abs() / (update_ema + 1e-8);
        double fall = (learning_gain_progress < 1.0)
            ? (1.0 - learning_gain_progress) : 0.0;
        auto gain = (1.0 + learning_gain_rise * coherence * fall)
            .clamp(1.0, learning_gain_cap);
        applied = delta_w * adaptive_factor * gain;
    } else {
        applied = delta_w * adaptive_factor;
    }
    weight.add_(applied);
    momentum.mul_(momentum_decay).add_(delta_w, 1.0 - momentum_decay);
    update_ema.mul_(update_ema_decay).add_(update_mag, 1.0 - update_ema_decay);

    // g. Homeostatic regulation (same as v1).
    auto homeostatic_force = set_point - weight;
    weight.add_(homeostatic_force, homeostatic_decay);

    // h. Set point adaptation (same as v1).
    auto sp_delta = weight - set_point;
    set_point.add_(sp_delta, set_point_adapt_rate);

    // i. Update prediction matrix toward gradient of squared prediction error.
    //    Audit 2026-05-11 fix: use clamped pred_error here too (the weight
    //    update already clamps via weighted_error). Inconsistent bounding
    //    let prediction take per-step jumps the weight matrix couldn't.
    auto pred_error_clamped = pred_error.clamp(-1.0, 1.0);
    auto delta_pred = torch::outer(output_mean, pred_error_clamped) * pred_learning_rate;
    prediction.add_(delta_pred);
    // Refinement 6: per-element clamp prevents the runaway prediction
    // growth that the M1 bounded-growth test surfaced under large input.
    prediction.clamp_(-prediction_clamp, prediction_clamp);

    // j. Precision EMA toward 1/error² — high precision for reliable inputs,
    //    low for noisy ones. Denominator floor 1e-3 (not 1e-8) prevents inf
    //    in the EMA target when err is small for some input dims; otherwise
    //    the running buffer can inherit inf even though the clamp would
    //    catch the final value.
    auto err_sq = pred_error.pow(2);
    auto precision_target = 1.0 / (err_sq + 1e-3);
    precision.mul_(precision_ema_decay).add_(
        precision_target, 1.0 - precision_ema_decay
    );
    precision.clamp_(precision_min, precision_max);

    // k. error_acc — per-output running prediction-error contribution.
    //    error_acc[j] tracks |output_mean[j]| * mean|pred_error|; mean of
    //    error_acc gives a salience scalar near v1's output-magnitude
    //    scale (refinement 2). Episode-store threshold transfers without
    //    rescaling.
    auto output_mag = output_mean.abs();             // [out]
    auto pred_err_mag = pred_error.abs().mean();     // scalar
    auto per_output_contrib = output_mag * pred_err_mag;
    error_acc.mul_(update_ema_decay).add_(
        per_output_contrib, 1.0 - update_ema_decay
    );

    auto salience = error_acc.mean();  // 0-dim tensor — caller does .item()
    // Applied-change reduction (spec §8 step 5): computed only when requested,
    // so legacy callers pay no extra reduction. A zero placeholder otherwise;
    // the dispatcher ignores it unless return_applied_change is set.
    Tensor applied_change = return_applied_change
        ? applied.abs().mean()
        : torch::zeros({}, weight.options());
    return std::make_tuple(salience, pred_error, applied_change);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "pc_self_modify",
        &pc_self_modify,
        "Fused predictive-coding self-modification (backend-agnostic)",
        py::arg("weight"),
        py::arg("prediction"),
        py::arg("set_point"),
        py::arg("momentum"),
        py::arg("update_ema"),
        py::arg("precision"),
        py::arg("error_acc"),
        py::arg("plasticity"),
        py::arg("x_flat"),
        py::arg("output"),
        py::arg("pc_rate"),
        py::arg("pred_learning_rate"),
        py::arg("homeostatic_decay"),
        py::arg("set_point_adapt_rate"),
        py::arg("momentum_decay"),
        py::arg("update_ema_decay"),
        py::arg("precision_ema_decay"),
        py::arg("precision_min"),
        py::arg("precision_max"),
        py::arg("prediction_clamp"),
        py::arg("sparse_gate") = c10::nullopt,
        py::arg("learning_gain_enabled") = false,
        py::arg("learning_gain_progress") = 0.0,
        py::arg("learning_gain_rise") = 2.0,
        py::arg("learning_gain_cap") = 3.0,
        py::arg("return_applied_change") = false
    );
}
