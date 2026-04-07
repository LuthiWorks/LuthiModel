/**
 * Fused living weight self-modification in C++.
 *
 * Replaces ~20 separate Python-level tensor operations with a single C++
 * function call. Uses the torch::Tensor API exclusively — no CUDA, no
 * vendor-specific code. Operations dispatch to whatever backend the tensors
 * live on (CUDA, ROCm, MPS, DirectML, CPU).
 *
 * Build with torch.utils.cpp_extension or include in setup.py.
 */

#include <torch/extension.h>

using torch::Tensor;
using c10::optional;

/**
 * Fused Hebbian self-modification for LivingLayerV6.
 *
 * Performs all per-forward-pass self-modification in a single call:
 *   1. Excitability factor computation (sigmoid mapping)
 *   2. Salience computation (mean absolute activation)
 *   3. Input magnitude EMA update (synaptic scaling)
 *   4. Hebbian update with metaplasticity gating
 *   5. Activity-dependent gating (optional, for spiking layers)
 *   6. Weight update
 *   7. Momentum update
 *   8. Homeostatic regulation (decay toward set point)
 *   9. Set point adaptation
 *  10. Excitability dynamics (sensitization/habituation)
 *
 * All tensors are modified in-place. Returns salience_scalar for episode
 * storage decisions (handled in Python).
 *
 * @param weight           [out, in] Living weight matrix
 * @param set_point        [out, in] Homeostatic set points
 * @param momentum         [out, in] Update momentum EMA
 * @param input_avg_mag    [in] Running average of input magnitudes
 * @param excitability_acc [out, in] Excitability accumulator
 * @param plasticity       [out, in] Per-weight learning rate multiplier
 * @param update_ema       [out, in] Metaplasticity update history
 * @param x_flat           [N, in] Flattened input (read-only)
 * @param output           [N, out] Layer output (read-only)
 * @param gate             Optional [out, in] activity-dependent gate (spike mask)
 * @param hebb_rate        Hebbian learning rate
 * @param homeostatic_decay Decay rate toward set point
 * @param set_point_adapt_rate Set point adaptation rate
 * @param momentum_decay   Momentum EMA decay
 * @param input_mag_decay  Input magnitude EMA decay
 * @param salience_threshold Threshold for excitability sensitization
 * @param excitability_min Minimum effective excitability
 * @param excitability_max Maximum effective excitability
 * @param update_ema_decay Metaplasticity EMA decay
 * @param eval_hebb_fraction Hebbian reduction factor in eval mode
 * @param training         Whether model is in training mode
 * @return salience_scalar as a 0-dim tensor (caller does .item() in Python)
 */
Tensor living_self_modify(
    Tensor weight,
    Tensor set_point,
    Tensor momentum,
    Tensor input_avg_mag,
    Tensor excitability_acc,
    Tensor plasticity,
    Tensor update_ema,
    Tensor x_flat,
    Tensor output,
    optional<Tensor> gate,
    double hebb_rate,
    double homeostatic_decay,
    double set_point_adapt_rate,
    double momentum_decay,
    double input_mag_decay,
    double salience_threshold,
    double excitability_min,
    double excitability_max,
    double update_ema_decay,
    double eval_hebb_fraction,
    bool training
) {
    // 1. Excitability factor: sigmoid mapping of accumulator
    double exc_span = excitability_max - excitability_min;
    auto exc = excitability_min + exc_span * torch::sigmoid(excitability_acc);

    // 2. Salience: mean absolute activation per output dimension
    auto salience_per_dim = output.abs().mean(0);  // [out]
    // Return salience as a tensor — .item() in Python avoids DirectML sync issues
    auto salience_scalar = salience_per_dim.mean();  // 0-dim tensor

    // 3. Input magnitude tracking (synaptic scaling)
    auto input_mag = x_flat.abs().mean(0);  // [in]
    input_avg_mag.mul_(input_mag_decay).add_(input_mag, 1.0 - input_mag_decay);

    // 4. Normalized input for Hebbian update
    auto normalized_input = x_flat / input_avg_mag.clamp_min(0.01);

    // 5. Per-weight salience (normalized by mean weight magnitude)
    auto weight_abs_mean = weight.abs().mean().clamp_min(0.01);
    auto weight_norm = weight.abs() / weight_abs_mean;
    auto per_weight_salience = weight_norm * x_flat.abs().mean(0).unsqueeze(0);

    // 6. Input signal: mean normalized input, broadcast to [1, in]
    auto input_signal = normalized_input.mean(0).unsqueeze(0);

    // 7. Hebbian update
    auto hebb_update = input_signal * per_weight_salience * exc * plasticity * hebb_rate;

    // 8. Metaplasticity: adaptive gating of unusually large updates
    auto update_mag = hebb_update.abs();
    auto ratio = update_mag / (update_ema + 1e-8);
    auto adaptive_factor = (2.0 / (1.0 + ratio)).clamp_max(1.0);

    // 9-12. Gate-dependent operations
    if (gate.has_value()) {
        auto g = gate.value();

        // Update EMA (delta formulation for gated mode)
        update_ema.add_(
            (update_mag - update_ema) * (1.0 - update_ema_decay) * g
        );

        // Eval mode reduction
        if (!training) {
            adaptive_factor = adaptive_factor * eval_hebb_fraction;
        }

        // Gated weight update
        weight.add_(hebb_update * adaptive_factor * g);

        // Gated momentum update (delta formulation)
        momentum.add_(
            (hebb_update - momentum) * (1.0 - momentum_decay) * g
        );
    } else {
        // Update EMA (standard EMA for ungated mode)
        update_ema.mul_(update_ema_decay).add_(
            update_mag, 1.0 - update_ema_decay
        );

        // Eval mode reduction
        if (!training) {
            adaptive_factor = adaptive_factor * eval_hebb_fraction;
        }

        // Ungated weight update
        weight.add_(hebb_update * adaptive_factor);

        // Ungated momentum update (standard EMA)
        momentum.mul_(momentum_decay).add_(
            hebb_update, 1.0 - momentum_decay
        );
    }

    // 10. Homeostatic regulation: decay toward set point
    auto homeostatic_force = set_point - weight;
    weight.add_(homeostatic_force, homeostatic_decay);

    // 11. Set point adaptation: slowly track current weight position
    auto sp_delta = weight - set_point;
    set_point.add_(sp_delta, set_point_adapt_rate);

    // 12. Excitability dynamics: sensitize on high salience, habituate on low
    auto salience_broadcast = salience_per_dim.unsqueeze(1).expand_as(
        excitability_acc
    );
    auto exc_update = torch::where(
        salience_broadcast > salience_threshold,
        torch::full_like(excitability_acc, 0.01),
        torch::full_like(excitability_acc, -0.005)
    );
    excitability_acc.add_(exc_update);

    return salience_scalar;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "living_self_modify",
        &living_self_modify,
        "Fused living weight self-modification (backend-agnostic)",
        py::arg("weight"),
        py::arg("set_point"),
        py::arg("momentum"),
        py::arg("input_avg_mag"),
        py::arg("excitability_acc"),
        py::arg("plasticity"),
        py::arg("update_ema"),
        py::arg("x_flat"),
        py::arg("output"),
        py::arg("gate"),
        py::arg("hebb_rate"),
        py::arg("homeostatic_decay"),
        py::arg("set_point_adapt_rate"),
        py::arg("momentum_decay"),
        py::arg("input_mag_decay"),
        py::arg("salience_threshold"),
        py::arg("excitability_min"),
        py::arg("excitability_max"),
        py::arg("update_ema_decay"),
        py::arg("eval_hebb_fraction"),
        py::arg("training")
    );
}
