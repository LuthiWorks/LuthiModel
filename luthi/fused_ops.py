"""Fused living weight operations with optional C++ acceleration.

Provides a unified interface for the self-modification block in
LivingLayerV6.forward(). Tries to JIT-compile the C++ extension
on first import; falls back to a pure Python implementation if
no C++ compiler is available.

Usage:
    from luthi.fused_ops import living_self_modify
    salience = living_self_modify(weight, set_point, ...)
"""

import os
import torch
from typing import Optional

# ---------------------------------------------------------------------------
# Try to load or compile C++ extension
# ---------------------------------------------------------------------------

_cpp_ops = None
_use_cpp = False


def _setup_msvc_env():
    """On Windows, activate MSVC environment if cl.exe isn't in PATH.

    Visual Studio Build Tools install cl.exe but require vcvarsall.bat
    to set PATH, INCLUDE, LIB, etc. This function finds and activates
    the environment, returning an env dict for subprocess use, or
    modifying os.environ directly for in-process compilation.
    """
    import shutil
    if shutil.which("cl") is not None:
        return  # Already available

    import subprocess, glob

    # Find vcvarsall.bat
    patterns = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\*\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\*\VC\Auxiliary\Build\vcvarsall.bat",
    ]
    vcvarsall = None
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            vcvarsall = matches[0]
            break

    if vcvarsall is None:
        return

    # Run vcvarsall and capture the resulting environment
    try:
        result = subprocess.run(
            f'cmd /c "call "{vcvarsall}" x64 && set"',
            shell=True, capture_output=True, text=True, timeout=60
        )
        for line in result.stdout.splitlines():
            if "=" in line:
                key, _, val = line.partition("=")
                os.environ[key] = val
    except Exception:
        pass


def _try_load_cpp():
    """Attempt JIT compilation of the C++ extension. Returns module or None."""
    try:
        from torch.utils.cpp_extension import load

        src_dir = os.path.join(os.path.dirname(__file__), "csrc")
        src_file = os.path.join(src_dir, "living_ops.cpp")

        if not os.path.exists(src_file):
            return None

        # Ensure MSVC environment is set up on Windows
        if os.name == "nt":
            _setup_msvc_env()

        module = load(
            name="living_ops",
            sources=[src_file],
            verbose=False,
        )
        return module
    except Exception:
        return None


# Attempt compilation at import time (cached by torch after first build)
_cpp_ops = _try_load_cpp()
_use_cpp = _cpp_ops is not None


# ---------------------------------------------------------------------------
# Pure Python fallback (identical math to the C++ version)
# ---------------------------------------------------------------------------

def _living_self_modify_python(
    weight: torch.Tensor,
    set_point: torch.Tensor,
    momentum: torch.Tensor,
    input_avg_mag: torch.Tensor,
    excitability_acc: torch.Tensor,
    plasticity: torch.Tensor,
    update_ema: torch.Tensor,
    x_flat: torch.Tensor,
    output: torch.Tensor,
    gate: Optional[torch.Tensor],
    hebb_rate: float,
    homeostatic_decay: float,
    set_point_adapt_rate: float,
    momentum_decay: float,
    input_mag_decay: float,
    salience_threshold: float,
    excitability_min: float,
    excitability_max: float,
    update_ema_decay: float,
    eval_hebb_fraction: float,
    training: bool,
) -> float:
    """Pure Python implementation of fused self-modification.

    All tensors are modified in-place. Returns salience_scalar.
    """
    # 1. Excitability factor
    exc_span = excitability_max - excitability_min
    exc = excitability_min + exc_span * torch.sigmoid(excitability_acc)

    # 2. Salience
    salience_per_dim = output.abs().mean(dim=0)  # [out]
    salience_scalar = salience_per_dim.mean().item()

    # 3. Input magnitude tracking
    input_mag = x_flat.abs().mean(dim=0)  # [in]
    input_avg_mag.mul_(input_mag_decay).add_(
        input_mag, alpha=1.0 - input_mag_decay
    )

    # 4. Normalized input
    normalized_input = x_flat / input_avg_mag.clamp(min=0.01)

    # 5. Per-weight salience
    weight_norm = weight.abs() / weight.abs().mean().clamp(min=0.01)
    per_weight_salience = weight_norm * x_flat.abs().mean(dim=0).unsqueeze(0)

    # 6. Input signal
    input_signal = normalized_input.mean(dim=0).unsqueeze(0)  # [1, in]

    # 7. Hebbian update
    hebb_update = (
        input_signal * per_weight_salience * exc * plasticity * hebb_rate
    )

    # 8. Metaplasticity
    update_mag = hebb_update.abs()
    ratio = update_mag / (update_ema + 1e-8)
    adaptive_factor = (2.0 / (1.0 + ratio)).clamp(max=1.0)

    # 9. Gate-dependent operations
    if gate is not None:
        update_ema.add_(
            (update_mag - update_ema) * (1.0 - update_ema_decay) * gate
        )
        if not training:
            adaptive_factor = adaptive_factor * eval_hebb_fraction
        weight.add_(hebb_update * adaptive_factor * gate)
        momentum.add_(
            (hebb_update - momentum) * (1.0 - momentum_decay) * gate
        )
    else:
        update_ema.mul_(update_ema_decay).add_(
            update_mag, alpha=1.0 - update_ema_decay
        )
        if not training:
            adaptive_factor = adaptive_factor * eval_hebb_fraction
        weight.add_(hebb_update * adaptive_factor)
        momentum.mul_(momentum_decay).add_(
            hebb_update, alpha=1.0 - momentum_decay
        )

    # 10. Homeostatic regulation
    homeostatic_force = set_point - weight
    weight.add_(homeostatic_force, alpha=homeostatic_decay)

    # 11. Set point adaptation
    sp_delta = weight - set_point
    set_point.add_(sp_delta, alpha=set_point_adapt_rate)

    # 12. Excitability dynamics
    salience_broadcast = salience_per_dim.unsqueeze(1).expand_as(
        excitability_acc
    )
    exc_update = torch.where(
        salience_broadcast > salience_threshold,
        torch.tensor(0.01, device=weight.device, dtype=weight.dtype),
        torch.tensor(-0.005, device=weight.device, dtype=weight.dtype),
    )
    excitability_acc.add_(exc_update)

    return salience_scalar


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def living_self_modify(
    weight: torch.Tensor,
    set_point: torch.Tensor,
    momentum: torch.Tensor,
    input_avg_mag: torch.Tensor,
    excitability_acc: torch.Tensor,
    plasticity: torch.Tensor,
    update_ema: torch.Tensor,
    x_flat: torch.Tensor,
    output: torch.Tensor,
    gate: Optional[torch.Tensor],
    hebb_rate: float,
    homeostatic_decay: float,
    set_point_adapt_rate: float,
    momentum_decay: float,
    input_mag_decay: float,
    salience_threshold: float,
    excitability_min: float,
    excitability_max: float,
    update_ema_decay: float,
    eval_hebb_fraction: float,
    training: bool,
) -> float:
    """Fused self-modification for living weights.

    Dispatches to C++ extension if available, otherwise pure Python.
    All tensors modified in-place. Returns salience_scalar for
    episode storage.
    """
    if _use_cpp:
        # C++ returns a 0-dim tensor to avoid .item() deadlock on DirectML
        salience_tensor = _cpp_ops.living_self_modify(
            weight, set_point, momentum, input_avg_mag,
            excitability_acc, plasticity, update_ema,
            x_flat, output, gate,
            hebb_rate, homeostatic_decay, set_point_adapt_rate,
            momentum_decay, input_mag_decay, salience_threshold,
            excitability_min, excitability_max, update_ema_decay,
            eval_hebb_fraction, training,
        )
        return salience_tensor.item()
    else:
        return _living_self_modify_python(
            weight, set_point, momentum, input_avg_mag,
            excitability_acc, plasticity, update_ema,
            x_flat, output, gate,
            hebb_rate, homeostatic_decay, set_point_adapt_rate,
            momentum_decay, input_mag_decay, salience_threshold,
            excitability_min, excitability_max, update_ema_decay,
            eval_hebb_fraction, training,
        )


def is_cpp_available() -> bool:
    """Check if C++ fused ops are compiled and loaded."""
    return _use_cpp
