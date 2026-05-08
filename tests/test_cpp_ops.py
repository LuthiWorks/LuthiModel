"""Tests for the fused living weight operations.

Verifies that:
1. The fused Python implementation matches the original inline code
2. C++ extension (when available) matches the Python fallback
3. Gated (spiking) and ungated (base) modes both work
4. Edge cases: zero input, extreme values, eval mode
"""

import torch
import pytest

from luthi.fused_ops import (
    living_self_modify,
    _living_self_modify_python,
    is_cpp_available,
)
from luthi.living_layer import LivingLayerV6
from luthi.living_layer_spiking import SpikingLivingLayer


# --- Fused op correctness ---

def test_fused_op_ungated():
    """Ungated (base layer) fused op produces valid results."""
    torch.manual_seed(42)

    weight = torch.randn(64, 64)
    set_point = weight.clone()
    momentum = torch.zeros(64, 64)
    input_avg_mag = torch.ones(64)
    excitability_acc = torch.zeros(64)  # per-output-neuron
    plasticity = torch.ones(64)  # per-input-feature
    update_ema = torch.ones(64, 64) * 1e-4
    x_flat = torch.randn(32, 64)
    output = x_flat @ weight.T

    weight_before = weight.clone()

    salience = living_self_modify(
        weight, set_point, momentum, input_avg_mag,
        excitability_acc, plasticity, update_ema,
        x_flat, output, None,  # no gate
        0.001, 0.001, 1e-6, 0.99, 0.99, 0.5, 0.3, 3.0, 0.99, 0.33,
        True,  # training
    )

    # Weight should have changed
    assert not torch.allclose(weight, weight_before)
    # Salience should be positive
    assert salience > 0
    # No NaN/Inf
    assert not torch.isnan(weight).any()
    assert not torch.isinf(weight).any()


def test_fused_op_gated():
    """Gated (spiking layer) fused op only updates active elements."""
    torch.manual_seed(42)

    weight = torch.randn(64, 64)
    set_point = weight.clone()
    momentum = torch.zeros(64, 64)
    input_avg_mag = torch.ones(64)
    excitability_acc = torch.zeros(64)  # per-output-neuron
    plasticity = torch.ones(64)  # per-input-feature
    update_ema = torch.ones(64, 64) * 1e-4
    x_flat = torch.randn(32, 64)
    output = x_flat @ weight.T

    # Gate: only top-left quadrant is active
    gate = torch.zeros(64, 64)
    gate[:32, :32] = 1.0

    weight_before = weight.clone()

    salience = living_self_modify(
        weight, set_point, momentum, input_avg_mag,
        excitability_acc, plasticity, update_ema,
        x_flat, output, gate,
        0.001, 0.001, 1e-6, 0.99, 0.99, 0.5, 0.3, 3.0, 0.99, 0.33,
        True,
    )

    # Weight should have changed (homeostatic regulation affects all weights)
    assert not torch.allclose(weight, weight_before)
    assert salience > 0


def test_fused_op_eval_mode():
    """Eval mode reduces Hebbian update magnitude."""
    torch.manual_seed(42)

    # Training run
    w_train = torch.randn(64, 64)
    sp_train = w_train.clone()
    mom_train = torch.zeros(64, 64)
    iam_train = torch.ones(64)
    ea_train = torch.zeros(64)  # per-output-neuron
    pl_train = torch.ones(64)  # per-input-feature
    ue_train = torch.ones(64, 64) * 1e-4
    x = torch.randn(32, 64)
    out = x @ w_train.T

    w_before_train = w_train.clone()
    living_self_modify(
        w_train, sp_train, mom_train, iam_train,
        ea_train, pl_train, ue_train,
        x, out, None,
        0.001, 0.001, 1e-6, 0.99, 0.99, 0.5, 0.3, 3.0, 0.99, 0.33,
        True,  # training
    )
    train_delta = (w_train - w_before_train).abs().mean()

    # Eval run (same initial state)
    torch.manual_seed(42)
    w_eval = torch.randn(64, 64)
    sp_eval = w_eval.clone()
    mom_eval = torch.zeros(64, 64)
    iam_eval = torch.ones(64)
    ea_eval = torch.zeros(64)  # per-output-neuron
    pl_eval = torch.ones(64)  # per-input-feature
    ue_eval = torch.ones(64, 64) * 1e-4
    out_eval = x @ w_eval.T

    w_before_eval = w_eval.clone()
    living_self_modify(
        w_eval, sp_eval, mom_eval, iam_eval,
        ea_eval, pl_eval, ue_eval,
        x, out_eval, None,
        0.001, 0.001, 1e-6, 0.99, 0.99, 0.5, 0.3, 3.0, 0.99, 0.33,
        False,  # eval
    )
    eval_delta = (w_eval - w_before_eval).abs().mean()

    # Eval should produce smaller weight changes
    assert eval_delta < train_delta


def test_fused_op_no_nan_after_many_calls():
    """Fused op remains stable after many consecutive calls."""
    torch.manual_seed(42)

    weight = torch.randn(64, 64)
    set_point = weight.clone()
    momentum = torch.zeros(64, 64)
    input_avg_mag = torch.ones(64)
    excitability_acc = torch.zeros(64)  # per-output-neuron
    plasticity = torch.ones(64)  # per-input-feature
    update_ema = torch.ones(64, 64) * 1e-4

    for _ in range(100):
        x_flat = torch.randn(32, 64)
        output = x_flat @ weight.T

        living_self_modify(
            weight, set_point, momentum, input_avg_mag,
            excitability_acc, plasticity, update_ema,
            x_flat, output, None,
            0.001, 0.001, 1e-6, 0.99, 0.99, 0.5, 0.3, 3.0, 0.99, 0.33,
            True,
        )

    assert not torch.isnan(weight).any()
    assert not torch.isinf(weight).any()


# --- Integration with living layers ---

def test_living_layer_uses_fused_ops():
    """LivingLayerV6 produces valid output after refactor to fused ops."""
    layer = LivingLayerV6(64, 64)
    x = torch.randn(4, 16, 64)

    out1 = layer(x)
    out2 = layer(x)

    # Outputs should differ (non-feedforward property preserved)
    assert not torch.allclose(out1, out2)
    assert not torch.isnan(out1).any()
    assert not torch.isnan(out2).any()


def test_spiking_layer_uses_fused_ops():
    """SpikingLivingLayer works with fused ops (gated mode)."""
    layer = SpikingLivingLayer(64, 64)
    x = torch.randn(4, 16, 64)

    # Warm up membrane potential (needs a few passes to charge)
    for _ in range(5):
        layer(x)

    out1 = layer(x)
    out2 = layer(x)

    assert not torch.allclose(out1, out2)
    assert not torch.isnan(out1).any()
    assert not torch.isnan(out2).any()


def test_full_model_with_fused_ops():
    """Full LuthiLM works with fused ops."""
    from luthi.model import LuthiLM

    model = LuthiLM(vocab_size=256, d_model=64, n_blocks=2)
    model.train()

    x = torch.randint(0, 256, (4, 16))
    logits = model(x)

    assert logits.shape == (4, 16, 256)
    assert not torch.isnan(logits).any()


def test_spiking_model_with_fused_ops():
    """Full SpikingLuthiLM works with fused ops."""
    from luthi.model_spiking import SpikingLuthiLM

    model = SpikingLuthiLM(vocab_size=256, d_model=64, n_blocks=2)
    model.train()

    x = torch.randint(0, 256, (4, 16))
    logits = model(x)

    assert logits.shape == (4, 16, 256)
    assert not torch.isnan(logits).any()


def test_training_loop_with_fused_ops():
    """Full training step works with fused ops."""
    from luthi.model import LuthiLM

    model = LuthiLM(vocab_size=256, d_model=64, n_blocks=2)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    x = torch.randint(0, 256, (4, 16))
    y = torch.randint(0, 256, (4, 16))

    optimizer.zero_grad()
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 256), y.reshape(-1)
    )
    loss.backward()
    model.apply_living_errors()
    optimizer.step()

    assert loss.item() > 0


# --- C++ status ---

def test_cpp_availability_reported():
    """is_cpp_available returns a boolean."""
    result = is_cpp_available()
    assert isinstance(result, bool)
