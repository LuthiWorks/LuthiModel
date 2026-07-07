"""The applied-change signal (spec §4 / §8 step 5).

momentum and update_ema record the *intended* (pre-gain) delta -- a measured
safety requirement (spec §4): post-gain would inflate update_ema and weaken the
refinement-6 spike guard ~2.3x. But the observation-only sinks (living-drift
eye, NREM day-accumulator) want the *actual* applied change
`delta_w * adaptive_factor * gain` -- they never feed back, so truth there costs
nothing and closes the intended-vs-applied gap the gain opens (up to 3x at
cap=3.0).

This suite pins:
 - pc_self_modify surfaces the applied-change reduction only when asked
   (return_applied_change=True), and it equals gain-off when gain==1.0 and
   exceeds it when gain>1.0 -- never below (pure amplifier).
 - the default 2-tuple return is unchanged (regime f / parity untouched).
 - the layer feeds the NREM accumulator (raw) + the fair-parallel _applied_ema
   (momentum_decay, for the eye) on the gain path, and leaves both inert off it.
 - the living-drift eye reads by EXPLICIT source config, never keyed to the flag.
"""

from __future__ import annotations

import torch

from luthi.v2 import PredictiveCodingLayer
from luthi.v2.pc_ops import pc_self_modify
from luthi.v2.slow_trace import SlowEMA, ReadResetAccumulator


def _buffers(seed: int = 0):
    torch.manual_seed(seed)
    return dict(
        weight=torch.randn(8, 16) * 0.1,
        prediction=torch.randn(8, 16) * 0.1,
        set_point=torch.randn(8, 16) * 0.1,
        momentum=torch.zeros(8, 16),
        update_ema=torch.ones(8, 16) * 1e-4,
        precision=torch.ones(16),
        error_acc=torch.zeros(8),
        plasticity=torch.ones(16),
        x_flat=torch.randn(4, 16),
        output=torch.randn(4, 8),
    )


_SCALARS = dict(
    pc_rate=0.01, pred_learning_rate=0.0001, homeostatic_decay=0.001,
    set_point_adapt_rate=1e-6, momentum_decay=0.9, update_ema_decay=0.99,
    precision_ema_decay=0.99, precision_min=0.1, precision_max=10.0,
    prediction_clamp=1.0,
)


# ---------------------------------------------------------------------------
# Op-level: the returned reduction.
# ---------------------------------------------------------------------------

def test_default_return_is_two_tuple_unchanged():
    """No return_applied_change -> 2-tuple, exactly as legacy callers expect."""
    bufs = _buffers()
    out = pc_self_modify(**bufs, **_SCALARS)
    assert isinstance(out, tuple) and len(out) == 2


def test_applied_change_returned_when_requested():
    bufs = _buffers()
    out = pc_self_modify(
        **bufs, **_SCALARS,
        learning_gain_enabled=True, learning_gain_progress=0.0,
        return_applied_change=True,
    )
    assert len(out) == 3
    salience, pred_error, applied = out
    assert isinstance(applied, float)
    assert applied >= 0.0


def test_applied_change_equals_gain_off_when_gain_is_one():
    """First step: momentum==0 -> coherence==0 -> gain==1.0, so the applied
    change equals the gain-off applied change (bit-for-bit same product)."""
    on = _buffers(seed=1)
    off = _buffers(seed=1)
    _, _, applied_on = pc_self_modify(
        **on, **_SCALARS,
        learning_gain_enabled=True, learning_gain_progress=0.0,
        return_applied_change=True,
    )
    _, _, applied_off = pc_self_modify(
        **off, **_SCALARS,
        learning_gain_enabled=False,
        return_applied_change=True,
    )
    assert applied_on == applied_off


def test_applied_change_exceeds_gain_off_when_gain_above_one():
    """With directional momentum (coherence>0) and progress<1 (fall~1), gain>1
    so the applied change is strictly larger than the gain-off applied change
    on the same buffers -- the amplifier at work."""
    on = _buffers(seed=2)
    off = _buffers(seed=2)
    # Seed coherent, directional momentum so coherence = |m|/(ema+eps) is large.
    on["momentum"] = torch.full((8, 16), 0.05)
    on["update_ema"] = torch.full((8, 16), 0.05)
    off["momentum"] = on["momentum"].clone()
    off["update_ema"] = on["update_ema"].clone()
    _, _, applied_on = pc_self_modify(
        **on, **_SCALARS,
        learning_gain_enabled=True, learning_gain_progress=0.0,
        learning_gain_rise=2.0, learning_gain_cap=3.0,
        return_applied_change=True,
    )
    _, _, applied_off = pc_self_modify(
        **off, **_SCALARS,
        learning_gain_enabled=False,
        return_applied_change=True,
    )
    assert applied_on > applied_off


# ---------------------------------------------------------------------------
# Layer-level: the sinks fed on the gain path, inert off it.
# ---------------------------------------------------------------------------

def _layer(gain: bool, seed: int = 0) -> PredictiveCodingLayer:
    torch.manual_seed(seed)
    return PredictiveCodingLayer(
        in_features=16, out_features=8, pc_rate=0.01,
        num_episodes=8, context_dim=8, learning_gain_enabled=gain,
    )


def test_layer_gain_on_feeds_sinks():
    layer = _layer(gain=True, seed=4)
    torch.manual_seed(31)
    x = torch.randn(4, 16)
    for _ in range(20):
        layer(x)
    # NREM day-integral fed the raw instantaneous applied change per step.
    assert layer._applied_change_accum.count == 20
    assert layer._applied_change_accum.total > 0.0
    # The eye's fair-parallel EMA (momentum_decay) was fed and is warm.
    assert layer._applied_ema._count == 20
    assert layer._applied_ema.value > 0.0


def test_layer_gain_off_sinks_inert():
    layer = _layer(gain=False, seed=4)
    torch.manual_seed(31)
    x = torch.randn(4, 16)
    for _ in range(20):
        layer(x)
    assert layer._applied_change_accum.count == 0
    assert layer._applied_ema._count == 0


def test_accumulator_read_and_reset_zeros_the_day():
    acc = ReadResetAccumulator()
    acc.add(1.0)
    acc.add(2.5)
    assert acc.read_and_reset() == 3.5
    assert acc.total == 0.0 and acc.count == 0


# ---------------------------------------------------------------------------
# The living-drift eye reads by EXPLICIT source (Fable step-8 ruling), not by
# attribute presence keyed to the gain flag.
# ---------------------------------------------------------------------------

def test_eye_reading_by_explicit_source():
    from luthi.v2.m9.runner import _living_drift_reading

    off = _layer(gain=False, seed=6)
    on = _layer(gain=True, seed=6)
    torch.manual_seed(41)
    x = torch.randn(4, 16)
    for _ in range(15):
        off(x)
        on(x)

    # source="momentum" reads mean |momentum| regardless of the gain flag.
    assert _living_drift_reading(off, "momentum") == off.momentum.abs().mean().item()
    assert _living_drift_reading(on, "momentum") == on.momentum.abs().mean().item()

    # source="applied_change" reads the fair-parallel EMA when it has been fed
    # (gain on), and None when it hasn't (gain off) -- NOT a momentum fallback,
    # so the band never mixes units.
    assert _living_drift_reading(on, "applied_change") == on._applied_ema.value
    assert _living_drift_reading(off, "applied_change") is None


# ---------------------------------------------------------------------------
# Persistence (regime i core): traces + accumulator survive checkpoint restore,
# and the collection is introspective -- every SlowEMA/accumulator attribute is
# captured, so forgetting to wire a new trace is a test failure.
# ---------------------------------------------------------------------------

def test_traces_and_accumulator_round_trip():
    from luthi.living_extra_state import (
        collect_living_extra_state, apply_living_extra_state,
    )
    import torch.nn as nn

    src = _layer(gain=True, seed=8)
    torch.manual_seed(51)
    x = torch.randn(4, 16)
    for _ in range(30):          # warm the traces, fill the accumulator
        src(x)
    assert src._err_long.is_warm()
    assert src._applied_change_accum.total > 0.0

    # Wrap in a module so named_modules() gives a path.
    holder_src = nn.Module()
    holder_src.add_module("layer", src)
    state = collect_living_extra_state(holder_src)

    # Introspective: every SlowEMA / ReadResetAccumulator attr on the layer
    # must appear in the collected slow_traces -- forgetting the wiring fails.
    trace_attrs = {
        name for name, obj in vars(src).items()
        if isinstance(obj, (SlowEMA, ReadResetAccumulator))
    }
    assert trace_attrs, "expected the layer to own slow-trace state"
    collected = state["layer"]["slow_traces"]
    assert trace_attrs <= set(collected)

    # Restore into a fresh layer mid-hard-growth: the resolution signal and the
    # day-accumulator must NOT reset (the silent-amnesia class, spec §5).
    dst = _layer(gain=True, seed=999)     # different init, cold traces
    holder_dst = nn.Module()
    holder_dst.add_module("layer", dst)
    apply_living_extra_state(holder_dst, state)

    assert dst._err_short.value == src._err_short.value
    assert dst._err_long.value == src._err_long.value
    assert dst._err_long._count == src._err_long._count
    assert dst._applied_change_accum.total == src._applied_change_accum.total
    assert dst._applied_change_accum.count == src._applied_change_accum.count
