"""Layer-level plumbing tests for the inverted-U gain (spec §8 step 4).

The op-level function is covered by test_learning_gain.py (a/b/c/e/g) and
test_learning_gain_integration.py (f/d). This suite covers the *layer*
wiring: PredictiveCodingLayer instantiates the resolution slow-traces, feeds
prediction error each forward, computes resolution_progress, and threads the
opt-in flag through to pc_self_modify.

Discipline held here:
 - Regime (f) at the layer: a layer with learning_gain_enabled=False evolves
   bit-identically to a plain default layer -- adding the (defaulted-off) gain
   params changed nothing.
 - Cold-start (regime e): the first forward has momentum==0 -> coherence==0 ->
   gain==1.0, so gain-on and gain-off are bit-identical on step one regardless
   of the flag. The gain only ever *amplifies*, never suppresses.
 - The traces warm on the layer's own forward clock and only when the gain is
   enabled (gain machinery is fully inert when off).
"""

from __future__ import annotations

import torch

from luthi.v2 import PredictiveCodingLayer


def _layer(gain: bool = False, seed: int = 0, **kw) -> PredictiveCodingLayer:
    torch.manual_seed(seed)
    return PredictiveCodingLayer(
        in_features=16,
        out_features=8,
        pc_rate=0.01,
        num_episodes=8,
        context_dim=8,
        learning_gain_enabled=gain,
        **kw,
    )


def _drive(layer: PredictiveCodingLayer, steps: int, seed: int = 1,
           fixed: bool = True) -> None:
    """Run `steps` forwards. fixed=True reuses one input each step so the
    error trend is establishment-shaped (coherent, resolving)."""
    torch.manual_seed(seed)
    x = torch.randn(4, layer.in_features)
    for i in range(steps):
        layer(x if fixed else torch.randn(4, layer.in_features))


# ---------------------------------------------------------------------------
# Regime (f) at the layer level: defaulted-off gain is a no-op.
# ---------------------------------------------------------------------------

def test_gain_off_bit_identical_to_default_layer():
    """A layer with learning_gain_enabled=False evolves bit-identically to a
    plain default layer -- the new (off) params touch nothing."""
    a = _layer(gain=False, seed=0)             # explicit flag off
    b = PredictiveCodingLayer(                  # constructed with no gain kwargs
        in_features=16, out_features=8, pc_rate=0.01,
        num_episodes=8, context_dim=8,
    )
    # Match b's random init to a's (same construction seed path).
    torch.manual_seed(0)
    b = PredictiveCodingLayer(
        in_features=16, out_features=8, pc_rate=0.01,
        num_episodes=8, context_dim=8,
    )
    _drive(a, 30)
    _drive(b, 30)
    assert torch.equal(a.weight, b.weight)
    assert torch.equal(a.momentum, b.momentum)
    assert torch.equal(a.update_ema, b.update_ema)


def test_gain_off_leaves_traces_and_sinks_inert():
    """Gain off -> resolution traces never fed, no applied-change recorded."""
    a = _layer(gain=False, seed=3)
    _drive(a, 30)
    assert not a._err_short.is_warm()
    assert not a._err_long.is_warm()
    assert a._err_short.value == 0.0 and a._err_long.value == 0.0
    assert a._last_applied_change is None
    assert a._applied_change_accum.count == 0


# ---------------------------------------------------------------------------
# Cold-start (regime e) at the layer: step one is bit-identical.
# ---------------------------------------------------------------------------

def test_first_forward_bit_identical_gain_on_vs_off():
    """On the first forward momentum==0 -> coherence==0 -> gain==1.0, so the
    gain-on layer's weight update is bit-identical to gain-off. The amplifier
    has nothing directional to amplify yet."""
    off = _layer(gain=False, seed=7)
    on = _layer(gain=True, seed=7)
    torch.manual_seed(11)
    x = torch.randn(4, 16)
    off(x)
    on(x)
    assert torch.equal(off.weight, on.weight)
    assert torch.equal(off.momentum, on.momentum)


# ---------------------------------------------------------------------------
# Traces warm on the forward clock, only when gain enabled.
# ---------------------------------------------------------------------------

def test_traces_warm_after_warmup_forwards():
    on = _layer(gain=True, seed=5, resolution_warmup=8)
    _drive(on, 7)
    assert not on._err_long.is_warm()   # 7 < warmup
    _drive(on, 2)                        # now 9 >= 8
    assert on._err_short.is_warm()
    assert on._err_long.is_warm()
    # progress is finite and non-negative once warm.
    from luthi.v2.slow_trace import resolution_progress
    p = resolution_progress(on._err_short, on._err_long)
    assert torch.isfinite(torch.tensor(p))
    assert p >= 0.0


# ---------------------------------------------------------------------------
# The gain amplifies directed novelty: gain-on moves the weight further than
# gain-off on coherent, resolving establishment -- and stays bounded.
# ---------------------------------------------------------------------------

def test_gain_amplifies_coherent_establishment_and_stays_bounded():
    off = _layer(gain=True, seed=9)   # same seed/init; toggle via flag below
    on = _layer(gain=True, seed=9)
    off.learning_gain_enabled = False

    torch.manual_seed(21)
    x = torch.randn(4, 16)             # fixed coherent input -> directional
    for _ in range(60):
        off(x)
        on(x)

    # After warmup, coherent directed change should have lifted the gain-on
    # weight displacement above gain-off's (pure amplification, never below).
    off_disp = (off.weight - off.set_point).abs().mean().item()
    on_disp = (on.weight - on.set_point).abs().mean().item()
    assert on_disp >= off_disp * 0.999   # never suppresses
    assert on_disp > off_disp            # amplifies on coherent novelty

    # Bounded: no runaway. cap=3.0 default; norm stays finite and comparable.
    assert torch.isfinite(on.weight).all()
    assert on.weight.norm().item() < 50.0
