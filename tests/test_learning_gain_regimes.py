"""Layer-level bounded-growth regimes for the inverted-U gain (spec §6 / §8
step 6). The op-level function regimes (a/b/c/e/g) live in test_learning_gain.py;
these are the ones the spec's Fable audit (2026-07-06) flagged as needing the
COMPOSED update, not just the pure function:

  (b) equilibrium-shift  -- homeostasis bounds the norm, so a 3x gain doesn't
      diverge; it *moves the operating point*. Assert the plateau-norm ratio
      gain_on/off is bounded (else downstream scales drift silently) AND the
      gain never suppresses (>= gain-off).
  (b) stacked-blocks     -- amplified layer-k outputs enlarge layer-(k+1)'s
      errors; single-layer boundedness does not compose for free. One
      multi-layer regime.
  (h) frozen-plasticity  -- lived re-encode under freeze_plasticity() stays a
      bit-identical no-self-mod even with the gain active (gate 3, executable).
  (i) persistence introspective -- collection captures slow-trace state BY TYPE,
      so wiring a new trace and forgetting to persist it is a test failure.
  (j) consolidation bypass -- the gain is NOT applied during
      consolidate_layer_attractor replay (placeholder; the real capture-vs-gain
      decision is deferred to the NREM spec).
  (k) oscillating-error  -- error oscillating between the two EMA horizons
      defeats the resolution detector; accept-and-document that the CAP
      governor still bounds it (the fall failing is not divergence).
"""

from __future__ import annotations

import torch

from luthi.v2 import PredictiveCodingLayer
from luthi.v2.slow_trace import SlowEMA, ReadResetAccumulator


def _layer(gain: bool, seed: int = 0, **kw) -> PredictiveCodingLayer:
    torch.manual_seed(seed)
    return PredictiveCodingLayer(
        in_features=16, out_features=16, pc_rate=0.02,
        num_episodes=8, context_dim=8, learning_gain_enabled=gain, **kw,
    )


def _plateau_norm(layer: PredictiveCodingLayer, xs, tail: int = 30) -> float:
    norms = []
    for i, x in enumerate(xs):
        layer(x)
        if i >= len(xs) - tail:
            norms.append(layer.weight.norm().item())
    return sum(norms) / len(norms)


# ---------------------------------------------------------------------------
# (b) equilibrium-shift: bounded operating-point move, never suppression.
# ---------------------------------------------------------------------------

def test_b_equilibrium_shift_bounded_and_never_suppresses():
    """Sustained high-error, non-resolving input. The gain-on plateau norm may
    move above gain-off (a 3x amplifier shifts the homeostatic operating point)
    but must stay bounded -- no runaway -- and never fall below gain-off (the
    amplifier cannot give up on hard growth)."""
    torch.manual_seed(100)
    # Non-resolving: fresh large-magnitude input every step so error never
    # settles (progress ~ 1 at the plateau -> fall -> 0 -> gain -> 1, but the
    # transient amplifies and homeostasis must still bound the result).
    xs = [torch.randn(4, 16) * 2.0 for _ in range(400)]

    off = _layer(gain=True, seed=1)
    on = _layer(gain=True, seed=1)
    off.learning_gain_enabled = False

    off_plateau = _plateau_norm(off, xs)
    on_plateau = _plateau_norm(on, xs)

    assert torch.isfinite(on.weight).all()
    assert on_plateau >= off_plateau * 0.999          # never suppresses
    # Bounded operating-point shift: the ratio is finite and modest, not a
    # runaway. cap=3.0; homeostasis pulls back, so the shift is far under cap.
    ratio = on_plateau / off_plateau
    assert 1.0 <= ratio <= 3.0, f"plateau ratio {ratio} outside bound"


def test_b_gain_returns_to_one_on_sustained_nonresolution():
    """The explicit fall: once effort is sustained-non-resolving, progress ~ 1
    so the gain decays back to ~1.0 (amplification off, never suppression). We
    read the effective gain the layer would compute at the plateau."""
    from luthi.v2.slow_trace import resolution_progress
    from luthi.v2.pc_ops import learning_gain

    on = _layer(gain=True, seed=2)
    torch.manual_seed(101)
    xs = [torch.randn(4, 16) * 2.0 for _ in range(400)]
    for x in xs:
        on(x)

    progress = resolution_progress(on._err_short, on._err_long)
    g = learning_gain(on.momentum, on.update_ema, progress,
                      rise=on.learning_gain_rise, cap=on.learning_gain_cap)
    assert (g >= 1.0).all()                            # never suppresses
    # Non-resolution drives the fall: mean gain should be near 1.0, well under
    # cap -- the amplifier stood down once effort stopped resolving.
    assert g.mean().item() < 1.5


# ---------------------------------------------------------------------------
# (b) stacked-blocks: amplification compounds across layers but stays bounded.
# ---------------------------------------------------------------------------

def test_b_stacked_layers_bounded():
    """Two chained living layers, gain on both. Layer-1's amplified output is
    layer-2's input, enlarging layer-2's errors -- the compounding single-layer
    boundedness does not give for free. Assert both stay finite and bounded,
    and the stack's norms track the gain-off stack within a bound."""
    torch.manual_seed(102)

    def _stack(gain):
        torch.manual_seed(5)
        l1 = PredictiveCodingLayer(16, 16, pc_rate=0.02, num_episodes=8,
                                   context_dim=8, learning_gain_enabled=gain)
        torch.manual_seed(6)
        l2 = PredictiveCodingLayer(16, 16, pc_rate=0.02, num_episodes=8,
                                   context_dim=8, learning_gain_enabled=gain)
        return l1, l2

    on1, on2 = _stack(True)
    off1, off2 = _stack(False)

    torch.manual_seed(103)
    xs = [torch.randn(4, 16) * 1.5 for _ in range(300)]
    for x in xs:
        on2(on1(x))
        off2(off1(x))

    for layer in (on1, on2, off1, off2):
        assert torch.isfinite(layer.weight).all()
        assert layer.weight.norm().item() < 100.0

    # Compounding stays bounded relative to the gain-off stack.
    assert on2.weight.norm().item() <= off2.weight.norm().item() * 4.0


# ---------------------------------------------------------------------------
# (h) frozen-plasticity: gain active but no self-mod under freeze_plasticity().
# ---------------------------------------------------------------------------

def test_h_frozen_plasticity_no_self_mod_with_gain_on():
    from luthi.v2.plasticity import freeze_plasticity

    layer = _layer(gain=True, seed=7)
    torch.manual_seed(104)
    x = torch.randn(4, 16)
    for _ in range(20):                 # warm traces + move buffers
        layer(x)

    snap = dict(
        weight=layer.weight.clone(),
        momentum=layer.momentum.clone(),
        update_ema=layer.update_ema.clone(),
        short=layer._err_short.value,
        long=layer._err_long.value,
        acc=layer._applied_change_accum.total,
        last=layer._last_applied_change,
    )

    with freeze_plasticity(layer):
        out1 = layer(x)
        out2 = layer(x)

    # No living buffer or trace moved under the freeze -- even with gain on.
    assert torch.equal(layer.weight, snap["weight"])
    assert torch.equal(layer.momentum, snap["momentum"])
    assert torch.equal(layer.update_ema, snap["update_ema"])
    assert layer._err_short.value == snap["short"]
    assert layer._err_long.value == snap["long"]
    assert layer._applied_change_accum.total == snap["acc"]
    assert layer._last_applied_change == snap["last"]
    # Two frozen forwards on identical input are bit-identical (no self-mod
    # between them) and grad-capable.
    assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# (i) persistence introspective: a NEW trace is captured automatically.
# ---------------------------------------------------------------------------

def test_i_collection_captures_traces_by_type():
    from luthi.living_extra_state import collect_living_extra_state
    import torch.nn as nn

    layer = _layer(gain=True, seed=8)
    # Attach a hypothetical future trace: the by-type collection must catch it
    # with no code change -- that is the "forgetting the wiring is a test
    # failure" guarantee made structural.
    layer._future_trace = SlowEMA(decay=0.95)
    layer._future_trace.update(0.3)

    holder = nn.Module()
    holder.add_module("layer", layer)
    collected = collect_living_extra_state(holder)["layer"]["slow_traces"]

    expected = {name for name, obj in vars(layer).items()
                if isinstance(obj, (SlowEMA, ReadResetAccumulator))}
    assert "_future_trace" in expected
    assert expected <= set(collected)


# ---------------------------------------------------------------------------
# (j) consolidation-replay bypass: the gain is NOT applied during replay.
# ---------------------------------------------------------------------------

def test_j_consolidation_replay_does_not_apply_gain(monkeypatch):
    """Placeholder per spec: consolidate_layer_attractor replay must not run
    the gain (capture-vs-gain is a NREM-spec decision). We spy on
    pc_self_modify and assert every replay call has the gain disabled, even
    when the layer itself has the gain enabled."""
    from luthi.v2 import consolidation
    from luthi.v2 import pc_ops

    layer = PredictiveCodingLayer(
        16, 16, pc_rate=0.02, num_episodes=8, context_dim=8,
        learning_gain_enabled=True, consolidation_enabled=True,
        consolidation_style="attractor",
    )
    # Give the layer a stored episode to replay.
    torch.manual_seed(105)
    x = torch.randn(4, 16)
    for _ in range(30):
        layer(x)
    # Force an episode to exist for replay.
    layer._store_episode(
        layer._compute_context(x), salience=1.0,
        input_pattern=x.mean(dim=0),
    )

    seen_gain_flags = []
    real = pc_ops.pc_self_modify

    def _spy(*args, **kwargs):
        seen_gain_flags.append(kwargs.get("learning_gain_enabled", False))
        return real(*args, **kwargs)

    # consolidate_layer_attractor does `from luthi.v2.pc_ops import
    # pc_self_modify` at call time, so patch the name in its source module.
    monkeypatch.setattr(pc_ops, "pc_self_modify", _spy)
    consolidation.consolidate_layer_attractor(layer, consolidation_rate_factor=0.1)

    assert seen_gain_flags, "expected at least one replay self-mod call"
    assert not any(seen_gain_flags), "gain must be bypassed during replay"


# ---------------------------------------------------------------------------
# (k) oscillating-error: the fall can be defeated, but the cap still bounds.
# ---------------------------------------------------------------------------

def test_k_oscillating_error_stays_bounded_under_cap():
    """Error oscillating at a period between the short and long EMA horizons
    defeats the resolution-progress detector (measured: fall>0.25 on 31% of
    steps at zero net resolution -- spec regime k). The accept-and-document
    safety claim is that the CAP governor still bounds the weight: the fall
    failing to engage is not divergence. Documented duty-cycle limitation;
    the workspace monitor (gate 2) is the designated discriminator."""
    on = _layer(gain=True, seed=9)
    torch.manual_seed(106)
    # Oscillate input magnitude with a ~24-step period: between short (~10) and
    # long (~100) horizons, so short and long EMAs stay roughly in phase and
    # the ratio never signals sustained non-resolution.
    import math
    for t in range(600):
        scale = 1.0 + 0.9 * math.sin(2 * math.pi * t / 24.0)
        on(torch.randn(4, 16) * scale)

    assert torch.isfinite(on.weight).all()
    assert on.weight.norm().item() < 100.0    # cap governor holds the line
