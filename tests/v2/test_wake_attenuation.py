"""The waking day: the substrate still moves, but far less.

Brian's ruling, 2026-08-14, as amended the same day -- "I don't think the
waking state should be completely frozen, but ... the susceptibility to
change should be greatly lessened so the lesson of the day can be
intentionally pondered rather than purely reacted to."

Two properties this pins:

1. **The day still writes.** `freeze_plasticity` looks like the right tool
   and is not: it also suppresses the episode write, because it was built
   for the momentary lived re-encode. Used as a daytime regime it would
   mean living a full day and storing none of it, leaving the rest phase
   nothing to integrate.
2. **The attenuation is uniform across all four weight-touching rates.**
   The taper deliberately scales the learning channels only ("stability is
   not what tapers"), which is safe at ~5x and NOT safe here: homeostasis
   pulls weight toward set_point with a time constant of
   ~1/homeostatic_decay forwards, so attenuating learning alone would make
   the day actively erase itself back to baseline rather than merely
   change less.
"""

from __future__ import annotations

import pytest
import torch

from luthi.v2.living_layer_pc import PredictiveCodingLayer
from luthi.v2.plasticity import (
    WAKE_ATTENUATION_DEFAULT,
    freeze_plasticity,
    set_wake_attenuation,
    wake_attenuated,
)

D = 16


def _layer(**kw):
    base = dict(
        in_features=D, out_features=D, num_episodes=8, context_dim=4,
        salience_threshold=0.0,          # store readily, so writes are visible
    )
    base.update(kw)
    torch.manual_seed(3)
    return PredictiveCodingLayer(**base)


def _drive(layer, steps=12, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(4, D)
    for _ in range(steps):
        layer(x)
    return x


def _movement(layer, steps, seed, factor=None):
    """How far the weight travels over `steps` forwards."""
    before = layer.weight.detach().clone()
    if factor is None:
        _drive(layer, steps=steps, seed=seed)
    else:
        with wake_attenuated(layer, factor):
            _drive(layer, steps=steps, seed=seed)
    return (layer.weight.detach() - before).abs().mean().item()


def test_the_day_still_moves_the_substrate():
    """A nonzero floor is the ruling, not an implementation detail: change
    stays automatic and unvetoed."""
    layer = _layer()
    _drive(layer)
    moved = _movement(layer, steps=40, seed=1, factor=WAKE_ATTENUATION_DEFAULT)
    assert moved > 0.0, "the waking day must not be completely frozen"


def test_the_day_moves_far_less_than_ordinary_plasticity():
    a, b = _layer(), _layer()
    _drive(a); _drive(b)
    ordinary = _movement(a, steps=40, seed=1)
    attenuated = _movement(b, steps=40, seed=1, factor=WAKE_ATTENUATION_DEFAULT)
    assert attenuated < ordinary / 10, (
        f"attenuated movement {attenuated:.3e} is not greatly lessened "
        f"against ordinary {ordinary:.3e}"
    )


def test_attenuation_is_monotone_in_the_factor():
    moves = []
    for f in (1.0, 1e-1, 1e-2, 1e-3):
        layer = _layer()
        _drive(layer)
        moves.append(_movement(layer, steps=30, seed=5, factor=f))
    assert moves == sorted(moves, reverse=True), f"not monotone: {moves}"


def test_uniform_scaling_does_not_drag_the_weight_to_the_set_point():
    """The failure mode of scaling the learning channels alone.

    With homeostasis left at full strength, a long attenuated stretch
    would collapse the weight onto set_point. Uniform scaling must leave
    the gap roughly intact.
    """
    layer = _layer()
    _drive(layer, steps=40)
    gap_before = (layer.weight - layer.set_point).abs().mean().item()
    with wake_attenuated(layer, 1e-3):
        _drive(layer, steps=300, seed=7)
    gap_after = (layer.weight - layer.set_point).abs().mean().item()
    assert gap_after > 0.5 * gap_before, (
        f"the day erased itself toward set_point: gap {gap_before:.3e} -> "
        f"{gap_after:.3e}. Homeostasis is outrunning the attenuated "
        f"learning channel."
    )


def test_wake_attenuation_still_writes_episodes():
    """The whole point. `freeze_plasticity` fails this by design."""
    layer = _layer()
    _drive(layer)
    before = int(layer.episode_writes.item())
    with wake_attenuated(layer):
        _drive(layer, steps=40, seed=2)
    assert int(layer.episode_writes.item()) > before, (
        "the day stored nothing -- it would be lost at rest"
    )


def test_plasticity_freeze_does_NOT_write_episodes():
    """The contrast that motivates a second regime."""
    layer = _layer()
    _drive(layer)
    before = int(layer.episode_writes.item())
    with freeze_plasticity(layer):
        _drive(layer, steps=40, seed=2)
    assert int(layer.episode_writes.item()) == before, (
        "freeze_plasticity is expected to suppress the episode write; if "
        "that changed, check before assuming wake attenuation is redundant"
    )


def test_the_noticing_machinery_runs_at_full_strength():
    """precision / error_acc decide salience, which decides what the day is
    worth storing. They must not be attenuated."""
    layer = _layer()
    _drive(layer)
    prec, err = layer.precision.detach().clone(), layer.error_acc.detach().clone()
    with wake_attenuated(layer, 1e-3):
        _drive(layer, steps=30, seed=4)
    assert not torch.equal(prec, layer.precision), "precision was attenuated"
    assert not torch.equal(err, layer.error_acc), "error_acc was attenuated"


def test_factor_one_is_the_legacy_regime_bit_identical():
    a, b = _layer(), _layer()
    _drive(a, steps=25, seed=6)
    with wake_attenuated(b, 1.0):
        _drive(b, steps=25, seed=6)
    assert torch.equal(a.weight, b.weight), "factor=1.0 is not legacy-identical"


def test_out_of_range_factor_fails_loud():
    layer = _layer()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        set_wake_attenuation(layer, 1.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        set_wake_attenuation(layer, -0.1)


def test_sweep_reaches_every_layer_and_reports_count():
    class _Trunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = _layer()
            self.b = _layer()

    trunk = _Trunk()
    assert set_wake_attenuation(trunk, 1e-3) == 2
    assert trunk.a._wake_attenuation == pytest.approx(1e-3)
    assert trunk.b._wake_attenuation == pytest.approx(1e-3)
    assert set_wake_attenuation(trunk, 1.0) == 2
    assert trunk.a._wake_attenuation == 1.0


def test_context_restores_prior_state_and_nests():
    layer = _layer()
    assert layer._wake_attenuation == 1.0
    with wake_attenuated(layer, 1e-2):
        assert layer._wake_attenuation == pytest.approx(1e-2)
        with wake_attenuated(layer, 1e-4):
            assert layer._wake_attenuation == pytest.approx(1e-4)
        assert layer._wake_attenuation == pytest.approx(1e-2), (
            "inner exit clobbered the outer regime"
        )
    assert layer._wake_attenuation == 1.0


def test_regime_is_not_persisted():
    """A regime, not lived state -- it must not enter the checkpoint."""
    layer = _layer()
    set_wake_attenuation(layer, 1e-3)
    assert "_wake_attenuation" not in layer.state_dict()
