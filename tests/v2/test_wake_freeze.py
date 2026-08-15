"""The waking day: substrate held still, the mind still noticing.

Brian's ruling, 2026-08-14 -- "data intake during the day, structural
cortical change over night." The regime must hold the living weights
still while keeping every mechanism that decides what the day MEANT
running, above all the episode write. If the day stores nothing, the
rest-phase pass has nothing to integrate and the day is simply lost.

The gotcha this exists to close: `freeze_plasticity` looks like the right
tool and is not. It also suppresses the episode write, because it was
built for the momentary lived re-encode. Used as a daytime regime it
would mean living a full day and storing none of it. These tests pin the
difference in both directions.
"""

from __future__ import annotations

import torch

from luthi.v2.living_layer_pc import PredictiveCodingLayer
from luthi.v2.plasticity import freeze_plasticity, set_wake_frozen, wake_freeze

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


def _substrate(layer):
    return (
        layer.weight.detach().clone(),
        layer.prediction.detach().clone(),
        layer.set_point.detach().clone(),
    )


def test_wake_freeze_leaves_the_substrate_bit_identical():
    layer = _layer()
    _drive(layer)                       # warm it so state is non-trivial
    before = _substrate(layer)
    with wake_freeze(layer):
        _drive(layer, steps=25, seed=1)
    after = _substrate(layer)
    for b, a, name in zip(before, after, ("weight", "prediction", "set_point")):
        assert torch.equal(b, a), f"{name} moved under wake_freeze"


def test_wake_freeze_still_writes_episodes():
    """The whole point. `freeze_plasticity` fails this by design."""
    layer = _layer()
    _drive(layer)
    writes_before = int(layer.episode_writes.item())
    with wake_freeze(layer):
        _drive(layer, steps=40, seed=2)
    writes_after = int(layer.episode_writes.item())
    assert writes_after > writes_before, (
        "wake_freeze stored nothing -- the day would be lost at midnight"
    )


def test_plasticity_freeze_does_NOT_write_episodes():
    """The contrast that motivates a second context manager."""
    layer = _layer()
    _drive(layer)
    writes_before = int(layer.episode_writes.item())
    with freeze_plasticity(layer):
        _drive(layer, steps=40, seed=2)
    assert int(layer.episode_writes.item()) == writes_before, (
        "freeze_plasticity is expected to suppress the episode write; "
        "if this changed, wake_freeze may be redundant -- check before "
        "deleting it"
    )


def test_wake_freeze_keeps_the_noticing_machinery_live():
    """precision / error_acc must keep moving: they decide salience, which
    decides what the day is worth storing."""
    layer = _layer()
    _drive(layer)
    prec_before = layer.precision.detach().clone()
    err_before = layer.error_acc.detach().clone()
    with wake_freeze(layer):
        _drive(layer, steps=30, seed=4)
    assert not torch.equal(prec_before, layer.precision), "precision froze"
    assert not torch.equal(err_before, layer.error_acc), "error_acc froze"


def test_recall_still_adapts_behaviour_within_the_day():
    """Within-day adaptation without structural change: the recall path
    blends a stored delta into the effective weight, so behaviour can move
    while `self.weight` does not."""
    layer = _layer()
    _drive(layer, steps=30)
    with wake_freeze(layer):
        w_before = layer.weight.detach().clone()
        torch.manual_seed(9)
        x = torch.randn(4, D)
        out = layer(x)
        assert torch.equal(w_before, layer.weight)
        assert torch.isfinite(out).all()


def test_sweep_reaches_every_layer_and_reports_count():
    class _Trunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = _layer()
            self.b = _layer()

    trunk = _Trunk()
    n = set_wake_frozen(trunk, True)
    assert n == 2, f"sweep reached {n} layers, expected 2"
    assert trunk.a._wake_frozen and trunk.b._wake_frozen
    assert set_wake_frozen(trunk, False) == 2
    assert not trunk.a._wake_frozen and not trunk.b._wake_frozen


def test_wake_freeze_restores_prior_state_and_nests():
    layer = _layer()
    assert not layer._wake_frozen
    with wake_freeze(layer):
        assert layer._wake_frozen
        with wake_freeze(layer):
            assert layer._wake_frozen
        assert layer._wake_frozen, "inner exit unfroze an outer freeze"
    assert not layer._wake_frozen


def test_wake_freeze_is_not_persisted():
    """A regime, not lived state -- it must not enter the checkpoint."""
    layer = _layer()
    set_wake_frozen(layer, True)
    assert "_wake_frozen" not in layer.state_dict()
