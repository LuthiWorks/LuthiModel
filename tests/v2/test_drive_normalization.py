"""Drive normalization: the living channel must not extinguish itself.

External review 2026-07-28, item 1.1. delta_w is driven by raw reconstruction
error, which any layer that is learning drives toward zero -- so the living
channel is self-extinguishing by construction. Measured on the v5 family:
update_ema fell 9.5e-5 -> 5.3e-9 monotonically and was still falling at step
72,000, while err_acc fell 45x. The last two-thirds of that run had an
arithmetically silent substrate.

Dividing the error by its running RMS makes the drive respond to relative
surprise, so it survives at any error scale.
"""

import torch

from luthi.v2.living_layer_pc import PredictiveCodingLayer


def _drive_decay(layer, steps=400, seed=0):
    """Train on a FIXED input distribution and report how far the update
    magnitude falls from its early value to its late one.

    Input scale is held constant on purpose. delta_w also carries an
    `output_mean` factor, so varying the input scale would measure that
    rather than the error-driven extinction this fix targets. The claim
    under test is specifically: as the layer LEARNS and its error shrinks,
    does its own update channel die?
    """
    g = torch.Generator().manual_seed(seed)
    early = None
    for i in range(steps):
        x = torch.randn(4, layer.in_features, generator=g)
        layer(x)
        if i == steps // 10:
            early = float(layer.update_ema.detach().mean().item())
    late = float(layer.update_ema.detach().mean().item())
    return early, late


def _layer(**kw):
    kw.setdefault("in_features", 16)
    kw.setdefault("out_features", 16)
    kw.setdefault("pc_rate", 0.01)
    return PredictiveCodingLayer(**kw)


def test_rms_tracker_follows_the_error_scale():
    """The mechanism: error_rms must track the actual RMS of pred_error, so
    dividing by it produces a unit-scale drive at any error magnitude."""
    for scale in (1.0, 0.01):
        layer = _layer(drive_normalize=True, drive_rms_decay=0.2)
        g = torch.Generator().manual_seed(7)
        for _ in range(60):
            layer(torch.randn(4, layer.in_features, generator=g) * scale)
        actual = float(layer._last_pred_error.detach().pow(2).mean().sqrt())
        tracked = float(layer.error_rms.item())
        assert tracked > 0
        assert 0.2 < tracked / max(actual, 1e-30) < 5.0, (
            f"rms tracker off by more than 5x at scale {scale}: "
            f"tracked {tracked:.3e} vs actual {actual:.3e}"
        )


def test_normalized_drive_is_scale_free():
    """The property the fix buys: the drive term entering delta_w has ~unit
    scale regardless of how large the raw error is. Without this, delta_w is
    proportional to an error that any learning layer drives toward zero.

    NOTE: the emergent claim -- that this keeps update_ema alive across a real
    72k-step run -- CANNOT be tested here. A toy on random inputs never learns
    enough for its error to collapse, so extinction does not reproduce in a
    unit test. That claim is pre-registered for a real run, per the review."""
    ratios = []
    for scale in (1.0, 0.01):
        layer = _layer(drive_normalize=True, drive_rms_decay=0.2)
        g = torch.Generator().manual_seed(11)
        for _ in range(60):
            layer(torch.randn(4, layer.in_features, generator=g) * scale)
        err = layer._last_pred_error.detach()
        drive = err / layer.error_rms.clamp(min=1e-12)
        ratios.append(float(drive.pow(2).mean().sqrt()))
    assert 0.2 < ratios[0] / ratios[1] < 5.0, (
        f"normalized drive scale differs across error magnitudes: {ratios}"
    )


def test_off_by_default_and_bit_identical():
    """Opt-in: with the flag off the update must be exactly what it was.
    Both layers start from identical state so the comparison is real."""
    a, b = _layer(), _layer(drive_normalize=False)
    b.load_state_dict(a.state_dict())
    g = torch.Generator().manual_seed(3)
    for _ in range(10):
        x = torch.randn(4, 16, generator=g)
        a(x.clone())
        b(x.clone())
    assert torch.equal(a.weight, b.weight)
    assert float(a.error_rms.item()) == 0.0, "rms tracked while disabled"


def test_precision_still_sees_raw_error():
    """Normalization applies ONLY to delta_w. Precision estimates real noise
    (it EMAs toward 1/err^2), so feeding it normalized error would destroy
    its meaning -- and the prediction matrix models the real input."""
    layer = _layer(drive_normalize=True)
    g = torch.Generator().manual_seed(5)
    for _ in range(30):
        layer(torch.randn(4, 16, generator=g) * 0.001)
    # tiny errors -> high precision; normalized error would have kept it ~1
    assert float(layer.precision.mean().item()) > 1.0, (
        "precision looks unaffected by the true error scale"
    )
