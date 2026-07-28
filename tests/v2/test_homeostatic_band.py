"""Homeostatic activity band -- the sparse gate's key.

The gate silences rows with low error. A COLLAPSED row has low error, so the
gate would freeze it collapsed forever: the cage. The band reopens rows that
are quiet for the wrong reason, and every one of Brian's required bounds
(2026-07-26 design conversation) is pinned here, because the mechanism is a
positive-feedback loop and an unbounded one would be worse than no key at all.

Every step below is explicitly seeded, including the probe. An earlier version
used global RNG for the final probe and passed alone but failed in-suite: with
the fast test decay, one random step revived the "dead" row. That was a test
bug, but it is also a live demonstration of why the design mandates a slow
timescale (band_decay ~1e-3 in production) -- a fast band chases noise.
"""

import torch

from luthi.v2.living_layer_pc import PredictiveCodingLayer


def _layer(**kw):
    kw.setdefault("homeostatic_band_enabled", True)
    kw.setdefault("in_features", 6)
    kw.setdefault("out_features", 8)
    kw.setdefault("band_warmup_steps", 5)
    kw.setdefault("band_decay", 0.5)          # fast, so tests converge quickly
    return PredictiveCodingLayer(**kw)


def _out(layer, dead_rows, gen):
    """One batch whose row activity varies except on `dead_rows`."""
    row_mean = torch.randn(layer.out_features, generator=gen)
    for r in dead_rows:
        row_mean[r] = 0.0
    return row_mean.unsqueeze(0).repeat(2, 1)


def _run(layer, dead_rows=(0, 1), steps=60, gate=None, seed=0):
    """Seed the activity estimate, then probe with a step of the same shape.
    The probe must respect the scenario: a row that is dead stays dead."""
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        layer._apply_activity_band(_out(layer, dead_rows, gen), None)
    return layer._apply_activity_band(_out(layer, dead_rows, gen), gate)


def test_disabled_is_a_no_op():
    layer = _layer(homeostatic_band_enabled=False)
    assert layer.homeostatic_band_enabled is False


def test_multiplier_is_bounded_at_both_ends():
    layer = _layer()
    h = _run(layer)
    assert float(h.min()) >= layer.band_h_min - 1e-6, "damping floor breached"
    assert float(h.max()) <= layer.band_h_max + 1e-6, "boost ceiling breached"


def test_dead_rows_are_boosted():
    layer = _layer()
    h = _run(layer, dead_rows=(0,))
    assert float(h[0]) > 1.0, "a row with no output variation was not boosted"


def test_healthy_layer_is_exactly_neutral():
    """Dead zone: inside the band the multiplier is exactly 1.0, so a healthy
    layer trains bit-identically to one with the band off."""
    layer = _layer()
    h = _run(layer, dead_rows=())
    assert torch.allclose(h, torch.ones_like(h)), f"band perturbed a healthy layer: {h}"


def test_rate_limit_caps_simultaneous_boosts():
    """A global dip must not reopen everything and undo the gate."""
    layer = _layer(out_features=40, band_max_boost_frac=0.05)
    _run(layer, dead_rows=tuple(range(30)))
    boosted = int(layer.band_boost_rows.item())
    assert boosted <= max(1, int(0.05 * 40)), (
        f"{boosted} rows boosted at once -- rate limit not enforced"
    )


def test_band_reopens_a_gated_row():
    """The key: a row the sparse gate has closed, and which has gone quiet,
    must get its gate forced open -- otherwise the band can only scale an
    update that is already zero, which is the trap it exists to prevent."""
    layer = _layer()
    gate = torch.zeros(layer.out_features)      # sparse gate closed everywhere
    combined = _run(layer, dead_rows=(0,), gate=gate)
    assert float(combined[0]) > 0.0, "dead gated row was never reopened"


def test_warmup_is_inert():
    layer = _layer(band_warmup_steps=1000)
    gate = torch.ones(layer.out_features)
    gen = torch.Generator().manual_seed(3)
    result = layer._apply_activity_band(_out(layer, (0,), gen), gate)
    assert torch.equal(result, gate), "band acted before its estimate was seeded"


def test_band_cannot_manufacture_an_update_from_nothing():
    """Multiplier-only, never additive: with a zero update there is nothing to
    amplify, so the band cannot create drift out of noise."""
    layer = _layer()
    h = _run(layer, dead_rows=(0,))
    delta_w = torch.zeros(layer.out_features, layer.in_features)
    assert torch.equal(delta_w * h.unsqueeze(1), delta_w)
