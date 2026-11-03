"""Seeding the living FFN's home (2026-11-03,
docs/research/2026-11-03_seeding-luthi.md, seed S3).

The living layer's `set_point` (its home, which homeostasis pulls the weight
back to) starts as a copy of the starting weight, so choosing the start
chooses the home. Pins:
- `living_home="random"` (the default) is the legacy draw, bit-identical.
- "orthogonal" and "identity" keep the random draw's Frobenius norm and
  change only directions and spectrum, so magnitude is not a confound.
- The home equals the start, after any μPC re-init.
- The dead control gets the same kind of start at its own draw's norm, so a
  living/dead pair differs as it always has, plus nothing.
- There is no zero home: a zero map outputs nothing and never writes.
"""

from __future__ import annotations

import pytest
import torch

from luthi.v2.hybrid_block_pc import LIVING_HOMES
from luthi.v2.living_layer_pc import PredictiveCodingLayer
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM

D = 32


def _model(seed: int = 0, **kw) -> MultimodalPredictiveCodingLM:
    torch.manual_seed(seed)
    return MultimodalPredictiveCodingLM(
        vocab_size=100, d_model=D, n_blocks=2, n_heads=4, ffn_expansion=1,
        max_seq_len=16, **kw,
    )


def test_random_home_is_the_legacy_draw_bit_identical():
    legacy, seeded = _model(), _model(living_home="random")
    for (k, a), (_, b) in zip(legacy.state_dict().items(), seeded.state_dict().items()):
        assert torch.equal(a, b), k


@pytest.mark.parametrize("mu_pc", [False, True])
@pytest.mark.parametrize("home", ["orthogonal", "identity"])
def test_structured_homes_keep_the_norm_and_become_the_home(home, mu_pc):
    torch.manual_seed(0)
    plain = _model(mu_pc_enabled=mu_pc)
    shaped = _model(mu_pc_enabled=mu_pc, living_home=home)
    first_plain = plain.blocks[0].living_ffn.weight
    for block in shaped.blocks:
        w = block.living_ffn.weight
        assert torch.equal(w, block.living_ffn.set_point), "the start is the home"
        sv = torch.linalg.svdvals(w)
        assert torch.allclose(sv, sv[0].expand_as(sv), rtol=1e-4), "a scaled isometry"
    # Block 0 is drawn before any structured init consumes the RNG, so its
    # norm can be compared with the legacy draw exactly.
    assert torch.allclose(shaped.blocks[0].living_ffn.weight.norm(), first_plain.norm())
    if home == "identity":
        w = shaped.blocks[0].living_ffn.weight
        assert torch.allclose(w, torch.diag(torch.diagonal(w)))


def test_the_random_home_is_ill_conditioned_which_is_what_the_seeds_change():
    sv = torch.linalg.svdvals(_model().blocks[0].living_ffn.weight)
    assert sv.min() < 0.1 * sv.max()


def test_the_dead_control_gets_the_same_kind_of_start():
    """The legacy living and dead layers already start from different draws
    (different init paths and norms). A seed gives the dead control the same
    KIND of start at its own draw's norm, so a pair differs as it always has,
    plus nothing."""
    legacy_dead = _model(dead_ffn=True).blocks[0].living_ffn.weight
    dead = _model(living_home="orthogonal", dead_ffn=True).blocks[0].living_ffn.weight
    sv = torch.linalg.svdvals(dead)
    assert torch.allclose(sv, sv[0].expand_as(sv), rtol=1e-4)
    assert torch.allclose(dead.norm(), legacy_dead.norm())


def test_unknown_or_zero_home_is_refused():
    assert "zero" not in LIVING_HOMES
    with pytest.raises(ValueError, match="living_home"):
        _model(living_home="zero")


def test_a_zero_map_would_never_write():
    """Why there is no zero home: the PC write is an outer product with the
    layer's own output."""
    torch.manual_seed(0)
    layer = PredictiveCodingLayer(D, D, num_episodes=4, context_dim=8, salience_threshold=1e9)
    with torch.no_grad():
        layer.weight.zero_()
        layer.set_point.zero_()
        for _ in range(20):
            layer(torch.randn(8, D))
    assert torch.count_nonzero(layer.weight) == 0


@pytest.mark.parametrize("rate", [1e-6, 1e-4, 1e-3])
def test_the_home_keeps_only_r_over_h_plus_r_of_any_change(rate):
    """What a living layer keeps for good. Homeostasis pulls the weight home
    at h = homeostatic_decay; the home follows the weight at r =
    set_point_adapt_rate. Of any displacement, the home ends up absorbing
    r(1-h) / (h + r - hr), and the rest decays away. At the legacy r = 1e-6
    that is 0.1%: everything else a living layer learns is gone within a few
    thousand forwards unless something else (Sanctuary's NREM
    consolidation) moves the home. The grown-home seed raises r before
    birth."""
    torch.manual_seed(0)
    layer = PredictiveCodingLayer(
        D, D, num_episodes=4, context_dim=8, salience_threshold=1e9,
        set_point_adapt_rate=rate,
    )
    layer.rate_scale = 0.0  # no writes: homeostasis and the home only
    disp = 0.1 * torch.randn(D, D)
    x = torch.randn(8, D)
    with torch.no_grad():
        home0 = layer.set_point.clone()
        layer.weight.copy_(home0 + disp)
        for _ in range(8000):
            layer(x)
    kept = ((layer.set_point - home0) * disp).sum() / (disp * disp).sum()
    h = layer.homeostatic_decay
    assert float(kept) == pytest.approx(rate * (1 - h) / (h + rate - h * rate), rel=0.06)


def test_the_follow_rate_can_be_raised_without_raising_the_top_down_push():
    """The legacy constant set_point_adapt_rate sets two things: how fast the
    home follows the weight, and the top-down push into the home. The push
    adds one error vector to every row (rank-1 by construction); raising the
    shared constant made it run away along one direction in block 0
    (docs/research/2026-11-03_stage0-the-home.md, step 6). home_follow_rate
    raises only the first."""
    from luthi.v2.backward_pass_pc import TopDownSignal

    def layer(**kw):
        torch.manual_seed(0)
        return PredictiveCodingLayer(D, D, num_episodes=4, context_dim=8, salience_threshold=1e9, **kw)

    legacy = layer()
    assert legacy.follow_rate == legacy.set_point_adapt_rate == 1e-6

    split = layer(home_follow_rate=1e-3)
    assert split.follow_rate == 1e-3 and split.set_point_adapt_rate == 1e-6

    signal = TopDownSignal(
        prediction_error=torch.ones(D), salience=torch.ones(D), modulation_strength=1.0,
    )
    for lay in (legacy, split):
        before = lay.set_point.clone()
        lay.apply_top_down(signal)
        assert torch.allclose(lay.set_point - before, torch.full((D, D), 1e-6 * 10.0), rtol=1e-2, atol=0)


def test_a_raised_follow_rate_sets_the_keep_fraction():
    torch.manual_seed(0)
    layer = PredictiveCodingLayer(
        D, D, num_episodes=4, context_dim=8, salience_threshold=1e9,
        home_follow_rate=1e-3,
    )
    layer.rate_scale = 0.0
    disp = 0.1 * torch.randn(D, D)
    x = torch.randn(8, D)
    with torch.no_grad():
        home0 = layer.set_point.clone()
        layer.weight.copy_(home0 + disp)
        for _ in range(8000):
            layer(x)
    kept = ((layer.set_point - home0) * disp).sum() / (disp * disp).sum()
    assert float(kept) == pytest.approx(0.5, rel=0.02)
