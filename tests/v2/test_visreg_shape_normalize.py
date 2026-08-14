"""VISReg's shape term scales with N unless normalized (audit B1).

Eq. 5 sums over N while L_scale means over D, so `l_shape` grows with the
sample count and the convex Eq. 9 mix at lambda = 0.6 buries `l_pred`.
Measured on the 768x8 family: VISReg was 98.6-99.99% of the objective for
the whole run and l_pred never exceeded 1.4%. The 2026-08-11 smoke saw the
same thing as a batch effect (l_shape 1,461,016 at batch 32 vs 693,472 at
batch 16 -- pure N-scaling).

`shape_normalize=True` makes l_shape a mean over N, so lambda becomes a
scale-free mixing weight. Default stays False: no completed family's
configuration may silently change meaning.
"""

import torch

from luthi.v2.visreg import VISReg

D = 16


def _z(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, D, generator=g)


def _collapsed_z(n, seed=0):
    """Rank-2 data in D dims -- the regime VISReg exists to penalize.

    Gaussian input is the wrong probe for the N-scaling property: random
    projections of Gaussian data are already Gaussian, so `l_shape`
    measures sampling noise that SHRINKS as N grows and the sum does not
    scale. On genuinely non-Gaussian data the per-element deviation
    persists and the sum-over-N shows its true behaviour -- which is the
    regime the 768x8 family was actually in (l_shape 1.36e6 at step 100).
    """
    g = torch.Generator().manual_seed(seed)
    basis = torch.randn(2, D, generator=g)
    coeffs = torch.randn(n, 2, generator=g)
    return coeffs @ basis


def test_unnormalized_shape_scales_with_sample_count():
    """The defect, pinned: double N, roughly double l_shape."""
    small = VISReg(num_proj=32)(_collapsed_z(256))["l_shape"].item()
    large = VISReg(num_proj=32)(_collapsed_z(512))["l_shape"].item()
    ratio = large / small
    assert 1.6 < ratio < 2.4, f"expected ~2x N-scaling, got {ratio:.2f}x"


def test_normalized_shape_is_stable_across_sample_count():
    small = VISReg(num_proj=32, shape_normalize=True)(
        _collapsed_z(256))["l_shape"].item()
    large = VISReg(num_proj=32, shape_normalize=True)(
        _collapsed_z(512))["l_shape"].item()
    ratio = large / small
    assert 0.75 < ratio < 1.35, f"expected scale-free, got {ratio:.2f}x"


def test_normalized_shape_is_the_unnormalized_one_divided_by_n():
    n = 128
    a = VISReg(num_proj=16)(_z(n, seed=3))["l_shape"].item()
    b = VISReg(num_proj=16, shape_normalize=True)(_z(n, seed=3))["l_shape"].item()
    assert abs(b - a / n) < 1e-4 * max(1.0, abs(b)), (
        f"normalized {b} should equal unnormalized {a} / N={n} = {a / n}"
    )


def test_default_is_unnormalized():
    """The opt-in contract: completed families keep their meaning."""
    assert VISReg().shape_normalize is False


def test_normalization_does_not_change_the_other_terms():
    n = 128
    plain = VISReg(num_proj=16)(_z(n, seed=5))
    norm = VISReg(num_proj=16, shape_normalize=True)(_z(n, seed=5))
    for k in ("l_scale", "l_center"):
        assert abs(plain[k].item() - norm[k].item()) < 1e-6, k


def test_gradient_still_flows_when_normalized():
    """The property VISReg exists for -- a non-vanishing gradient -- must
    survive the normalization."""
    z = _z(128, seed=7).requires_grad_(True)
    out = VISReg(num_proj=16, shape_normalize=True)(z)
    out["l_reg"].backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0
