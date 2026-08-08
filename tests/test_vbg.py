"""Tests for the variance-budget governor (VBG).

Spec: docs/reviews/2026-08-07_variance-budget-governor-spec-for-opus.md §4.
Five contracts: cap zero below budget / positive above; share-term scale
invariance; power-iteration accuracy vs exact SVD; the fail-loud raise; and
defaults-off (no behaviour change for existing arms).
"""
import math

import pytest
import torch

from luthi.v2.jepa_loss import (
    sketched_isotropy_penalty,
    soloist_cap_penalty,
    top_direction_share,
)

K = 64
D = 128


def _sketch(d: int = D, k: int = K) -> torch.Tensor:
    g = torch.Generator().manual_seed(20260807)
    return torch.randn(d, k, generator=g) / math.sqrt(d)


def _unit_vec(k: int = K) -> torch.Tensor:
    g = torch.Generator().manual_seed(20260808)
    v = torch.randn(k, generator=g)
    return v / v.norm()


# ---------------------------------------------------------------------------
# Term A: the cap
# ---------------------------------------------------------------------------

def test_cap_is_zero_below_budget():
    """A direction is allowed its budget -- this is a cap, not a kill."""
    for share in (0.0, 0.01, 0.049, 0.05):
        pen = soloist_cap_penalty(torch.tensor(share), cap=0.05)
        assert float(pen) == 0.0, f"cap fired below budget at share={share}"


def test_cap_is_positive_above_budget_and_grows_quadratically():
    cap = 0.05
    p1 = float(soloist_cap_penalty(torch.tensor(cap + 0.10), cap))
    p2 = float(soloist_cap_penalty(torch.tensor(cap + 0.20), cap))
    assert p1 > 0.0
    # relu(excess)^2 -- doubling the excess quadruples the penalty.
    assert p2 == pytest.approx(4.0 * p1, rel=1e-5)


def test_cap_is_differentiable_above_budget():
    share = torch.tensor(0.30, requires_grad=True)
    soloist_cap_penalty(share, cap=0.05).backward()
    assert share.grad is not None and float(share.grad) > 0.0


# ---------------------------------------------------------------------------
# Term B: scale invariance
# ---------------------------------------------------------------------------

def test_share_term_is_scale_invariant():
    """z and 100z must give the identical trace-normalized penalty.

    This is the whole point of Term B: press on SHAPE, not on scale, so the
    penalty stops fighting the trunk's native std band (0.25-0.35).
    """
    torch.manual_seed(0)
    z = torch.randn(512, D)
    s = _sketch()
    a = sketched_isotropy_penalty(z, s, trace_normalized=True)
    b = sketched_isotropy_penalty(z * 100.0, s, trace_normalized=True)
    assert float(a) == pytest.approx(float(b), rel=1e-4)


def test_raw_share_term_is_NOT_scale_invariant():
    """Guards the contrast: the raw form is the one that fights scale.

    Kept so a future refactor that silently normalizes the raw path fails
    here instead of quietly changing every legacy arm's dose.
    """
    torch.manual_seed(0)
    z = torch.randn(512, D)
    s = _sketch()
    a = sketched_isotropy_penalty(z, s, trace_normalized=False)
    b = sketched_isotropy_penalty(z * 100.0, s, trace_normalized=False)
    assert float(b) > 10.0 * float(a)


# ---------------------------------------------------------------------------
# The power-iteration estimate vs exact SVD
# ---------------------------------------------------------------------------

def _exact_share(z: torch.Tensor, sketch: torch.Tensor) -> float:
    flat = z.reshape(-1, z.shape[-1]) @ sketch
    flat = flat - flat.mean(dim=0, keepdim=True)
    cov = (flat.t() @ flat) / max(flat.shape[0] - 1, 1)
    sv = torch.linalg.svdvals(cov).clamp(min=1e-12)
    return float(sv.max() / sv.sum())


def test_power_iteration_matches_svd_on_random_latents():
    torch.manual_seed(1)
    z = torch.randn(1024, D)
    s = _sketch()
    est, _ = top_direction_share(z, s, _unit_vec(), n_iter=25)
    exact = _exact_share(z, s)
    assert float(est) == pytest.approx(exact, rel=0.05)


def test_power_iteration_matches_svd_on_rank_one_latents():
    """The case the governor exists for: one direction holding everything."""
    torch.manual_seed(2)
    direction = torch.randn(D)
    direction = direction / direction.norm()
    coeffs = torch.randn(1024, 1)
    z = coeffs * direction.unsqueeze(0) + 0.01 * torch.randn(1024, D)
    s = _sketch()
    est, _ = top_direction_share(z, s, _unit_vec(), n_iter=25)
    exact = _exact_share(z, s)
    assert float(est) == pytest.approx(exact, rel=0.05)
    assert exact > 0.5, "rank-1 fixture should be dominated by one direction"


def test_power_iteration_warm_start_converges_in_three_steps():
    """Spec §1: warm-starting is what makes n_iter=3 sufficient.

    Cold-start at 3 iterations is allowed to be loose; after one warm cycle
    the estimate must be within 5% of exact, which is the property the
    persistent buffer buys.
    """
    torch.manual_seed(3)
    direction = torch.randn(D)
    direction = direction / direction.norm()
    z = torch.randn(1024, 1) * direction.unsqueeze(0) + 0.05 * torch.randn(1024, D)
    s = _sketch()
    exact = _exact_share(z, s)
    v = _unit_vec()
    for _ in range(4):                       # simulate successive steps
        est, v = top_direction_share(z, s, v, n_iter=3)
    assert float(est) == pytest.approx(exact, rel=0.05)


def test_power_iteration_gradient_flows_to_latents():
    torch.manual_seed(4)
    z = torch.randn(256, D, requires_grad=True)
    s = _sketch()
    share, _ = top_direction_share(z, s, _unit_vec(), n_iter=3)
    share.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert float(z.grad.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# Fail-loud contract and defaults-off
# ---------------------------------------------------------------------------

class _StubEncoder(torch.nn.Module):
    """Minimal stand-in exposing what JEPALoss touches at construction."""

    def __init__(self, interior_blocks=()):
        super().__init__()
        self.d_model = D
        self.n_heads = 4
        self.max_seq_len = 128
        self.max_audio_tokens = 128
        self.max_vision_tokens = 128
        self.interior_latent_blocks = tuple(interior_blocks)


def _loss_module(**kw):
    from luthi.v2.jepa_loss import JEPALoss
    return JEPALoss(online_encoder=_StubEncoder((0, 3, 6)), **kw)


def test_governor_raises_when_no_interior_latents_are_produced():
    """Fail loud: a silently inert regularizer is the forbidden failure."""
    mod = _loss_module(vbg_cap_weight=1.0, vbg_share_weight=1.0)
    with pytest.raises(RuntimeError, match="silently inert"):
        # The governor block runs off online_result; an empty dict is what a
        # model without interior_latent_blocks configured actually returns.
        mod._raise_if_no_interior({})


def test_defaults_are_off_and_register_no_governor_state():
    mod = _loss_module()
    assert mod.vbg_cap_weight == 0.0
    assert mod.vbg_share_weight == 0.0
    assert mod._vbg_on is False
    assert not hasattr(mod, "vbg_power_vecs")


def test_power_vec_buffer_is_non_persistent():
    """Old checkpoints must load with no strict=False concession."""
    mod = _loss_module(vbg_cap_weight=1.0)
    assert hasattr(mod, "vbg_power_vecs")
    assert "vbg_power_vecs" not in mod.state_dict()


def test_identity_creation_survives_the_directml_eye_landmine():
    """torch.eye(n, device=dml) returns EMPTY; identities are CPU-created.

    Asserted on the penalty's output shape rather than the backend, so the
    contract is checked everywhere the suite runs.
    """
    torch.manual_seed(5)
    z = torch.randn(256, D)
    out = sketched_isotropy_penalty(z, _sketch(), trace_normalized=True)
    assert out.ndim == 0 and torch.isfinite(out)
