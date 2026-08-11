"""Tests for VISReg (arXiv 2606.02572) and its JEPALoss replacement path.

Rulings: docs/reviews/2026-08-10_pruning-and-visreg-brief-for-opus.md.
The one test that carries the module's reason to exist is
test_collapse_gradient_nonvanishing: the sorted-quantile objective must
keep gradient signal on a near-collapsed input, which is precisely where
the Epps-Pulley statistic's gradient vanishes (the paper's motivation,
and our measured depth-8 disease).
"""
import math

import pytest
import torch

from luthi.v2.jepa_loss import JEPALoss
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM
from luthi.v2.visreg import VISReg

VOCAB = 256


def _model(**over):
    mk = dict(
        vocab_size=VOCAB, d_model=32, n_blocks=2, n_heads=2, ffn_expansion=1,
        max_seq_len=16, backward_pass_enabled=False, consolidation_enabled=False,
        learning_gain_enabled=False, relative_trust=True,
        episode_recall_threshold=0.7, mu_pc_enabled=False,
    )
    mk.update(over)
    torch.manual_seed(0)
    return MultimodalPredictiveCodingLM(**mk)


def _toks(b=2, t=12):
    torch.manual_seed(1)
    return torch.randint(0, VOCAB, (b, t))


# ---------------------------------------------------------------------------
# The module itself
# ---------------------------------------------------------------------------

def test_quantiles_hand_computed():
    """N=3 plotting positions i/(N+1) = 0.25, 0.5, 0.75 -> known icdf values."""
    vr = VISReg(num_proj=8)
    q = vr._quantiles(3, torch.device("cpu"), torch.float32)
    assert q[0].item() == pytest.approx(-0.6744898, abs=1e-4)
    assert q[1].item() == pytest.approx(0.0, abs=1e-6)
    assert q[2].item() == pytest.approx(0.6744898, abs=1e-4)


def test_gaussian_input_terms_near_zero():
    torch.manual_seed(7)
    vr = VISReg(num_proj=64)
    out = vr(torch.randn(2048, 32))
    assert float(out["l_scale"]) < 0.01
    assert float(out["l_center"]) < 0.05
    # Shape: empirical order statistics of a true Gaussian sample sit close
    # to the quantiles; the summed-over-N form still stays small.
    assert float(out["l_shape"]) < 5.0


def test_scale_term_fires_on_shrink():
    torch.manual_seed(7)
    vr = VISReg(num_proj=64)
    out = vr(0.01 * torch.randn(2048, 32))
    # Every dim's std ~0.01 -> (1 - 0.01)^2 ~ 0.98.
    assert float(out["l_scale"]) == pytest.approx(0.98, abs=0.02)


def test_center_term_fires_on_offset():
    torch.manual_seed(7)
    vr = VISReg(num_proj=64)
    d = 32
    out = vr(torch.randn(2048, d) + 3.0)
    # ||mu||^2 ~ D * 9 (paper form, not divided by D).
    assert float(out["l_center"]) == pytest.approx(d * 9.0, rel=0.05)


def test_shape_term_fires_on_rank_collapse():
    torch.manual_seed(7)
    vr = VISReg(num_proj=64)
    n, d = 2048, 32
    gauss = float(vr(torch.randn(n, d))["l_shape"])
    direction = torch.randn(d)
    direction = direction / direction.norm()
    rank1 = torch.randn(n, 1) * direction  # rank-1: all dims one signal
    collapsed = float(vr(rank1)["l_shape"])
    # After per-dim standardization every projection of a rank-1 cloud is a
    # (scaled) copy of ONE Gaussian sample vector -- marginals are Gaussian
    # in law, but the per-dim std division cannot fix cross-dim structure:
    # projections have wildly varying variance across slices, so sorted
    # values miss the quantiles badly on most slices.
    assert collapsed > gauss * 5


def test_collapse_gradient_nonvanishing():
    """THE property VISReg is bought for (paper's motivation, our disease):
    gradient signal must survive deep collapse instead of vanishing the way
    the Epps-Pulley CF statistic's does."""
    torch.manual_seed(11)
    vr = VISReg(num_proj=64)
    vr.eval()  # fixed projections: compare gradients, not RNG draws
    n, d = 1024, 32
    direction = torch.randn(d)
    direction = direction / direction.norm()

    def grad_norm(eps: float) -> float:
        torch.manual_seed(13)
        z = (torch.randn(n, 1) * direction + eps * torch.randn(n, d))
        z = z.detach().requires_grad_(True)
        out = vr(z)
        (g,) = torch.autograd.grad(out["l_reg"], z)
        return float(g.norm())

    mild = grad_norm(0.3)
    severe = grad_norm(0.01)
    deepest = grad_norm(0.001)
    # Non-vanishing: the near-floor gradients hold the same order of
    # magnitude as the mild-collapse gradient rather than dying with eps.
    assert severe > 0.1 * mild
    assert deepest > 0.1 * mild
    assert deepest > 1e-3


def test_global_rng_stream_untouched():
    """The 2026-08-01 SIGReg lesson, not reintroduced: a forward must not
    advance the global RNG stream."""
    vr = VISReg(num_proj=32)
    torch.manual_seed(123)
    z = torch.randn(64, 16)
    torch.manual_seed(123)
    _ = torch.randn(64, 16)
    baseline = torch.randn(5)
    torch.manual_seed(123)
    z2 = torch.randn(64, 16)
    vr(z2)
    after = torch.randn(5)
    assert torch.equal(z, z2)
    assert torch.equal(baseline, after)


def test_eval_does_not_advance_step_train_does():
    vr = VISReg(num_proj=32)
    z = torch.randn(64, 16)
    vr.train()
    vr(z)
    assert int(vr.global_step) == 1
    vr.eval()
    vr(z)
    assert int(vr.global_step) == 1


def test_bad_input_fails_loud():
    vr = VISReg(num_proj=32)
    with pytest.raises(ValueError, match="N >= 2"):
        vr(torch.randn(1, 16))
    with pytest.raises(ValueError, match=r"\(N, D\)"):
        vr(torch.randn(2, 3, 16))


# ---------------------------------------------------------------------------
# JEPALoss integration: replacement, convex mix, fail-loud contracts
# ---------------------------------------------------------------------------

def test_defaults_off_no_visreg_module_and_no_terms():
    m = _model()
    loss = JEPALoss(online_encoder=m)
    assert not hasattr(loss, "visreg")
    out = loss.compute_modality_loss("text", {"text_tokens": _toks()})
    assert out["l_visreg"] is None
    assert out["l_vis_scale"] is None
    assert out["l_sigreg"] is not None  # SIGReg path ran, unchanged


def test_defaults_off_total_is_additive_sigreg_form():
    m = _model()
    loss = JEPALoss(online_encoder=m)
    out = loss.compute_modality_loss("text", {"text_tokens": _toks()})
    want = float(out["l_pred"]) + loss.sigreg_lambd * float(out["l_sigreg"])
    assert float(out["loss"]) == pytest.approx(want, rel=1e-5)


def test_visreg_replaces_sigreg_and_convex_mix_holds():
    m = _model()
    loss = JEPALoss(
        online_encoder=m, sigreg_projection="none",
        visreg_lambda=0.6, visreg_num_proj=64,
    )
    out = loss.compute_modality_loss("text", {"text_tokens": _toks()})
    assert out["l_sigreg"] is None            # replaced, not zero-weighted
    assert out["l_visreg"] is not None
    for k in ("l_vis_scale", "l_vis_shape", "l_vis_center"):
        assert out[k] is not None
    want = 0.4 * float(out["l_pred"]) + 0.6 * float(out["l_visreg"])
    assert float(out["loss"]) == pytest.approx(want, rel=1e-5)


def test_visreg_component_sum_matches_l_reg():
    m = _model()
    loss = JEPALoss(
        online_encoder=m, sigreg_projection="none",
        visreg_lambda=0.6, visreg_num_proj=64,
    )
    out = loss.compute_modality_loss("text", {"text_tokens": _toks()})
    want = (
        float(out["l_vis_scale"]) + float(out["l_vis_shape"])
        + float(out["l_vis_center"])
    )
    assert float(out["l_visreg"]) == pytest.approx(want, rel=1e-5)


def test_visreg_gradient_reaches_trunk_and_predictor():
    m = _model()
    loss = JEPALoss(
        online_encoder=m, sigreg_projection="none",
        visreg_lambda=0.6, visreg_num_proj=64,
    )
    out = loss.compute_modality_loss("text", {"text_tokens": _toks()})
    out["loss"].backward()
    trunk = [p for p in m.parameters() if p.requires_grad and p.grad is not None]
    pred = [p for p in loss.predictor.parameters()
            if p.requires_grad and p.grad is not None]
    assert any(float(p.grad.abs().sum()) > 0 for p in trunk)
    assert any(float(p.grad.abs().sum()) > 0 for p in pred)


def test_visreg_through_projection_head_fails_loud():
    """Trunk-latents ruling: the learnable head absorbs scale (measured);
    VISReg behind it would be blind and the failure misattributed."""
    with pytest.raises(RuntimeError, match="trunk latents"):
        JEPALoss(
            online_encoder=_model(), sigreg_projection="linear",
            visreg_lambda=0.6,
        )


def test_visreg_lambda_outside_convex_range_fails_loud():
    for bad in (1.0, 1.5, -0.2):
        with pytest.raises((ValueError, RuntimeError)):
            JEPALoss(
                online_encoder=_model(), sigreg_projection="none",
                visreg_lambda=bad,
            )
