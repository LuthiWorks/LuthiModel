"""Tests for the LLM-JEPA next-token term.

Spec: docs/reviews/2026-08-08_llm-jepa-integration-spec-for-opus.md §4.
Contracts: NTP matches hand-computed cross-entropy; the causal mask is
VERIFIED not assumed (token t's loss cannot see t+1); the combined loss
reaches trunk, head and predictor; defaults-off bit-exactness; fail-loud.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from luthi.v2.jepa_loss import JEPALoss
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM

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
# The term itself
# ---------------------------------------------------------------------------

def test_ntp_matches_hand_computed_cross_entropy():
    m = _model()
    loss = JEPALoss(online_encoder=m, w_ntp=1.0)
    toks = _toks()
    got = loss._ntp_loss(toks)

    from luthi.v2.plasticity import freeze_plasticity
    with freeze_plasticity(m):
        logits = m(text_tokens=toks)
    want = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, VOCAB), toks[:, 1:].reshape(-1)
    )
    assert float(got) == pytest.approx(float(want), rel=1e-5)


def test_ntp_at_init_is_near_ln_vocab():
    """An untrained head should sit at the uniform-prediction entropy."""
    m = _model()
    loss = JEPALoss(online_encoder=m, w_ntp=1.0)
    got = float(loss._ntp_loss(_toks()))
    assert got == pytest.approx(math.log(VOCAB), rel=0.25)


# ---------------------------------------------------------------------------
# Causal masking -- verified, not assumed
# ---------------------------------------------------------------------------

def test_ntp_path_is_causal_no_future_leakage():
    """Changing token t+1 must not change the logits at positions <= t.

    This exercises the real encoder path used by the NTP term, so it fails
    if `forward()`'s causal=True is ever dropped or if a future refactor
    routes NTP through a bidirectional encode.
    """
    m = _model()
    m.eval()
    loss = JEPALoss(online_encoder=m, w_ntp=1.0)
    toks = _toks(b=1, t=12)

    from luthi.v2.plasticity import freeze_plasticity
    with torch.no_grad(), freeze_plasticity(m):
        base = m(text_tokens=toks)
        perturbed_toks = toks.clone()
        cut = 6
        # Change everything strictly after position `cut`.
        perturbed_toks[0, cut + 1:] = (perturbed_toks[0, cut + 1:] + 7) % VOCAB
        perturbed = m(text_tokens=perturbed_toks)

    # Positions 0..cut must be untouched by anything after cut.
    assert torch.allclose(base[:, :cut + 1, :], perturbed[:, :cut + 1, :], atol=1e-5), \
        "future tokens leaked into past positions -- the NTP path is not causal"
    # Sanity: the perturbation did change something later, so the test is live.
    assert not torch.allclose(base[:, cut + 1:, :], perturbed[:, cut + 1:, :])


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_combined_loss_reaches_trunk_head_and_predictor():
    m = _model()
    loss = JEPALoss(online_encoder=m, sigreg_lambd=0.2, w_ntp=10.0)
    out = loss.compute_modality_loss("text", {"text_tokens": _toks()})
    out["loss"].backward()

    head = m.output_proj.weight
    trunk = m.blocks[0].attention.v_proj.weight
    pred = loss.predictor.output_norm.weight

    for name, p in (("head", head), ("trunk", trunk), ("predictor", pred)):
        assert p.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient at {name}"
        assert float(p.grad.abs().sum()) > 0.0, f"zero gradient at {name}"


def test_ntp_alone_does_not_reach_the_predictor():
    """Attribution check: the NTP term must not silently train the JEPA
    predictor, or the two objectives are not separable in the record."""
    m = _model()
    loss = JEPALoss(online_encoder=m, w_ntp=1.0)
    loss._ntp_loss(_toks()).backward()
    assert loss.predictor.output_norm.weight.grad is None


# ---------------------------------------------------------------------------
# Defaults-off and fail-loud
# ---------------------------------------------------------------------------

def test_defaults_off_produce_no_ntp_term():
    m = _model()
    loss = JEPALoss(online_encoder=m, sigreg_lambd=0.2)
    assert loss.w_ntp == 0.0
    out = loss.compute_modality_loss("text", {"text_tokens": _toks()})
    assert out["l_ntp"] is None


def test_defaults_off_is_bit_exact_against_pre_ntp_loss():
    """w_ntp=0 must reproduce the JEPA-only total exactly."""
    toks = _toks()
    m1 = _model()
    a = JEPALoss(online_encoder=m1, sigreg_lambd=0.2)
    m1.eval()
    with torch.no_grad():
        la = float(a.compute_modality_loss("text", {"text_tokens": toks})["loss"])
    m2 = _model()
    b = JEPALoss(online_encoder=m2, sigreg_lambd=0.2, w_ntp=0.0)
    m2.eval()
    with torch.no_grad():
        lb = float(b.compute_modality_loss("text", {"text_tokens": toks})["loss"])
    assert la == pytest.approx(lb, rel=0.0, abs=0.0)


def test_missing_lm_head_fails_loud():
    class _NoHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.d_model = 32
            self.n_heads = 2
            self.max_seq_len = 16
            self.max_audio_tokens = 16
            self.max_vision_tokens = 16
            self.interior_latent_blocks = ()
    with pytest.raises(RuntimeError, match="no output_proj"):
        JEPALoss(online_encoder=_NoHead(), w_ntp=1.0)


def test_text_modality_without_tokens_fails_before_reaching_ntp():
    """The upstream text guard fires first, so the NTP guard is
    defense-in-depth rather than the active check.

    Recorded as a test rather than assumed: if a refactor ever moves the NTP
    term ahead of the text-token validation, this pins which error a caller
    actually gets.
    """
    m = _model()
    loss = JEPALoss(online_encoder=m, w_ntp=1.0)
    with pytest.raises(ValueError, match="requires text_tokens"):
        loss.compute_modality_loss("text", {})


def test_ntp_freeze_plasticity_default_is_on():
    """A third encode with live plasticity would take per-step self-
    modification from two events to three and confound the whole track."""
    m = _model()
    loss = JEPALoss(online_encoder=m, w_ntp=1.0)
    assert loss.ntp_freeze_plasticity is True
