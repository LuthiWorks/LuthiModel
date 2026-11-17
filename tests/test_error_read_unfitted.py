"""Addendum 2 to brief 8 (2026-11-17): the held score, the kinds of the
error read that need nothing fitted, and a read in context.

The fitted kinds and the read's own guards are in test_error_map_read.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from luthi.v2 import adaptation_eval as ae  # noqa: E402
from luthi.v2 import error_map_read as emr  # noqa: E402
from luthi.v2.adaptation_eval import adaptation_eval, holdout_documents  # noqa: E402
from luthi.v2.jepa_loss import JEPALoss  # noqa: E402
from luthi.v2.living_weight_gradient import living_layers  # noqa: E402
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM  # noqa: E402
from tests.test_error_map_read import (  # noqa: E402
    D, L, V, _assert_untouched, _everything, _loss, _windows,
)


def _dead() -> JEPALoss:
    torch.manual_seed(0)
    m = MultimodalPredictiveCodingLM(vocab_size=V, d_model=D, n_blocks=2, n_heads=4,
                                     max_seq_len=L, dead_ffn=True)
    return JEPALoss(m, visreg_lambda=0.6, sigreg_projection="none").eval()


def _docs(n_docs: int = 4) -> list[torch.Tensor]:
    return holdout_documents(_windows(n_docs * 3 * 2), n_read=2, n_score=1, n_docs=n_docs)


# ---- the held score ------------------------------------------------------

def test_the_held_score_is_off_by_default_and_changes_nothing_else():
    plain = adaptation_eval(_loss(), _docs(), n_read=2)
    held = adaptation_eval(_loss(), _docs(), n_read=2, held_score=True)
    assert "held" not in plain["per_doc"][0]["A"] and "held" not in plain["summary"]
    for p, h in zip(plain["per_doc"], held["per_doc"]):
        for arm in "ABC":
            for key in ("l_pred", "nmse", "target_eff_rank"):
                assert p[arm][key] == h[arm][key]
    assert held["settings"]["held_score"] is True


def test_for_arm_a_the_held_score_is_nmse_and_nothing_has_drifted():
    res = adaptation_eval(_loss(), _docs(), n_read=2, held_score=True)
    for p in res["per_doc"]:
        assert p["A"]["held"] == pytest.approx(p["A"]["nmse"], rel=1e-5)
        assert p["A"]["drift"] == 0.0
        assert p["B"]["drift"] > 0.0, "a read moved the weights, so the targets moved"
        assert p["B"]["held"] != p["B"]["nmse"]
    assert set(res["summary"]["held"]) == {"A_mean", "B_minus_A", "C_minus_A", "B_minus_C"}
    assert res["summary"]["drift"]["B"]["mean"] > 0.0


def test_a_dead_arm_is_still_an_exact_null_with_the_held_score_on():
    res = adaptation_eval(_dead(), _docs(), n_read=2, held_score=True)
    assert res["summary"]["exact_null"] is True
    assert all(p["B"]["drift"] == 0.0 and p["B"]["held"] == p["A"]["held"] for p in res["per_doc"])


def test_the_held_score_holds_the_target_and_nmse_does_not():
    # Move the living weights by hand and score both ways. The held score
    # must be the moved model's prediction against the UNMOVED model's
    # targets; nmse is against its own.
    lm = _loss()
    tail = _windows(2, seed=6)
    from luthi.v2.plasticity import freeze_plasticity

    def frozen_pass() -> dict:
        with freeze_plasticity(lm.online_encoder), torch.no_grad():
            return lm.compute_modality_loss("text", {"text_tokens": tail})

    a, ref = ae._score_held(lm, tail, None)
    r = frozen_pass()
    target_a = r["target_latents"][:, r["ctx_len"]:, :].clone()
    for layer in living_layers(lm.online_encoder):
        with torch.no_grad():
            layer.weight.mul_(1.05)
    b, _ = ae._score_held(lm, tail, ref)
    r = frozen_pass()
    var_a = (target_a - target_a.mean(dim=(0, 1), keepdim=True)).pow(2).mean()
    expected = float((r["predicted_target"] - target_a).pow(2).mean() / var_a)
    drift = float((r["target_latents"][:, r["ctx_len"]:, :] - target_a).pow(2).mean() / var_a)
    assert b["held"] == pytest.approx(expected, rel=1e-6)
    assert b["drift"] == pytest.approx(drift, rel=1e-6) and drift > 0
    assert b["nmse"] == pytest.approx(ae._score(lm, tail)["nmse"], rel=1e-6)
    assert abs(b["held"] - b["nmse"]) > 1e-6


# ---- the kinds that need nothing fitted ----------------------------------

def test_the_error_kind_moves_the_weight_by_the_error_times_the_mean_context_input():
    window = _windows(1, seed=4)
    moved = {}
    for kind in ("error", "error_turned"):
        lm = _loss()
        e, x_mean, _ = emr._window_signals(lm, window, wanted=False)
        layers = living_layers(lm.online_encoder)
        was = [layer.weight.clone() for layer in layers]
        shares: list = []
        emr.read_by_error_map(lm, window, emr.unfitted(kind), rate=0.5, step_shares=shares)
        moved[kind] = [(layer.weight - w).double() for layer, w in zip(layers, was)]
        sign = -0.5 if kind == "error" else +0.5
        for change, x, w, share in zip(moved[kind], x_mean, was, shares[0]):
            assert torch.allclose(change, sign * torch.outer(e, x), rtol=1e-4, atol=1e-7)
            assert share == pytest.approx(
                float(0.5 * e.norm() * x.norm() / w.double().norm()), rel=1e-6)
    for a, b in zip(moved["error"], moved["error_turned"]):
        assert torch.allclose(a, -b, rtol=1e-4, atol=1e-7)


def test_the_size_a_rate_is_set_from_is_the_size_of_a_step():
    lm = _loss()
    window = _windows(1, seed=4)
    unit = emr.step_share_at_rate_one(lm, window)
    shares: list = []
    emr.read_by_error_map(lm, window, emr.unfitted("error"), rate=1.0, step_shares=shares)
    assert unit == pytest.approx(shares[0], rel=1e-6)


def test_the_shuffled_kind_takes_its_error_from_the_other_window():
    lm = _loss()
    window, other = _windows(1, seed=4), _windows(1, seed=5)
    e_other, _, _ = emr._window_signals(lm, other, wanted=False)
    _, x_mean, _ = emr._window_signals(lm, window, wanted=False)
    layers = living_layers(lm.online_encoder)
    was = [layer.weight.clone() for layer in layers]
    emr.read_by_error_map(lm, window, emr.unfitted("error_shuffled"), rate=0.5, errors_from=other)
    for layer, w, x in zip(layers, was, x_mean):
        assert torch.allclose((layer.weight - w).double(), -0.5 * torch.outer(e_other, x),
                              rtol=1e-4, atol=1e-7)
    with pytest.raises(ValueError, match="errors_from is given"):
        emr.read_by_error_map(lm, window, emr.unfitted("error"), rate=0.5, errors_from=other)
    with pytest.raises(ValueError, match="errors_from is given"):
        emr.read_by_error_map(lm, window, emr.unfitted("error_shuffled"), rate=0.5)


def test_the_bias_kind_shifts_the_output_and_leaves_the_weight_alone():
    lm = _loss()
    before = _everything(lm)
    window = _windows(1, seed=4)
    bias = emr.LivingBias(lm.online_encoder)
    try:
        frozen = ae._score(lm, window)                        # a zero shift changes nothing
        e, x_mean, _ = emr._window_signals(lm, window, wanted=False)
        emr.read_by_error_map(lm, window, emr.unfitted("bias"), rate=0.5, bias=bias)
        _assert_untouched(lm, before)                         # no weight, no buffer
        for vector, x in zip(bias.vectors, x_mean):
            assert torch.allclose(vector, -0.5 * float(x @ x) * e, rtol=1e-6, atol=1e-12)
        assert ae._score(lm, window) != frozen, "the shift reaches the score"
        bias.zero()
        assert ae._score(lm, window) == frozen
    finally:
        bias.remove()
    assert all(len(layer._forward_hooks) == 0 for layer in living_layers(lm.online_encoder))


def test_the_bias_and_the_error_kinds_make_the_same_shift_for_the_mean_input():
    lm = _loss()
    window = _windows(1, seed=4)
    e, x_mean, _ = emr._window_signals(lm, window, wanted=False)
    layers = living_layers(lm.online_encoder)
    was = [layer.weight.clone() for layer in layers]
    emr.read_by_error_map(lm, window, emr.unfitted("error"), rate=0.5)
    for layer, w, x in zip(layers, was, x_mean):
        shift = (layer.weight - w).double() @ x
        assert torch.allclose(shift, -0.5 * float(x @ x) * e, rtol=1e-4, atol=1e-9)


@pytest.mark.parametrize("kind", ["error", "error_turned", "error_shuffled", "bias"])
def test_the_model_is_bit_identical_after_an_eval_of_each_unfitted_kind(kind):
    lm = _loss()
    before = _everything(lm)
    res = adaptation_eval(lm, _docs(), n_read=2, read_by_error_map=emr.unfitted(kind),
                          error_map_rate=1.0, held_score=True)
    _assert_untouched(lm, before)
    assert all(len(layer._forward_hooks) == 0 for layer in living_layers(lm.online_encoder))
    assert res["summary"]["exact_null"] is False
    assert res["settings"]["read_by_error_map"]["kind"] == kind
    assert res["settings"]["read_by_error_map"]["fitted_on"] == 0


def test_turning_the_sign_turns_the_change_in_the_held_score_at_a_small_rate():
    docs = _docs()
    by_kind = {
        kind: adaptation_eval(_loss(), docs, n_read=2, held_score=True,
                              read_by_error_map=emr.unfitted(kind), error_map_rate=1e-3)
        for kind in ("error", "error_turned")}
    for p, q in zip(by_kind["error"]["per_doc"], by_kind["error_turned"]["per_doc"]):
        down, up = p["B"]["held"] - p["A"]["held"], q["B"]["held"] - q["A"]["held"]
        assert down != 0 and up != 0 and (down > 0) != (up > 0), (down, up)


def test_the_shuffled_kind_needs_a_third_document():
    with pytest.raises(ValueError, match="at least 4 documents"):
        adaptation_eval(_loss(), _docs(2), n_read=2,
                        read_by_error_map=emr.unfitted("error_shuffled"), error_map_rate=1.0)


# ---- once in context -----------------------------------------------------

def test_a_read_in_context_is_the_plastic_read_and_then_the_step():
    docs = _docs()
    alone = adaptation_eval(_loss(), docs, n_read=2, read_by_error_map=emr.unfitted("error"),
                            error_map_rate=1.0)
    lm = _loss()
    before = _everything(lm)
    both = adaptation_eval(lm, docs, n_read=2, read_by_error_map=emr.unfitted("error"),
                           error_map_rate=1.0, error_map_in_context=True)
    _assert_untouched(lm, before)
    plastic = adaptation_eval(_loss(), docs, n_read=2)
    assert both["settings"]["read_by_error_map"]["in_context"] is True
    b = [p["B"]["nmse"] for p in both["per_doc"]]
    assert b != [p["B"]["nmse"] for p in alone["per_doc"]]
    assert b != [p["B"]["nmse"] for p in plastic["per_doc"]]
    # At rate 0 the step is nothing, and a read in context is the plastic read.
    zero = adaptation_eval(_loss(), docs, n_read=2, read_by_error_map=emr.unfitted("error"),
                           error_map_rate=0.0, error_map_in_context=True)
    assert [p["B"] for p in zero["per_doc"]] == [p["B"] for p in plastic["per_doc"]]
    with pytest.raises(ValueError, match="read alone"):
        adaptation_eval(_loss(), docs, n_read=2, read_by_error_map=emr.unfitted("bias"),
                        error_map_rate=1.0, error_map_in_context=True)
