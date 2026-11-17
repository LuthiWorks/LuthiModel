"""A read by a fixed map of the prediction's error (brief 8, Part 2, 2026-11-17).

What must hold:

* the model is bit-identical after the eval, and after the fit;
* the fit and the read share no document;
* the read takes no backward pass, and moves the living weights and nothing
  else;
* it is refused in training mode, and it is off by default.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from luthi.living_extra_state import collect_living_extra_state  # noqa: E402
from luthi.v2 import error_map_read as emr  # noqa: E402
from luthi.v2.adaptation_eval import (  # noqa: E402
    adaptation_eval, holdout_document_ranges, holdout_documents,
)
from luthi.v2.jepa_loss import JEPALoss  # noqa: E402
from luthi.v2.living_weight_gradient import living_layers  # noqa: E402
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM  # noqa: E402

V, D, L = 64, 32, 16


def _loss() -> JEPALoss:
    torch.manual_seed(0)
    m = MultimodalPredictiveCodingLM(vocab_size=V, d_model=D, n_blocks=2, n_heads=4, max_seq_len=L)
    return JEPALoss(m, visreg_lambda=0.6, sigreg_projection="none").eval()


def _windows(n: int, seed: int = 1) -> torch.Tensor:
    return torch.randint(0, V, (n, L), generator=torch.Generator().manual_seed(seed))


def _everything(lm) -> tuple[dict, dict]:
    return ({k: v.detach().clone() for k, v in lm.state_dict().items()},
            copy.deepcopy(collect_living_extra_state(lm.online_encoder)))


def _same(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _assert_untouched(lm, before) -> None:
    state, extra = _everything(lm)
    for k in before[0]:
        assert torch.equal(before[0][k], state[k]), k
    assert _same(before[1], extra)
    assert not any(b.requires_grad for b in lm.buffers())
    assert all(p.grad is None for p in lm.parameters())


def test_the_fit_changes_nothing_and_says_how_well_it_reads_on_windows_held_back():
    lm = _loss()
    before = _everything(lm)
    maps = emr.fit_error_maps(lm, _windows(24))
    _assert_untouched(lm, before)
    assert maps.fitted_on == 18 and maps.check["windows_held_back"] == 6
    assert len(maps.maps) == 2 and all(F.shape == (D, D) for F in maps.maps)
    assert all(-1.0 <= c["cosine"] <= 1.0 for c in maps.check["median_by_layer"])


def test_the_model_is_bit_identical_after_an_eval_that_reads_by_error_map():
    lm = _loss()
    maps = emr.fit_error_maps(lm, _windows(24, seed=9))
    docs = holdout_documents(_windows(24), n_read=2, n_score=1, n_docs=4)
    before = _everything(lm)
    res = adaptation_eval(lm, docs, n_read=2, read_by_error_map=maps, error_map_rate=1.0)
    _assert_untouched(lm, before)
    assert res["summary"]["exact_null"] is False              # it did move something
    told = res["settings"]["read_by_error_map"]
    assert told["kind"] == "map" and told["rate"] == 1.0 and told["fitted_on"] == 18
    shares = res["settings"]["living_step_share_of_weight"]
    assert len(shares) == 2 and all(s["median"] > 0 for s in shares)
    assert shares[0]["steps"] == 4 * 2 * 2                     # documents, B and C, windows


def test_a_read_moves_the_living_weights_and_nothing_else_with_no_backward_pass(monkeypatch):
    lm = _loss()
    maps = emr.fit_error_maps(lm, _windows(24, seed=9))
    before, _ = _everything(lm)

    def no_backward(*args, **kwargs):
        raise AssertionError("a backward pass was taken during the read")

    monkeypatch.setattr(torch.autograd, "grad", no_backward)
    monkeypatch.setattr(torch.Tensor, "backward", no_backward)
    emr.read_by_error_map(lm, _windows(3, seed=4), maps, rate=1.0)
    monkeypatch.undo()
    after, _ = _everything(lm)
    moved = sorted(k for k in before if not torch.equal(before[k], after[k]))
    assert moved == sorted(f"online_encoder.blocks.{i}.living_ffn.weight" for i in range(2))
    assert not any(layer._plasticity_frozen for layer in living_layers(lm.online_encoder))


def test_a_step_is_the_map_of_the_error_times_the_mean_context_input():
    lm = _loss()
    maps = emr.fit_error_maps(lm, _windows(24, seed=9))
    window = _windows(1, seed=4)
    e, x_mean, _ = emr._window_signals(lm, window, wanted=False)
    layers = living_layers(lm.online_encoder)
    was = [layer.weight.clone() for layer in layers]
    emr.read_by_error_map(lm, window, maps, rate=0.5)
    for layer, w, F, u, x in zip(layers, was, maps.maps, maps.wanted_mean, x_mean):
        expected = -0.5 * torch.outer(u + F @ (e - maps.e_mean), x)
        assert torch.allclose((layer.weight - w).double(), expected, rtol=1e-4, atol=1e-7)


def test_the_fitted_offset_is_close_to_the_gradient_it_stands_for():
    # For one window the gradient of the score (target held) for a living
    # weight is close to outer(summed output gradient, mean input): the
    # offset the rule is built on. Held here on the toy so that a change to
    # the signals cannot quietly make the rule about something else.
    from luthi.v2.living_weight_gradient import living_weights_on_the_graph

    lm = _loss()
    window = _windows(1, seed=4)
    _, x_mean, summed = emr._window_signals(lm, window, wanted=True)
    model = lm.online_encoder
    ctx = int(L * lm.context_fraction)
    with living_weights_on_the_graph(model) as layers, torch.enable_grad():
        context = model.encode(text_tokens=window[:, :ctx], causal=False)["per_modality"]["text"]
        full = model.encode(text_tokens=window, causal=False)["per_modality"]["text"]
        target = full[:, ctx:, :].detach()
        positions = torch.arange(ctx, L).unsqueeze(0)
        predicted = lm.predictor(context, positions, lm.action_token)
        var = (target - target.mean(dim=(0, 1), keepdim=True)).pow(2).mean()
        held = (predicted - target).pow(2).mean() / var
        grads = torch.autograd.grad(held, [layer.weight for layer in layers])
    for g, u, x in zip(grads, summed, x_mean):
        offset = torch.outer(u, x)
        cos = float((g.double() * offset).sum() / (g.double().norm() * offset.norm()))
        assert cos > 0.5, cos
        assert 0.3 < float(offset.norm() / g.double().norm()) < 3.0


def test_the_three_kinds_differ_and_the_common_kind_has_no_map():
    lm = _loss()
    fit = _windows(24, seed=9)
    by_kind = {kind: emr.fit_error_maps(lm, fit, kind=kind) for kind in emr.KINDS}
    assert float(by_kind["common"].maps[0].abs().sum()) == 0.0
    assert not torch.equal(by_kind["map"].maps[0], by_kind["shuffled"].maps[0])
    for kind in emr.KINDS:
        assert torch.equal(by_kind[kind].wanted_mean[0], by_kind["map"].wanted_mean[0])
    with pytest.raises(ValueError, match="kind must be one of"):
        emr.fit_error_maps(lm, fit, kind="other")


def test_the_fit_and_the_read_share_no_document():
    total = 400
    taken = [r for phase in (0, 1) for r in holdout_document_ranges(
        total, n_read=6, n_score=2, n_docs=20, phase=phase)]
    fit_at = emr.windows_outside_documents(total, taken, 40)
    assert len(fit_at) == len(set(fit_at)) == 40
    in_a_document = {i for first, stop in taken for i in range(first, stop)}
    assert not in_a_document & set(fit_at)
    # And the ranges are the documents: holdout_documents cuts the same windows.
    windows = torch.arange(total).unsqueeze(1).expand(total, 3)
    for phase in (0, 1):
        docs = holdout_documents(windows, n_read=6, n_score=2, n_docs=20, phase=phase)
        ranges = holdout_document_ranges(total, n_read=6, n_score=2, n_docs=20, phase=phase)
        assert [(int(d[0, 0]), int(d[-1, 0]) + 1) for d in docs] == ranges
    with pytest.raises(ValueError, match="are in no document"):
        emr.windows_outside_documents(total, taken, 81)


def test_refused_in_training_mode_and_off_by_default():
    lm = _loss()
    maps = emr.fit_error_maps(lm, _windows(24, seed=9))
    lm.train()
    with pytest.raises(RuntimeError, match="training mode"):
        emr.read_by_error_map(lm, _windows(2), maps, rate=1.0)
    with pytest.raises(RuntimeError, match="training mode"):
        emr.fit_error_maps(lm, _windows(24))
    lm.eval()
    docs = holdout_documents(_windows(24), n_read=2, n_score=1, n_docs=4)
    plain = adaptation_eval(_loss(), docs, n_read=2)
    assert plain["settings"]["read_by_error_map"] is None
    with pytest.raises(ValueError, match="needs error_map_rate"):
        adaptation_eval(lm, docs, n_read=2, read_by_error_map=maps)
    with pytest.raises(ValueError, match="needs read_by_error_map"):
        adaptation_eval(lm, docs, n_read=2, error_map_rate=1.0)
    with pytest.raises(ValueError, match="not both"):
        adaptation_eval(lm, docs, n_read=2, read_by_error_map=maps, error_map_rate=1.0,
                        read_by_gradient=1e-3)
