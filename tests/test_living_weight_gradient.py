"""The guards on a gradient taken with respect to a living weight.

Brian's ruling of 2026-11-16, 10:29 (docs/DECISIONS.md) allows a measurement
to take one, under three guards. Each has a test here:

1. measurement code only, never a Parameter or an optimizer;
2. refused in training mode;
3. the model is bit-identical afterwards, and no buffer is left requiring a
   gradient.

And the gradient has to be the right one, or everything read from it is
about something else: it is checked against the score itself.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from luthi.living_extra_state import collect_living_extra_state  # noqa: E402
from luthi.v2 import adaptation_eval as ae  # noqa: E402
from luthi.v2 import living_weight_gradient as lwg  # noqa: E402
from luthi.v2.adaptation_eval import adaptation_eval, holdout_documents  # noqa: E402
from luthi.v2.jepa_loss import JEPALoss  # noqa: E402
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM  # noqa: E402

V, D, L = 64, 32, 16


def _loss(**kw) -> JEPALoss:
    torch.manual_seed(0)
    m = MultimodalPredictiveCodingLM(
        vocab_size=V, d_model=D, n_blocks=2, n_heads=4, max_seq_len=L, **kw,
    )
    return JEPALoss(m, visreg_lambda=0.6, sigreg_projection="none").eval()


def _tail(n: int = 3) -> torch.Tensor:
    return torch.randint(0, V, (n, L), generator=torch.Generator().manual_seed(3))


def _docs(n_docs: int = 4, per: int = 3) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    windows = torch.randint(0, V, (n_docs * per * 2, L), generator=g)
    return holdout_documents(windows, n_read=2, n_score=1, n_docs=n_docs)


def _everything(lm: nn.Module) -> tuple[dict, dict]:
    return ({k: v.detach().clone() for k, v in lm.state_dict().items()},
            copy.deepcopy(collect_living_extra_state(lm.online_encoder)))


def _same(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(
            _same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _assert_untouched(lm: nn.Module, before: tuple[dict, dict]) -> None:
    state, extra = _everything(lm)
    for k in before[0]:
        assert torch.equal(before[0][k], state[k]), k
    assert _same(before[1], extra)
    left = [n for n, b in lm.named_buffers() if b.requires_grad]
    assert left == [], f"buffers left requiring a gradient: {left}"
    assert all(layer.weight.grad is None
               for layer in lwg.living_layers(lm.online_encoder))


# ---- guard 2: refused in training mode ---------------------------------

def test_refused_in_training_mode():
    lm = _loss().train()
    with pytest.raises(RuntimeError, match="training mode"):
        lwg.score_and_its_gradient(lm, _tail())
    with pytest.raises(RuntimeError, match="training mode"):
        lwg.step_living_weights_down_the_loss(lm, _tail(1), rate=1e-3)
    with pytest.raises(RuntimeError, match="training mode"):
        with lwg.living_weights_on_the_graph(lm.online_encoder):
            pass
    assert not any(b.requires_grad for b in lm.buffers())


def test_refused_when_only_the_encoder_is_in_training_mode():
    lm = _loss()
    lm.online_encoder.train()
    with pytest.raises(RuntimeError, match="training mode"):
        with lwg.living_weights_on_the_graph(lm.online_encoder):
            pass


def test_a_dead_arm_has_no_living_weight_and_says_so():
    with pytest.raises(RuntimeError, match="no living layer"):
        lwg.score_and_its_gradient(_loss(dead_ffn=True), _tail())


# ---- guard 3: bit-identical afterwards ---------------------------------

def test_the_model_is_bit_identical_after_a_gradient_is_taken():
    lm = _loss()
    before = _everything(lm)
    score, grads = lwg.score_and_its_gradient(lm, _tail())
    assert len(grads) == 2 and all(float(g.abs().max()) > 0 for g in grads)
    assert not any(g.requires_grad for g in grads)
    _assert_untouched(lm, before)


def test_the_weights_come_off_the_graph_when_the_block_raises():
    lm = _loss()
    with pytest.raises(ZeroDivisionError):
        with lwg.living_weights_on_the_graph(lm.online_encoder):
            1 / 0
    assert not any(b.requires_grad for b in lm.buffers())
    assert not any(m._plasticity_frozen for m in lwg.living_layers(lm.online_encoder))


def test_plasticity_is_frozen_while_a_weight_is_on_the_graph():
    lm = _loss()
    with lwg.living_weights_on_the_graph(lm.online_encoder) as layers:
        assert all(layer._plasticity_frozen for layer in layers)
        assert all(layer.weight.requires_grad for layer in layers)


def test_the_eval_that_steps_living_weights_restores_everything():
    lm = _loss()
    before = _everything(lm)
    res = adaptation_eval(lm, _docs(), n_read=2, read_by_gradient=1e-2,
                          gradient_read_what="living")
    _assert_untouched(lm, before)
    assert res["summary"]["exact_null"] is False       # it did step something
    assert res["settings"]["gradient_read_what"] == "living"
    shares = res["settings"]["living_step_share_of_weight"]
    assert len(shares) == 2 and all(s["median"] > 0 for s in shares)
    assert shares[0]["steps"] == 4 * 2 * 2             # documents, B and C, windows


# ---- guard 1: measurement code only -------------------------------------

def test_a_living_weight_stays_a_buffer_and_no_optimizer_is_involved():
    lm = _loss()
    params_before = [n for n, _ in lm.named_parameters()]
    lwg.score_and_its_gradient(lm, _tail())
    lwg.step_living_weights_down_the_loss(lm, _tail(1), rate=1e-3)
    assert [n for n, _ in lm.named_parameters()] == params_before
    for layer in lwg.living_layers(lm.online_encoder):
        assert "weight" in dict(layer.named_buffers(recurse=False))
        assert "weight" not in dict(layer.named_parameters(recurse=False))
    source = Path(lwg.__file__).read_text(encoding="utf-8")
    assert "torch.optim" not in source and "nn.Parameter(" not in source


def test_nothing_in_the_training_path_imports_the_module():
    # The trainer and the model must not be able to reach it: it is imported
    # by measurement code only (adaptation_eval, lazily, and probes).
    allowed = {"adaptation_eval.py", "living_weight_gradient.py"}
    users = []
    for path in list((REPO_ROOT / "luthi").rglob("*.py")) + list(
            (REPO_ROOT / "scripts").glob("*.py")):
        if path.name in allowed:
            continue
        if "living_weight_gradient" in path.read_text(encoding="utf-8", errors="replace"):
            users.append(str(path.relative_to(REPO_ROOT)))
    assert users == []


def test_a_step_moves_the_living_weights_and_nothing_else():
    lm = _loss()
    before, _ = _everything(lm)
    shares = lwg.step_living_weights_down_the_loss(lm, _tail(1), rate=1e-2)
    after, _ = _everything(lm)
    moved = sorted(k for k in before if not torch.equal(before[k], after[k]))
    assert moved == sorted(f"online_encoder.blocks.{i}.living_ffn.weight" for i in range(2))
    assert len(shares) == 2 and all(s > 0 for s in shares)
    assert not any(b.requires_grad for b in lm.buffers())


# ---- the gradient is the score's ----------------------------------------

def test_the_score_is_the_harness_score():
    lm = _loss()
    tail = _tail()
    score, _ = lwg.score_and_its_gradient(lm, tail)
    harness = ae._score(lm, tail)
    assert score["nmse"] == pytest.approx(harness["nmse"], rel=1e-5)
    assert score["l_pred"] == pytest.approx(harness["l_pred"], rel=1e-5)


def test_the_gradient_predicts_what_a_small_change_does_to_the_score():
    lm = _loss().double()
    tail = _tail()
    if ae._score(lm, tail)["nmse"] != ae._score(lm, tail)["nmse"]:
        # Found 2026-11-17: under torch 2.4.1+cpu (Sanctuary's interpreter)
        # the predictor gives nan in float64 for this split, before any
        # gradient is asked for. Under torch 2.9.1 (the interpreter runs use)
        # it does not. The check needs float64, so it cannot be made there.
        pytest.skip(f"the harness's own score is nan in float64 under torch {torch.__version__}")
    _, grads = lwg.score_and_its_gradient(lm, tail)
    layers = lwg.living_layers(lm.online_encoder)
    gen = torch.Generator().manual_seed(5)
    for layer, g in zip(layers, grads):
        direction = torch.randn(g.shape, generator=gen, dtype=g.dtype)
        direction *= 1e-6 * layer.weight.norm() / direction.norm()
        home = layer.weight.clone()
        with torch.no_grad():
            layer.weight.copy_(home + direction)
            up = ae._score(lm, tail)["nmse"]
            layer.weight.copy_(home - direction)
            down = ae._score(lm, tail)["nmse"]
            layer.weight.copy_(home)
        measured = (up - down) / 2
        predicted = float((g * direction).sum())
        assert abs(predicted) > 0
        assert measured == pytest.approx(predicted, rel=1e-3)


def test_the_default_read_by_gradient_is_unchanged():
    # "trained" is the default and steps no living weight.
    a = adaptation_eval(_loss(), _docs(), n_read=2, read_by_gradient=1e-2)
    b = adaptation_eval(_loss(), _docs(), n_read=2, read_by_gradient=1e-2,
                        gradient_read_what="trained")
    assert a["per_doc"] == b["per_doc"]
    assert a["settings"]["living_step_share_of_weight"] is None


def test_living_needs_a_rate():
    with pytest.raises(ValueError, match="needs read_by_gradient"):
        adaptation_eval(_loss(), _docs(), n_read=2, gradient_read_what="living")
