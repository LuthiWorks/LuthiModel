"""Inference-time adaptation eval (2026-10-19): the harness's own contracts.

The eval exists to measure the README's near-term goal on checkpoints, so
the harness must be trustworthy before any number it produces is read:

* POSITIVE CONTROL -- a dead arm has no living state, so frozen / read-own /
  read-other scores must be BITWISE equal (``exact_null``).
* RESTORATION -- after the eval the model is bitwise the model it was given;
  the read conditions never leak into each other or into the caller.
* SENSITIVITY -- on a living arm, reading actually changes the score
  (otherwise the harness would be measuring nothing, the failure class this
  repo keeps paying for).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from luthi.v2.adaptation_eval import adaptation_eval, holdout_documents  # noqa: E402
from luthi.v2.jepa_loss import JEPALoss  # noqa: E402
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM  # noqa: E402

V, D, L = 64, 32, 16


def _loss(**kw) -> JEPALoss:
    torch.manual_seed(0)
    m = MultimodalPredictiveCodingLM(
        vocab_size=V, d_model=D, n_blocks=2, n_heads=4, max_seq_len=L, **kw,
    )
    return JEPALoss(m, visreg_lambda=0.6, sigreg_projection="none")


def _docs(n_docs: int = 4, per: int = 3) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    windows = torch.randint(0, V, (n_docs * per * 2, L), generator=g)
    return holdout_documents(windows, n_read=2, n_score=1, n_docs=n_docs)


def test_dead_arm_is_an_exact_null():
    res = adaptation_eval(_loss(dead_ffn=True), _docs(), n_read=2)
    assert res["summary"]["exact_null"] is True
    assert res["settings"]["living_layers"] == 0


def test_model_restored_bitwise_after_eval():
    lm = _loss()
    before = {k: v.clone() for k, v in lm.online_encoder.state_dict().items()}
    adaptation_eval(lm, _docs(), n_read=2, read_repeats=2)
    after = lm.online_encoder.state_dict()
    for k in before:
        assert torch.equal(before[k], after[k]), k


def test_reading_changes_the_score_on_a_living_arm():
    res = adaptation_eval(_loss(), _docs(), n_read=2, read_repeats=3,
                          rate_scale=10.0)
    assert res["summary"]["exact_null"] is False
    deltas = [p["B"]["l_pred"] - p["A"]["l_pred"] for p in res["per_doc"]]
    assert any(abs(d) > 0 for d in deltas)


def test_rate_scale_is_restored():
    lm = _loss()
    layers = [b.living_ffn for b in lm.online_encoder.blocks]
    adaptation_eval(lm, _docs(), n_read=2, rate_scale=0.2)
    assert all(l.rate_scale == 1.0 for l in layers)


def test_holdout_documents_refuses_a_short_holdout():
    with pytest.raises(ValueError, match="holdout has"):
        holdout_documents(torch.zeros(5, L, dtype=torch.long),
                          n_read=2, n_score=1, n_docs=4)


def test_needs_two_documents():
    with pytest.raises(ValueError, match="at least 2"):
        adaptation_eval(_loss(), _docs()[:1], n_read=2)
