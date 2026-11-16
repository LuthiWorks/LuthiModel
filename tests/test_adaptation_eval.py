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


def test_document_phases_are_disjoint_and_phase0_is_unchanged():
    # Window i holds the value i, so a document's windows name themselves.
    windows = torch.arange(96).unsqueeze(1).expand(96, L).clone()
    base = dict(n_read=4, n_score=2, n_docs=4)          # stride 24, per 6
    stride, per = 24, 6
    legacy = [windows[i * stride: i * stride + per] for i in range(4)]
    p0 = holdout_documents(windows, **base)
    assert all(torch.equal(a, b) for a, b in zip(p0, legacy))
    seen = []
    for phase in range(4):                               # 4 * 6 = 24 fits
        ids = {int(d[w, 0]) for d in holdout_documents(windows, **base, phase=phase)
               for w in range(per)}
        assert len(ids) == 4 * per
        seen.append(ids)
    for a in range(4):
        for b in range(a + 1, 4):
            assert not (seen[a] & seen[b]), (a, b)
    with pytest.raises(ValueError, match="does not fit"):
        holdout_documents(windows, **base, phase=4)


# ---------------------------------------------------------------------------
# Opt-in switches (2026-11-16): the pull off, consolidation off, a repeat
# read, a read by gradient. Each is off by default, and the default path is
# the path as it was.
# ---------------------------------------------------------------------------

import copy  # noqa: E402

from luthi.living_extra_state import collect_living_extra_state  # noqa: E402
from luthi.v2 import adaptation_eval as ae  # noqa: E402


def _the_eval_as_it_was(loss_module, documents, *, n_read, read_repeats=1):
    """The A / B / C loop exactly as it stood before the switches, written
    out from the 2026-10-19 module, so the default path has something to be
    identical to."""
    model = loss_module.online_encoder
    d = len(documents)
    shift = max(1, d // 2)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    extra = copy.deepcopy(collect_living_extra_state(model))

    def restore():
        model.load_state_dict(state)
        ae.apply_living_extra_state(model, extra, source="test")
        model.clear_forward_cache()

    def read(head):
        was = model.training
        model.train(False)
        with torch.no_grad():
            for _ in range(read_repeats):
                for w in head:
                    model.encode(text_tokens=w.unsqueeze(0), causal=False)
                    model.clear_forward_cache()
        model.train(was)

    out = []
    for i, doc in enumerate(documents):
        head, tail = doc[:n_read], doc[n_read:]
        other_head = documents[(i + shift) % d][:n_read]
        a = ae._score(loss_module, tail)
        read(head)
        b = ae._score(loss_module, tail)
        restore()
        read(other_head)
        c = ae._score(loss_module, tail)
        restore()
        out.append({"A": a, "B": b, "C": c})
    return out


def test_default_path_is_the_path_as_it_was():
    was = _the_eval_as_it_was(_loss(), _docs(), n_read=2, read_repeats=2)
    now = adaptation_eval(_loss(), _docs(), n_read=2, read_repeats=2)
    assert now["summary"]["exact_null"] is False     # the reads did something
    for old, new in zip(was, now["per_doc"]):
        for cond in "ABC":
            assert old[cond] == new[cond], cond
    s = now["settings"]
    assert (s["homeostatic_decay"], s["read_consolidation"], s["read_episode_writes"],
            s["read_what"], s["read_by_gradient"], s["gradient_steps"]) == (
        None, True, True, "head", None, None)


def test_pull_off_is_recorded_restored_and_changes_the_read():
    lm = _loss()
    layers = [b.living_ffn for b in lm.online_encoder.blocks]
    decays = [layer.homeostatic_decay for layer in layers]
    assert all(d > 0 for d in decays)
    off = adaptation_eval(lm, _docs(), n_read=2, read_repeats=3, homeostatic_decay=0.0)
    assert [layer.homeostatic_decay for layer in layers] == decays
    assert off["settings"]["homeostatic_decay"] == 0.0
    on = adaptation_eval(_loss(), _docs(), n_read=2, read_repeats=3)
    assert [p["B"] for p in off["per_doc"]] != [p["B"] for p in on["per_doc"]]


def test_with_every_way_off_a_read_changes_nothing_the_score_reads():
    # The write, the pull, consolidation and the episode writes, all off: a
    # living arm must then be the exact null a dead arm is. If it is not,
    # there is another way and these switches do not name everything. The
    # first version of this test had three ways and failed: the fourth, the
    # episode store, was found by it.
    lm = _loss(consolidation_enabled=True)
    res = adaptation_eval(lm, _docs(), n_read=2, read_repeats=3, rate_scale=0.0,
                          homeostatic_decay=0.0, read_consolidation=False,
                          read_episode_writes=False)
    assert res["settings"]["living_layers"] == 2
    assert res["summary"]["exact_null"] is True


def test_the_episode_store_alone_carries_a_read_to_the_score():
    # The weight routes off, the episode writes left on: no weight moves and
    # the score still does. This is the fourth way, and rate_scale does not
    # scale it.
    lm = _loss(consolidation_enabled=True)
    model = lm.online_encoder
    layers = [b.living_ffn for b in model.blocks]
    weights = [layer.weight.clone() for layer in layers]
    for layer in layers:
        layer.rate_scale, layer.homeostatic_decay = 0.0, 0.0
    ae._read(model, _docs()[0][:2], repeats=3, read_train_mode=False,
             consolidation=False)
    assert all(torch.equal(w, layer.weight) for w, layer in zip(weights, layers))

    res = adaptation_eval(_loss(consolidation_enabled=True), _docs(), n_read=2,
                          read_repeats=3, rate_scale=0.0, homeostatic_decay=0.0,
                          read_consolidation=False)
    assert res["summary"]["exact_null"] is False


def test_episode_writes_off_is_undone_after_the_read():
    lm = _loss(consolidation_enabled=True)
    model = lm.online_encoder
    from luthi.episode_store import EpisodeStore
    stores = [m for m in model.modules() if isinstance(m, EpisodeStore)]
    layers = [b.living_ffn for b in model.blocks]
    assert stores
    counts = [int(s.episode_count) for s in stores]
    ae._read(model, _docs()[0][:2], repeats=3, read_train_mode=False,
             episode_writes=False)
    assert [int(s.episode_count) for s in stores] == counts      # nothing written
    assert all(s._plasticity_frozen is False for s in stores)
    assert all("_store_episode" not in vars(layer) for layer in layers)
    ae._read(model, _docs()[0][:2], repeats=3, read_train_mode=False)
    assert [int(s.episode_count) for s in stores] != counts      # on: it writes


def test_consolidation_off_holds_the_trigger_still_and_puts_it_back():
    lm = _loss(consolidation_enabled=True)
    model = lm.online_encoder
    layers = [b.living_ffn for b in model.blocks]
    trackers = [layer._consolidation_tracker for layer in layers]
    assert all(t is not None for t in trackers)
    head = _docs()[0][:2]

    ae._read(model, head, repeats=2, read_train_mode=False, consolidation=False)
    assert all(layer._consolidation_tracker is t for layer, t in zip(layers, trackers))
    assert all(len(t._history) == 0 for t in trackers)              # not advanced

    ae._read(model, head, repeats=2, read_train_mode=False)
    assert all(len(t._history) == 4 for t in trackers)              # on: it advances

    res = adaptation_eval(lm, _docs(), n_read=2, read_consolidation=False)
    assert res["settings"]["read_consolidation"] is False
    assert all(layer._consolidation_tracker is t for layer, t in zip(layers, trackers))


def test_a_repeat_read_reads_the_scored_windows(monkeypatch):
    # Window i holds the value i, so what was read names itself.
    n_docs, n_read, n_score = 4, 2, 2
    per = n_read + n_score
    windows = (torch.arange(n_docs * per).unsqueeze(1).expand(-1, L) % V).clone()
    docs = holdout_documents(windows, n_read=n_read, n_score=n_score, n_docs=n_docs)
    read = []
    real = ae._read

    def spy(model, head, **kw):
        read.append([int(w[0]) for w in head])
        return real(model, head, **kw)

    monkeypatch.setattr(ae, "_read", spy)
    res = adaptation_eval(_loss(), docs, n_read=n_read, read_what="tail")
    assert res["settings"]["read_what"] == "tail"
    shift = n_docs // 2
    for i in range(n_docs):
        own_tail = [int(w[0]) for w in docs[i][n_read:]]
        other_tail = [int(w[0]) for w in docs[(i + shift) % n_docs][n_read:]]
        assert read[2 * i] == own_tail          # B read what it is scored on
        assert read[2 * i + 1] == other_tail    # C read the other document's
    with pytest.raises(ValueError, match="read_what"):
        adaptation_eval(_loss(), docs, n_read=n_read, read_what="middle")


def _whole_state(lm):
    return ({k: v.clone() for k, v in lm.state_dict().items()},
            copy.deepcopy(collect_living_extra_state(lm.online_encoder)))


def test_a_read_by_gradient_learns_and_leaves_nothing_behind():
    lm = _loss(consolidation_enabled=True)
    before, extra_before = _whole_state(lm)
    grads_before = [p.requires_grad for p in lm.parameters()]
    res = adaptation_eval(lm, _docs(), n_read=2, read_by_gradient=1e-2,
                          gradient_steps=2)

    # It learned something: the score moved.
    assert res["summary"]["exact_null"] is False
    assert any(p["B"]["l_pred"] != p["A"]["l_pred"] for p in res["per_doc"])
    s = res["settings"]
    assert (s["read_by_gradient"], s["gradient_steps"]) == (1e-2, 2)
    assert s["gradient_parameters"] == sum(
        p.numel() for p in ae.gradient_read_parameters(lm))

    # Every parameter, every buffer and every piece of living state is back:
    # encoder, predictor, projection heads, regulariser counters.
    after, extra_after = _whole_state(lm)
    assert before.keys() == after.keys()
    for k in before:
        assert torch.equal(before[k], after[k]), k
    assert extra_before == extra_after
    assert [p.requires_grad for p in lm.parameters()] == grads_before
    assert all(p.grad is None for p in lm.parameters())

    # And it was back before each next document: every document's frozen
    # score A is the score an untouched model gives.
    untouched = adaptation_eval(_loss(consolidation_enabled=True), _docs(), n_read=2)
    assert [p["A"] for p in res["per_doc"]] == [p["A"] for p in untouched["per_doc"]]


def test_a_read_by_gradient_moves_parameters_and_no_living_state():
    lm = _loss()
    before, extra_before = _whole_state(lm)
    ae._read_by_gradient(lm, _docs()[0][:2], rate=1e-2, steps=1, repeats=1)
    after, extra_after = _whole_state(lm)
    stepped = {"online_encoder." + n for n, p in lm.online_encoder.named_parameters()
               if p.requires_grad} | {"predictor." + n for n, p in
                                      lm.predictor.named_parameters() if p.requires_grad}
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    assert changed, "the gradient read changed nothing"
    # Only what it steps: so no buffer, and a living weight is a buffer.
    assert changed <= stepped, sorted(changed - stepped)[:5]
    assert extra_before == extra_after


def test_a_read_by_gradient_shows_on_a_dead_arm_too():
    # The dead arm is the harness's negative control for a plastic read. A
    # gradient read is not plastic: it must show there as well, or the
    # control is not reaching the score.
    res = adaptation_eval(_loss(dead_ffn=True), _docs(), n_read=2,
                          read_by_gradient=1e-2)
    assert res["summary"]["exact_null"] is False
