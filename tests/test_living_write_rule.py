"""The living write rule, the episode context, and the block output store
(2026-10-19, docs/research/2026-10-19_the-living-write-is-mean-field.md).

Two kinds of test live here, deliberately side by side:

* CHARACTERIZATION tests pin properties of the LEGACY machinery that the
  2026-10-19 diagnosis rests on -- the mean-field write is blind to
  everything but batch means, accumulates rank-1 along the mean output, the
  legacy episode context is near-identical across batches, and the legacy
  block store blends into nearly every forward. If one of these starts
  failing, the diagnosis no longer describes the code and the research log
  must be revisited, not the test loosened.
* CONTRACT tests pin the new opt-in options: legacy bit-identity when they
  are off, per-token content-sensitivity when on, bounded growth, no silent
  C++ fallback, persistence, and frozen-forward purity.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from luthi.living_extra_state import (  # noqa: E402
    apply_living_extra_state,
    collect_living_extra_state,
)
from luthi.v2 import pc_ops  # noqa: E402
from luthi.v2.consolidation import consolidate_layer_attractor  # noqa: E402
from luthi.v2.hybrid_block_pc import PredictiveCodingBlock  # noqa: E402
from luthi.v2.living_layer_pc import PredictiveCodingLayer  # noqa: E402
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM  # noqa: E402
from luthi.v2.plasticity import freeze_plasticity  # noqa: E402

_RATES = dict(
    pc_rate=0.01, pred_learning_rate=0.001, homeostatic_decay=0.001,
    set_point_adapt_rate=1e-6, momentum_decay=0.9, update_ema_decay=0.99,
    precision_ema_decay=0.99, precision_min=0.1, precision_max=10.0,
    prediction_clamp=10.0,
)


def _bufs(n_in: int = 16, n_out: int = 16, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n_out, n_in, generator=g) * 0.2
    return dict(
        weight=w.clone(), prediction=torch.zeros(n_out, n_in),
        set_point=w.clone(), momentum=torch.zeros(n_out, n_in),
        update_ema=torch.ones(n_out, n_in) * 1e-4,
        precision=torch.ones(n_in), error_acc=torch.zeros(n_out),
        plasticity=torch.ones(n_in),
    )


def _step(b: dict, x: torch.Tensor, **kw):
    out = x @ b["weight"].T
    return pc_ops._pc_self_modify_python(
        b["weight"], b["prediction"], b["set_point"], b["momentum"],
        b["update_ema"], b["precision"], b["error_acc"], b["plasticity"],
        x, out, **_RATES, **kw,
    )


def _offset_batches(n: int, n_tok: int = 256, n_in: int = 16, seed: int = 1):
    """Batches sharing a large constant offset plus per-token noise -- the
    regime the production trunk is in (offset dominance 0.79-0.999)."""
    g = torch.Generator().manual_seed(seed)
    offset = torch.randn(n_in, generator=g) * 3.0
    return [offset + torch.randn(n_tok, n_in, generator=g) for _ in range(n)]


# ------------------------------------------------------------------ op level


def test_default_is_bit_identical_to_explicit_mean_field():
    a, b = _bufs(), _bufs()
    for x in _offset_batches(20):
        _step(a, x)
        _step(b, x, write_rule="mean_field")
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_unknown_write_rule_raises():
    b = _bufs()
    with pytest.raises(ValueError, match="write_rule"):
        _step(b, _offset_batches(1)[0], write_rule="hebbian")
    with pytest.raises(ValueError, match="write_rule"):
        PredictiveCodingLayer(8, 8, write_rule="hebbian")


def test_CHARACTERIZATION_mean_field_write_sees_only_batch_means():
    """Two batches with IDENTICAL per-dimension means but different token
    structure produce BITWISE-identical mean-field writes. This is the whole
    diagnosis in one assertion: the rule cannot see content."""
    g = torch.Generator().manual_seed(3)
    x1 = torch.randn(64, 16, generator=g)
    x2 = torch.randn(64, 16, generator=g)
    x2 = x2 - x2.mean(0, keepdim=True) + x1.mean(0, keepdim=True)
    # Different tokens, same means.
    assert not torch.allclose(x1, x2)
    # Outputs depend on the tokens; make their means identical too by
    # using a linear layer (mean output = mean input @ W.T).
    w1, w2 = _bufs(), _bufs()
    _step(w1, x1)
    _step(w2, x2)
    assert torch.allclose(w1["weight"], w2["weight"], atol=1e-6)
    # The covariance rule sees the difference.
    c1, c2 = _bufs(), _bufs()
    _step(c1, x1, write_rule="covariance")
    _step(c2, x2, write_rule="covariance")
    assert (c1["weight"] - c2["weight"]).abs().max() > 1e-4


def _spectrum(d: torch.Tensor) -> tuple[float, torch.Tensor]:
    u, s, _ = torch.linalg.svd(d)
    return float(s[0] ** 2 / (s ** 2).sum()), u[:, 0]


def test_CHARACTERIZATION_mean_field_change_is_rank1_along_mean_output():
    b = _bufs()
    w0 = b["weight"].clone()
    mean_outs = []
    for x in _offset_batches(200):
        # The write's output side is the batch-mean output AT WRITE TIME.
        mean_outs.append(x.mean(0) @ b["weight"].T)
        _step(b, x)
    share, u1 = _spectrum(b["weight"] - w0)
    mean_out = torch.stack(mean_outs).mean(0)
    align = float((u1 @ (mean_out / mean_out.norm())).abs())
    assert share > 0.95, share
    assert align > 0.9, align


def test_covariance_change_is_not_rank1():
    mf, cv = _bufs(), _bufs()
    w0 = mf["weight"].clone()
    for x in _offset_batches(200):
        _step(mf, x)
        _step(cv, x, write_rule="covariance")
    mf_share, _ = _spectrum(mf["weight"] - w0)
    cv_share, _ = _spectrum(cv["weight"] - w0)
    assert cv_share < 0.5, cv_share
    assert cv_share < mf_share


_PROD_RATES = dict(_RATES, pc_rate=0.001, pred_learning_rate=0.0001)


def _max_norm(rule: str, scale: float, steps: int, rates: dict) -> tuple[float, float, dict]:
    b = _bufs()
    g = torch.Generator().manual_seed(7)
    mix = torch.randn(16, 16, generator=g)
    n0, mx = float(b["weight"].norm()), 0.0
    for _ in range(steps):
        x = (torch.randn(128, 16, generator=g) @ mix) * scale
        out = x @ b["weight"].T
        pc_ops._pc_self_modify_python(
            b["weight"], b["prediction"], b["set_point"], b["momentum"],
            b["update_ema"], b["precision"], b["error_acc"], b["plasticity"],
            x, out, **rates, write_rule=rule,
        )
        mx = max(mx, float(b["weight"].norm()))
    return n0, mx, b


@pytest.mark.parametrize("scale", [1.0, 20.0])
def test_covariance_bounded_growth_at_production_rates(scale):
    """Refinement-6 style: structured input (LayerNorm scale and 20x) for
    5,000 steps at the production rates. The capped covariance rule must stay
    finite and grow no more than the legacy rule does on the same stream
    (measured: 3.09 vs 3.68 at 1x; 5.84 vs 9.92 at 20x, from 3.01)."""
    n0, mx_cv, b = _max_norm("covariance", scale, 5000, _PROD_RATES)
    _, mx_mf, _ = _max_norm("mean_field", scale, 5000, _PROD_RATES)
    assert all(torch.isfinite(b[k]).all() for k in b)
    assert mx_cv <= mx_mf * 1.05, (n0, mx_cv, mx_mf)
    assert float(b["prediction"].abs().max()) <= _RATES["prediction_clamp"] + 1e-6


def test_covariance_write_budget_never_exceeds_mean_field():
    """The cap: per step, the covariance write's Frobenius norm is at most
    the mean-field write the same batch would have produced. Checked from a
    shared starting state so the two writes are directly comparable."""
    for x in _offset_batches(10, n_tok=128):
        base = _bufs(seed=5)
        a, c = copy.deepcopy(base), copy.deepcopy(base)
        rates = dict(_RATES, homeostatic_decay=0.0, set_point_adapt_rate=0.0)
        out = x @ base["weight"].T
        for bufs, rule in ((a, "mean_field"), (c, "covariance")):
            pc_ops._pc_self_modify_python(
                bufs["weight"], bufs["prediction"], bufs["set_point"],
                bufs["momentum"], bufs["update_ema"], bufs["precision"],
                bufs["error_acc"], bufs["plasticity"], x, out, **rates,
                write_rule=rule,
            )
        d_mf = (a["weight"] - base["weight"]).norm()
        d_cv = (c["weight"] - base["weight"]).norm()
        assert float(d_cv) <= float(d_mf) * (1 + 1e-5), (float(d_cv), float(d_mf))


def test_covariance_composes_with_relative_trust_and_surprise_drive():
    b = _bufs()
    st = dict(
        drive_ref=torch.tensor(0.0), drive_ref_drift=torch.tensor(0.0),
        drive_dev=torch.tensor(0.0), drive_calls=torch.tensor(0),
        drive_gain_out=torch.tensor(0.0),
        drive_fire_count=torch.tensor(0), drive_gain_sum=torch.tensor(0.0),
    )
    for x in _offset_batches(300):
        _step(b, x, write_rule="covariance", relative_trust=True,
              drive_mode="surprise", drive_warmup_calls=50, **st)
    assert int(st["drive_calls"]) == 300
    assert all(torch.isfinite(b[k]).all() for k in b)


def test_covariance_precision_tracks_per_token_error_energy():
    """Under the covariance rule precision EMAs toward 1/mean_t(e_t^2): with
    a constant mean and large token spread, the per-token error is large even
    though the batch-mean error is ~0, so precision must FALL below 1 -- the
    mean-field rule, which only sees the mean error, drives it UP."""
    mf, cv = _bufs(), _bufs()
    for x in _offset_batches(300):
        _step(mf, x)
        _step(cv, x, write_rule="covariance")
    assert float(cv["precision"].mean()) < float(mf["precision"].mean())


def test_covariance_never_runs_on_the_cpp_path(monkeypatch):
    class _Boom:
        def pc_self_modify(self, *a, **k):  # pragma: no cover - must not run
            raise AssertionError("covariance rule reached the C++ kernel")

    monkeypatch.setattr(pc_ops, "_use_cpp", True)
    monkeypatch.setattr(pc_ops, "_cpp_ops", _Boom())
    b = _bufs()
    x = _offset_batches(1)[0]
    pc_ops.pc_self_modify(
        b["weight"], b["prediction"], b["set_point"], b["momentum"],
        b["update_ema"], b["precision"], b["error_acc"], b["plasticity"],
        x, x @ b["weight"].T, **_RATES, write_rule="covariance",
    )


# --------------------------------------------------------------- layer level


def _layer(**kw) -> PredictiveCodingLayer:
    torch.manual_seed(0)
    return PredictiveCodingLayer(16, 16, num_episodes=8, **kw)


def test_layer_default_bit_identical_to_explicit_legacy_options():
    a = _layer()
    b = _layer(write_rule="mean_field", context_mode="mean")
    for x in _offset_batches(30, n_tok=64):
        a(x)
        b(x)
    sa, sb = a.state_dict(), b.state_dict()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


def test_CHARACTERIZATION_mean_context_is_near_identical_across_batches():
    layer = _layer()
    ctxs = [layer._compute_context(x) for x in _offset_batches(8, n_tok=256)]
    sims = [float(ctxs[0] @ c) for c in ctxs[1:]]
    assert min(sims) > 0.99, sims


def test_deviation_context_differs_across_batches_after_warmup():
    layer = _layer(context_mode="deviation", context_warmup_calls=16)
    batches = _offset_batches(80, n_tok=256)
    for x in batches[:64]:
        layer._compute_context(x)          # warm the reference
    ctxs = [layer._compute_context(x) for x in batches[64:]]
    sims = [abs(float(ctxs[0] @ c)) for c in ctxs[1:]]
    assert max(sims) < 0.9, sims


def test_deviation_context_warmup_falls_back_to_legacy_context():
    legacy = _layer()
    dev = _layer(context_mode="deviation", context_warmup_calls=4)
    x = _offset_batches(1, n_tok=64)[0]
    assert torch.allclose(dev._compute_context(x), legacy._compute_context(x))


def test_deviation_reference_untouched_by_frozen_forward():
    layer = _layer(context_mode="deviation", context_warmup_calls=2)
    batches = _offset_batches(10, n_tok=64)
    for x in batches[:5]:
        layer(x)
    ref, count = layer.context_ref.clone(), layer._context_ref_count
    with freeze_plasticity(layer):
        layer(batches[5])
    assert torch.equal(layer.context_ref, ref)
    assert layer._context_ref_count == count


def test_deviation_reference_persists_through_living_extra_state():
    src = _layer(context_mode="deviation")
    for x in _offset_batches(12, n_tok=64):
        src(x)
    state = collect_living_extra_state(src)
    dst = _layer(context_mode="deviation")
    dst.load_state_dict(src.state_dict())   # strict: the buffer is non-persistent
    apply_living_extra_state(dst, state)
    assert torch.equal(dst.context_ref, src.context_ref)
    assert dst._context_ref_count == src._context_ref_count


def test_legacy_layer_adds_no_extra_state_keys():
    layer = _layer()
    for x in _offset_batches(5, n_tok=32):
        layer(x)
    entry = collect_living_extra_state(layer).get("", {})
    assert "context_ref" not in entry


# --------------------------------------------------------------- block level


def _block(**kw) -> PredictiveCodingBlock:
    torch.manual_seed(0)
    return PredictiveCodingBlock(16, n_heads=2, num_episodes=8,
                                 episode_recall_threshold=0.7, **kw)


def test_CHARACTERIZATION_legacy_block_store_blends_into_nearly_every_forward():
    block = _block()
    for x in _offset_batches(40, n_tok=128):
        block(x.reshape(4, 32, 16), causal=False)
    a = block.aliveness()
    assert a["output_episode_store"] == 1.0
    assert a["output_episode_recall_duty"] > 0.9, a
    assert a["output_episode_blend_share"] > 0.05, a


def test_block_store_off_removes_the_store_and_says_so():
    block = _block(block_episode_store="off")
    assert block.episode_store is None
    block(torch.randn(2, 8, 16), causal=False)
    assert block.aliveness()["output_episode_store"] == 0.0
    with pytest.raises(ValueError, match="block_episode_store"):
        _block(block_episode_store="sometimes")


def test_block_store_off_changes_only_the_store():
    """Same seed, store on vs off: the living FFN's first write is identical
    (the store acts AFTER the FFN), so the option isolates the store."""
    on, off = _block(), _block(block_episode_store="off")
    x = torch.randn(2, 8, 16)
    on(x, causal=False)
    off(x, causal=False)
    assert torch.equal(on.living_ffn.weight, off.living_ffn.weight)


# ------------------------------------------------------------- consolidation


def test_attractor_replay_preserves_relative_trust_ledger():
    """Regression (2026-10-19): replay used to run with relative_trust=False
    and clamp the freed precision ledger back into [0.1, 10]."""
    layer = _layer(relative_trust=True, consolidation_enabled=False)
    with torch.no_grad():
        layer.precision.copy_(torch.logspace(-2, 6, 16))
        layer.episode_inputs[0].copy_(torch.randn(16))
        layer.episode_saliences[0] = 1.0
        layer.episode_count.fill_(1)
    replayed = consolidate_layer_attractor(layer)
    assert replayed == 1
    assert float(layer.precision.max()) > 10.0
    assert float(layer.precision.min()) < 0.1


# ------------------------------------------------------------------ plumbing


def test_model_plumbs_options_to_every_block():
    m = MultimodalPredictiveCodingLM(
        vocab_size=50, d_model=16, n_blocks=3, n_heads=2, max_seq_len=8,
        write_rule="covariance", context_mode="deviation",
        block_episode_store="off",
    )
    for blk in m.blocks:
        assert blk.living_ffn.write_rule == "covariance"
        assert blk.living_ffn.context_mode == "deviation"
        assert blk.episode_store is None
    m.train()
    m.encode(text_tokens=torch.randint(0, 50, (2, 8)))


def test_dead_arm_unaffected_by_living_options():
    d = MultimodalPredictiveCodingLM(
        vocab_size=50, d_model=16, n_blocks=2, n_heads=2, max_seq_len=8,
        dead_ffn=True, write_rule="covariance", block_episode_store="off",
    )
    for blk in d.blocks:
        assert isinstance(blk.living_ffn, torch.nn.Linear)
        assert blk.episode_store is None
