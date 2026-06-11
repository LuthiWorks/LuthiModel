"""Unit tests for ValueHead and EFEEvaluator (step 1).

Run from project root:
    python -m luthi.v2.m9.test_efe

Tests target spec-correctness properties:
- ValueHead: scalar per batch element; gradient flows.
- EFEEvaluator: G aggregates pragmatic cost only at step 1;
  beta_epi != 0 raises NotImplementedError; candidate batching
  matches per-candidate calls.
- ACTION SENSITIVITY: predicted next-state varies with action
  (the structural property the kill-5 redux metric reads --
  Var_a[s_hat] near zero would mean the predictor is action-
  ignoring and planning is broken).
"""

from __future__ import annotations

import torch

from luthi.v2.jepa_loss import JEPAPredictor
from luthi.v2.m9.efe import EFEEvaluator
from luthi.v2.m9.preferences import Preferences
from luthi.v2.m9.value_head import ValueHead


D = 16
B = 4
CTX = 8
TGT = 6
K = 5


def _build_predictor() -> JEPAPredictor:
    return JEPAPredictor(
        d_model=D,
        n_layers=1,
        n_heads=2,
        ffn_expansion=2,
        max_target_len=32,
    )


def _build_preferences() -> Preferences:
    return Preferences(d_model=D, engagement_target_magnitude=0.5)


def _context_and_positions(device="cpu"):
    context = torch.randn(B, CTX, D, device=device)
    target_positions = torch.arange(CTX, CTX + TGT, device=device).unsqueeze(0).expand(B, -1)
    return context, target_positions


# ----------------------- ValueHead -----------------------

def test_value_head_shape():
    v = ValueHead(d_model=D)
    s = torch.randn(B, D)
    out = v(s)
    assert out.shape == (B,), f"expected [B], got {out.shape}"


def test_value_head_gradient_flows():
    v = ValueHead(d_model=D)
    s = torch.randn(B, D, requires_grad=True)
    out = v(s).sum()
    out.backward()
    assert s.grad is not None and s.grad.abs().sum() > 0


# ----------------------- EFEEvaluator -----------------------

def test_efe_step1_pragmatic_only_shape():
    predictor = _build_predictor()
    prefs = _build_preferences()
    efe = EFEEvaluator(predictor, prefs)
    context, target_positions = _context_and_positions()
    s_t = torch.randn(B, D)
    a_t = torch.randn(B, D)
    out = efe.compute_g(
        s_t=s_t,
        a_t=a_t,
        context_latents=context,
        target_positions=target_positions,
    )
    assert out["G"].shape == (B,)
    assert out["s_hat_next"].shape == (B, D)
    assert out["c_eng"].shape == (B,)
    # P2/P3/P4 zeroed because no decoder/counterpart/reencoded args.
    assert torch.all(out["c_coh"] == 0.0)
    assert torch.all(out["c_con"] == 0.0)
    assert torch.all(out["c_truth"] == 0.0)


def test_efe_step1_blocks_epistemic():
    predictor = _build_predictor()
    prefs = _build_preferences()
    efe = EFEEvaluator(predictor, prefs)
    context, target_positions = _context_and_positions()
    try:
        efe.compute_g(
            s_t=torch.randn(B, D),
            a_t=torch.randn(B, D),
            context_latents=context,
            target_positions=target_positions,
            beta_epi=0.1,
        )
        assert False, "beta_epi != 0 should raise NotImplementedError at step 1"
    except NotImplementedError:
        pass


def test_efe_value_head_bootstrap():
    predictor = _build_predictor()
    prefs = _build_preferences()
    v = ValueHead(d_model=D)
    efe = EFEEvaluator(predictor, prefs, value_head=v)
    context, target_positions = _context_and_positions()
    out = efe.compute_g(
        s_t=torch.zeros(B, D),
        a_t=torch.randn(B, D),
        context_latents=context,
        target_positions=target_positions,
    )
    assert "v_hat" in out
    assert out["v_hat"].shape == (B,)


def test_efe_action_sensitivity():
    """Critical property: predicted next-state varies with action.

    Var_a[s_hat] across the K candidate actions per the kill-5 redux
    metric. Near-zero variance would mean the predictor is ignoring
    its action input -- planning would have nothing to optimize over.
    """
    torch.manual_seed(0)
    predictor = _build_predictor()
    prefs = _build_preferences()
    efe = EFEEvaluator(predictor, prefs)
    context, target_positions = _context_and_positions()
    s_t = torch.zeros(B, D)
    # K distinct random candidate actions per batch element.
    candidates = torch.randn(B, K, D) * 2.0  # scale up to see action effect
    result = efe.compute_g_candidates(
        s_t=s_t,
        candidate_actions=candidates,
        context_latents=context,
        target_positions=target_positions,
    )
    s_hats = result["s_hat_next"]  # [B, K, D]
    # Variance across the K-axis per batch per dim, then averaged.
    var_a = s_hats.var(dim=1).mean()
    # Sigmoid(0) = 0.5 init on the mask, so action is half-passed-through;
    # variance should be clearly above zero.
    assert var_a > 1e-4, (
        f"Var_a[s_hat] = {var_a:.6f} too small -- predictor is "
        "action-ignoring (would trip kill-5 redux). With sigmoid(0)=0.5 "
        "init the action should pass through enough to produce variance."
    )


def test_efe_candidate_batching_matches_per_candidate():
    """compute_g_candidates(k loop) == individual compute_g(k) per k."""
    torch.manual_seed(0)
    predictor = _build_predictor()
    prefs = _build_preferences()
    efe = EFEEvaluator(predictor, prefs)
    predictor.eval()
    context, target_positions = _context_and_positions()
    s_t = torch.zeros(B, D)
    candidates = torch.randn(B, K, D)

    # Batched.
    with torch.no_grad():
        batched = efe.compute_g_candidates(
            s_t=s_t,
            candidate_actions=candidates,
            context_latents=context,
            target_positions=target_positions,
        )

    # Per-candidate.
    with torch.no_grad():
        for k in range(K):
            single = efe.compute_g(
                s_t=s_t,
                a_t=candidates[:, k, :],
                context_latents=context,
                target_positions=target_positions,
            )
            assert torch.allclose(batched["G"][:, k], single["G"], atol=1e-5)
            assert torch.allclose(
                batched["s_hat_next"][:, k, :], single["s_hat_next"], atol=1e-5
            )


def test_efe_threads_through_observation_kwargs():
    """Decoder re-encodes / counterpart / re-encoded a_t threaded through."""
    predictor = _build_predictor()
    prefs = _build_preferences()
    efe = EFEEvaluator(predictor, prefs)
    context, target_positions = _context_and_positions()
    s_t = torch.zeros(B, D)
    a_t = torch.randn(B, D)
    decoder_reencodes = {
        "text": torch.zeros(B, D),
        "attention": torch.full((B, D), 0.1),
    }
    counterpart = torch.ones(B)
    elapsed = torch.full((B,), 2.0)
    a_re = torch.zeros(B, D)  # identical to a_t? no, a_t is random
    out = efe.compute_g(
        s_t=s_t,
        a_t=a_t,
        context_latents=context,
        target_positions=target_positions,
        decoder_reencodes=decoder_reencodes,
        counterpart_present=counterpart,
        time_since_emission=elapsed,
        a_reencoded=a_re,
    )
    # All four features should now contribute non-zero (or be plausibly
    # non-zero given the inputs).
    assert torch.all(out["c_coh"] > 0.0), "P2 should be > 0 with disagreeing modalities"
    assert torch.all(out["c_con"] == 2.0), f"P3 = counterpart*elapsed = 1*2 = 2; got {out['c_con']}"
    # P4: a_t random vs a_re zero -> non-zero.
    assert torch.all(out["c_truth"] > 0.0)


def main() -> int:
    tests = [
        test_value_head_shape,
        test_value_head_gradient_flows,
        test_efe_step1_pragmatic_only_shape,
        test_efe_step1_blocks_epistemic,
        test_efe_value_head_bootstrap,
        test_efe_action_sensitivity,
        test_efe_candidate_batching_matches_per_candidate,
        test_efe_threads_through_observation_kwargs,
    ]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed.append((t.__name__, str(e)))
            print(f"  FAIL  {t.__name__}: {e}")
    if failed:
        print(f"\n{len(failed)} test(s) failed")
        return 1
    print(f"\nAll {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
