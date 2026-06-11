"""Unit tests for HabitNet (Fountas-style amortized proposal distribution).

Run from project root:
    python -m luthi.v2.m9.test_habit_net

Spec-correctness properties:
- Forward returns (mean, log_std) both shape [B, D].
- Sample returns K candidates with reparameterized gradient flow.
- Sample log-prob matches the closed-form Gaussian density.
- log_std is clamped to the configured bounds.
- Entropy matches the closed-form Gaussian entropy.
- Across many samples, the empirical mean approaches the parametric
  mean (sanity-check that sampling actually uses the Gaussian).
"""

from __future__ import annotations

import math

import torch

from luthi.v2.m9.habit_net import HabitNet


D = 16
B = 4
K = 8


def test_forward_shape():
    net = HabitNet(d_model=D)
    s = torch.randn(B, D)
    mean, log_std = net(s)
    assert mean.shape == (B, D)
    assert log_std.shape == (B, D)


def test_sample_candidate_shape():
    net = HabitNet(d_model=D)
    s = torch.randn(B, D)
    out = net.sample(s, K=K)
    assert out["candidates"].shape == (B, K, D)
    assert out["log_prob"].shape == (B, K)
    assert out["mean"].shape == (B, D)
    assert out["log_std"].shape == (B, D)


def test_sample_reparam_gradient_flows():
    """Reparam trick: gradient on a candidate-sum flows to net params."""
    net = HabitNet(d_model=D)
    s = torch.randn(B, D)
    out = net.sample(s, K=K)
    loss = out["candidates"].sum()
    loss.backward()
    # At least the mean head should have non-zero grad.
    assert net.mean_head.weight.grad is not None
    assert net.mean_head.weight.grad.abs().sum() > 0, (
        "Reparam grad not flowing into mean head"
    )


def test_log_prob_matches_closed_form():
    """Verify that log_prob is the correct Gaussian log-density."""
    torch.manual_seed(0)
    net = HabitNet(d_model=D, log_std_init=0.0)
    s = torch.randn(B, D)
    out = net.sample(s, K=K)
    mean, log_std = out["mean"], out["log_std"]
    candidates = out["candidates"]
    reported_log_prob = out["log_prob"]

    # Closed-form recomputation:
    # log p(x) = -0.5 * sum_i[((x_i - mu_i) / sigma_i)^2 + log(2*pi) + 2*log(sigma_i)]
    std = log_std.exp()  # [B, D]
    z = (candidates - mean.unsqueeze(1)) / std.unsqueeze(1)  # [B, K, D]
    closed_form = (
        -0.5 * z.pow(2).sum(dim=-1)
        - 0.5 * D * math.log(2.0 * math.pi)
        - log_std.sum(dim=-1).unsqueeze(1)
    )  # [B, K]
    assert torch.allclose(reported_log_prob, closed_form, atol=1e-5), (
        f"log_prob mismatch:\n  reported={reported_log_prob}\n"
        f"  closed_form={closed_form}"
    )


def test_log_std_clamped():
    """log_std clamped to [min, max]; an aggressive head bias is suppressed."""
    net = HabitNet(d_model=D, log_std_min=-1.0, log_std_max=1.0)
    # Force the bias far outside the clamp.
    with torch.no_grad():
        net.log_std_head.bias.fill_(50.0)
    s = torch.randn(B, D)
    _, log_std = net(s)
    assert log_std.max().item() <= 1.0 + 1e-6
    assert log_std.min().item() >= -1.0 - 1e-6


def test_entropy_matches_closed_form():
    net = HabitNet(d_model=D, log_std_init=0.0)
    s = torch.randn(B, D)
    h = net.entropy(s)
    _, log_std = net(s)
    expected = 0.5 * D * math.log(2.0 * math.pi * math.e) + log_std.sum(dim=-1)
    assert torch.allclose(h, expected, atol=1e-5)


def test_empirical_mean_approaches_parametric_mean():
    """Sanity: averaging many samples approaches the Gaussian mean."""
    torch.manual_seed(0)
    net = HabitNet(d_model=D, log_std_init=0.0)
    s = torch.randn(B, D)
    K_large = 4096
    out = net.sample(s, K=K_large)
    empirical = out["candidates"].mean(dim=1)  # [B, D]
    parametric = out["mean"]  # [B, D]
    # With K_large samples, the empirical mean is within ~std/sqrt(K)
    # of parametric. With log_std_init=0, std=1; tolerance ~0.05 plenty.
    err = (empirical - parametric).abs().max().item()
    assert err < 0.1, f"empirical mean too far from parametric (err={err})"


def test_distinct_samples_from_same_state():
    """K samples from a single state should not collapse to a point."""
    net = HabitNet(d_model=D, log_std_init=0.0)
    s = torch.randn(B, D)
    out = net.sample(s, K=K)
    cands = out["candidates"]  # [B, K, D]
    # Per-batch variance across K should be > 0.
    var_k = cands.var(dim=1).mean(dim=-1)  # [B]
    assert torch.all(var_k > 0.1), (
        f"Samples collapsed for some batch element: var_k={var_k}"
    )


def main() -> int:
    tests = [
        test_forward_shape,
        test_sample_candidate_shape,
        test_sample_reparam_gradient_flows,
        test_log_prob_matches_closed_form,
        test_log_std_clamped,
        test_entropy_matches_closed_form,
        test_empirical_mean_approaches_parametric_mean,
        test_distinct_samples_from_same_state,
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
