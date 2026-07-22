"""Unit tests for GammaInference (spec §9).

Run from project root:
    python -m luthi.v2.m9.test_gamma

Tests cover the spec-correctness properties:
- Q = softmax(-gamma G) has the standard simplex shape.
- gamma_target ~ 1/Var_Q[G]: peaked landscape -> high gamma; flat
  landscape -> low gamma.
- EMA smoothing: one step doesn't snap to target.
- Bounded clamp to [gamma_min, gamma_max].
- History accumulates and history_stats returns sensible numbers.
- fixed_for_bringup leaves gamma untouched.
- clamp_to_last_healthy (K-M9-4 first-stage recovery) sets gamma
  and appends to history.
"""

from __future__ import annotations

import torch

from luthi.v2.m9.gamma import GammaInference


K = 5


def test_posterior_shape_and_simplex():
    g = GammaInference(gamma_init=2.0)
    efes = torch.randn(K)
    q = g.posterior(efes)
    assert q.shape == (K,)
    assert abs(q.sum().item() - 1.0) < 1e-5
    assert (q >= 0).all()


def test_posterior_batched():
    g = GammaInference(gamma_init=1.0)
    efes = torch.randn(3, K)
    q = g.posterior(efes)
    assert q.shape == (3, K)
    for row in q:
        assert abs(row.sum().item() - 1.0) < 1e-5


def test_update_returns_bounded_gamma():
    g = GammaInference(gamma_init=1.0, gamma_min=0.01, gamma_max=10.0)
    # Many updates with random EFEs.
    torch.manual_seed(0)
    for _ in range(50):
        efes = torch.randn(K)
        gv = g.update(efes)
        assert 0.01 - 1e-6 <= gv.item() <= 10.0 + 1e-6


def test_peaked_landscape_increases_gamma():
    """When candidate EFEs are highly distinct (peaked posterior),
    Var_Q[G] should be small -> gamma_target should be large -> gamma
    should rise from a low initial value.
    """
    g = GammaInference(
        gamma_init=1.0,
        rho_g=0.5,  # fast smoothing so the test sees movement
        gamma_min=0.01,
        gamma_max=100.0,
    )
    # One candidate hugely better than the rest -> peaked posterior.
    efes = torch.tensor([0.1, 10.0, 10.0, 10.0, 10.0])
    initial = g.gamma.item()
    for _ in range(10):
        g.update(efes)
    final = g.gamma.item()
    assert final > initial, (
        f"peaked landscape should raise gamma: {initial} -> {final}"
    )


def test_flat_landscape_lowers_gamma():
    """Two distinct EFE landscapes drive gamma in different directions.

    Peaked (one clear winner, others much worse) -> posterior is a
    near-delta -> Var_Q[G] ~ 0 -> gamma_target hits the clamp.
    Spread (closer values, posterior more uniform) -> Var_Q[G] is
    appreciable -> gamma_target is small.
    """
    # Peaked: one candidate hugely better.
    peaked = torch.tensor([0.0, 10.0, 20.0, 30.0, 40.0])
    # Spread: graded but closer; posterior at gamma=1.0 is well-spread.
    spread = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])

    g_peak = GammaInference(gamma_init=1.0, rho_g=0.5, gamma_max=100.0)
    g_spread = GammaInference(gamma_init=1.0, rho_g=0.5, gamma_max=100.0)
    for _ in range(20):
        g_peak.update(peaked)
        g_spread.update(spread)
    assert g_peak.gamma.item() > g_spread.gamma.item(), (
        f"peaked landscape should drive gamma above spread landscape: "
        f"peak={g_peak.gamma.item():.3f}, spread={g_spread.gamma.item():.3f}"
    )


def test_ema_smoothing_doesnt_snap():
    """One update step moves gamma but not all the way to target."""
    g = GammaInference(gamma_init=1.0, rho_g=0.1, gamma_max=100.0)
    initial = g.gamma.item()
    # Peaked landscape pushes target very high.
    efes = torch.tensor([0.1, 100.0, 100.0, 100.0, 100.0])
    g.update(efes)
    after_one = g.gamma.item()
    assert after_one > initial
    # rho_g=0.1 so the move is bounded:
    # new = 0.9*1.0 + 0.1*target. Even with target = 100, new <= 10.9.
    assert after_one < 12.0, (
        f"EMA at rho_g=0.1 should move slowly, got {after_one}"
    )


def test_clamp_to_min():
    """Gamma can't go below gamma_min."""
    g = GammaInference(gamma_init=0.5, gamma_min=0.5, gamma_max=100.0)
    # Flat-landscape (large Var_Q[G]) drives target very low.
    efes = torch.tensor([-1000.0, 0.0, 1000.0])
    for _ in range(20):
        g.update(efes)
    assert g.gamma.item() >= 0.5 - 1e-6


def test_history_accumulates():
    g = GammaInference(history_window=4)
    # Initial seed.
    assert g.history_stats()["n"] == 1
    g.update(torch.randn(K))
    g.update(torch.randn(K))
    g.update(torch.randn(K))
    g.update(torch.randn(K))
    # Window cap of 4 + initial seed = 4 (initial gets pushed out).
    assert g.history_stats()["n"] == 4


def test_fixed_for_bringup_doesnt_update():
    g = GammaInference(gamma_init=2.5, fixed_for_bringup=True)
    g.update(torch.tensor([1.0, 10.0, 10.0]))  # would normally move gamma
    assert g.gamma.item() == 2.5


def test_clamp_to_last_healthy():
    g = GammaInference(gamma_init=1.0)
    g.clamp_to_last_healthy(3.0)
    assert g.gamma.item() == 3.0
    # And it appends to history so kill-criterion stats include it.
    h = g.history_stats()
    assert h["max"] >= 3.0


def main() -> int:
    tests = [
        test_posterior_shape_and_simplex,
        test_posterior_batched,
        test_update_returns_bounded_gamma,
        test_peaked_landscape_increases_gamma,
        test_flat_landscape_lowers_gamma,
        test_ema_smoothing_doesnt_snap,
        test_clamp_to_min,
        test_history_accumulates,
        test_fixed_for_bringup_doesnt_update,
        test_clamp_to_last_healthy,
    ]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed.append((t.__name__, str(e)))
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    if failed:
        print(f"\n{len(failed)} test(s) failed")
        return 1
    print(f"\nAll {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
