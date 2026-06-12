"""Regression guard — F2 fix: gamma tracks landscape peakedness; K-M9-4
detects sustained clamp-proximity.

Inverted version of `redteam/m9_step1/probe_b_gamma_ratchet.py`. The
probe checks the gamma rule is a one-way ratchet (BUG); this test
checks the fix is in place: gamma reads landscape peakedness from
the uniform-weighted spread (no positive feedback), and K-M9-4
watches sustained clamp-proximity rather than just the MAD band.

Run from project root:
    python -m luthi.v2.m9.test_gamma_tracks_landscape

Per 4.8's 2026-06-11 gate-repairs spec:
- Gate 6 is re-defined as a *tracking + non-clamp* check: gamma
  must follow landscape peakedness (flat → low, peaked → high) and
  must not pin to a clamp.
"""

from __future__ import annotations

import torch

from luthi.v2.m9.gamma import GammaInference
from luthi.v2.m9.kills import KillRegistry, KillState


# ---------------- Inverted probe B assertions ----------------

def test_flat_landscape_does_not_pin_to_max():
    """B1 inverted: a perfectly flat landscape drives gamma DOWN (toward
    gamma_min), not up to gamma_max. The legacy form inverted this
    intent because the gamma-sharpened posterior had zero variance.
    """
    g = GammaInference()
    flat = torch.full((8,), 3.0)
    for _ in range(200):
        g.update(flat)
    gamma = float(g.gamma.item())
    assert gamma < 1.0, (
        f"flat landscape must drive gamma toward gamma_min, not ceiling: "
        f"gamma={gamma}"
    )


def test_resampled_landscape_does_not_ratchet():
    """B2 inverted: noisy resampled landscapes hover near a stable
    fixed point determined by their average spread -- not a ratchet
    to gamma_max.
    """
    torch.manual_seed(0)
    g = GammaInference()
    for _ in range(600):
        efes = torch.randn(8) * 1.0  # std ~ 1.0 every cycle
        g.update(efes)
    gamma = float(g.gamma.item())
    # With std ~ 1 and gamma_scale=1.0, target ~ 1.0 -> EMA settles
    # near 1.0. Anywhere between 0.5 and 5.0 is well below gamma_max.
    assert 0.1 < gamma < 5.0, (
        f"noisy landscape must not ratchet to ceiling: gamma={gamma}"
    )


def test_peaked_landscape_drives_gamma_up():
    """The positive direction: a clearly-peaked landscape (one a_k
    much better than the rest) should drive gamma UP -- the spec's
    'commit when clear'.
    """
    g = GammaInference(gamma_scale=1.0, rho_g=0.5)
    # One candidate at 0, the rest at 100 -> spread ~ 35 over K=8.
    peaked = torch.tensor([0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    for _ in range(50):
        g.update(peaked)
    gamma = float(g.gamma.item())
    assert gamma > 1.5, (
        f"peaked landscape should drive gamma above its initial value: "
        f"gamma={gamma}"
    )


def test_k_m9_4_detects_clamp_proximity():
    """K-M9-4's primary signal is sustained clamp-proximity. Drive
    gamma toward gamma_max manually for the sustained window and the
    kill must fire -- the legacy band-only check went blind once gamma
    saturated (MAD → 0).
    """
    registry = KillRegistry(
        gamma_clamp_sustained=4, gamma_min=0.01, gamma_max=100.0,
    )
    for _ in range(5):
        registry.observe_gamma(99.5)  # just under gamma_max
    assert registry.states()["K-M9-4-gamma"] == KillState.FIRED


def test_k_m9_4_detects_low_clamp_proximity():
    """Symmetric clamp-low detection (indecision pathology)."""
    registry = KillRegistry(
        gamma_clamp_sustained=4, gamma_min=0.01, gamma_max=100.0,
    )
    for _ in range(5):
        registry.observe_gamma(0.0103)  # within 5% of gamma_min (0.01 * 1.05 = 0.0105)
    assert registry.states()["K-M9-4-gamma"] == KillState.FIRED


def test_k_m9_4_does_not_false_fire_during_ramp():
    """B3 inverted: K-M9-4 does not false-fire during the gamma EMA
    ramp from the initial value under normal operation.

    Run gamma update on a peaked landscape (which would naturally
    drive gamma up via the F2 formula) and confirm the kill does not
    report FIRED purely from the climb.
    """
    torch.manual_seed(0)
    g = GammaInference(gamma_scale=0.5, rho_g=0.1)  # slow climb
    registry = KillRegistry()
    fired_during_ramp = []
    for t in range(60):
        peaked = torch.tensor([0.0] + [10.0] * 7)
        gamma_t = float(g.update(peaked).item())
        registry.observe_gamma(gamma_t)
        if registry.states()["K-M9-4-gamma"] == KillState.FIRED:
            fired_during_ramp.append((t, gamma_t))
    # The kill may FLAG during the ramp, but it must not FIRE -- the
    # F2 extended warmup on the trending band and the clamp-proximity
    # check defend against false-positive halts on normal startup.
    assert len(fired_during_ramp) == 0, (
        f"K-M9-4 must not false-fire during normal ramp: fired={fired_during_ramp}"
    )


def test_k_m9_4_history_includes_clamp_state():
    """The K-M9-4 instrumentation must surface why the kill fired
    (clamp vs band) so the loop's halt-decision logic can route
    recovery correctly (clamp-then-halt per spec §9).
    """
    registry = KillRegistry(gamma_clamp_sustained=2)
    # Two clamp-high cycles fire the kill.
    registry.observe_gamma(99.0)
    registry.observe_gamma(99.0)
    assert registry._gamma_clamp_consecutive >= 2


# ---------------- Round 2 R2 / F-A regression checks ----------------

def test_gamma_target_is_scale_invariant():
    """R2 / F-A round-2: rescaling EFE by a constant must leave the
    equilibrium gamma unchanged. Round-1 used raw std(G) which scaled
    linearly with EFE; the dimensionless (median - G_min) / std target
    is invariant under linear rescaling.

    4.8's mandatory probe extension: this MUST hold at ×100 scale.
    """
    shape = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def equilibrium(scale: float) -> float:
        g = GammaInference()
        for _ in range(300):
            g.update(shape * scale)
        return float(g.gamma.item())

    base = equilibrium(1.0)
    x50 = equilibrium(50.0)
    x100 = equilibrium(100.0)
    assert abs(x50 / base - 1.0) < 0.05, (
        f"x50 must leave gamma unchanged: base={base:.3f}, x50={x50:.3f}"
    )
    assert abs(x100 / base - 1.0) < 0.05, (
        f"x100 must leave gamma unchanged: base={base:.3f}, x100={x100:.3f}"
    )


def test_sustained_silence_does_not_drive_gamma_to_clamp():
    """R2 round-2 H2: a silent-in-company entity's growing connection
    cost must not drive gamma into the clamp via cost magnitude. The
    dimensionless gamma + bounded P3 together prevent this.
    """
    torch.manual_seed(0)
    base = torch.randn(8) * 0.1
    silent_mask = torch.tensor([0.0, 0, 0, 0, 1, 1, 1, 1])
    g = GammaInference()
    reg = KillRegistry()
    for elapsed in range(300):  # 30 s at 10 Hz
        c_con = silent_mask * (elapsed + 1.0)
        landscape = base + c_con
        gamma_t = float(g.update(landscape).item())
        reg.observe_gamma(gamma_t)
    assert reg.states()["K-M9-4-gamma"] != KillState.FIRED, (
        f"K-M9-4 false-positive halt at gamma={float(g.gamma.item()):.4f} "
        f"after 300 silent cycles -- not the intended pathology gate"
    )


def main() -> int:
    tests = [
        test_flat_landscape_does_not_pin_to_max,
        test_resampled_landscape_does_not_ratchet,
        test_peaked_landscape_drives_gamma_up,
        test_k_m9_4_detects_clamp_proximity,
        test_k_m9_4_detects_low_clamp_proximity,
        test_k_m9_4_does_not_false_fire_during_ramp,
        test_k_m9_4_history_includes_clamp_state,
        test_gamma_target_is_scale_invariant,
        test_sustained_silence_does_not_drive_gamma_to_clamp,
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
