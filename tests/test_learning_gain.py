"""Bounded-growth suite for the inverted-U learning gain (function level).

Gate 1 (2026-07-05_inverted-u-gain-spec.md §7): these bounded-growth tests
exist BEFORE the gain ships. This file covers the invariants that live in the
pure gain function -- regimes (a) rise/plateau, (b) two-sided range, (c)
thrash, (e) cold-start, (g) cap/overshoot. The integration regimes (d) spike
3-way, (f) legacy bit-identity, (h) frozen-plasticity, (i) persistence, (j)
consolidation-replay attach to pc_self_modify in the next step.

The load-bearing invariant is (b): gain(t) >= 1.0 for ALL inputs. That is
Brian's "the substrate never gives up on hard growth because it's hard," made
structural -- the function cannot express suppression.
"""

from __future__ import annotations

import torch

from luthi.v2.pc_ops import learning_gain

RISE, CAP = 2.0, 3.0


def _g(momentum, update_ema, progress):
    return learning_gain(
        torch.as_tensor(momentum, dtype=torch.float32),
        torch.as_tensor(update_ema, dtype=torch.float32),
        progress,
        rise=RISE, cap=CAP,
    )


def test_gain_is_bounded_one_to_cap_for_all_inputs():
    """(b) + structural: over a wide grid including extremes, 1.0 <= gain <= cap.
    No input -- adversarial or degenerate -- can push the gain below legacy."""
    torch.manual_seed(0)
    for progress in (-1.0, 0.0, 0.3, 1.0, 2.0, 100.0):
        mom = (torch.randn(64, 64) * 10.0)
        ema = (torch.rand(64, 64) * 5.0)
        g = _g(mom, ema, progress)
        assert torch.isfinite(g).all()
        assert (g >= 1.0).all(), f"gain dropped below 1.0 at progress={progress}"
        assert (g <= CAP).all(), f"gain exceeded cap at progress={progress}"


def test_gain_rises_on_coherent_resolving_novelty():
    """(a) rise: directed change (|momentum| ~ update_ema, coherence ~ 1) that
    is RESOLVING (progress < 1) lifts gain above 1.0."""
    mom = torch.full((8, 8), 0.9)
    ema = torch.full((8, 8), 1.0)   # coherence ~ 0.9
    g = _g(mom, ema, progress=0.2)  # resolving
    assert (g > 1.05).all()


def test_gain_decays_to_one_as_momentum_settles():
    """(a) plateau: once a concept establishes and the weight stops moving
    consistently (momentum -> 0), coherence -> 0 and gain returns to legacy."""
    ema = torch.full((8, 8), 1.0)
    settling = [0.9, 0.5, 0.1, 0.01, 0.0]
    gains = [float(_g(torch.full((8, 8), m), ema, 0.2).mean()) for m in settling]
    assert gains == sorted(gains, reverse=True), "gain should fall as momentum settles"
    assert abs(gains[-1] - 1.0) < 1e-6, "settled weight -> legacy gain 1.0"


def test_nonresolution_returns_to_one_never_below():
    """(b) two-sided core: sustained effort that is NOT resolving (progress ~ 1)
    must return the gain to 1.0 -- amplification off -- but NEVER below. High
    coherence + non-resolution = legacy strength, not suppression."""
    mom = torch.full((8, 8), 5.0)   # very coherent
    ema = torch.full((8, 8), 1.0)
    g = _g(mom, ema, progress=1.0)  # non-resolving
    assert torch.allclose(g, torch.ones_like(g), atol=1e-6)


def test_worsening_does_not_amplify():
    """Fall, progress > 1 (error growing): amplification off (gain = 1.0), not
    negative feedback. The clamp on (1 - progress) keeps it at legacy."""
    g = _g(torch.full((8, 8), 5.0), torch.full((8, 8), 1.0), progress=3.0)
    assert torch.allclose(g, torch.ones_like(g), atol=1e-6)


def test_thrash_stays_near_one():
    """(c): undirected change (|momentum| << update_ema, coherence ~ 0) is not
    amplified even while resolving -- the gain rewards learning-shaped change,
    not thrash."""
    mom = torch.full((8, 8), 0.01)
    ema = torch.full((8, 8), 1.0)   # coherence ~ 0.01
    g = _g(mom, ema, progress=0.0)
    assert (g < 1.05).all() and (g >= 1.0).all()


def test_cold_start_is_one_no_nan():
    """(e): a dead/fresh weight (momentum = update_ema = 0) -> coherence 0/eps
    -> gain exactly 1.0, no NaN. The 3 a.m. bug."""
    g = _g(torch.zeros(8, 8), torch.zeros(8, 8), progress=0.0)
    assert torch.isfinite(g).all()
    assert torch.allclose(g, torch.ones_like(g))


def test_cap_binds_on_coherence_overshoot():
    """(g): decay mismatch can push coherence transiently > 1; with the [1, cap]
    range the governor cap is what bounds it. Huge coherence + resolving -> cap."""
    mom = torch.full((8, 8), 1000.0)
    ema = torch.full((8, 8), 1.0)   # coherence ~ 1000
    g = _g(mom, ema, progress=0.0)
    assert torch.allclose(g, torch.full_like(g, CAP))


def test_nan_progress_fails_safe_to_legacy():
    """A corrupt (NaN) progress signal must yield gain = 1.0 everywhere --
    a deliberate fail-safe, pinned so a future refactor of `fall` can't
    silently flip it to NaN (Fable audit 2026-07-06)."""
    g = _g(torch.full((8, 8), 5.0), torch.full((8, 8), 1.0),
           progress=float("nan"))
    assert torch.isfinite(g).all()
    assert torch.allclose(g, torch.ones_like(g))


def test_gain_monotonic_nondecreasing_in_coherence():
    """Rise shape: more directedness -> at least as much gain, up to the cap."""
    ema = torch.ones(1)
    prev = 0.0
    for m in (0.0, 0.1, 0.5, 1.0, 2.0, 5.0):
        g = float(_g(torch.full((1,), m), ema, progress=0.0))
        assert g >= prev - 1e-6, f"gain not monotone in coherence at |m|={m}"
        prev = g
