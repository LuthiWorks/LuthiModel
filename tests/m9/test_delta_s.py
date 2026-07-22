"""Unit tests for §A.2 DeltaSInternal and DeltaSBand.

Run from project root:
    python -m luthi.v2.m9.test_delta_s

Spec-correctness properties (per 4.8's 2026-06-11 gate-repairs):
- DeltaSInternal.compute() applies the internal-dim mask and
  normalizes by sqrt(effective_dim) (closes N3 + de-saturates P1)
- DeltaSBand provides the engagement_target() that replaces P1's
  fixed 0.5 -- the F1.A5 saturation fix
- DeltaSBand.is_silent_per_batch() is the K-M9-5 internal-stasis
  signal; identical scalar must reach both consumers (the §A.2
  fan-out property)
- Before band warmup, the silent predicate is all-False (kill can't
  fire against an uncalibrated band)
- After warmup, P1's hinge target is the silent threshold, not 0.5
- Preferences.engagement_cost_from_delta_s and the pragmatic_cost
  precomputed-Δs path produce the same result as the s_t/s_hat path
  when Δs is computed the same way -- the equivalence the loop's
  §A.2 assertion will rely on
"""

from __future__ import annotations

import torch

from luthi.v2.m9.delta_s import DeltaSBand, DeltaSInternal
from luthi.v2.m9.preferences import Preferences


D = 16
B = 4


# ---------- DeltaSInternal ----------

def test_compute_shape():
    ds = DeltaSInternal(d_model=D)
    s_t = torch.randn(B, D)
    s_hat = torch.randn(B, D)
    out = ds.compute(s_t, s_hat)
    assert out.shape == (B,)
    assert (out >= 0).all()


def test_compute_zero_at_stasis():
    ds = DeltaSInternal(d_model=D)
    s_t = torch.randn(B, D)
    out = ds.compute(s_t, s_t.clone())
    assert torch.allclose(out, torch.zeros(B), atol=1e-6)


def test_internal_mask_applied():
    """Mask all dims to zero -- ‖Δs_internal‖ should be 0 regardless of Δs."""
    mask = torch.zeros(D)
    ds = DeltaSInternal(d_model=D, internal_mask=mask)
    s_t = torch.zeros(B, D)
    s_hat = torch.randn(B, D) * 10.0  # big delta
    out = ds.compute(s_t, s_hat)
    # eff_dim = 0 falls back to raw_norm * mask = 0; module returns raw_norm
    # (which is also 0 when mask zeros it). Both should be 0 here.
    assert torch.allclose(out, torch.zeros(B), atol=1e-6)


def test_internal_mask_subset():
    """Mask half the dims; verify the other half drives the magnitude."""
    mask = torch.zeros(D)
    mask[: D // 2] = 1.0  # only first half internal
    ds = DeltaSInternal(d_model=D, internal_mask=mask)
    s_t = torch.zeros(B, D)
    s_hat = torch.zeros(B, D)
    s_hat[:, D // 2:] = 5.0  # change only the masked-out half
    out = ds.compute(s_t, s_hat)
    # The external-half change should be masked out -> ‖Δs_internal‖ = 0.
    assert torch.allclose(out, torch.zeros(B), atol=1e-6)


def test_normalization_by_effective_dim():
    """Norm divided by sqrt(eff_dim) -- so scale is comparable across mask sparsity."""
    ds_full = DeltaSInternal(d_model=D, internal_mask=torch.ones(D))
    half_mask = torch.zeros(D)
    half_mask[: D // 2] = 1.0
    ds_half = DeltaSInternal(d_model=D, internal_mask=half_mask)
    s_t = torch.zeros(B, D)
    # Set first half to 1.0 in s_hat -- both masks include those dims.
    s_hat = torch.zeros(B, D)
    s_hat[:, : D // 2] = 1.0
    out_full = ds_full.compute(s_t, s_hat)
    out_half = ds_half.compute(s_t, s_hat)
    # raw_norm is sqrt(D/2) for both. After dividing by sqrt(eff_dim):
    #   full: sqrt(D/2) / sqrt(D)   = sqrt(0.5)
    #   half: sqrt(D/2) / sqrt(D/2) = 1.0
    # half-mask should be ~sqrt(2) bigger -- normalization meaningful.
    assert (out_half > out_full).all()


# ---------- DeltaSBand ----------

def test_band_disarmed_before_warmup_uses_absolute_floor():
    """R1 round-2: even before the band warms, the absolute floor
    fires internal_silent for born-still (‖Δs‖ == 0). The band
    alone recalibrates to a sustained-constant catatonia (probe_g
    analog); the floor catches born-catatonia immediately.
    """
    band = DeltaSBand(min_warmup=8, absolute_silent_floor=1e-4)
    for _ in range(7):
        band.observe(1.0)
    assert not band.is_warm()
    # ‖Δs‖ at 0 is below the absolute floor -> silent even before warmup.
    delta_zero = torch.tensor([0.0, 0.0, 0.0, 0.0])
    assert torch.all(band.is_silent_per_batch(delta_zero))
    # ‖Δs‖ above the floor -> not silent before warmup (band not yet usable).
    delta_above = torch.tensor([1.0, 1.0, 1.0, 1.0])
    assert not torch.any(band.is_silent_per_batch(delta_above))


def test_band_armed_after_warmup():
    band = DeltaSBand(min_warmup=4)
    for _ in range(8):
        band.observe(1.0)
    assert band.is_warm()


def test_silent_per_batch_below_threshold():
    band = DeltaSBand(min_warmup=4, silence_k=0.5)
    for _ in range(8):
        band.observe(2.0)
    # Threshold ~ median - 0.5*MAD ~ 2 - 0 = 2.0 (MAD = 0 on identical samples).
    # Below 2.0 - epsilon should be silent.
    delta = torch.tensor([0.0, 1.0, 3.0, 5.0])
    silent = band.is_silent_per_batch(delta)
    # First two are below threshold = silent; last two above = not silent.
    # (Edge case: at MAD=0, threshold = median exactly; strict < makes equal
    # not silent.)
    assert silent.tolist() == [True, True, False, False]


def test_engagement_target_after_warmup_is_silent_threshold():
    """P1's hinge target = silent_threshold once warm."""
    band = DeltaSBand(min_warmup=4, silence_k=0.5)
    for _ in range(8):
        band.observe(2.0)
    assert band.engagement_target() == band.silent_threshold()


def test_engagement_target_before_warmup_is_floor():
    band = DeltaSBand(min_warmup=8)
    # No observations yet.
    assert band.engagement_target(floor=0.5) == 0.5


# ---------- §A.2 fan-out equivalence ----------

def test_engagement_cost_from_delta_s_matches_compute_path():
    """P1 evaluated via precomputed Δs gives the same result as via
    (s_t, s_hat_next) when both compute Δs the same way -- the
    equivalence the loop will assert.
    """
    prefs = Preferences(d_model=D, engagement_target_magnitude=1.0)
    s_t = torch.randn(B, D)
    s_hat = torch.randn(B, D)
    # Compute Δs the same way the legacy engagement_cost does
    # (full-mask, no normalization).
    delta_legacy = (s_hat - s_t).norm(dim=-1)
    c_compute = prefs.engagement_cost(s_t, s_hat)
    c_precomputed = prefs.engagement_cost_from_delta_s(delta_legacy)
    assert torch.allclose(c_compute, c_precomputed, atol=1e-5)


def test_engagement_cost_uses_passed_target():
    """Passing target overrides the fixed engagement_target_magnitude."""
    prefs = Preferences(d_model=D, engagement_target_magnitude=0.5)
    delta = torch.tensor([0.0, 0.5, 1.0, 1.5])
    # With target=2.0, deficit = max(2 - delta, 0) -> [2, 1.5, 1, 0.5]
    c = prefs.engagement_cost_from_delta_s(delta, target=2.0)
    expected = torch.tensor([4.0, 2.25, 1.0, 0.25])
    assert torch.allclose(c, expected, atol=1e-5)


def test_pragmatic_cost_uses_precomputed_delta_s():
    """pragmatic_cost on the §A.2 path (passing delta_s_internal +
    engagement_target) produces the same P1 cost as the legacy path
    when given equivalent inputs.
    """
    prefs = Preferences(d_model=D, engagement_target_magnitude=1.0)
    s_t = torch.zeros(B, D)
    s_hat = torch.zeros(B, D)  # stasis
    delta = (s_hat - s_t).norm(dim=-1)
    out_legacy = prefs.pragmatic_cost(s_t=s_t, s_hat_next=s_hat)
    out_a2 = prefs.pragmatic_cost(
        s_t=s_t, s_hat_next=s_hat,
        delta_s_internal=delta, engagement_target=1.0,
    )
    assert torch.allclose(out_legacy["c_eng"], out_a2["c_eng"], atol=1e-5)
    assert torch.allclose(out_legacy["total"], out_a2["total"], atol=1e-5)


def test_observe_accepts_tensor_or_float():
    band = DeltaSBand()
    band.observe(1.0)
    band.observe(torch.tensor(2.0))
    band.observe(torch.tensor([1.0, 2.0, 3.0]))
    # No exception thrown.
    assert band.band.median() > 0


def main() -> int:
    tests = [
        test_compute_shape,
        test_compute_zero_at_stasis,
        test_internal_mask_applied,
        test_internal_mask_subset,
        test_normalization_by_effective_dim,
        test_band_disarmed_before_warmup_uses_absolute_floor,
        test_band_armed_after_warmup,
        test_silent_per_batch_below_threshold,
        test_engagement_target_after_warmup_is_silent_threshold,
        test_engagement_target_before_warmup_is_floor,
        test_engagement_cost_from_delta_s_matches_compute_path,
        test_engagement_cost_uses_passed_target,
        test_pragmatic_cost_uses_precomputed_delta_s,
        test_observe_accepts_tensor_or_float,
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
