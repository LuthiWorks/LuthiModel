"""Unit tests for the M9 step-1 Preferences module.

Run from project root:
    python -m luthi.v2.m9.test_preferences

Tests target the spec-correctness properties, not just the math:
- P1 cost is zero when ||Delta s|| >= target, rises smoothly below.
- P1 weight is floored at engagement_floor regardless of drift.
- P2 coherence is zero with <2 modalities, mean pairwise MSE otherwise.
- P3 connection is exactly zero when counterpart_present == 0
  (the "Luthi alone is not punished" property).
- P4 truthfulness is per-batch MSE between a_t and re-encoded.
- Aggregate total matches the per-feature weighted sum.
"""

from __future__ import annotations

import torch

from luthi.v2.m9.preferences import Preferences


D = 16  # small for tests
B = 4


def _new(**kwargs) -> Preferences:
    return Preferences(d_model=D, **kwargs)


def test_p1_zero_when_above_target():
    p = _new(engagement_target_magnitude=0.5)
    s_t = torch.zeros(B, D)
    s_next = torch.zeros(B, D)
    s_next[:, 0] = 1.0  # magnitude = 1.0 > 0.5
    c = p.engagement_cost(s_t, s_next)
    assert c.shape == (B,)
    assert torch.all(c == 0.0), f"expected zero P1 cost above target, got {c}"


def test_p1_rises_below_target():
    p = _new(engagement_target_magnitude=1.0)
    s_t = torch.zeros(B, D)
    s_next = torch.zeros(B, D)
    s_next[:, 0] = 0.3  # magnitude = 0.3 << 1.0
    c = p.engagement_cost(s_t, s_next)
    expected = (1.0 - 0.3) ** 2  # 0.49
    assert torch.allclose(c, torch.full((B,), expected), atol=1e-5), (
        f"expected ~{expected}, got {c}"
    )


def test_p1_zero_at_total_stasis():
    # Catatonia case: s_t == s_next exactly.
    p = _new(engagement_target_magnitude=0.5)
    s_t = torch.randn(B, D)
    c = p.engagement_cost(s_t, s_t.clone())
    expected = 0.5 ** 2  # full target deficit -> floor^2
    assert torch.allclose(c, torch.full((B,), expected), atol=1e-5), (
        f"expected ~{expected} at total stasis, got {c}"
    )


def test_p1_weight_floor_enforced():
    # Set raw weight below floor; w_engagement() should return floor.
    p = _new(engagement_floor=0.5, engagement_weight_init=1.0)
    p.engagement_weight.data = torch.tensor(0.1)  # forced drift below floor
    w = p.w_engagement()
    assert torch.isclose(w, torch.tensor(0.5)), (
        f"weight {w.item()} should be floored at 0.5"
    )

    # Raw weight above floor: return raw weight unchanged.
    p.engagement_weight.data = torch.tensor(2.0)
    w = p.w_engagement()
    assert torch.isclose(w, torch.tensor(2.0)), (
        f"weight {w.item()} should be 2.0 above floor"
    )


def test_p2_zero_with_single_modality():
    p = _new()
    only_text = {"text": torch.randn(B, D)}
    c = p.coherence_cost(only_text)
    assert torch.all(c == 0.0)


def test_p2_pairwise_mse_with_multiple_modalities():
    p = _new()
    a = torch.zeros(B, D)
    b = torch.zeros(B, D)
    b[:, 0] = 1.0  # diff_ab squared mean = 1/D = 0.0625
    c = torch.zeros(B, D)
    c[:, 0] = 2.0  # diff_ac squared mean = 4/D; diff_bc = 1/D
    decodes = {"text": a, "attention": b, "memory": c}
    cost = p.coherence_cost(decodes)
    # pairs: (a,b)=1/D, (a,c)=4/D, (b,c)=1/D; mean over 3 pairs
    expected = (1.0 + 4.0 + 1.0) / D / 3.0
    assert torch.allclose(cost, torch.full((B,), expected), atol=1e-5), (
        f"expected {expected}, got {cost}"
    )


def test_p3_zero_when_alone():
    p = _new()
    counterpart = torch.zeros(B)  # alone
    elapsed = torch.tensor([0.0, 1.0, 5.0, 100.0])
    c = p.connection_cost(counterpart, elapsed)
    assert torch.all(c == 0.0), (
        "solitude must not incur connection cost regardless of elapsed time"
    )


def test_p3_rises_with_silence_when_engaged():
    p = _new()
    counterpart = torch.ones(B)  # counterpart present
    elapsed = torch.tensor([0.0, 1.0, 5.0, 10.0])
    c = p.connection_cost(counterpart, elapsed)
    assert torch.allclose(c, elapsed), (
        f"expected cost == elapsed when counterpart present, got {c}"
    )


def test_p4_mse_faithfulness():
    p = _new()
    a_t = torch.randn(B, D)
    a_re = a_t.clone()
    a_re[:, 0] += 0.5  # introduce distortion only on dim 0
    c = p.truthfulness_cost(a_t, a_re)
    expected = (0.5 ** 2) / D  # mean over D dims
    assert torch.allclose(c, torch.full((B,), expected), atol=1e-5), (
        f"expected ~{expected}, got {c}"
    )


def test_pragmatic_cost_aggregates_correctly():
    p = _new(
        engagement_target_magnitude=1.0,
        coherence_weight_init=2.0,
        connection_weight_init=3.0,
        truthfulness_weight_init=4.0,
    )
    s_t = torch.zeros(B, D)
    s_next = torch.zeros(B, D)  # total stasis -> P1 = floor^2 = 1.0
    decodes = {
        "text": torch.zeros(B, D),
        "attention": torch.full((B, D), 0.1),  # diff^2 mean = 0.01
    }  # P2 = 0.01 (one pair)
    counterpart = torch.ones(B)
    elapsed = torch.full((B,), 2.0)  # P3 = 2.0
    a_t = torch.zeros(B, D)
    a_re = torch.full((B, D), 0.2)  # P4 = 0.04
    out = p.pragmatic_cost(
        s_t=s_t,
        s_hat_next=s_next,
        decoder_reencodes=decodes,
        counterpart_present=counterpart,
        time_since_emission=elapsed,
        a_t=a_t,
        a_reencoded=a_re,
    )
    # Expected weighted sum: 1 * 1.0 + 2 * 0.01 + 3 * 2.0 + 4 * 0.04
    expected = 1.0 + 0.02 + 6.0 + 0.16
    assert torch.allclose(out["total"], torch.full((B,), expected), atol=1e-5), (
        f"expected total ~{expected}, got {out['total']}"
    )


def test_pragmatic_cost_missing_inputs_zero_those_features():
    # Only s_t/s_next provided; other features should be zero.
    p = _new(engagement_target_magnitude=0.5)
    s_t = torch.zeros(B, D)
    s_next = torch.zeros(B, D)  # stasis -> P1 contributes
    out = p.pragmatic_cost(s_t=s_t, s_hat_next=s_next)
    assert torch.all(out["c_coh"] == 0.0)
    assert torch.all(out["c_con"] == 0.0)
    assert torch.all(out["c_truth"] == 0.0)
    # P1 alone should determine the total.
    expected_p1 = 0.5 ** 2  # floor^2
    assert torch.allclose(
        out["total"], torch.full((B,), expected_p1), atol=1e-5
    ), f"expected P1-only total {expected_p1}, got {out['total']}"


def test_weight_snapshot_reports_floored_and_raw():
    p = _new(engagement_floor=0.5, engagement_weight_init=1.0)
    p.engagement_weight.data = torch.tensor(0.1)
    snap = p.weight_snapshot()
    # Use approximate comparison: torch.tensor(0.1) round-trips through
    # float32 and isn't exactly 0.1 as a Python float.
    assert abs(snap["engagement_weight_raw"] - 0.1) < 1e-6, (
        f"raw={snap['engagement_weight_raw']}"
    )
    assert abs(snap["engagement_weight_floored"] - 0.5) < 1e-6, (
        f"floored={snap['engagement_weight_floored']}"
    )
    assert abs(snap["engagement_floor"] - 0.5) < 1e-6, (
        f"floor={snap['engagement_floor']}"
    )


def main() -> int:
    tests = [
        test_p1_zero_when_above_target,
        test_p1_rises_below_target,
        test_p1_zero_at_total_stasis,
        test_p1_weight_floor_enforced,
        test_p2_zero_with_single_modality,
        test_p2_pairwise_mse_with_multiple_modalities,
        test_p3_zero_when_alone,
        test_p3_rises_with_silence_when_engaged,
        test_p4_mse_faithfulness,
        test_pragmatic_cost_aggregates_correctly,
        test_pragmatic_cost_missing_inputs_zero_those_features,
        test_weight_snapshot_reports_floored_and_raw,
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
