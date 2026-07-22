"""Unit tests for MIProbe + ActionLog.

Run from project root:
    python -m luthi.v2.m9.test_instrumentation

MIProbe properties:
- estimate returns a finite scalar.
- Linearly-predictable target gives a *higher* MI proxy than a
  random target (the structural property: high target predictability
  = high MI signal).
- Band lower bound stays below median; upper above median.
- Snapshot reports always-present keys.

ActionLog properties:
- write produces one JSONL record per call.
- Records round-trip through json.loads.
- Tensors in the record are converted to JSON-safe values.
- Context manager closes the file.
"""

from __future__ import annotations

import json
import os
import tempfile

import torch

from luthi.v2.m9.instrumentation import ActionLog, MIProbe


# ---------- MIProbe ----------

def test_mi_probe_estimate_finite():
    p = MIProbe()
    trunk = torch.randn(64, 16)
    target = torch.randn(64, 1)
    s = p.estimate(trunk, target)
    assert isinstance(s, float)
    assert s == s  # not NaN
    # Some torch versions return -inf for degenerate solves; sanity:
    assert s > -1e9


def test_mi_probe_linearly_predictable_target_scores_higher():
    """Construct a target that's a known linear function of trunk;
    contrast against a random target. Predictable -> higher MI.
    """
    torch.manual_seed(0)
    p_pred = MIProbe(ridge_lambda=1e-3)
    p_rand = MIProbe(ridge_lambda=1e-3)
    trunk = torch.randn(128, 16)
    # Predictable target: linear function of trunk + small noise.
    w_true = torch.randn(16, 1)
    predictable = trunk @ w_true + 0.01 * torch.randn(128, 1)
    random_target = torch.randn(128, 1) * predictable.std()

    s_pred = p_pred.estimate(trunk, predictable)
    s_rand = p_rand.estimate(trunk, random_target)
    assert s_pred > s_rand, (
        f"predictable target should score higher: pred={s_pred:.4f}, "
        f"rand={s_rand:.4f}"
    )


def test_mi_probe_band_orders_correctly():
    p = MIProbe(window=16, band_k=2.0)
    torch.manual_seed(0)
    for _ in range(16):
        trunk = torch.randn(32, 8)
        target = torch.randn(32, 1)
        p.observe(trunk, target)
    lower, med, upper = p.bounds()
    assert lower <= med <= upper


def test_mi_probe_snapshot_keys():
    p = MIProbe()
    for _ in range(5):
        p.observe(torch.randn(16, 8), torch.randn(16, 1))
    snap = p.snapshot()
    for key in (
        "mi_latest", "mi_median", "mi_band_lower", "mi_band_upper",
        "mi_n_samples",
    ):
        assert key in snap


def test_mi_probe_too_small_batch_returns_zero():
    p = MIProbe()
    s = p.estimate(torch.randn(2, 8), torch.randn(2, 1))
    assert s == 0.0


# ---------- ActionLog ----------

def test_action_log_writes_one_record_per_call():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "action_log.jsonl")
        log = ActionLog(path)
        for cycle in range(3):
            log.write({"cycle": cycle, "gamma": 1.0 + cycle})
        log.close()
        # Read it back.
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 3
        for i, line in enumerate(lines):
            rec = json.loads(line)
            assert rec["cycle"] == i


def test_action_log_tensors_jsonable():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "action_log.jsonl")
        log = ActionLog(path)
        log.write({
            "cycle": 0,
            "scalar_tensor": torch.tensor(3.14),
            "vector_tensor": torch.tensor([1.0, 2.0, 3.0]),
            "nested": {
                "inner_tensor": torch.tensor([[1, 2], [3, 4]]),
            },
            "kill_states": {"K-M9-2-entropy": "healthy"},
        })
        log.close()
        with open(path, "r", encoding="utf-8") as f:
            rec = json.loads(f.readline())
        assert abs(rec["scalar_tensor"] - 3.14) < 1e-5
        assert rec["vector_tensor"] == [1.0, 2.0, 3.0]
        assert rec["nested"]["inner_tensor"] == [[1, 2], [3, 4]]
        assert rec["kill_states"]["K-M9-2-entropy"] == "healthy"


def test_action_log_context_manager_closes():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "action_log.jsonl")
        with ActionLog(path) as log:
            log.write({"cycle": 0})
            assert not log._file.closed
        assert log._file.closed


def test_action_log_n_records_counts():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "action_log.jsonl")
        log = ActionLog(path)
        for _ in range(5):
            log.write({"cycle": 0})
        assert log.n_records == 5
        log.close()


def main() -> int:
    tests = [
        test_mi_probe_estimate_finite,
        test_mi_probe_linearly_predictable_target_scores_higher,
        test_mi_probe_band_orders_correctly,
        test_mi_probe_snapshot_keys,
        test_mi_probe_too_small_batch_returns_zero,
        test_action_log_writes_one_record_per_call,
        test_action_log_tensors_jsonable,
        test_action_log_context_manager_closes,
        test_action_log_n_records_counts,
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
