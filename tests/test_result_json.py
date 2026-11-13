"""An instrument's result is strict JSON (2026-11-13). All six dead-control
results from Brian's card held a bare NaN: the t-value of a difference that
is exactly 0 everywhere."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from luthi.v2 import result_json
from luthi.v2.adaptation_eval import _summ

REPO_ROOT = Path(__file__).resolve().parent.parent


def _strict(text: str):
    def refuse(name: str):
        raise ValueError(f"not JSON: {name}")

    return json.loads(text, parse_constant=refuse)


def test_a_dead_controls_summary_is_written_as_strict_json():
    stats = _summ([0.0, 0.0, 0.0, 0.0])      # every difference exactly 0
    assert stats["t"] != stats["t"], "the instrument's t is 0 divided by 0"
    with pytest.raises(ValueError, match="not JSON"):
        _strict(json.dumps({"B_minus_A": stats}))
    written = _strict(result_json.dumps({"B_minus_A": stats}))
    assert written == {"B_minus_A": {"mean": 0.0, "se": 0.0, "t": None, "n": 4}}


def test_numbers_are_written_as_they_are_and_only_non_numbers_become_null():
    result = {"a": 1.5, "b": [0.1, float("inf"), -float("inf")], "c": {"d": float("nan"), "e": 3},
              "f": "NaN", "g": (2.0, None), "h": True}
    assert _strict(result_json.dumps(result)) == {
        "a": 1.5, "b": [0.1, None, None], "c": {"d": None, "e": 3},
        "f": "NaN", "g": [2.0, None], "h": True,
    }


def test_each_instrument_writes_through_it():
    for name in ("living_write_probe.py", "adaptation_eval.py", "per_block_probe.py"):
        text = (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "result_json.dumps(" in text and "write_text(json.dumps(" not in text, name
