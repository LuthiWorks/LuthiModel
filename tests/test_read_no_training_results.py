"""The reader of the collected E1, E2 and E4 results (2026-11-13) applies
the registered rules. Pinned on made-up results, and once on the results
committed from Brian's card, so a later edit to either is noticed."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.read_no_training_results import COLLECTED, read_all, read_e1, read_e4  # noqa: E402


def _stat(mean: float, t: float | None) -> dict:
    return {"mean": mean, "se": 1.0, "t": t, "n": 32}


def _e1(run: Path, name: str, specific: float, t: float | None, rank_drop: float = 0.0) -> None:
    summary = {
        "exact_null": t is None,
        "nmse": {"A_mean": 0.5, "B_minus_A": _stat(specific, t), "C_minus_A": _stat(0.0, 0.0),
                 "B_minus_C": _stat(specific, t)},
        "target_eff_rank": {"A_mean": 40.0, "B_minus_C": _stat(-rank_drop, -1.0)},
    }
    docs = [{"doc": 0, "A": {"nmse": 0.5}, "B": {"nmse": 0.5 + specific}, "C": {"nmse": 0.5}}]
    (run / name).write_text(json.dumps(
        {"summary": summary, "per_doc": docs, "settings": {"rate_scale": None}}
    ))


def _run(tmp_path: Path, name: str) -> Path:
    run = tmp_path / name
    run.mkdir()
    return run


def test_e1_needs_both_document_sets(tmp_path):
    both = _run(tmp_path, "living_a_512d_seed1")
    _e1(both, "adaptation_eval.json", -0.01, -2.5)
    _e1(both, "adaptation_eval_phase1.json", -0.01, -2.0)
    one = _run(tmp_path, "living_b_512d_seed1")
    _e1(one, "adaptation_eval.json", -0.01, -3.0)
    _e1(one, "adaptation_eval_phase1.json", -0.01, -1.9)
    worse = _run(tmp_path, "living_c_512d_seed1")           # a change for the worse
    _e1(worse, "adaptation_eval.json", +0.01, +3.0)
    _e1(worse, "adaptation_eval_phase1.json", +0.01, +3.0)
    hollow = _run(tmp_path, "living_d_512d_seed1")          # bought by losing rank
    _e1(hollow, "adaptation_eval.json", -0.01, -3.0, rank_drop=2.5)
    _e1(hollow, "adaptation_eval_phase1.json", -0.01, -3.0, rank_drop=2.5)
    dead = _run(tmp_path, "dead_4x_512d_seed1")
    _e1(dead, "adaptation_eval.json", 0.0, None)
    _e1(dead, "adaptation_eval_phase1.json", 0.0, None)

    verdicts = {p.name: read_e1(p)["verdict"] for p in (both, one, worse, hollow)}
    assert verdicts == {
        "living_a_512d_seed1": "meets", "living_b_512d_seed1": "candidate",
        "living_c_512d_seed1": "does not meet", "living_d_512d_seed1": "does not meet",
    }
    report = read_all(tmp_path)["verdicts"]
    assert report["e1_controls_exact_null"] is True
    assert report["e1_checkpoints_counted"] == 4
    assert report["e1_meet"] == ["living_a_512d_seed1"]
    assert report["e1_candidates"] == ["living_b_512d_seed1"]


def _e4(run: Path, emb: float, blocks: dict[str, float]) -> None:
    rows = [{"block": "emb", "probe_top1": emb, "n_pairs": 10_000}] + [
        {"block": b, "probe_top1": v, "n_pairs": 10_000} for b, v in blocks.items()
    ]
    for name in ("offset+1", "offset-1", "offset0", "causal"):
        (run / f"per_block_probe_{name}.json").write_text(
            json.dumps({"blocks": rows, "settings": {"fit_device": "cpu"}})
        )


def test_e4_is_refuted_only_by_the_same_block_in_two_seeds(tmp_path):
    margin = 3 * math.sqrt(0.1 * 0.9 / 10_000)            # 0.009
    over, under = 0.1 + margin + 0.001, 0.1 + margin - 0.001
    for seed, blocks in ((1, {"b0": over, "b1": under}), (2, {"b0": under, "b1": over}),
                         (3, {"b0": under, "b1": under})):
        _e4(_run(tmp_path, f"probe_768_x_768d_seed{seed}"), 0.1, blocks)
    assert read_e4(tmp_path / "probe_768_x_768d_seed1")["+1"]["exceed"] == ["b0"]
    assert read_all(tmp_path)["verdicts"]["e4"] == "confirmed"     # b0 once, b1 once

    _e4(tmp_path / "probe_768_x_768d_seed3", 0.1, {"b0": over, "b1": under})
    report = read_all(tmp_path)["verdicts"]
    assert report["e4"] == "refuted"
    assert list(report["e4_same_block_exceeds_in_two_seeds"]) == ["+1 b0", "-1 b0"]


def test_the_first_read_of_the_results_from_brians_card():
    """docs/research/2026-11-13_e1-e2-e4-first-read.md states these."""
    report = read_all(COLLECTED)
    v = report["verdicts"]
    assert v["e1_controls_exact_null"] is True
    assert v["e1_checkpoints_counted"] == 7 and not v["e1_meet"] and not v["e1_candidates"]
    assert v["e1_reported_not_counted"] == [
        "probe_768_visreg_768d_seed46", "probe_768_visreg_768d_seed95",
    ]
    assert v["e4"] == "confirmed" and v["e4_seeds_read"] == 3 and v["e4_seeds_admissible"] == 1
    misses = v["e2_768_family_misses"]["probe_768_visreg_768d_seed97"]
    assert len(misses) == 12
    assert sum("spectrum" in m for m in misses) == 8 and sum("store" in m for m in misses) == 4
    floors = [
        run["e1"]["at_taper_floor"]["median_change_relative_to_full_rate"]
        for run in report["runs"].values() if run["e1"] and "at_taper_floor" in run["e1"]
    ]
    # A fifth of the rate, and the change is not a fifth: it is not the write.
    assert len(floors) == 7 and all(0.8 < f < 1.2 for f in floors)
