"""Rebuilding runs made before model settings were recorded (2026-11-13).

Three arms were checked against history and can be rebuilt. Any other is
refused, as before. The frozen settings do not follow the live registry.
"""

from __future__ import annotations

import json
import random
from argparse import Namespace
from pathlib import Path

import pytest
import torch

import scripts.jepa_pilot_driver as driver
from luthi.v2.run_rebuild import PRE_PROVENANCE_ARMS, PRE_PROVENANCE_NOTE, rebuild_run

REPO = Path(__file__).resolve().parents[1]


def _old_run(root: Path, monkeypatch, arm: str) -> Path:
    """A tiny run of ``arm``, with its recorded model settings removed, as a
    run from before 2026-08-05 has them."""
    monkeypatch.setenv("LUTHI_CORPUS_CACHE_DIR", str(root / "cache"))
    corpus = root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    words = "river stone light memory window garden quiet music signal".split()
    (corpus / "text.txt").write_text(" ".join(rng.choice(words) for _ in range(3000)), encoding="utf-8")
    monkeypatch.setattr(driver, "OUTPUT_ROOT", root / "runs")
    monkeypatch.setattr(driver, "ARM_FILELIST", {})
    driver._run_one(arm, 32, 42, Namespace(
        data_dir=str(corpus), tokenizer=str(REPO / "corpus_build" / "tokenizer_32k.json"),
        n_blocks=1, epochs=1, batch_size=2, seq_len=32, stride=32, lr=3e-4,
        holdout_fraction=0.2, heldout_batches=1, probe_batches=1, kill_warmup=10**9,
        checkpoint_interval=900, checkpoint_slots=3, max_batches_per_epoch=3,
    ))
    run = root / "runs" / f"{arm}_32d_seed42"
    path = run / "pilot_result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    recorded = result["config"].pop("model_kwargs")
    path.write_text(json.dumps(result), encoding="utf-8")
    return run, recorded


@pytest.mark.parametrize("arm", ["dead_4x", "living_v5_4x_d4"])
def test_a_checked_arm_is_rebuilt_as_the_driver_built_it(tmp_path: Path, monkeypatch, arm):
    run, recorded = _old_run(tmp_path, monkeypatch, arm)
    rebuilt = rebuild_run(run, device=torch.device("cpu"))
    used = rebuilt.result["config"]["model_kwargs"]
    assert {k: v for k, v in recorded.items() if k != "vocab_size"} == used, (
        "the reconstruction is what the driver records for this arm today"
    )
    assert rebuilt.result["config"]["model_kwargs_source"] == PRE_PROVENANCE_NOTE
    assert rebuilt.model is not None    # and the checkpoint loaded strictly


def test_an_unchecked_arm_is_still_refused(tmp_path: Path, monkeypatch):
    run, _ = _old_run(tmp_path, monkeypatch, "living")
    with pytest.raises(RuntimeError, match="not one whose earlier configuration has been checked"):
        rebuild_run(run, device=torch.device("cpu"))


def test_the_frozen_settings_do_not_follow_the_registry(monkeypatch):
    frozen = {arm: dict(kw) for arm, kw in PRE_PROVENANCE_ARMS.items()}
    monkeypatch.setitem(driver.ARM_CONFIGS, "dead_4x", {"dead_ffn": False, "n_heads": 16})
    assert PRE_PROVENANCE_ARMS == frozen
    assert PRE_PROVENANCE_ARMS["living_v5_4x_d4"] == PRE_PROVENANCE_ARMS["living_v5_4x_d4_rerun"]
