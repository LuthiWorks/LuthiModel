"""A smaller episode store, opt-in on the training driver (2026-11-11).

Absent, nothing about a run changes. Set, the run records the setting, the
store is smaller, and the run loads back and lives.
"""

from __future__ import annotations

import random
from argparse import Namespace
from pathlib import Path

import pytest
import torch

import scripts.jepa_pilot_driver as driver
from luthi import sanctuary_interface as L
from luthi.v2.living_layer_pc import PredictiveCodingLayer

REPO = Path(__file__).resolve().parents[1]


def _train(root: Path, monkeypatch, **extra) -> Path:
    monkeypatch.setenv("LUTHI_CORPUS_CACHE_DIR", str(root / "cache"))
    corpus = root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    words = "river stone light memory window garden quiet music signal".split()
    (corpus / "text.txt").write_text(" ".join(rng.choice(words) for _ in range(3000)), encoding="utf-8")
    monkeypatch.setattr(driver, "OUTPUT_ROOT", root / "runs")
    driver._run_one("living", 32, 42, Namespace(
        data_dir=str(corpus), tokenizer=str(REPO / "corpus_build" / "tokenizer_32k.json"),
        n_blocks=1, epochs=1, batch_size=2, seq_len=32, stride=32, lr=3e-4,
        holdout_fraction=0.2, heldout_batches=1, probe_batches=1, kill_warmup=10**9,
        checkpoint_interval=900, checkpoint_slots=3, max_batches_per_epoch=3, **extra,
    ))
    return root / "runs" / "living_32d_seed42"


def _recorded(run: Path) -> dict:
    """The model settings the run wrote down, which a loader rebuilds from."""
    import json

    result = json.loads((run / "pilot_result.json").read_text(encoding="utf-8"))
    return result["config"]["model_kwargs"]


def _stores(model) -> list[torch.Tensor]:
    return [m.episode_values for m in model.modules() if isinstance(m, PredictiveCodingLayer)]


def test_without_the_flags_a_run_is_as_it_was(tmp_path: Path, monkeypatch):
    run = _train(tmp_path, monkeypatch)
    loaded = L.load_m9_run(run, life_dir=tmp_path / "life", device="cpu")
    assert "compressed_episodes" not in _recorded(run)
    assert "num_episodes" not in _recorded(run)
    assert all(s.dtype == torch.float32 and s.shape[0] == 64 for s in _stores(loaded.model))
    loaded.trainer.action_log.close()


def test_with_the_flags_the_store_is_smaller_and_the_run_loads_and_lives(tmp_path: Path, monkeypatch):
    run = _train(tmp_path, monkeypatch, compressed_episodes=True, num_episodes=8)
    loaded = L.load_m9_run(run, life_dir=tmp_path / "life", device="cpu")
    assert _recorded(run)["compressed_episodes"] is True
    assert _recorded(run)["num_episodes"] == 8
    stores = _stores(loaded.model)
    assert stores and all(s.dtype == torch.int8 and s.shape[0] == 8 for s in stores)
    full = 64 * stores[0][0].numel() * 4
    assert stores[0].numel() * stores[0].element_size() == full // 32, "a quarter the bytes, an eighth the slots"
    state = L.perceive(loaded.model, text_tokens=torch.randint(0, 100, (1, 16)))
    assert torch.isfinite(state.s_t).all()
    loaded.trainer.action_log.close()


def test_a_store_of_no_slots_is_refused(tmp_path: Path, monkeypatch):
    with pytest.raises(ValueError, match="at least 1"):
        _train(tmp_path, monkeypatch, num_episodes=0)
