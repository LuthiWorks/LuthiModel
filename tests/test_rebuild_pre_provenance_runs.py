"""Rebuilding runs made before model settings were recorded (2026-11-13).

Three arms were checked against history and can be rebuilt. Any other is
refused, as before. The frozen settings do not follow the live registry.

A run of July 2026 differs from a run of today in four ways, and the old run
made here has all four: it recorded no model settings; its projection head
was a linear layer and a BatchNorm; its living layers lack the state added
between 2026-07-27 and 2026-07-29; and it may record its corpus at a path
from before the repo moved. The first version of this file made its old run
with today's code and only removed the settings, so it could not show the
other three. Brian's runs showed them.

What this file cannot show is that today's code computes from such a
checkpoint what July's code computed. That was measured once, with the code
of 2026-07-26 itself: docs/research/2026-11-13_rebuilding-july-runs.md.
"""

from __future__ import annotations

import json
import random
from argparse import Namespace
from pathlib import Path

import pytest
import torch

import scripts.jepa_pilot_driver as driver
import luthi.v2.run_rebuild as run_rebuild
from luthi.v2.living_layer_pc import PredictiveCodingLayer
from luthi.v2.run_rebuild import (
    CORPUS_MOVED_NOTE,
    LIVING_STATE_SINCE_2026_07_27,
    PRE_PROVENANCE_ARMS,
    PRE_PROVENANCE_NOTE,
    rebuild_run,
)

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
    monkeypatch.setitem(driver.ARM_SIGREG_PROJ, arm, "linear_bn")   # as before 2026-07-29
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
    result["config"] = {k: v for k, v in result["config"].items() if k in JULY_CONFIG}
    path.write_text(json.dumps(result), encoding="utf-8")
    for ckpt in (run / "checkpoints").glob("ckpt_*.pt"):
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        state["online_state_dict"] = {
            k: v for k, v in state["online_state_dict"].items()
            if not (".living_ffn." in k and k.rpartition(".")[2] in LIVING_STATE_SINCE_2026_07_27)
        }
        torch.save(state, ckpt)
    return run, recorded


# What a result of 2026-07-26 recorded about its run, and nothing more.
JULY_CONFIG = {
    "batch_size", "cosine_lr", "data_dir", "epochs", "file_list",
    "holdout_fraction", "lr", "lr_total_steps", "n_blocks", "seq_len",
    "sigreg_lambd", "stride",
}


@pytest.mark.parametrize("arm", ["dead_4x", "living_v5_4x_d4"])
def test_a_checked_arm_is_rebuilt_as_the_driver_built_it(tmp_path: Path, monkeypatch, arm):
    run, recorded = _old_run(tmp_path, monkeypatch, arm)
    rebuilt = rebuild_run(run, device=torch.device("cpu"))
    used = rebuilt.result["config"]["model_kwargs"]
    assert {k: v for k, v in recorded.items() if k != "vocab_size"} == used, (
        "the reconstruction is what the driver records for this arm today"
    )
    assert rebuilt.result["config"]["model_kwargs_source"] == PRE_PROVENANCE_NOTE
    cfg = rebuilt.result["config"]
    assert cfg["sigreg_projection"] == "linear_bn"
    assert cfg["taper"] is (arm != "dead_4x")
    if arm == "dead_4x":
        assert "living_state_absent_at_training" not in cfg
    else:
        assert cfg["living_state_absent_at_training"] == sorted(LIVING_STATE_SINCE_2026_07_27)
        assert len(LIVING_STATE_SINCE_2026_07_27) == 27


def test_an_unchecked_arm_is_still_refused(tmp_path: Path, monkeypatch):
    run, _ = _old_run(tmp_path, monkeypatch, "living")
    with pytest.raises(RuntimeError, match="not one whose earlier configuration has been checked"):
        rebuild_run(run, device=torch.device("cpu"))


def test_the_frozen_settings_do_not_follow_the_registry(monkeypatch):
    frozen = {arm: dict(kw) for arm, kw in PRE_PROVENANCE_ARMS.items()}
    monkeypatch.setitem(driver.ARM_CONFIGS, "dead_4x", {"dead_ffn": False, "n_heads": 16})
    assert PRE_PROVENANCE_ARMS == frozen
    assert PRE_PROVENANCE_ARMS["living_v5_4x_d4"] == PRE_PROVENANCE_ARMS["living_v5_4x_d4_rerun"]


def _living(model) -> list[PredictiveCodingLayer]:
    return [m for m in model.modules() if isinstance(m, PredictiveCodingLayer)]


def _scramble(layer: PredictiveCodingLayer) -> None:
    for i, name in enumerate(sorted(LIVING_STATE_SINCE_2026_07_27)):
        value = getattr(layer, name)
        if value.is_floating_point():
            # Spread over four orders of magnitude, so a per-row value can
            # fall outside any band drawn around the layer's median.
            spread = torch.rand(value.shape, generator=torch.Generator().manual_seed(i))
            value.copy_(spread ** 6 * 30 + 1e-3)
        elif name == "salience_window_pos":
            value.fill_(7)              # an index, so it has to stay inside the window
        else:
            value.fill_(10_000 + i)     # past every warm-up, so a mechanism that is on reads the rest


def test_the_state_an_old_run_lacks_changes_nothing_it_computes(tmp_path: Path, monkeypatch):
    """The allowance rests on this. The same old run is rebuilt twice, and in
    one every value it lacked is overwritten with nonsense. Every forward,
    with the living layers writing, must come out the same to the bit, and so
    must everything else the model holds afterwards."""
    run, _ = _old_run(tmp_path, monkeypatch, "living_v5_4x_d4")
    plain = rebuild_run(run, device=torch.device("cpu"))
    odd = rebuild_run(run, device=torch.device("cpu"))
    # The tiny run's moments were too ordinary to be stored. Both copies live
    # two moments with the bar for storing lowered, so recall has episodes to
    # read. They are still the same model after it.
    tokens = torch.randint(0, 30000, (8, 2, 32), generator=torch.Generator().manual_seed(5))

    def live(rebuilt, step: int, batch: torch.Tensor) -> dict:
        torch.manual_seed(100 + step)
        with torch.no_grad():
            return rebuilt.loss_module.compute_modality_loss("text", {"text_tokens": batch})

    for rebuilt in (plain, odd):
        for layer in _living(rebuilt.model):
            layer.salience_threshold = 0.0
        for step in (0, 1):
            live(rebuilt, step, tokens[step])
    layers = _living(odd.model)
    assert layers and all(int(layer.episode_count) > 0 for layer in layers), (
        "the stores hold episodes, so recall has something to read"
    )
    for layer in layers:
        _scramble(layer)

    before = {k: v.clone() for k, v in plain.model.state_dict().items()}
    for step in range(2, 8):
        for rebuilt in (plain, odd):
            rebuilt.loss_module.train(step >= 5)    # three in eval mode, three in train
        outs = [live(rebuilt, step, tokens[step]) for rebuilt in (plain, odd)]
        for key in ("loss", "l_pred", "target_latents", "predicted_target"):
            assert torch.equal(torch.as_tensor(outs[0][key]), torch.as_tensor(outs[1][key])), (step, key)

    after, after_odd = plain.model.state_dict(), odd.model.state_dict()
    rest = [k for k in after if k.rpartition(".")[2] not in LIVING_STATE_SINCE_2026_07_27]
    assert all(torch.equal(after[k], after_odd[k]) for k in rest)
    wrote = [k for k in rest if ".living_ffn." in k and not torch.equal(after[k], before[k])]
    assert wrote, "the living layers were writing, so this compared layers at work"


def test_only_that_state_may_be_missing(tmp_path: Path, monkeypatch):
    run, _ = _old_run(tmp_path, monkeypatch, "living_v5_4x_d4")
    ckpt = next((run / "checkpoints").glob("ckpt_*.pt"))
    original = torch.load(ckpt, map_location="cpu", weights_only=False)

    def saved_with(change) -> None:
        state = dict(original, online_state_dict=dict(original["online_state_dict"]))
        change(state["online_state_dict"])
        torch.save(state, ckpt)

    saved_with(lambda online: online.pop("blocks.0.living_ffn.set_point"))
    with pytest.raises(RuntimeError, match=r"does not fit.*living_ffn\.set_point"):
        rebuild_run(run, device=torch.device("cpu"))

    saved_with(lambda online: online.update({"blocks.0.living_ffn.something_else": torch.zeros(1)}))
    with pytest.raises(RuntimeError, match="Not expected.*something_else"):
        rebuild_run(run, device=torch.device("cpu"))

    saved_with(lambda online: online.update(
        {"blocks.0.living_ffn.weight": torch.zeros(3, 3)}
    ))
    with pytest.raises(RuntimeError, match="size mismatch"):
        rebuild_run(run, device=torch.device("cpu"))


def test_a_layer_that_reads_that_state_may_not_lack_it(tmp_path: Path, monkeypatch):
    """If one of the mechanisms were on, a fresh value would be a wrong value."""
    run, _ = _old_run(tmp_path, monkeypatch, "living_v5_4x_d4")
    monkeypatch.setitem(
        run_rebuild.PRE_PROVENANCE_ARMS, "living_v5_4x_d4",
        dict(PRE_PROVENANCE_ARMS["living_v5_4x_d4"], adaptive_episodes=True),
    )
    with pytest.raises(RuntimeError, match="does not fit.*episode_steps"):
        rebuild_run(run, device=torch.device("cpu"))


def test_a_run_made_today_still_loads_strictly(tmp_path: Path, monkeypatch):
    """The allowance is for runs with no recorded settings, and no other."""
    run, recorded = _old_run(tmp_path, monkeypatch, "living_v5_4x_d4")
    path = run / "pilot_result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result["config"].update(model_kwargs=recorded, sigreg_projection="linear_bn")
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Missing key"):
        rebuild_run(run, device=torch.device("cpu"))


# -- the corpus, when the recorded place is gone -------------------------


@pytest.fixture
def repo_with_a_list(tmp_path: Path, monkeypatch) -> Path:
    here = tmp_path / "repo"
    (here / "corpus_build").mkdir(parents=True)
    (here / "corpus_build" / "a_filelist.txt").write_text("one.txt\n", encoding="utf-8")
    monkeypatch.setattr(run_rebuild, "REPO_ROOT", here)
    return here


@pytest.mark.parametrize("recorded", [
    r"C:\Some\Old Place\LuthiModel\LuthiModel\corpus_build\a_filelist.txt",
    "/some/old place/LuthiModel/corpus_build/a_filelist.txt",
])
def test_a_list_recorded_at_a_place_that_is_gone_is_found_in_this_repo(repo_with_a_list, recorded):
    cfg = {"file_list": recorded}
    found = run_rebuild._where_the_corpus_is(recorded, cfg, "file_list")
    assert found == repo_with_a_list / "corpus_build" / "a_filelist.txt"
    assert cfg == {
        "file_list": str(found), "file_list_recorded": recorded,
        "file_list_source": CORPUS_MOVED_NOTE,
    }


def test_a_recorded_place_that_exists_is_used_as_recorded(repo_with_a_list, tmp_path):
    elsewhere = tmp_path / "corpus_build"
    elsewhere.mkdir()
    (elsewhere / "a_filelist.txt").write_text("two.txt\n", encoding="utf-8")
    cfg = {"file_list": str(elsewhere / "a_filelist.txt")}
    found = run_rebuild._where_the_corpus_is(cfg["file_list"], cfg, "file_list")
    assert found == elsewhere / "a_filelist.txt" and list(cfg) == ["file_list"]


@pytest.mark.parametrize("recorded", [
    r"C:\Some\Old Place\data\gutenberg",              # not under corpus_build
    r"C:\Some\Old Place\corpus_build\not_in_this_repo.txt",
])
def test_a_place_this_repo_cannot_stand_in_for_is_refused(repo_with_a_list, recorded):
    cfg = {"data_dir": recorded}
    with pytest.raises(FileNotFoundError, match="nothing to rebuild its data from"):
        run_rebuild._where_the_corpus_is(recorded, cfg, "data_dir")
    assert cfg == {"data_dir": recorded}
