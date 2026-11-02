"""M9 on the training path, and handed to Sanctuary (Brian, 2026-11-02:
"get this on the training path and connect it to sanctuary").

Pins:
- Without ``--m9`` the pilot driver builds ``JEPATrainer`` exactly as before
  and writes no ``m9`` block.
- With ``--m9`` it trains ``M9Trainer`` through ``JEPATrainer.run()`` (which
  could not drive an M9Trainer before: the logging flags), and the run's
  checkpoint carries the trained heads.
- M9 rides along without touching the trunk. On CPU, the trunk at the end
  of an ``--m9`` run is bit-identical to the same arm without it; only the
  text readout ``output_proj``, which M9 trains, differs. This is the check
  on the stop-grad discipline, the private random stream and the held-out
  retention batch together: a leak through any of them shows here.
- ``load_m9_run`` refuses runs without M9, refuses a life directory inside
  the run, restores the trained heads, never writes into the run, and a
  life saved with ``save_life`` continues on the next load.
- ``decode_action`` reports what an action does, in range.
"""

from __future__ import annotations

import json
import random
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scripts.jepa_pilot_driver as driver  # noqa: E402
from luthi.sanctuary_interface import (  # noqa: E402
    decode_action,
    load_m9_run,
    save_life,
)
from luthi.v2.jepa_runner import JEPATrainer  # noqa: E402
from luthi.v2.m9.runner import M9Trainer  # noqa: E402

ARM, D_MODEL, SEED = "living", 32, 42
WORDS = (
    "river stone light memory window garden quiet music signal field "
    "morning shadow pattern voice letter bridge winter colour thread"
).split()


def _args(data_dir: Path, *, m9: bool) -> Namespace:
    return Namespace(
        data_dir=str(data_dir),
        tokenizer=str(REPO_ROOT / "corpus_build" / "tokenizer_32k.json"),
        n_blocks=1, epochs=1, batch_size=4, seq_len=32, stride=32, lr=3e-4,
        holdout_fraction=0.2, heldout_batches=2, probe_batches=2,
        kill_warmup=10**9, checkpoint_interval=900, checkpoint_slots=3,
        max_batches_per_epoch=6, m9=m9,
    )


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """One small run of the same arm and seed, with and without --m9."""
    root = tmp_path_factory.mktemp("m9_path")
    corpus = root / "corpus"
    corpus.mkdir()
    rng = random.Random(0)
    (corpus / "text.txt").write_text(
        " ".join(rng.choice(WORDS) for _ in range(12000)), encoding="utf-8",
    )
    mp = pytest.MonkeyPatch()
    mp.setenv("LUTHI_ALLOW_CPU", "1")
    mp.setenv("LUTHI_CORPUS_CACHE_DIR", str(root / "cache"))
    built: dict[str, type] = {}
    real_jepa_init = JEPATrainer.__init__

    def recording_init(self, *a, **k):
        built.setdefault(type(self).__name__, type(self))
        real_jepa_init(self, *a, **k)

    mp.setattr(JEPATrainer, "__init__", recording_init)
    out = {}
    for name, m9 in (("default", False), ("m9", True)):
        mp.setattr(driver, "OUTPUT_ROOT", root / name)
        built.clear()
        result = driver._run_one(ARM, D_MODEL, SEED, _args(corpus, m9=m9))
        out[name] = {
            "dir": root / name / driver._run_name(ARM, D_MODEL, SEED),
            "result": result,
            "trainer": next(iter(built)),
        }
    mp.undo()
    out["root"] = root
    return out


def _ckpt(run_dir: Path) -> dict:
    [path] = sorted((run_dir / "checkpoints").glob("ckpt_*.pt"))[-1:]
    return torch.load(path, map_location="cpu", weights_only=False)


def test_default_path_builds_jepa_trainer_and_records_no_m9(runs):
    d = runs["default"]
    assert d["trainer"] == "JEPATrainer"
    assert "m9" not in d["result"]
    assert not (d["dir"] / "m9_action_log.jsonl").exists()
    assert not any(k.startswith("m9_") for k in _ckpt(d["dir"]))


def test_m9_path_trains_the_heads_and_records_them(runs):
    m = runs["m9"]
    assert m["trainer"] == "M9Trainer"
    assert m["result"]["outcome"] == "completed"
    assert m["result"]["m9"]["rng_seed"] == 9000 + SEED
    on_disk = json.loads((m["dir"] / "pilot_result.json").read_text())
    assert on_disk["m9"] == m["result"]["m9"]
    state = _ckpt(m["dir"])
    for key in ("m9_v_head_state_dict", "m9_habit_net_state_dict",
                "m9_decoder_attention_state_dict", "m9_optimizer_state_dict",
                "m9_lived_optimizer_state_dict", "m9_rng_state"):
        assert key in state, key
    log = (m["dir"] / "m9_action_log.jsonl").read_text().splitlines()
    assert len(log) == 6, "one action-log record per training step"


def _same(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def test_m9_rides_along_without_touching_the_trunk(runs):
    plain, m9 = _ckpt(runs["default"]["dir"]), _ckpt(runs["m9"]["dir"])
    assert plain["global_step"] == m9["global_step"] == 6

    readout = {k for k in plain["online_state_dict"] if k.startswith("output_proj.")}
    assert readout, "the trunk has a text readout to exclude"
    for k, v in plain["online_state_dict"].items():
        if k in readout:
            continue
        assert torch.equal(v, m9["online_state_dict"][k]), f"trunk differs at {k}"
    for part in ("predictor_state_dict", "projection_heads_state_dict",
                 "loss_module_buffers", "living_extra_state", "rng_state",
                 "data_loader_state"):
        assert _same(plain[part], m9[part]), f"{part} differs"
    assert not all(
        torch.equal(plain["online_state_dict"][k], m9["online_state_dict"][k])
        for k in readout
    ), "M9 trains the text readout, so it should differ"


def test_load_refuses_a_run_trained_without_m9(runs, tmp_path):
    with pytest.raises(ValueError, match="not trained with --m9"):
        load_m9_run(runs["default"]["dir"], life_dir=tmp_path / "life", device="cpu")


def test_load_refuses_a_life_inside_the_run(runs):
    with pytest.raises(ValueError, match="never writes into the run"):
        load_m9_run(runs["m9"]["dir"], life_dir=runs["m9"]["dir"] / "life", device="cpu")


def _tree(path: Path) -> dict:
    return {p: p.stat().st_mtime_ns for p in sorted(path.rglob("*"))}


@pytest.fixture
def cache_env(runs, monkeypatch):
    monkeypatch.setenv("LUTHI_CORPUS_CACHE_DIR", str(runs["root"] / "cache"))


def test_a_life_starts_from_the_run_and_continues_after_a_save(runs, tmp_path, cache_env):
    run_dir = runs["m9"]["dir"]
    before = _tree(run_dir)
    life = tmp_path / "life"

    first = load_m9_run(run_dir, life_dir=life, device="cpu")
    assert isinstance(first.trainer, M9Trainer)
    assert first.continuing_life is False
    assert first.loaded_from.parent == run_dir / "checkpoints"
    assert first.speech_trained is False, "a pure-JEPA run does not speak"
    assert first.config == {"d_model": D_MODEL, "n_blocks": 1, "seq_len": 32}
    trained = _ckpt(run_dir)["m9_v_head_state_dict"]
    for k, v in first.trainer.v_head.state_dict().items():
        assert torch.equal(v, trained[k]), "trained heads restored, not fresh ones"

    first.trainer._lived_step_count = 7  # stand-in for lived experience
    saved = save_life(first.trainer)
    assert saved.parent == life / "checkpoints"

    second = load_m9_run(run_dir, life_dir=life, device="cpu")
    assert second.continuing_life is True
    assert second.loaded_from == saved
    assert second.trainer._lived_step_count == 7
    assert second.trainer.global_step == first.trainer.global_step

    assert _tree(run_dir) == before, "a life never writes into its run"


def test_decode_action_reports_each_channel(runs, tmp_path, cache_env):
    loaded = load_m9_run(runs["m9"]["dir"], life_dir=tmp_path / "life", device="cpu")
    tokens = torch.randint(0, 100, (1, 16))
    from luthi.sanctuary_interface import encode_state, select_action
    from luthi.v2.m9.s_t import compute_s_t

    latents = encode_state(loaded.model, text_tokens=tokens, pool=False)
    s_t = compute_s_t(latents)
    selection = select_action(loaded.trainer, s_t, context_latents=latents, budget=2)
    decoded = decode_action(
        loaded.trainer, selection.action, s_t=s_t, context_latents=latents,
    )
    assert set(decoded["active"]) == {"text", "attention", "memory"}
    assert all(isinstance(v, bool) for v in decoded["active"].values())
    assert set(decoded["attention_gates"]) == {"text", "audio", "vision"}
    assert all(0.0 <= g <= 1.0 for g in decoded["attention_gates"].values())
    assert 0.0 <= decoded["memory_salience"] <= 1.0
    assert set(decoded["bands_armed"]) == {"text", "attention", "memory"}
