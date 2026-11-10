"""A run cut off and resumed ends where an uninterrupted run ends (2026-11-10).

Brian paused training in August partly because a long run can lose power.
The driver already resumes an interrupted seed from its newest rolling
checkpoint. This pins that the resumed run is the *same run*: every tensor
and counter in its final checkpoint equals the uninterrupted run's.

The cut falls three steps after the last checkpoint, so the resumed run
retrains steps it had already taken once.

The run is the registered living arm (``lc768_living``: living layers, block
store, VISReg, the taper and cosine schedules, M9) through the real driver,
at test size on CPU.

Pins:
- Uncapped (how registered families launch): cut and resumed == straight.
- Capped with ``--max-batches-per-epoch``: cut and resumed == straight, and
  it stops at the cap. Before 2026-11-10 the cap counted from the resume, so
  a run whose last checkpoint was at step k trained cap + k steps.
- A checkpoint written before the fix still loads, counts its cap from the
  resume as it always did, and says so.

Not covered: a real GPU. ROCm kernels are not guaranteed bit-reproducible
run to run, so on the card "the same run" can only mean "within the noise of
two uninterrupted runs". Measure that there before reading this as proof.
"""

from __future__ import annotations

import logging
import random
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scripts.jepa_pilot_driver as driver  # noqa: E402
from luthi.v2.jepa_runner import JEPATrainer  # noqa: E402

ARM, D_MODEL, SEED = "lc768_living", 32, 46
CUT_AT = 7
SAVED_AT = CUT_AT - 3   # the last checkpoint before the cut: three steps are lost
WORDS = (
    "river stone light memory window garden quiet music signal field "
    "morning shadow pattern voice letter bridge winter colour thread"
).split()
# Differ between any two runs by construction, and are not the run's state.
# The RNG states are the run's state and are compared (Opus review,
# 2026-11-10: a lossy resume would show there first).
NOT_STATE = ("wall_clock_seconds", "reason")


class _PowerCut(BaseException):
    """Not an Exception, so nothing in the trainer catches it and writes a
    parting checkpoint. A power cut doesn't either."""


def _args(corpus: Path, cap: int, epochs: int = 1) -> Namespace:
    return Namespace(
        data_dir=str(corpus),
        tokenizer=str(REPO_ROOT / "corpus_build" / "tokenizer_32k.json"),
        n_blocks=2, epochs=epochs, batch_size=4, seq_len=32, stride=32, lr=3e-4,
        holdout_fraction=0.2, heldout_batches=2, probe_batches=2,
        kill_warmup=10**9,
        checkpoint_interval=10**9,   # no timed saves; see _cut
        checkpoint_slots=3, max_batches_per_epoch=cap,
    )


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("resume")
    corpus = root / "corpus"
    corpus.mkdir()
    rng = random.Random(0)
    (corpus / "text.txt").write_text(
        " ".join(rng.choice(WORDS) for _ in range(3000)), encoding="utf-8",
    )
    mp = pytest.MonkeyPatch()
    mp.setenv("LUTHI_ALLOW_CPU", "1")
    mp.setenv("LUTHI_CORPUS_CACHE_DIR", str(root / "cache"))
    # The arm's registered file list names the 768 corpus on E:. Here it
    # reads the small corpus above.
    mp.delitem(driver.ARM_FILELIST, ARM, raising=False)
    yield root, corpus, mp
    mp.undo()


def _cut(world, cap: int, epochs: int = 1, cut_at: int = CUT_AT) -> None:
    """Start a run, save once three steps before ``cut_at`` where the
    trainer saves, and cut the power at ``cut_at``. The steps between are
    lost and must be retrained."""
    _, corpus, mp = world
    real_step, real_due = JEPATrainer.train_step, JEPATrainer._checkpoint_if_due

    def due(self):
        if self.global_step == cut_at - 3:
            self._checkpoint(reason="before_the_cut")
        return real_due(self)

    def dying(self, *a, **k):
        if self.global_step >= cut_at:
            raise _PowerCut()
        return real_step(self, *a, **k)

    mp.setattr(JEPATrainer, "_checkpoint_if_due", due)
    mp.setattr(JEPATrainer, "train_step", dying)
    with pytest.raises(_PowerCut):
        driver._run_one(ARM, D_MODEL, SEED, _args(corpus, cap, epochs))
    mp.setattr(JEPATrainer, "train_step", real_step)
    mp.setattr(JEPATrainer, "_checkpoint_if_due", real_due)


def _run(
    world, name: str, cap: int, cut: bool = False, epochs: int = 1,
    cut_at: int = CUT_AT,
) -> Path:
    root, corpus, mp = world
    mp.setattr(driver, "OUTPUT_ROOT", root / name)
    run_dir = root / name / driver._run_name(ARM, D_MODEL, SEED)
    if cut:
        _cut(world, cap, epochs, cut_at)
        saved = _final(run_dir)
        assert saved["global_step"] == cut_at - 3, "steps were lost"
    driver._run_one(ARM, D_MODEL, SEED, _args(corpus, cap, epochs))
    return run_dir


def _final(run_dir: Path) -> dict:
    path = sorted((run_dir / "checkpoints").glob("ckpt_*.pt"))[-1]
    return torch.load(path, map_location="cpu", weights_only=False)


def _flat(obj, at=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flat(v, f"{at}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _flat(v, f"{at}[{i}]")
    else:
        yield at, obj


def _differences(a: dict, b: dict) -> list[str]:
    fa, fb = dict(_flat(a)), dict(_flat(b))
    out = sorted(set(fa) ^ set(fb))
    for key, x in fa.items():
        if key not in fb or any(s in key for s in NOT_STATE):
            continue
        y = fb[key]
        if torch.is_tensor(x):
            same = torch.is_tensor(y) and x.shape == y.shape and torch.equal(x, y)
        elif isinstance(x, float) and x != x:
            same = isinstance(y, float) and y != y   # NaN is NaN
        else:
            same = x == y
        if not same:
            out.append(key)
    return out


def test_an_uncapped_run_cut_and_resumed_is_the_same_run(world):
    straight = _final(_run(world, "straight", cap=-1))
    resumed = _final(_run(world, "cut", cap=-1, cut=True))
    assert straight["global_step"] > CUT_AT + 3, "the cut fell inside the run"
    assert _differences(straight, resumed) == []


def test_a_run_cut_after_the_taper_has_started_is_the_same_run(world):
    """Found by the Opus review, 2026-11-10. The plasticity taper starts at
    half the run. Its scale sits on the living layers as a plain attribute
    and was only applied after a step, so the first step after a resume ran
    untapered. One-epoch runs this short never reach the taper; this one is
    two epochs, cut in the second."""
    straight = _final(_run(world, "straight_2", cap=-1, epochs=2))
    cut_at = straight["global_step"] // 2 + 5
    resumed = _final(
        _run(world, "cut_2", cap=-1, cut=True, epochs=2, cut_at=cut_at)
    )
    assert _differences(straight, resumed) == []


def test_the_divergence_guards_memory_survives_a_resume(world):
    """Same review. The guards' state is in the checkpoint, so it is
    compared by the tests above. This pins that it is there at all."""
    state = _final(_run(world, "guard", cap=-1))
    guard = state["divergence_guard"]
    assert set(guard) == {"losses", "baseline", "run", "over_since"}
    assert "device_rng_state" in state and state["device_rng_state"] is None, (
        "no device generator on CPU"
    )


def test_a_capped_run_cut_and_resumed_stops_at_its_cap(world):
    cap = 14
    straight = _final(_run(world, "straight_cap", cap=cap))
    resumed = _final(_run(world, "cut_cap", cap=cap, cut=True))
    assert straight["global_step"] == cap
    assert resumed["global_step"] == cap, (
        "the cap counts the whole epoch, not the steps since the resume"
    )
    assert _differences(straight, resumed) == []


def test_a_checkpoint_from_before_the_fix_loads_and_says_what_it_lacks(
    world, caplog,
):
    root, corpus, mp = world
    cap = 14
    mp.setattr(driver, "OUTPUT_ROOT", root / "old")
    _cut(world, cap)

    run_dir = root / "old" / driver._run_name(ARM, D_MODEL, SEED)
    for path in (run_dir / "checkpoints").glob("ckpt_*.pt"):
        state = torch.load(path, map_location="cpu", weights_only=False)
        del state["steps_this_epoch"]
        torch.save(state, path)

    with caplog.at_level(logging.WARNING):
        driver._run_one(ARM, D_MODEL, SEED, _args(corpus, cap))
    assert _final(run_dir)["global_step"] == cap + SAVED_AT, (
        "an old checkpoint resumes as it always did"
    )
    assert any(
        "steps_this_epoch" in r.message and "past its cap" in r.message
        for r in caplog.records
    )
