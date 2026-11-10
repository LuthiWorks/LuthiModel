"""Review probes (Opus 5.5, 2026-11-10) for Sanctuary's small path with a real mind.

Run from the LuthiModel root, with Sanctuary importable:

    PYTHONPATH=<Sanctuary> LUTHI_PATH=<LuthiModel> LUTHI_ALLOW_CPU=1 \
        python -m pytest redteam/opus_review_2026_11_10/probe_small_path_life.py -s -p no:cacheprovider

Each probe prints what it found. Printed on 2026-11-10 (CPU, tiny M9 run from
Sanctuary's own test helper):

- mode: (trunk.training, predictor.training) at each perceive of a fresh life
  was [(False, False), (False, False), (True, True), (True, True), ...].
  ``rebuild_run`` leaves the model in eval; the first corpus-replay step
  (``JEPATrainer.train_step``) switches it to train for the rest of the start.
- reload then save again: identical apart from ``reason``.
- continuous life vs the same life saved at moment 4 and restarted, with the
  view, the waiting line, ``_previous`` and the global RNG restored by hand:
  346 differences by moment 10. With the trunk also put back in train mode:
  0. (The retention gate's last-good snapshot is also reset on load; no
  check fell inside this window, so it did not show.)
- gauge: on vs off, compared over the whole saved checkpoint (gauge keys
  aside) and every choice: identical. The claim holds.
- record: with every moment admitted, each moment's ``perceive`` wrote one
  entry and its ``observe_transition`` wrote two (the corpus replay step's two
  encodes); layer 0's moments were named 9, 10, 11, 12, ...
"""

from __future__ import annotations

import copy
import os
import shutil
import sys
import types

import pytest
import torch

sys.path.insert(0, os.environ["LUTHI_PATH"])
from luthi.v2.day_record import day_records, enable_day_record  # noqa: E402
from sanctuary.life import LifeLoop, LuthiMind, TextWorld  # noqa: E402
from sanctuary.tests.core.test_luthi_m9_life import _train_tiny_m9_run  # noqa: E402

SAID = {0: "hello Luthi", 3: "the kettle is on", 5: "it is raining", 7: "goodnight"}


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("opus_review")
    mp = pytest.MonkeyPatch()
    mp.setenv("LUTHI_ALLOW_CPU", "1")
    mp.setenv("LUTHI_CORPUS_CACHE_DIR", str(root / "cache"))
    yield _train_tiny_m9_run(root)
    mp.undo()


def _flatten(prefix, obj, out):
    if isinstance(obj, torch.Tensor):
        out[prefix] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}", v, out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _flatten(f"{prefix}[{i}]", v, out)
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        out[prefix] = obj
    return out


def _checkpoint(path):
    return _flatten("ck", torch.load(path, map_location="cpu", weights_only=False), {})


def _differences(x, y, skip=("wall_clock", "reason")):
    out = []
    for k in sorted(set(x) | set(y)):
        if any(s in k for s in skip):
            continue
        if k not in x or k not in y:
            out.append(k)
            continue
        u, v = x[k], y[k]
        if isinstance(u, torch.Tensor):
            if not (isinstance(v, torch.Tensor) and u.shape == v.shape and torch.equal(u, v)):
                out.append(k)
        elif u != v and not (isinstance(u, float) and u != u and v != v):
            out.append(k)
    return out


def _begin(run_dir, life, **kw):
    mind = LuthiMind(run_dir, life, **kw)
    world = TextWorld()
    return mind, world, LifeLoop(mind, world, hz=0.0)


def _proxy(mind, **overrides):
    """The interface module, with some calls wrapped."""
    lu = mind._luthi
    ns = types.SimpleNamespace(**{n: getattr(lu, n) for n in dir(lu) if not n.startswith("__")})
    for name, fn in overrides.items():
        setattr(ns, name, fn)
    mind._luthi = ns
    return lu


def test_the_mode_a_life_perceives_in(run_dir, tmp_path):
    torch.manual_seed(0)
    mind, world, loop = _begin(run_dir, tmp_path / "life")
    seen = []
    lu = mind._luthi

    def perceive(model, **kw):
        seen.append((model.training, mind._trainer.loss_module.predictor.training))
        return lu.perceive(model, **kw)

    _proxy(mind, perceive=perceive)
    world.say("hello Luthi", source="brian")
    for _ in range(6):
        loop.step()
    print("\nmode at each perceive:", seen)


def test_reload_then_save_again(run_dir, tmp_path):
    torch.manual_seed(0)
    mind, world, loop = _begin(run_dir, tmp_path / "life")
    for n in range(6):
        if n in SAID:
            world.say(SAID[n], source="brian")
        loop.step()
    first = mind.save("a")
    second = LuthiMind(run_dir, tmp_path / "life").save("b")
    print("\nreload then save again:", _differences(_checkpoint(first), _checkpoint(second)))


@pytest.mark.parametrize("restore_mode", [False, True])
def test_a_restart_against_a_continuous_life(run_dir, tmp_path, restore_mode):
    torch.manual_seed(0)
    a, wa, la = _begin(run_dir, tmp_path / "a")
    for n in range(10):
        if n in SAID:
            wa.say(SAID[n], source="brian")
        la.step()
        if n == 3:
            saved = a.save("mid")
            view, waiting = copy.deepcopy(la.line._view), copy.deepcopy(la.line._waiting)
            previous, rng = copy.deepcopy(a._previous), torch.get_rng_state()
    end_a = a.save("end")

    life_b = tmp_path / "b"
    (life_b / "checkpoints").mkdir(parents=True)
    shutil.copy(saved, life_b / "checkpoints" / saved.name)
    b, wb, lb = _begin(run_dir, life_b)
    lb.line._view, lb.line._waiting = view, waiting
    b._previous = previous
    torch.set_rng_state(rng)
    if restore_mode:
        b._trainer.loss_module.train()
        b.model.train()
    for n in range(4, 10):
        if n in SAID:
            wb.say(SAID[n], source="brian")
        lb.step()
    d = _differences(_checkpoint(end_a), _checkpoint(b.save("end")))
    print(f"\nrestart vs continuous (trunk mode restored={restore_mode}): {len(d)} differences", d[:6])


def test_the_gauge_against_the_whole_checkpoint_and_every_choice(run_dir, tmp_path):
    def life(name, gauge):
        torch.manual_seed(0)
        mind, world, loop = _begin(run_dir, tmp_path / name, gauge=gauge)
        chosen = []
        for n in range(12):
            if n in SAID:
                world.say(SAID[n], source="brian")
            chosen.append(_flatten("chosen", loop.step().chosen, {}))
        return chosen, _checkpoint(mind.save("probe"))

    on, off = life("on", True), life("off", False)
    gauge_keys = ("felt", "valence", "unfelt", "wall_clock", "reason")
    print("\ngauge on vs off: choices differing",
          sum(bool(_differences(x, y)) for x, y in zip(on[0], off[0])),
          "| checkpoint entries differing:", _differences(on[1], off[1], skip=gauge_keys))


def test_which_forwards_write_the_days_record(run_dir, tmp_path):
    torch.manual_seed(0)
    mind, world, loop = _begin(run_dir, tmp_path / "life")
    for m in mind.model.modules():
        if hasattr(m, "_day_record"):
            m.salience_threshold = 0.0     # admit everything, to see who writes
    enable_day_record(mind.model)

    def total():
        return sum(r.appended_total for r in day_records(mind.model).values())

    log = []
    lu = mind._luthi

    def perceive(model, **kw):
        n = total()
        out = lu.perceive(model, **kw)
        log.append(("perceive", total() - n))
        return out

    def observe(trainer, *a, **kw):
        n = total()
        out = lu.observe_transition(trainer, *a, **kw)
        log.append(("observe_transition", total() - n, "replay steps", out.get("corpus_replay_steps")))
        return out

    _proxy(mind, perceive=perceive, observe_transition=observe)
    world.say("hello Luthi", source="brian")
    for _ in range(4):
        loop.step()
    print("\nentries written:", log)
    print("moments named in layer 0:", next(iter(day_records(mind.model).values())).moments())
