"""Review probe (Opus 5.5, 2026-11-10): does a resume stay exact past one epoch?

Run from the LuthiModel root, on CPU:

    LUTHI_ALLOW_CPU=1 python -m pytest redteam/opus_review_2026_11_10/probe_resume_past_the_taper.py -s -p no:cacheprovider

Builds on tests/test_resume_matches_uninterrupted.py (same arm, corpus and
driver), and adds what those tests leave out:

- Two epochs, with the cut in the second. The plasticity taper starts at half
  the run (``TaperConfig.start_fraction = 0.5``), so this cut falls after it.
- A cut right after an epoch's end-of-epoch checkpoint (the brief's third
  least-sure item).
- The RNG states are compared too (the test excludes ``rng_state``).
- The same two-epoch cut with ``_apply_taper()`` called at the end of
  ``resume()``.

What it printed on 2026-11-10 (CPU, torch 2.10):

    two epochs, cut at step 33 of 56: differences in online_state_dict (189),
    living_extra_state (440), optimizer (212), the M9 heads ... and the loss
    differs from the first step after the resume
    cut right after end_of_epoch_0, capped and uncapped: 0 differences
    two epochs with the taper re-applied at resume: 0 differences

Cause: ``rate_scale`` on each living layer is a plain attribute, not saved, and
``_apply_taper()`` runs only after each training step. The first step after a
resume runs at ``rate_scale = 1.0`` where the uninterrupted run used the
tapered value (down to the 0.2 floor).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import scripts.jepa_pilot_driver as driver  # noqa: E402
import tests.test_resume_matches_uninterrupted as T  # noqa: E402
from luthi.v2.jepa_runner import JEPATrainer  # noqa: E402
from tests.test_resume_matches_uninterrupted import world  # noqa: E402,F401  (fixture)

IGNORE = ("wall_clock_seconds", "reason")   # rng_state is compared on purpose


def _args(corpus, cap, epochs):
    a = T._args(corpus, cap)
    a.epochs = epochs
    return a


def _run(world, name, cap, epochs, cut=None):
    root, corpus, mp = world
    mp.setattr(driver, "OUTPUT_ROOT", root / name)
    run_dir = root / name / driver._run_name(T.ARM, T.D_MODEL, T.SEED)
    if cut is not None:
        cut(mp, corpus, cap, epochs)
    driver._run_one(T.ARM, T.D_MODEL, T.SEED, _args(corpus, cap, epochs))
    return T._final(run_dir)


def _differences(a, b):
    fa, fb = dict(T._flat(a)), dict(T._flat(b))
    out = sorted(set(fa) ^ set(fb))
    for k, x in fa.items():
        if k not in fb or any(s in k for s in IGNORE):
            continue
        y = fb[k]
        if torch.is_tensor(x):
            same = torch.is_tensor(y) and x.shape == y.shape and torch.equal(x, y)
        elif isinstance(x, float) and x != x:
            same = isinstance(y, float) and y != y
        else:
            same = x == y
        if not same:
            out.append(k)
    return out


def _summary(diffs):
    groups = {}
    for k in diffs:
        groups.setdefault(k.split("/")[1].split("[")[0], 0)
        groups[k.split("/")[1].split("[")[0]] += 1
    return groups


def _cut_mid(step):
    """Save three steps before ``step`` and cut the power at ``step``."""
    def cut(mp, corpus, cap, epochs):
        real_step, real_due = JEPATrainer.train_step, JEPATrainer._checkpoint_if_due

        def due(self):
            if self.global_step == step - 3:
                self._checkpoint(reason="before_the_cut")
            return real_due(self)

        def dying(self, *a, **k):
            if self.global_step >= step:
                raise T._PowerCut()
            return real_step(self, *a, **k)

        mp.setattr(JEPATrainer, "_checkpoint_if_due", due)
        mp.setattr(JEPATrainer, "train_step", dying)
        with pytest.raises(T._PowerCut):
            driver._run_one(T.ARM, T.D_MODEL, T.SEED, _args(corpus, cap, epochs))
        mp.setattr(JEPATrainer, "train_step", real_step)
        mp.setattr(JEPATrainer, "_checkpoint_if_due", real_due)
    return cut


def _cut_after_epoch_end(mp, corpus, cap, epochs):
    """Cut right after end_of_epoch_0 is written, before its held-out eval."""
    real_eval = JEPATrainer.evaluate_heldout

    def dying_eval(self, *a, **k):
        if not k.get("quick", False) and self.epoch == 0:
            raise T._PowerCut()
        return real_eval(self, *a, **k)

    mp.setattr(JEPATrainer, "evaluate_heldout", dying_eval)
    with pytest.raises(T._PowerCut):
        driver._run_one(T.ARM, T.D_MODEL, T.SEED, _args(corpus, cap, epochs))
    mp.setattr(JEPATrainer, "evaluate_heldout", real_eval)


def test_two_epochs_cut_after_the_taper_has_started(world):
    straight = _run(world, "s2", -1, 2)
    cut_at = straight["global_step"] // 2 + 5
    resumed = _run(world, "c2", -1, 2, cut=_cut_mid(cut_at))
    d = _differences(straight, resumed)
    print(f"\ntwo epochs, cut at step {cut_at} of {straight['global_step']}: {_summary(d)}")


@pytest.mark.parametrize("cap", [-1, 14])
def test_a_cut_right_after_an_epochs_end(world, cap):
    straight = _run(world, f"se{cap}", cap, 2)
    resumed = _run(world, f"ce{cap}", cap, 2, cut=_cut_after_epoch_end)
    assert resumed["global_step"] == straight["global_step"]
    print(f"\ncut right after end_of_epoch_0, cap={cap}: {_summary(_differences(straight, resumed))}")


def test_two_epochs_with_the_taper_reapplied_at_resume(world):
    _, _, mp = world
    real_resume = JEPATrainer.resume

    def resume(self, path):
        real_resume(self, path)
        self._apply_taper()

    mp.setattr(JEPATrainer, "resume", resume)
    straight = _run(world, "s2fix", -1, 2)
    resumed = _run(world, "c2fix", -1, 2, cut=_cut_mid(straight["global_step"] // 2 + 5))
    print(f"\ntwo epochs with the taper re-applied at resume: {_summary(_differences(straight, resumed))}")
