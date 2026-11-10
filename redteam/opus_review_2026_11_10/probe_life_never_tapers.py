"""Follow-up probe (Opus 5.5, 2026-11-10): what living rate does a life loaded
from a tapered run start with, before and after the taper fix?

Run from the LuthiModel root, on CPU:

    LUTHI_ALLOW_CPU=1 python -m pytest redteam/opus_review_2026_11_10/probe_life_never_tapers.py -s -p no:cacheprovider

Trains ``lc768_living`` at test size (it tapers in training) for one and for
two epochs, then loads each as a life with ``load_m9_run``.

Printed on 2026-11-10, at LuthiModel 4c99fb1:

    epochs=1: life starts with rate_scale [1.0]; taper enabled False, epoch 0/3
    epochs=1: without _apply_taper (before the fix): rate_scale [1.0]
    epochs=2: life starts with rate_scale [1.0]; taper enabled False, epoch 1/3
    epochs=2: without _apply_taper (before the fix): rate_scale [1.0]

The trainer a life gets carries no taper settings and the default
``max_epochs``, so the fix changes nothing in a life, and a life always runs at
the untapered rate. For a run that finished its taper at the 0.2 floor, that is
five times the rate its living layers ended training at.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import scripts.jepa_pilot_driver as driver  # noqa: E402
import tests.test_resume_matches_uninterrupted as T  # noqa: E402
from luthi.sanctuary_interface import load_m9_run  # noqa: E402
from luthi.v2.jepa_runner import JEPATrainer  # noqa: E402
from luthi.v2.living_layer_pc import PredictiveCodingLayer  # noqa: E402
from tests.test_resume_matches_uninterrupted import world  # noqa: E402,F401  (fixture)


def _scales(model):
    return sorted({round(float(m.rate_scale), 4) for m in model.modules()
                   if isinstance(m, PredictiveCodingLayer)})


@pytest.mark.parametrize("epochs", [1, 2])
def test_the_rate_a_life_starts_at(world, tmp_path, epochs):
    root, corpus, mp = world
    mp.setattr(driver, "OUTPUT_ROOT", root / f"taper{epochs}")
    driver._run_one(T.ARM, T.D_MODEL, T.SEED, T._args(corpus, -1, epochs))
    run_dir = root / f"taper{epochs}" / driver._run_name(T.ARM, T.D_MODEL, T.SEED)
    life = load_m9_run(run_dir, life_dir=tmp_path / "life", device="cpu")
    trainer = life.trainer
    print(f"\nepochs={epochs}: life starts with rate_scale {_scales(life.model)}; "
          f"taper enabled {trainer.config.taper.enabled}, "
          f"epoch {trainer.epoch}/{trainer.config.epoch.max_epochs}")
    mp.setattr(JEPATrainer, "_apply_taper", lambda self: None)
    before = load_m9_run(run_dir, life_dir=tmp_path / "before", device="cpu")
    print(f"epochs={epochs}: without _apply_taper (before the fix): rate_scale {_scales(before.model)}")
