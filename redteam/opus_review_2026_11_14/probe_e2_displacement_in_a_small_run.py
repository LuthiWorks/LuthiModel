"""Review probe (Opus 5.5, 2026-11-14): in a small run, is E2's living change
(weight - set_point) one direction, and does the top-down sweep change that?

Run from the LuthiModel root, on CPU (about a minute):

    LUTHI_ALLOW_CPU=1 python -m pytest redteam/opus_review_2026_11_14/probe_e2_displacement_in_a_small_run.py -s -p no:cacheprovider

Trains ``lc768_living`` at test size for two epochs (the setup of
tests/test_resume_matches_uninterrupted.py), with the top-down sweep on and
off, and prints ``living_change_spectrum`` for every living block.

Printed on 2026-11-14:

    backward pass True:  top_share 0.999 to 1.000 in all eight blocks; rel_norm 0.14 to 0.21
    backward pass False: top_share 0.999 to 1.000 in all eight blocks; rel_norm 0.15 to 0.23

So early in a small run the displacement is one direction, as the diagnosis
says, whether or not the top-down sweep runs. (out_align prints NaN here: the
rebuilt model has run no self-modifying forward yet.)
"""

import copy
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import scripts.jepa_pilot_driver as driver  # noqa: E402
import tests.test_resume_matches_uninterrupted as T  # noqa: E402
from luthi.v2.living_diagnostics import living_change_spectrum, living_layers  # noqa: E402
from luthi.v2.run_rebuild import rebuild_run  # noqa: E402
from tests.test_resume_matches_uninterrupted import world  # noqa: E402,F401  (fixture)


@pytest.mark.parametrize("backward", [True, False])
def test_the_displacement(world, backward):
    root, corpus, mp = world
    cfg = copy.deepcopy(driver.ARM_CONFIGS[T.ARM])
    cfg["backward_pass_enabled"] = backward
    mp.setitem(driver.ARM_CONFIGS, T.ARM, cfg)
    mp.setattr(driver, "OUTPUT_ROOT", root / f"bw{int(backward)}")
    driver._run_one(T.ARM, T.D_MODEL, T.SEED, T._args(corpus, -1, 2))
    model = rebuild_run(root / f"bw{int(backward)}" / driver._run_name(T.ARM, T.D_MODEL, T.SEED),
                        device="cpu").model
    print(f"\nbackward pass {backward}:")
    for name, layer in living_layers(model):
        s = living_change_spectrum(layer)
        print(f"   {name}: top_share {s['top_share']:.3f} eff_rank {s['eff_rank']:.2f} "
              f"rel_norm {s['rel_norm']:.2e}")
