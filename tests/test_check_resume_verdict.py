"""The on-device resume check's verdict (scripts/check_resume_on_device.py).

The script's runs need the training machine. Its judgment is pure and is
pinned here: counts and generator states must match exactly, and measured
values (float tensors and plain floats) may differ only within the backend's
own run-to-run noise.

The first version filed every differing float as a "counter" and failed on
it. A real checkpoint is full of floats (each step's loss, the kill history,
the bands), so on a backend that does not repeat itself the check could
never pass (Opus review, 2026-11-10, B1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.check_resume_on_device import compare, verdict  # noqa: E402


def _ckpt(
    shift: float = 0.0, step: int = 10, count: int = 20, draws: int = 3,
) -> dict:
    return {
        "global_step": step,
        "wall_clock_seconds": 1.0 + shift,
        "online_state_dict": {"w": torch.ones(4) + shift},
        "smoothed_loss_buf": [0.5 + shift, 0.4 + shift],
        "living_extra_state": {"blocks.0": {"sparse_step_count": count}},
        "device_rng_state": torch.arange(draws, draws + 4, dtype=torch.uint8),
    }


def test_identical_runs_are_the_same_run():
    a = _ckpt()
    ok, why = verdict(compare(a, _ckpt()), compare(a, _ckpt()), 10, 10)
    assert ok and "identical" in why


def test_a_difference_within_the_backends_noise_passes():
    a = _ckpt()
    ok, _ = verdict(compare(a, _ckpt(1e-6)), compare(a, _ckpt(3e-6)), 10, 10)
    assert ok


def test_a_difference_far_above_the_noise_fails():
    a = _ckpt()
    ok, why = verdict(compare(a, _ckpt(1e-6)), compare(a, _ckpt(1e-2)), 10, 10)
    assert not ok and "noise" in why


def test_a_difference_where_the_backend_repeats_itself_fails():
    a = _ckpt()
    ok, _ = verdict(compare(a, _ckpt()), compare(a, _ckpt(1e-7)), 10, 10)
    assert not ok


def test_a_changed_counter_fails_however_small_the_tensors():
    a = _ckpt()
    ok, why = verdict(compare(a, _ckpt()), compare(a, _ckpt(count=27)), 10, 10)
    assert not ok and "sparse_step_count" in why


def test_running_past_the_cap_fails():
    a = _ckpt()
    ok, why = verdict(compare(a, _ckpt()), compare(a, _ckpt()), 10, 17)
    assert not ok and "step 17" in why


def test_floats_in_the_log_are_judged_against_noise_like_tensors():
    a = _ckpt()
    noise, resumed = compare(a, _ckpt(1e-6)), compare(a, _ckpt(3e-6))
    assert noise["measured_differing"] == 3 and not noise["exact"]
    ok, why = verdict(noise, resumed, 10, 10)
    assert ok and "within ten times" in why


def test_a_generator_state_that_differs_fails():
    a = _ckpt()
    ok, why = verdict(compare(a, _ckpt(1e-6)), compare(a, _ckpt(1e-6, draws=4)), 10, 10)
    assert not ok and "device_rng_state" in why


def test_a_count_that_two_uninterrupted_runs_also_disagree_on_is_not_held_against_the_resume():
    a = _ckpt()
    noise = compare(a, _ckpt(1e-6, count=21))
    ok, _ = verdict(noise, compare(a, _ckpt(1e-6, count=22)), 10, 10)
    assert ok
