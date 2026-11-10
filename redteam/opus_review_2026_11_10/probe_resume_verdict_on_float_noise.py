"""Review probe (Opus 5.5, 2026-11-10): can check_resume_on_device.py ever pass on a
backend that does not repeat itself bit for bit?

Run from the LuthiModel root:

    python redteam/opus_review_2026_11_10/probe_resume_verdict_on_float_noise.py

A real checkpoint carries plain floats: every step's loss in
``smoothed_loss_buf``, the kill history, the pilot observations, M9's activity
bands. ``compare()`` files a float that differs as a "counter", as the string
``"key: a != b"``, and ``verdict()`` fails on any such string that is not also
in the noise pair's list. The noise pair differs at the same keys by different
amounts, so the strings never match.

Printed on 2026-11-10:

    (False, '2 counter(s) differ after the resume, e.g. /smoothed_loss_buf[0]: 0.5 != 0.500003')

The tensors here differ by three times the noise, which the ten-times rule
would pass. The float log entries fail it first.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.check_resume_on_device import compare, verdict  # noqa: E402


def checkpoint(shift: float = 0.0) -> dict:
    return {
        "global_step": 10,
        "online_state_dict": {"w": torch.ones(4) + shift},
        "smoothed_loss_buf": [0.5 + shift, 0.4 + shift],
    }


if __name__ == "__main__":
    a = checkpoint()
    print(verdict(compare(a, checkpoint(1e-6)), compare(a, checkpoint(3e-6)), 10, 10))
