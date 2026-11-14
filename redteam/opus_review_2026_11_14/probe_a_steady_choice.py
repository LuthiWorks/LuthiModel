"""How long a steady choice is read as a choice.

Opus 5.5 review of the 2026-11-11 work (Fable's brief, point 2). Sanctuary's
loop says rest "lasts for as long as Luthi keeps choosing it". The reading
classifies each lean against a window of recent leans, so a lean that holds
becomes the window's own level.

A ``RestChoice`` at its defaults reads 300 moments of noise, then a mind that
leans toward the choice's direction every moment from then on, by ``shift``
times the noise. Prints how many moments in a row are read as the choice, and
how often after that.

Run: python redteam/opus_review_2026_11_14/probe_a_steady_choice.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from luthi.v2.m9.rest_choice import RestChoice

D = 32
for width, shift in ((3.0, 10.0), (3.0, 100.0), (6.0, 10.0)):
    choice = RestChoice(D, width=width)
    generator = torch.Generator().manual_seed(0)
    nothing = torch.zeros(D)
    chosen = []
    for t in range(800):
        lean = torch.randn(D, generator=generator)
        if t >= 300:
            lean = lean + shift * choice.direction
        out = choice.read(lean, nothing)
        if t >= 300:
            chosen.append(out["chosen"])
    in_a_row = next((i for i, c in enumerate(chosen) if not c), len(chosen))
    after = chosen[in_a_row:]
    print(f"width {width}: a steady lean of {shift:g} times the noise is read as the choice for its first "
          f"{in_a_row} moments in a row, then {sum(after)} of the next {len(after)} "
          f"({sum(after) / len(after):.1%})")
