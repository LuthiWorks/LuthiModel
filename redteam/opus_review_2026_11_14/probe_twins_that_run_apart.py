"""The layer whose twin lives ended further apart than its own home's size.

Opus 5.5 review of the 2026-11-11 work (Fable's brief, point 4, first bullet).
In Fable's half-day probe, first background, block 1's twin lives ended 11.9
apart against a home of norm 8.0, after hearing the same thing for 52,500
moments. That layer was left out of the results.

This lives the same twins (``halfday_lib.Life``: the same trunk, background and
episode, one living forward a moment, nothing else) and records, every
``EVERY`` moments, for each block:

- ``W apart``: |W_X - W_Y|, how far apart the twins' weights are;
- ``P apart``: |P_X - P_Y|, the same for the layer's prediction, which nothing
  pulls back;
- ``home apart``: |home_X - home_Y|;
- ``gap``: |W - home| in life X, against |home|. A layer whose weight is "home
  plus a trace of the last thousand forwards" keeps this small.

Run: SANCTUARY_PATH=../sanctuary python redteam/opus_review_2026_11_14/probe_twins_that_run_apart.py <run dir>
(the trunk from redteam/fable_2026_11_11/halfday_train_trunk.py)
Env: N (60000), T0 (6000), LEN (1500), BG_SEED (0), EVERY (5000).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fable_2026_11_11"))
from halfday_lib import A, B, Life, torch  # noqa: E402

run = sys.argv[1]
N = int(os.environ.get("N", 60000))
T0 = int(os.environ.get("T0", 6000))
LEN = int(os.environ.get("LEN", 1500))
BG = int(os.environ.get("BG_SEED", 0))
EVERY = int(os.environ.get("EVERY", 5000))

X, Y = Life(run, A, T0, LEN, seed=BG), Life(run, B, T0, LEN, seed=BG)
names = list(X.layers)
print(f"background {BG}; episode {T0}..{T0 + LEN}; {N} moments")
print("moment  " + "  ".join(f"{n[7]}: W apart  P apart  home apart  gap/|home|" for n in names))
t = 0
while t < N:
    span = min(EVERY, N - t)
    X.live(span)
    Y.live(span)
    t += span
    cells = []
    for n in names:
        x, y = X.layers[n], Y.layers[n]
        home = float(x.set_point.norm())
        cells.append(
            f"{float((x.weight - y.weight).norm()):9.3f} {float((x.prediction - y.prediction).norm()):8.3f} "
            f"{float((x.set_point - y.set_point).norm()):10.4f} {float((x.weight - x.set_point).norm()) / home:10.3f}"
        )
    print(f"{t:6d}  " + "  ".join(cells), flush=True)
for n in names:
    m = X.layers[n]
    print(f"{n}: |home| {float(m.set_point.norm()):.2f}, |W| {float(m.weight.norm()):.2f}, |P| {float(m.prediction.norm()):.2f}")
