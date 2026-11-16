"""How many documents each rung allows, with E1's two disjoint document sets.

Opus 5.5 review of 2026-11-16 (Fable's draft Amendment 5, (b) and (g)). E1
counts a checkpoint only when the effect holds on two disjoint document sets
(phases 0 and 1). ``holdout_documents`` raises when a set does not fit, so
the held-out set fixes how many documents each rung can have. Uses the
function itself, on the held-out sizes the card's results record (July runs
7,841 windows; 768 seed 97 17,657).

Two ways to reach 1,024 forwards are compared: 256 windows read four times
(the draft's top rung), and 64 windows read sixteen times.

Run from the repo root:  python redteam/opus_review_2026_11_16/probe_how_many_documents_fit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from luthi.v2.adaptation_eval import holdout_documents

HELD_OUT = {"July runs, 512": 7841, "768 seed 97": 17657}


def most(total: int, n_read: int, n_score: int, sets: int) -> int:
    windows = torch.zeros(total, 1, dtype=torch.long)
    best = 0
    for n in range(1, total // (n_read + n_score) + 1):
        try:
            for phase in range(sets):
                holdout_documents(windows, n_read=n_read, n_score=n_score, n_docs=n, phase=phase)
        except ValueError:
            break
        best = n
    return best


def main() -> None:
    print("documents per set, with two disjoint sets (one set in brackets)")
    for name, total in HELD_OUT.items():
        print(f"{name} ({total:,} held-out windows):")
        for n_read, repeats in ((4, 1), (64, 1), (256, 1), (256, 4), (64, 4), (64, 16)):
            cells = []
            for n_score in (2, 8):
                cells.append(f"{n_score} scored: {most(total, n_read, n_score, 2):3d} ({most(total, n_read, n_score, 1):3d})")
            print(f"  read {n_read:3d} x {repeats:2d} = {n_read * repeats:5d} forwards   " + "   ".join(cells))


if __name__ == "__main__":
    main()
