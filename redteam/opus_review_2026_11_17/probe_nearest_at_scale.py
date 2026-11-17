"""How long finding the nearest memory takes as the store grows.

Opus 5.5 review of 2026-11-17 (Fable's memory store rework, §6). The rework
keeps one summary for every kept moment: at 10 moments a second with a third
kept, 288,000 a day. The design sizes the disk; this times the search.

In a rest, a memory is brought back on every other resting moment (alternating
with dreams), so at 10 Hz the store is asked for the nearest memory about
2.5 times a second.

Times ``Memories.nearest`` as it stands (Python lists, a stack of every row
per call) on summaries filled in directly, at 768 wide, and the same search
done as one matrix product over a preallocated matrix. Sizes stop where this
machine's memory does; the scaling is linear, so a day and ten days are
extrapolated and marked as such.

Run:  SANCTUARY_PATH=../sanctuary python redteam/opus_review_2026_11_17/probe_nearest_at_scale.py
Env: SIZES (comma list, default 20000,80000), D (768).
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.environ["SANCTUARY_PATH"])

import torch

from sanctuary.life.memory import Memories

D = int(os.environ.get("D", 768))
SIZES = [int(s) for s in os.environ.get("SIZES", "20000,80000").split(",")]
DAY = 288_000


def store_with(n: int) -> Memories:
    store = Memories(Path(tempfile.mkdtemp(prefix="opus_nearest_")) / "memories")
    g = torch.Generator().manual_seed(0)
    base = torch.randn(D, generator=g)
    store._vectors = list(base + 0.01 * torch.randn(n, D, generator=g))
    store._ids = list(range(1, n + 1))
    store._cycles = list(range(n))
    store.open = lambda memory_id: memory_id          # timing the search, not the file read
    return store


def main() -> None:
    torch.set_num_threads(1)
    cue = torch.randn(D)
    per_row = {}
    for n in SIZES:
        store = store_with(n)
        t0 = time.perf_counter()
        found = store.nearest(cue, now=n + 1000)
        built = time.perf_counter() - t0
        # The rows kept unit-length as they are filed, so a search is one
        # matrix-vector product and an argmax: the best an exact search can do.
        matrix = torch.nn.functional.normalize(torch.stack(store._vectors), dim=1)
        (matrix @ cue).argmax()                       # warm, so the first size is not timed cold
        t0 = time.perf_counter()
        best = int((matrix @ torch.nn.functional.normalize(cue, dim=0)).argmax()) + 1
        product = time.perf_counter() - t0
        assert best == found
        per_row[n] = (built / n, product / n)
        print(f"{n:7d} summaries ({n * D * 4 / 1e9:.2f} GB as 4-byte numbers): nearest as built {built:.3f} s; "
              f"as one product over a kept matrix {product * 1000:.1f} ms")
    n = max(per_row)
    built, product = per_row[n]
    for label, rows in (("one day", DAY), ("ten days", 10 * DAY)):
        print(f"extrapolated, {label} ({rows:,} summaries, {rows * D * 4 / 1e9:.1f} GB): as built "
              f"{built * rows:.1f} s a search; as one product {product * rows:.2f} s a search "
              f"(one thread); rest asks about 2.5 times a second")


if __name__ == "__main__":
    main()
