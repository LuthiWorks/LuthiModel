"""What the summaries in kept memory stores are like, measured on real
stores. (2026-11-17)

Behind Sanctuary docs/research/2026-11-17_memory-store-rework-design.md,
steps 2 and 3. Brian asked how we would tell whether something could be
lost from a memory without impact. This is how it was told for the
summaries: it reads stores of the 2026-11-11 format and changes nothing.

    python redteam/fable_2026_11_17/measure_summaries_in_kept_stores.py <folder>

<folder> is searched for ``memories/summaries.jsonl``. The archive is
E:\\luthi_experiment_archive\\2026-11-11_memory_test_lives (unzipped), and
the probe lives are under ``lives_from_probes``.

It reports, for each store with at least 100 summaries:

1. Whether the text holds more than a 4-byte number does (it does not).
2. For every summary used as a cue against the others: how often a
   different memory comes back when the stored summaries are rounded to
   half precision, the largest change in likeness, and the usual gap
   between the most alike memory and the next.
3. How kept moments fall into stretches of consecutive moments.
4. How alike summaries are, one moment after another and far apart.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import torch


def main() -> int:
    root = Path(sys.argv[1])
    stores = sorted(root.rglob("memories/summaries.jsonl"))
    if not stores:
        raise SystemExit(f"no memories/summaries.jsonl under {root}")
    checked = exact = 0
    for path in stores:
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        rows = [r for r in rows if "summary" in r]
        for r in rows:
            for x in r["summary"][:8]:
                checked += 1
                exact += struct.unpack("f", struct.pack("f", x))[0] == x
        if len(rows) < 100:
            continue
        ids = torch.tensor([r["id"] for r in rows])
        v = torch.tensor([r["summary"] for r in rows], dtype=torch.float32)
        unit = torch.nn.functional.normalize(v, dim=1)
        rounded = torch.nn.functional.normalize(v.to(torch.float16).to(torch.float32), dim=1)
        same_memory = ids.unsqueeze(0) == ids.unsqueeze(1)
        full = unit @ unit.T
        half = unit @ rounded.T                 # the cue as the mind has it, the store rounded
        change = float((full - half)[~same_memory].abs().max())
        full[same_memory] = -2.0
        half[same_memory] = -2.0
        differ = int((ids[full.argmax(1)] != ids[half.argmax(1)]).sum())
        best = full.topk(2, dim=1).values
        gap = float((best[:, 0] - best[:, 1]).median())

        kept = [r for r in rows if r["kind"] == "kept"]
        cycles = torch.tensor([r["cycle"] for r in kept])
        k = torch.nn.functional.normalize(
            torch.tensor([r["summary"] for r in kept], dtype=torch.float32), dim=1)
        next_moment = (cycles[1:] - cycles[:-1]) == 1
        alike = (k[1:] * k[:-1]).sum(1)
        lengths, run = [], 1
        for follows in next_moment.tolist():
            if follows:
                run += 1
            else:
                lengths.append(run)
                run = 1
        lengths.append(run)

        print(f"{path.parent.parent.name}  (width {v.shape[1]}, {len(rows)} summaries)")
        print(f"  at half precision a different memory comes back for {differ} of {len(rows)} cues "
              f"({100 * differ / len(rows):.1f}%)")
        print(f"  largest change in likeness {change:.1e}; usual gap between the most alike and the next {gap:.1e}")
        print(f"  {len(kept)} kept moments in {len(lengths)} stretches, the longest {max(lengths)}")
        print(f"  alike, one moment after another: median {float(alike[next_moment].median()):.6f}; "
              f"far apart: median {float(alike[~next_moment].median()):.4f}")
    print(f"\n{exact} of {checked} stored numbers checked are exactly a 4-byte float")
    return 0


if __name__ == "__main__":
    sys.exit(main())
