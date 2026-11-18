"""What the reworked memory store costs: bytes a day, and the exact search.

Brief 9 (2026-11-17) asks for two numbers:

- bytes a day at 768 wide, words only, against the design's estimate (about
  1 GB);
- the exact search's time after a day's memories at full width, against
  Opus's 0.08 s (``redteam/opus_review_2026_11_17/probe_nearest_at_scale.py``).

No mind is run and no life is lived, so there is no life folder to keep. The
summaries are random 4-byte numbers with a large shared part, as the stores of
2026-11-11 had (alike at 0.999). Random summaries say nothing about which
memory comes back; they are here for size and time only.

**Size.** ``--kept`` moments are kept through the real store, in stretches
whose lengths are drawn to match the archived stores (491 kept moments in 151
stretches, 698 in 85: about 3 to 8 a stretch). Each moment's view is a window
of ``--window`` tokens that moved by ``--new`` tokens. The bytes are read from
the disk, both as file sizes and as whole 4096-byte clusters, and scaled to a
day: 10 moments a second, a third of them kept, 288,000 kept moments.

**Search.** A matrix of a day's summaries is filled in memory (the store's own
cache, filled directly) and the store's own ``nearest`` arithmetic is timed,
on one thread and on torch's default.

    C:/Dev/Sanctuary/.venv/Scripts/python.exe redteam/terminal_fable_2026_11_18/probe_memory_store_size_and_search.py
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
INTO = REPO_ROOT / "runs_meta" / "memory_store_2026_11_18"
KEPT_A_DAY = 288_000          # 10 moments a second, a third kept
CLUSTER = 4096


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sanctuary", default=r"C:\Dev\Sanctuary")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--new", type=int, default=1, help="tokens that join the view each moment")
    ap.add_argument("--kept", type=int, default=20_000)
    ap.add_argument("--searches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.path.insert(0, args.sanctuary)
    from sanctuary.life.memory import Memories

    out = INTO / f"size_and_search_w{args.width}_new{args.new}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting it")
    dice = random.Random(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    shared = torch.randn(args.width, generator=gen) * 30

    def summary() -> torch.Tensor:
        return shared + torch.randn(args.width, generator=gen)

    # ---- Size ----
    with tempfile.TemporaryDirectory() as scratch:
        store = Memories(Path(scratch) / "memories", limit_bytes=None, drive_margin_bytes=0)
        view = [dice.randrange(256) for _ in range(args.window)]
        cycle, lengths, started = 0, [], time.perf_counter()
        while store.kept < args.kept:
            length = max(1, round(dice.expovariate(1 / 4.5)))
            lengths.append(length)
            for step in range(length):
                fresh = [dice.randrange(256) for _ in range(args.new)]
                view = (view + fresh)[-args.window:]
                store.keep(cycle=cycle, summary=summary(), tokens=view, new=args.new if step else None,
                           signals={"salience": dice.random()})
                cycle += 1
            for _ in range(dice.randrange(1, 12)):          # moments not kept, between stretches
                view = (view + [dice.randrange(256) for _ in range(args.new)])[-args.window:]
                cycle += 1
        keep_seconds = time.perf_counter() - started
        files = [p for p in (Path(scratch) / "memories").rglob("*") if p.is_file()]
        sizes = [p.stat().st_size for p in files]
        records = [p.stat().st_size for p in files if p.suffix == ".rec"]
        as_files = sum(sizes)
        as_clusters = sum(-(-s // CLUSTER) * CLUSTER for s in sizes)
        assert store.bytes == as_files, (store.bytes, as_files)
        started = time.perf_counter()
        again = Memories(Path(scratch) / "memories", limit_bytes=None, drive_margin_bytes=0)
        open_seconds = time.perf_counter() - started
        assert again._rows == store._rows
        scale = KEPT_A_DAY / store.kept
        size = {
            "kept_moments": store.kept, "memories": len(store),
            "kept_moments_a_memory_mean": statistics.mean(lengths),
            "record_bytes_median": statistics.median(records), "record_bytes_mean": statistics.mean(records),
            "bytes_as_file_sizes": as_files, "bytes_as_4096_clusters": as_clusters, "files": len(files),
            "a_day": {
                "kept_moments": KEPT_A_DAY,
                "gb_as_file_sizes": as_files * scale / 1024**3,
                "gb_as_4096_clusters": as_clusters * scale / 1024**3,
                "files": round(len(files) * scale),
                "seconds_to_open_the_store": open_seconds * scale,
            },
            "seconds_a_keep": keep_seconds / store.kept,
            "seconds_to_open_this_store": open_seconds,
        }
        del store, again
    print(json.dumps(size, indent=2), flush=True)

    # ---- Search ----
    with tempfile.TemporaryDirectory() as scratch:
        store = Memories(Path(scratch) / "memories", limit_bytes=None, drive_margin_bytes=0, recent=100)
        store.keep(cycle=0, summary=summary(), tokens=[1])      # one real memory, so there is one to open
        store._matrix = shared + torch.randn(KEPT_A_DAY, args.width, generator=gen)
        store._row_id = torch.ones(KEPT_A_DAY, dtype=torch.int64)
        store._row_seq = torch.ones(KEPT_A_DAY, dtype=torch.int64)
        store._rows = KEPT_A_DAY
        store._sum = store._matrix.sum(dim=0, dtype=torch.float64)
        alike = torch.nn.functional.cosine_similarity(store._matrix[:2000], store._matrix[2000:4000], dim=1)
        search = {"summaries": KEPT_A_DAY, "width": args.width,
                  "matrix_gb": KEPT_A_DAY * args.width * 4 / 1024**3,
                  "summaries_alike_as_they_stand_median": float(alike.median())}
        default_threads = torch.get_num_threads()
        for label, threads in (("one_thread", 1), (f"default_{default_threads}_threads", default_threads)):
            torch.set_num_threads(threads)
            took = []
            for _ in range(args.searches):
                cue = summary()
                started = time.perf_counter()
                found = store.nearest(cue, now=10**9)
                took.append(time.perf_counter() - started)
                assert found is not None
            # Opus's figure was one product of 4-byte numbers over rows made unit length beforehand.
            unit = torch.nn.functional.normalize(store._matrix, dim=1)
            plain = []
            for _ in range(args.searches):
                cue = torch.nn.functional.normalize(summary(), dim=0)
                started = time.perf_counter()
                int((unit @ cue).argmax())
                plain.append(time.perf_counter() - started)
            del unit
            search[label] = {
                "exact_8_byte_mean_removed_seconds": statistics.median(took),
                "one_4_byte_product_over_unit_rows_seconds": statistics.median(plain),
            }
            print(label, json.dumps(search[label]), flush=True)
        torch.set_num_threads(default_threads)
        del store

    INTO.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "what": "the reworked memory store's size a day and its exact search, brief 9",
        "settings": vars(args), "torch": torch.__version__, "size": size, "search": search,
        "note": "random summaries; size and time only, nothing about which memory comes back",
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
