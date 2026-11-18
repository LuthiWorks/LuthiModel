"""The screened search as built in the store: the same answer as the full
search, and how many rows it still computes exactly.

Build brief 10, part 2, must-holds: "the same memory, the same moment, the
same likeness, next_likeness and beat as the full search, on every store under
E:/luthi_experiment_archive/2026-11-11_memory_test_lives, and on a made-up
store built to be hard". The hard made-up stores are in Sanctuary's tests
(``test_memory_store.py``, part G). This is the archived stores, and made-up
stores at full width and a day's size, where the tests cannot go.

The store's own search code is run (``Memories._find``, screened and full) on
a search matrix filled directly with the summaries: no memory is written, and
the archive is read from its zip and not changed.

**Archived stores** (old format; ``summaries.jsonl`` only). Every store with
at least ``--least`` summaries. Each summary is its own memory, the newest
``--recent`` are still the present, and the cues are the newest summaries (the
mind recalling from where it is) and summaries from anywhere. The screen's
reference is taken when the first half of the rows is in, so the mean has
moved since.

**Made-up stores,** 768 wide, ``--rows`` rows: summaries with a shared part 30
times their own (as the size probe's), a slow walk, and summaries alike at
0.99999. Times are given only if ``--time`` is passed, and say what else was
running.

    C:/Dev/Sanctuary/.venv/Scripts/python.exe redteam/terminal_fable_2026_11_18/probe_screened_search_as_built.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
INTO = REPO_ROOT / "runs_meta" / "memory_store_2026_11_18"
ARCHIVE = r"E:\luthi_experiment_archive\2026-11-11_memory_test_lives\2026-11-11_memory_test_lives.zip"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sanctuary", default=r"C:\Dev\Sanctuary")
    ap.add_argument("--archive", default=ARCHIVE)
    ap.add_argument("--least", type=int, default=60)
    ap.add_argument("--recent", type=int, default=20)
    ap.add_argument("--cues", type=int, default=60)
    ap.add_argument("--rows", type=int, default=288_000)
    ap.add_argument("--time", default="", help="what else was running, if times are to be kept")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    sys.path.insert(0, args.sanctuary)
    import sanctuary.life.memory as store_code
    from sanctuary.life.memory import Memories

    out = INTO / f"screened_search_as_built{args.tag}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting it")
    gen = torch.Generator().manual_seed(0)

    def filled(scratch: str, rows: torch.Tensor, refer_at: int) -> Memories:
        """The store's search state holding ``rows``, each its own memory, with
        the screen's reference taken when ``refer_at`` rows were in."""
        store = Memories(Path(scratch) / "memories", limit_bytes=None, drive_margin_bytes=0)
        n = rows.shape[0]
        store._matrix = rows.clone()
        store._row_id = torch.arange(1, n + 1, dtype=torch.int64)
        store._row_seq = torch.ones(n, dtype=torch.int64)
        store._rows = refer_at
        store._sum = rows[:refer_at].sum(dim=0, dtype=torch.float64)
        store._refer()
        for at in range(refer_at, n):                      # the rest, as the store adds a row
            wide = rows[at].to(torch.float64)
            store._row_len[at] = wide.norm()
            store._row_far[at] = (wide - store._ref_mean).pow(2).sum()
            store._sum = store._sum + wide
        store._rows = n
        return store

    def compare(store: Memories, cues: list[torch.Tensor], able: torch.Tensor, timed: bool) -> dict:
        ids = store._row_id[: store._rows]
        differ, candidates, chunks, screened_s, full_s = 0, [], [], [], []
        for cue in cues:
            vector = store_code._as_summary(cue)
            started = time.perf_counter()
            a = store._find(vector, able, ids, screened=True)
            screened_s.append(time.perf_counter() - started)
            started = time.perf_counter()
            b = store._find(vector, able, ids, screened=False)
            full_s.append(time.perf_counter() - started)
            differ += int(a[:4] != b[:4])
            candidates.append(a[4])
        result = {"cues": len(cues), "answers_that_differ": differ,
                  "rows_past_the_screen_median": statistics.median(candidates),
                  "rows_past_the_screen_most": max(candidates), "eligible_rows": int(able.sum())}
        if timed:
            result["seconds_screened_median"] = statistics.median(screened_s)
            result["seconds_full_median"] = statistics.median(full_s)
        return result

    archived, skipped = [], 0
    with tempfile.TemporaryDirectory() as scratch, zipfile.ZipFile(args.archive) as z:
        for k, name in enumerate(sorted(n for n in z.namelist() if n.endswith("summaries.jsonl"))):
            lines = [json.loads(t) for t in z.read(name).decode("utf-8").splitlines() if t.strip()]
            vectors = [line["summary"] for line in lines if "summary" in line]
            if len(vectors) < args.least or len({len(v) for v in vectors}) != 1:
                skipped += 1
                continue
            rows = torch.tensor(vectors, dtype=torch.float64).to(torch.float32)
            n = rows.shape[0]
            store = filled(f"{scratch}/{k}", rows, refer_at=n // 2)
            able = torch.ones(n, dtype=torch.bool)
            able[n - args.recent:] = False
            cues = [rows[n - 1 - q % args.recent] if q % 2 == 0 else rows[int(torch.randint(n, (1,), generator=gen))]
                    for q in range(args.cues)]
            archived.append({"store": name.split("/memories/")[0].split("2026-11-11_memory_test_lives/")[1],
                             "summaries": n, "width": rows.shape[1], **compare(store, cues, able, False)})
    archived.sort(key=lambda s: -s["summaries"])
    for s in archived:
        print(f"{s['store'][-36:]:36s} n={s['summaries']:4d}: differ {s['answers_that_differ']} of {s['cues']} | "
              f"past the screen: median {s['rows_past_the_screen_median']:.0f}, most {s['rows_past_the_screen_most']} "
              f"of {s['eligible_rows']}", flush=True)

    made = []
    width, n = 768, args.rows
    shared = torch.randn(width, generator=gen, dtype=torch.float64)
    kinds = {
        "shared_part_30_times_their_own": lambda: (shared * 30 + torch.randn(n, width, generator=gen)).to(torch.float32),
        "a_slow_walk": lambda: (shared / shared.norm() + _walk(n, width, gen)).to(torch.float32),
        "alike_at_0.99999": lambda: (shared / shared.norm()
                                     + torch.randn(n, width, generator=gen) * 1.6e-4).to(torch.float32),
    }
    with tempfile.TemporaryDirectory() as scratch:
        for k, (kind, make) in enumerate(kinds.items()):
            rows = make()
            store = filled(f"{scratch}/{k}", rows, refer_at=int(n / 1.5))
            able = torch.ones(n, dtype=torch.bool)
            able[n - 300:] = False                         # the last half minute of keeps is the present
            spread = float(rows[:2000].std(dim=0).mean())
            cues = []
            for q in range(args.cues // 2):
                i = n - 1 - q if q % 2 == 0 else int(torch.randint(n, (1,), generator=gen))
                cues.append(rows[i] + torch.randn(width, generator=gen) * spread * (0.0, 0.01, 0.3)[q % 3])
            alike = torch.nn.functional.cosine_similarity(rows[:2000].double(), rows[2000:4000].double(), dim=1)
            row = {"store": kind, "summaries": n, "width": width,
                   "alike_as_they_stand_median": float(alike.median()),
                   "length_over_distance_from_the_mean": float(
                       (store._row_len[:2000] / store._row_far[:2000].sqrt().clamp_min(1e-300)).median()),
                   **compare(store, cues, able, bool(args.time))}
            made.append(row)
            print(json.dumps(row), flush=True)
            del store, rows

    result = {"what": __doc__.splitlines()[0], "settings": vars(args), "torch": torch.__version__,
              "threads": torch.get_num_threads(),
              "archived_stores_read": len(archived), "archived_stores_too_small_or_mixed": skipped,
              "archived_answers_that_differ": sum(s["answers_that_differ"] for s in archived),
              "archived": archived, "made_up": made,
              "note": "the made-up stores say nothing about which memory a mind would get back; size and agreement only"}
    INTO.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


def _walk(n: int, width: int, gen: torch.Generator) -> torch.Tensor:
    steps = torch.randn(n, width, generator=gen) * 2e-4
    walk = steps.to(torch.float64).cumsum(0)
    return walk / walk.norm(dim=1).max() * 0.045


if __name__ == "__main__":
    sys.exit(main())
