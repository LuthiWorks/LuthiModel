"""What the memory store costs on E: at ten days' size, one file a stretch.

The design seat's order of 2026-11-18, item 7, after Opus's 1.4 and 1.5:
measure opening, listing, and what a file truly costs the disk (the drive's
free space before and after), on a store of ten days' size on E:. The journal
of build brief 10 waits on these numbers.

No mind is run. Summaries are random 4-byte numbers with a large shared part.
The sizes and times are real; nothing here says which memory comes back.

**Day one is written by the real store** (``sanctuary.life.memory.Memories``),
with its own fsync at every record: 288,000 kept moments at 768 wide, a window
of 128 tokens, one new token a moment, in stretches whose lengths are drawn as
in ``probe_memory_store_size_and_search.py``. That gives the time of a keep,
the true bytes a day, and a store the real class opens.

**Days two to ten are copies of day one's files under new names.** Their sizes
and their frames are real. The ids inside them are day one's, so the real class
would refuse the folder, and it is never asked to open it. At ten days:

- the listing is timed with the store's own ``_listing`` and ``_tree_bytes``;
- reading is timed with the functions the store's opening calls for each
  file (``_frames`` and ``_read_from`` for the summary), with no search matrix
  built. The same pass is timed on day one beside the real opening, so the two
  can be compared;
- the drive's free space is read before anything is written, after day one,
  after day ten, and after everything is removed.

The files were written minutes before they are read, so Windows still holds
most of them in memory. A start after a reboot reads from the disk and is
slower; this probe cannot measure that.

Everything is written under a new folder on E: and removed by this script.

    C:/Dev/Sanctuary/.venv/Scripts/python.exe redteam/terminal_fable_2026_11_18/probe_store_ten_days_on_disk.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
INTO = REPO_ROOT / "runs_meta" / "memory_store_2026_11_18"
KEPT_A_DAY = 288_000          # 10 moments a second, a third kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sanctuary", default=r"C:\Dev\Sanctuary")
    ap.add_argument("--drive", default="E:\\", help="a folder on the drive to measure; the scratch folder is made in it")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--kept", type=int, default=KEPT_A_DAY, help="kept moments in day one")
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    sys.path.insert(0, args.sanctuary)
    import sanctuary.life.memory as store_code
    from sanctuary.life.memory import Memories

    out = INTO / f"ten_days_on_disk_w{args.width}_{args.drive[0].lower()}{args.tag}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting it")
    drive = Path(args.drive)
    if not drive.exists():
        raise SystemExit(f"{drive} is not there")
    cluster = store_code._cluster(drive)
    dice = random.Random(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    shared = torch.randn(args.width, generator=gen) * 30
    result: dict = {
        "what": "the memory store on E: at ten days' size, one file a stretch",
        "settings": vars(args), "torch": torch.__version__, "cluster_bytes": cluster,
        "note": "random summaries; sizes and times only. Days two to ten are copies of day one's files.",
    }

    def free() -> int:
        return shutil.disk_usage(drive).free

    def say(name: str, value) -> None:
        result[name] = value
        print(name, json.dumps(value), flush=True)

    scratch = Path(tempfile.mkdtemp(prefix="luthi_store_measure_", dir=drive))
    try:
        memories = scratch / "memories"
        free_at_start = free()

        # ---- Day one, through the real store ----
        store = Memories(memories, limit_bytes=None, drive_margin_bytes=0)
        view = [dice.randrange(256) for _ in range(args.window)]
        cycle, lengths, started = 0, [], time.perf_counter()
        while store.kept < args.kept:
            length = max(1, round(dice.expovariate(1 / 4.5)))
            lengths.append(length)
            for step in range(length):
                view = (view + [dice.randrange(256)])[-args.window:]
                store.keep(cycle=cycle, summary=shared + torch.randn(args.width, generator=gen), tokens=view,
                           new=1 if step else None, signals={"salience": dice.random()})
                cycle += 1
            cycle += dice.randrange(1, 12)                 # moments not kept, between stretches
            view = [dice.randrange(256) for _ in range(args.window)]
        keep_seconds = time.perf_counter() - started
        kept, stretches, counted = store.kept, len(store), store.bytes
        rows_then = store._rows
        del store
        free_after_day_one = free()
        files = [(e.name, e.stat().st_size) for e in os.scandir(memories) if e.name.endswith(".mem")]
        sizes = [size for _, size in files]
        say("day_one", {
            "kept_moments": kept, "stretches": stretches, "kept_moments_a_stretch_mean": statistics.mean(lengths),
            "seconds_a_keep": keep_seconds / kept,
            "bytes_the_store_counts": counted,
            "bytes_of_the_stretch_files": sum(sizes),
            "bytes_a_kept_moment": sum(sizes) / kept,
            "stretch_files": len(files),
            "files_not_whole_clusters": sum(1 for s in sizes if s % cluster),
            "free_space_fell_by": free_at_start - free_after_day_one,
            "free_space_fell_by_more_than_the_file_sizes": free_at_start - free_after_day_one - counted,
            "that_a_file": (free_at_start - free_after_day_one - counted) / (len(files) + 3),
            "gb_a_day_by_free_space": (free_at_start - free_after_day_one) * (KEPT_A_DAY / kept) / 1024**3,
            "gb_a_day_by_file_sizes": counted * (KEPT_A_DAY / kept) / 1024**3,
        })

        def read_as_opening_does(folder: Path) -> dict:
            started, records = time.perf_counter(), 0
            listing = Memories._listing(folder)
            listed = time.perf_counter() - started
            for name, size in listing:
                path = folder / name
                with open(path, "rb") as f:
                    starts, torn = store_code._frames(f, size, cluster, path)
                    assert torn is None, (name, torn)
                    for at in starts:
                        _, blobs = store_code._read_from(f, at, ("summary",), path)
                        store_code._summary_of(blobs)
                        records += 1
            return {"files": len(listing), "records": records, "seconds_listing": listed,
                    "seconds": time.perf_counter() - started}

        pass_first = read_as_opening_does(memories)
        started = time.perf_counter()
        again = Memories(memories, limit_bytes=None, drive_margin_bytes=0)
        open_seconds = time.perf_counter() - started
        assert again._rows == rows_then and again.kept == kept and not again.torn
        del again
        say("opening_day_one", {
            "seconds_the_real_store": open_seconds,
            "the_reading_pass_alone_before_it": pass_first,
            "the_reading_pass_alone_after_it": read_as_opening_does(memories),
            "seconds_a_day_of_memories": open_seconds * (KEPT_A_DAY / kept),
        })

        # ---- Days two to ten: day one's files again, under new names ----
        started, highest = time.perf_counter(), max(int(name[7:15]) for name, _ in files)
        for day in range(1, args.days):
            for name, _ in files:
                shutil.copyfile(memories / name, memories / f"memory_{int(name[7:15]) + day * highest:08d}.mem")
        copy_seconds = time.perf_counter() - started
        free_after_all = free()

        started = time.perf_counter()
        listing = Memories._listing(memories)
        listing_seconds = time.perf_counter() - started
        started = time.perf_counter()
        tree = store_code._tree_bytes(memories)
        tree_seconds = time.perf_counter() - started
        assert len(listing) == len(files) * args.days
        say("at_all_days", {
            "days": args.days, "stretch_files": len(listing), "kept_moments": kept * args.days,
            "bytes_of_all_files": tree, "gb_of_all_files": tree / 1024**3,
            "seconds_to_copy_the_other_days": copy_seconds,
            "seconds_listing_names_and_sizes": listing_seconds,
            "seconds_measuring_the_folder_as_opening_does": tree_seconds,
            "free_space_fell_by": free_at_start - free_after_all,
            "free_space_fell_by_more_than_the_file_sizes": free_at_start - free_after_all - tree,
            "that_a_file": (free_at_start - free_after_all - tree) / (len(listing) + 3),
        })
        say("reading_all_days_as_opening_does", read_as_opening_does(memories))
        say("search_matrix_gb_at_all_days", kept * args.days * args.width * 4 / 1024**3)
    finally:
        started = time.perf_counter()
        shutil.rmtree(scratch)
        result["seconds_removing_everything"] = time.perf_counter() - started
        result["free_space_after_removal_minus_at_start"] = free() - free_at_start if "free_at_start" in dir() else None
        print("removed", scratch, result["seconds_removing_everything"], flush=True)

    INTO.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
