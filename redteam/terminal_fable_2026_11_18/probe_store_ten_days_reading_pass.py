"""The step ``probe_store_ten_days_on_disk.py`` did not reach on C:.

That run was stopped by the harness for low memory during its last step, the
reading pass over all ten days, and left its scratch folder in place. This
reads that folder as the store's opening does (``_listing``, then ``_frames``
and ``_read_from`` for each record's summary, with no search matrix built),
then removes the folder (Brian, 2026-11-18: finish the reading pass, then
remove it), and writes the times beside the partial result.

The files were written and copied about forty minutes before this pass and
have been read once or not at all since, so this is closer to a first read
than the other probe's pass straight after writing would have been. It is
still not a start after a reboot.

    C:/Dev/Sanctuary/.venv/Scripts/python.exe redteam/terminal_fable_2026_11_18/probe_store_ten_days_reading_pass.py --folder C:/Dev/luthi_store_measure_yup0pkma
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INTO = REPO_ROOT / "runs_meta" / "memory_store_2026_11_18"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sanctuary", default=r"C:\Dev\Sanctuary")
    ap.add_argument("--folder", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.sanctuary)
    import sanctuary.life.memory as store_code
    from sanctuary.life.memory import Memories

    scratch = Path(args.folder)
    if not scratch.name.startswith("luthi_store_measure_"):
        raise SystemExit(f"{scratch} is not a scratch folder of the ten-day probe; nothing was touched")
    memories = scratch / "memories"
    out = INTO / "ten_days_on_disk_w768_c_reading_pass.json"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting it")
    cluster = json.loads((memories / "store.json").read_text(encoding="utf-8"))["frame"]
    free_before = shutil.disk_usage(scratch).free

    started = time.perf_counter()
    listing = Memories._listing(memories)
    listed = time.perf_counter() - started
    records, at_file = 0, {}
    for n, (name, size) in enumerate(listing, start=1):
        path = memories / name
        with open(path, "rb") as f:
            starts, torn = store_code._frames(f, size, cluster, path)
            if torn is not None:
                raise RuntimeError(f"{name}: {torn}")
            for at in starts:
                _, blobs = store_code._read_from(f, at, ("summary",), path)
                store_code._summary_of(blobs)
                records += 1
        if n in (len(listing) // 10, len(listing) // 2):
            at_file[n] = time.perf_counter() - started
            print("files", n, "seconds", at_file[n], flush=True)
    seconds = time.perf_counter() - started
    result = {
        "what": "reading a ten-day store on C: as opening does; the step the stopped run did not reach",
        "folder": str(scratch), "frame_bytes": cluster,
        "files": len(listing), "records": records,
        "seconds_listing": listed, "seconds": seconds,
        "seconds_a_file": seconds / len(listing), "seconds_a_day_of_memories": seconds * 288_000 / records,
        "seconds_at_file": at_file,
        "note": "no search matrix was built; the real opening also fills one (8.2 GB at this size)",
    }
    print(json.dumps(result), flush=True)

    started = time.perf_counter()
    shutil.rmtree(scratch)
    result["seconds_removing_everything"] = time.perf_counter() - started
    result["free_space_rose_by_on_removal"] = shutil.disk_usage(scratch.parent).free - free_before
    INTO.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"removed {scratch}; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
