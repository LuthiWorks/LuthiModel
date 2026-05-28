"""Build the M7 1024d training file list.

Selected for the M7 training run (decided 2026-05-25):
- Stage 3 (psychology) - full
- Stage 7 (fantasy) - full
- Stage 8 (substack_essays) - full
- Stage 1 (science_philosophy) subset: Philosophy, Philosophy_of_Mind,
  Consciousness, Neuroscience, Ethics, Logic, Linguistics

Stage 2 (code) is explicitly excluded - Brian's call: "I don't want this
version of luthimodel to have any coding knowledge at all whatsoever."

Output: corpus_build/m7_filelist.txt, formatted for m5_runner --file-list.
"""

import os
from pathlib import Path

CORPUS_ROOT = Path("E:/data/clean_corpus")
OUTPUT_FILE = Path(__file__).parent / "m7_filelist.txt"

SCIENCE_SUBJECTS = [
    "Philosophy",
    "Philosophy_of_Mind",
    "Consciousness",
    "Neuroscience",
    "Ethics",
    "Logic",
    "Linguistics",
]

STAGE_DIRS = [
    ("fantasy", CORPUS_ROOT / "fantasy_corpus", None),
    ("substack_essays", CORPUS_ROOT / "substack_corpus", None),
    ("psychology", CORPUS_ROOT / "psychology_corpus", None),
    (
        "science_philosophy_subset",
        CORPUS_ROOT / "academic_corpus",
        SCIENCE_SUBJECTS,
    ),
]


def walk_directory(root: Path, subdirs_filter: list[str] | None) -> list[Path]:
    """Walk a directory and return all files. If subdirs_filter is given,
    only walk those subdirectories of root."""
    files: list[Path] = []
    if subdirs_filter is not None:
        for sub in subdirs_filter:
            d = root / sub
            if not d.exists():
                print(f"  MISSING: {d}")
                continue
            for dirpath, _, filenames in os.walk(d):
                for fname in filenames:
                    files.append(Path(dirpath) / fname)
    else:
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                files.append(Path(dirpath) / fname)
    return files


def main() -> None:
    print("Building M7 file list...")
    print()

    all_entries: list[tuple[str, Path]] = []

    for stage_name, root, subdirs in STAGE_DIRS:
        print(f"[{stage_name}] scanning {root}")
        files = walk_directory(root, subdirs)
        total_bytes = 0
        for fp in files:
            try:
                total_bytes += fp.stat().st_size
            except OSError:
                pass
            all_entries.append((stage_name, fp))
        size_mb = total_bytes / 1024 / 1024
        size_str = f"{size_mb / 1024:.2f} GB" if size_mb > 1024 else f"{size_mb:.1f} MB"
        print(f"  -> {len(files):,} files, {size_str}")

    print()
    print(f"Total: {len(all_entries):,} files")
    total_bytes = sum(fp.stat().st_size for _, fp in all_entries if fp.exists())
    print(f"Total size: {total_bytes / 1024 / 1024 / 1024:.2f} GB")

    print()
    print(f"Writing {OUTPUT_FILE}...")

    last_stage = None
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for stage_name, fp in all_entries:
            if stage_name != last_stage:
                stage_count = sum(1 for s, _ in all_entries if s == stage_name)
                f.write(f"# === Stage: {stage_name} ({stage_count} files) ===\n")
                last_stage = stage_name
            f.write(str(fp) + "\n")

    print(f"  -> wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
