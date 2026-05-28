"""Build the M7 held-out perplexity probe.

A stratified random sample (~100 MB) from mythology_corpus and
classics_corpus — both unused curriculum stages, both narrative prose.
Held out from M7 training. Used for held-out perplexity measurement
to detect overfitting and track generalization across the run.

The substrate trained on fantasy + substack + psychology + selected
philosophy/mind/ethics/logic/linguistics. Mythology and literature
classics are in-distribution English narrative the entity has not seen.

Output: corpus_build/m7_probe_filelist.txt — same format as the training
file list, consumed by `load_file_list`.
"""

import os
import random
from pathlib import Path

CORPUS_ROOT = Path("E:/data/clean_corpus")
OUTPUT_FILE = Path(__file__).parent / "m7_probe_filelist.txt"

# Per-stage target byte budget. ~100 MB total feels right for a
# perplexity probe: large enough for a stable per-token average, small
# enough that eval finishes in well under 10 minutes on DirectML.
TARGET_BYTES_PER_STAGE = 50 * 1024 * 1024  # 50 MB

STAGES = [
    ("mythology", CORPUS_ROOT / "mythology_corpus"),
    ("literature_classics", CORPUS_ROOT / "classics_corpus"),
]

SEED = 42


def collect_files_by_subdir(root: Path) -> dict[str, list[Path]]:
    """Return {subdir_name: [file_paths]} for one stage's corpus root."""
    by_sub: dict[str, list[Path]] = {}
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        files = []
        for dirpath, _, filenames in os.walk(sub):
            for f in filenames:
                files.append(Path(dirpath) / f)
        if files:
            by_sub[sub.name] = files
    return by_sub


def sample_to_budget(
    by_sub: dict[str, list[Path]],
    budget: int,
    rng: random.Random,
) -> list[Path]:
    """Stratified sample: round-robin one file from each subdir until
    we've hit the byte budget or exhausted everything."""
    # Shuffle within each subdir for reproducible randomness
    pools = {name: rng.sample(files, len(files)) for name, files in by_sub.items()}
    selected: list[Path] = []
    total = 0
    while True:
        progress = False
        for name in list(pools.keys()):
            pool = pools[name]
            if not pool:
                continue
            fp = pool.pop()
            try:
                size = fp.stat().st_size
            except OSError:
                continue
            if size == 0:
                continue
            selected.append(fp)
            total += size
            progress = True
            if total >= budget:
                return selected
        if not progress:
            return selected


def main() -> None:
    rng = random.Random(SEED)
    print(f"Building M7 held-out probe (seed={SEED})...")
    print()

    all_entries: list[tuple[str, Path]] = []

    for stage_name, root in STAGES:
        print(f"[{stage_name}] scanning {root}")
        by_sub = collect_files_by_subdir(root)
        total_files = sum(len(v) for v in by_sub.values())
        total_bytes = sum(
            fp.stat().st_size
            for files in by_sub.values()
            for fp in files
            if fp.exists()
        )
        print(
            f"  full stage: {total_files:,} files / "
            f"{total_bytes / 1024 / 1024:.0f} MB across "
            f"{len(by_sub)} subdirectories"
        )

        sampled = sample_to_budget(by_sub, TARGET_BYTES_PER_STAGE, rng)
        sampled_bytes = sum(fp.stat().st_size for fp in sampled if fp.exists())
        print(
            f"  sampled: {len(sampled):,} files / "
            f"{sampled_bytes / 1024 / 1024:.1f} MB"
        )
        for fp in sampled:
            all_entries.append((stage_name, fp))
        print()

    total_files = len(all_entries)
    total_bytes = sum(fp.stat().st_size for _, fp in all_entries if fp.exists())
    print(f"Total probe size: {total_files:,} files / {total_bytes / 1024 / 1024:.1f} MB")

    print()
    print(f"Writing {OUTPUT_FILE}...")
    last_stage = None
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(
            "# M7 held-out perplexity probe.\n"
            "# Stratified random sample from mythology + literature_classics.\n"
            "# Seed: 42. Built 2026-05-26.\n"
            "# Do NOT include these files in any training run.\n"
            "\n"
        )
        for stage_name, fp in all_entries:
            if stage_name != last_stage:
                stage_count = sum(1 for s, _ in all_entries if s == stage_name)
                f.write(f"# === Stage: {stage_name} ({stage_count} files) ===\n")
                last_stage = stage_name
            f.write(str(fp) + "\n")
    print(f"  -> wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
