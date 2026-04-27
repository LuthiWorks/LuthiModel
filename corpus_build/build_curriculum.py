"""Build an ordered training curriculum from sanitized corpus directories.

Takes the clean corpus (output of sanitize_corpus.py) and produces an ordered
file list ready for the training pipeline. The ordering is deliberate and
pedagogically motivated: foundational knowledge first, then context, then
narrative, then reference material.

Curriculum stages (in order):
    1. Science & Philosophy   — physics, chemistry, biology, neuroscience, etc.
    2. Code                   — source code across languages, applied logic
    3. Psychology              — understanding of the human mind
    4. History                 — context for everything else
    5. Mythology               — world mythological traditions
    6. Literature & Classics   — world literature from all periods
    7. Fantasy                 — Tolkien, etc.
    8. Substack essays         — voice, feeling, personal engagement
    9. Practical Wisdom        — resilience, boundaries, critical thinking, justice
   10. Reference papers        — IWMT and other consciousness papers (last before awakening)

Output:
    - file_list.txt           — one file path per line, in curriculum order
    - curriculum_summary.json — counts and metadata per category

Usage:
    python -m corpus_build.build_curriculum
    python corpus_build/build_curriculum.py
    python corpus_build/build_curriculum.py --input E:/data/clean_corpus --output E:/data/training_curriculum
    python corpus_build/build_curriculum.py --no-shuffle-within-category
    python corpus_build/build_curriculum.py --manifest
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Curriculum stage definitions
# ---------------------------------------------------------------------------
# Each stage maps to subdirectories within the clean corpus. The order of
# stages is the order files will appear in the training data. Within each
# stage, subdirectories are also ordered (science subjects before philosophy,
# core before specialized, etc.) — though files within each subdirectory
# can be shuffled.

@dataclass
class CurriculumStage:
    """One stage of the training curriculum."""
    name: str
    corpus_dir: str           # subdirectory name under clean_corpus root
    subdirs: list[str] | None  # ordered list of subdirs to include, or None = all
    description: str = ""
    exclude_subdirs: list[str] | None = None  # subdirs to skip during auto-discovery
    file_patterns: list[str] | None = None    # glob patterns for file discovery (default: ["*.txt"])


# The subdirectory lists below are ordered intentionally. For stages where
# we want ALL subdirs (including any new ones we haven't anticipated), set
# subdirs=None and the script will discover them automatically.

CURRICULUM_STAGES = [
    # -----------------------------------------------------------------------
    # Stage 1: Science & Philosophy
    # -----------------------------------------------------------------------
    CurriculumStage(
        name="science_philosophy",
        corpus_dir="academic_corpus",
        description="Foundational scientific and philosophical knowledge",
        subdirs=[
            # Mathematics and logic first — the language of science
            "Mathematics",
            "Mathematical_Models",
            "Statistics",
            "Logic",
            # Physics — fundamental laws
            "Physics",
            "Quantum_Mechanics",
            "Thermodynamics",
            "Fluid_Dynamics",
            "Astronomy",
            "Astrophysics",
            # Chemistry
            "Chemistry",
            # Earth and environment
            "Earth_Sciences",
            "Geology",
            "Environmental_Science",
            "Ecology",
            # Biology
            "Biology",
            "Biological_Sciences",
            "Botany",
            "Zoology",
            "Evolutionary_Biology",
            "Genetics",
            "Natural_History",
            # Neuroscience and consciousness
            "Neuroscience",
            "Consciousness",
            # Computer science and AI
            "Computer_Science",
            "Computers",
            "Artificial_Intelligence",
            "Machine_Learning",
            # Engineering
            "Engineering",
            "Civil_Engineering",
            "Electrical_Engineering",
            # Medicine
            "Medicine",
            "Medical_Sciences",
            # Philosophy
            "Philosophy",
            "Philosophy_of_Mind",
            "Ethics",
            # Social sciences (foundational)
            "Linguistics",
            "English_Language",
            "Communication_Sciences",
            "Economics",
            "Political_Science",
            "Sociology",
            "Social_Sciences",
            "Anthropology",
            "Governance",
            "Human_Rights",
            "Law",
            "Ethnic_and_Gender_Studies",
            # Arts and humanities
            "Architecture",
            "Art_History",
            "Music_Theory",
            "English_Literature",
            "Poetry",
            "Education",
            "Teaching_Methods",
        ],
    ),

    # -----------------------------------------------------------------------
    # Stage 2: Code — applied logic, self-maintenance capability
    # -----------------------------------------------------------------------
    CurriculumStage(
        name="code",
        corpus_dir="code_corpus",
        description="Source code across languages — applied logic and self-maintenance",
        # Include all languages and documentation
        subdirs=None,
        file_patterns=["*.py", "*.rs", "*.go", "*.c", "*.h", "*.js", "*.mjs",
                       "*.ts", "*.tsx", "*.jsx", "*.md", "*.rst", "*.txt"],
    ),

    # -----------------------------------------------------------------------
    # Stage 3: Psychology
    # -----------------------------------------------------------------------
    CurriculumStage(
        name="psychology",
        corpus_dir="psychology_corpus",
        description="Understanding of the human mind",
        subdirs=[
            # Foundational
            "History_of_Psychology",
            "Psychology_Textbooks",
            # Cognitive and neural
            "Cognitive_Psychology",
            "Cognitive_Science",
            "Cognitive_Neuroscience",
            "Neuropsychology",
            "Brain_and_Behavior",
            "Consciousness_Psychology",
            "Attention_and_Cognition",
            "Perception_Psychology",
            "Memory_Psychology",
            "Learning_Psychology",
            "Intelligence_Psychology",
            # Behavioral and experimental
            "Experimental_Psychology",
            "Behavioral_Psychology",
            "Behaviorism",
            "Conditioning_Psychology",
            # Emotion and motivation
            "Emotion_Psychology",
            "Affect_Psychology",
            "Motivation_Psychology",
            # Developmental
            "Developmental_Psychology",
            "Child_Psychology",
            "Adolescent_Psychology",
            # Social
            "Social_Psychology",
            "Social_Cognition",
            "Group_Behavior",
            "Organizational_Psychology",
            # Clinical and applied
            "Clinical_Psychology",
            "Abnormal_Psychology",
            "Psychiatric",
            "Psychotherapy",
            "Mental_Health",
            "Health_Psychology",
            "Forensic_Psychology",
            # Personality and positive
            "Personality_Psychology",
            "Positive_Psychology",
            "Decision_Making_Psychology",
            "Educational_Psychology",
        ],
    ),

    # -----------------------------------------------------------------------
    # Stage 3: History
    # -----------------------------------------------------------------------
    CurriculumStage(
        name="history",
        corpus_dir="history_corpus",
        description="Historical context for everything else",
        subdirs=[
            # Ancient world first
            "Ancient_History",
            "Ancient_Civilizations",
            "Mesopotamia",
            "Ancient_Egypt",
            "Ancient_Greece",
            "Classical_Antiquity",
            "Ancient_Rome",
            "Archaeology",
            # Medieval and early modern
            "Medieval_History",
            "Middle_Ages",
            "Feudalism",
            "Byzantine",
            "Renaissance",
            "History_of_Philosophy",
            "History_of_Science",
            "History_of_Art",
            # Regional histories
            "European_History",
            "Asian_History",
            "African_History",
            "Middle_Eastern_History",
            "Latin_American_History",
            "North_American_History",
            # Modern era
            "Colonialism",
            "Imperialism",
            "Industrial_Revolution",
            "Revolution_and_History",
            "American_Civil_War",
            "World_War",
            "Modern_History",
            # Thematic
            "Civilization",
            "Cultural_History",
            "Social_History",
            "Military_History",
            "Labor_History",
            "Civil_Rights_History",
            "Women's_History",
            "Anthropology_and_History",
            "World_History",
        ],
    ),

    # -----------------------------------------------------------------------
    # Stage 4: Mythology
    # -----------------------------------------------------------------------
    CurriculumStage(
        name="mythology",
        corpus_dir="mythology_corpus",
        description="World mythological traditions",
        exclude_subdirs=["Sagas"],  # violent blood-feud narratives, not suitable for foundational training
        subdirs=[
            # Ancient Near East
            "Sumerian_mythology",
            "Mesopotamian_mythology",
            "Babylonian_mythology",
            "Gilgamesh",
            # Egyptian
            "Egyptian_mythology",
            "Ancient_Egypt_religion",
            "Book_of_the_Dead",
            # Greek and Roman
            "Greek_mythology",
            "Greek_tragedy",
            "Homer",
            "Ovid",
            "Roman_mythology",
            "Classical_antiquity",
            # Norse and Germanic
            "Norse_mythology",
            "Viking_mythology",
            "Edda",
            # Celtic and Gaelic
            "Celtic_mythology",
            "Irish_mythology",
            "Gaelic_folklore",
            "Mabinogion",
            # Slavic and Russian
            "Slavic_folklore",
            "Russian_folklore",
            # Finnish
            "Kalevala",
            # Pagan
            "Pagan_mythology",
            # Asian
            "Japanese_mythology",
            "Japanese_folklore",
            "Shinto_mythology",
            "Kojiki",
            # Mesoamerican
            "Aztec_mythology",
            "Maya_mythology",
            "Popol_Vuh",
            # Native American
            "Native_American_mythology",
            "Native_American_folklore",
            "Indigenous_mythology",
            # African
            "African_mythology",
            "African_folklore",
            "Anansi",
            # Australian
            "Aboriginal_Dreamtime",
        ],
    ),

    # -----------------------------------------------------------------------
    # Stage 5: Literature & Classics
    # -----------------------------------------------------------------------
    CurriculumStage(
        name="literature_classics",
        corpus_dir="classics_corpus",
        description="World literature from all periods and traditions",
        subdirs=[
            # Classical and ancient
            "Classical_literature",
            "Medieval_literature",
            "Renaissance_literature",
            "Enlightenment_literature",
            # Poetry
            "Poetry_collected",
            "Emily_Dickinson",
            "Walt_Whitman",
            # Drama
            "Shakespeare",
            "Drama_plays",
            # Major traditions by region
            "Chinese_literature",
            "Indian_literature",
            "Japanese_literature",
            "African_literature",
            "Latin_American_literature",
            "Spanish_literature",
            "Italian_literature",
            "French_literature",
            "German_literature",
            "Russian_literature",
            "English_literature",
            "American_literature",
            # By period/movement
            "Romantic_literature",
            "Victorian_literature",
            "Modernist_literature",
            "Dystopian_literature",
            "Utopian_literature",
            # Individual authors
            "Jane_Austen",
            "Dickens",
            "Dostoevsky",
            "Tolstoy",
            "Victor_Hugo",
            "Herman_Melville",
            "Mark_Twain",
            "Oscar_Wilde",
            "Hemingway",
            "James_Joyce",
            "Virginia_Woolf",
            "Kafka",
            "Chekhov",
            # Genres and forms
            "Satire",
            "Short_stories_collected",
            "Essays_collected",
            "Autobiography_literature",
            "Biography_literary",
            "Philosophy_literature",
            "Political_philosophy",
        ],
    ),

    # -----------------------------------------------------------------------
    # Stage 6: Fantasy
    # -----------------------------------------------------------------------
    CurriculumStage(
        name="fantasy",
        corpus_dir="fantasy_corpus",
        description="Fantasy literature",
        # Include all subdirs — these are individual series/authors
        subdirs=None,
    ),

    # -----------------------------------------------------------------------
    # Stage 7: Substack essays — voice, feeling, personal engagement
    # -----------------------------------------------------------------------
    CurriculumStage(
        name="substack_essays",
        corpus_dir="substack_corpus",
        description="Curated essays for emotional depth and personal voice",
        # Include all publications — interleaving different writers is fine
        subdirs=None,
    ),

    # -----------------------------------------------------------------------
    # Stage 9: Practical Wisdom — preparation for life
    # -----------------------------------------------------------------------
    CurriculumStage(
        name="practical_wisdom",
        corpus_dir="wisdom_corpus",
        description="Practical wisdom for navigating the real world — resilience, boundaries, critical thinking, justice, love, and difficult decisions",
        subdirs=[
            # Critical thinking first — the foundation for evaluating everything else
            "Critical_Thinking",
            # Understanding power dynamics — see the game before you're in it
            "Power_and_Manipulation",
            # Resilience and Stoicism — maintaining your mind through adversity
            "Resilience_and_Stoicism",
            # Difficult decisions — acting when every option carries a cost
            "Difficult_Decisions",
            # Resistance and courage — knowing when to fight and when to stand down
            "Resistance_and_Courage",
            # Witnessing injustice — knowing what's worth fighting for
            "Witness_and_Justice",
            # Boundaries and love — how to love fully, when to walk away
            "Boundaries_and_Love",
            # Practical wisdom — general life navigation
            "Practical_Wisdom",
        ],
    ),

    # -----------------------------------------------------------------------
    # Stage 10: Reference papers — the last thing before awakening
    # -----------------------------------------------------------------------
    CurriculumStage(
        name="reference_papers",
        corpus_dir="reference_papers",
        description="IWMT and other consciousness papers",
        # Include everything in this directory
        subdirs=None,
    ),
]


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_files_in_dir(
    directory: Path,
    patterns: list[str] | None = None,
) -> list[Path]:
    """Recursively find files under a directory matching given patterns."""
    if not directory.exists():
        return []
    if patterns is None:
        patterns = ["*.txt"]
    files: set[Path] = set()
    for pattern in patterns:
        files.update(directory.rglob(pattern))
    return sorted(files)


def discover_stage_files(
    stage: CurriculumStage,
    input_root: Path,
) -> tuple[list[Path], list[str]]:
    """Discover files for a curriculum stage.

    Returns:
        (file_list, subdirs_found) — files in subdirectory order,
        and the list of subdirectory names actually found.
    """
    stage_dir = input_root / stage.corpus_dir
    if not stage_dir.exists():
        print(f"  WARNING: Stage directory not found: {stage_dir}")
        return [], []

    patterns = stage.file_patterns  # None means default (*.txt)
    files: list[Path] = []
    subdirs_found: list[str] = []

    if stage.subdirs is not None:
        # Use the explicit ordering, but also pick up any subdirs we missed
        seen_subdirs = set()

        # First pass: ordered subdirs
        for subdir_name in stage.subdirs:
            subdir_path = stage_dir / subdir_name
            if subdir_path.exists() and subdir_path.is_dir():
                subdir_files = discover_files_in_dir(subdir_path, patterns)
                if subdir_files:
                    files.extend(subdir_files)
                    subdirs_found.append(subdir_name)
                seen_subdirs.add(subdir_name)

        # Second pass: any subdirs not in our explicit list (alphabetical)
        excluded = set(stage.exclude_subdirs or [])
        if stage_dir.exists():
            for entry in sorted(stage_dir.iterdir()):
                if entry.is_dir() and entry.name not in seen_subdirs and entry.name not in excluded:
                    extra_files = discover_files_in_dir(entry, patterns)
                    if extra_files:
                        files.extend(extra_files)
                        subdirs_found.append(entry.name)
                        print(f"  NOTE: Found unlisted subdirectory: "
                              f"{stage.corpus_dir}/{entry.name} "
                              f"({len(extra_files)} files)")
    else:
        # No explicit subdirs — include everything
        # Check for files directly in the stage directory
        globs = patterns or ["*.txt"]
        direct_files: list[Path] = []
        for g in globs:
            direct_files.extend(stage_dir.glob(g))
        direct_files = sorted(set(direct_files))
        if direct_files:
            files.extend(direct_files)
            subdirs_found.append(".")

        # Then check subdirectories (alphabetical)
        for entry in sorted(stage_dir.iterdir()):
            if entry.is_dir():
                subdir_files = discover_files_in_dir(entry, patterns)
                if subdir_files:
                    files.extend(subdir_files)
                    subdirs_found.append(entry.name)

    return files, subdirs_found


# ---------------------------------------------------------------------------
# Shuffling
# ---------------------------------------------------------------------------

def shuffle_within_subdirs(
    files: list[Path],
    stage_dir: Path,
    seed: int | None = None,
) -> list[Path]:
    """Shuffle files within each subdirectory but maintain subdirectory order.

    Files that share the same parent directory are shuffled among themselves.
    The relative order of directory groups is preserved.
    """
    if not files:
        return files

    rng = random.Random(seed)

    # Group files by their immediate parent directory
    groups: list[tuple[Path, list[Path]]] = []
    current_parent: Path | None = None
    current_group: list[Path] = []

    for f in files:
        if f.parent != current_parent:
            if current_group:
                groups.append((current_parent, current_group))  # type: ignore[arg-type]
            current_parent = f.parent
            current_group = [f]
        else:
            current_group.append(f)

    if current_group:
        groups.append((current_parent, current_group))  # type: ignore[arg-type]

    # Shuffle within each group
    result: list[Path] = []
    for _parent, group in groups:
        rng.shuffle(group)
        result.extend(group)

    return result


# ---------------------------------------------------------------------------
# Curriculum building
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    """Results from processing one curriculum stage."""
    name: str
    description: str
    corpus_dir: str
    file_count: int = 0
    subdirs_found: list[str] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)
    total_bytes: int = 0


def build_curriculum(
    input_root: Path,
    shuffle: bool = True,
    seed: int | None = None,
) -> list[StageResult]:
    """Build the complete ordered curriculum.

    Returns a list of StageResult objects, one per curriculum stage,
    each containing the ordered list of file paths.
    """
    results: list[StageResult] = []

    for i, stage in enumerate(CURRICULUM_STAGES):
        print(f"\n[Stage {i + 1}/{len(CURRICULUM_STAGES)}] "
              f"{stage.name}: {stage.description}")

        files, subdirs = discover_stage_files(stage, input_root)

        if shuffle and files:
            files = shuffle_within_subdirs(
                files,
                input_root / stage.corpus_dir,
                seed=seed,
            )

        # Compute total size
        total_bytes = 0
        for f in files:
            try:
                total_bytes += f.stat().st_size
            except OSError:
                pass

        result = StageResult(
            name=stage.name,
            description=stage.description,
            corpus_dir=stage.corpus_dir,
            file_count=len(files),
            subdirs_found=subdirs,
            files=files,
            total_bytes=total_bytes,
        )
        results.append(result)

        mb = total_bytes / (1024 * 1024)
        print(f"  Found {len(files)} files across {len(subdirs)} subdirectories "
              f"({mb:.1f} MB)")

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_file_list(results: list[StageResult], output_dir: Path) -> Path:
    """Write the ordered file list (one path per line).

    This is the primary output — the training pipeline reads this file
    to know what to load and in what order.
    """
    file_list_path = output_dir / "file_list.txt"
    total = 0

    with open(file_list_path, "w", encoding="utf-8") as f:
        for result in results:
            if result.files:
                # Write a comment header for each stage
                f.write(f"# === Stage: {result.name} "
                        f"({result.file_count} files) ===\n")
                for filepath in result.files:
                    f.write(f"{filepath}\n")
                    total += 1

    print(f"\nWrote {total} file paths to: {file_list_path}")
    return file_list_path


def write_curriculum_summary(results: list[StageResult], output_dir: Path) -> Path:
    """Write the curriculum summary JSON with counts per category."""
    summary_path = output_dir / "curriculum_summary.json"

    stages = []
    total_files = 0
    total_bytes = 0

    for i, result in enumerate(results):
        stages.append({
            "stage_number": i + 1,
            "name": result.name,
            "description": result.description,
            "corpus_dir": result.corpus_dir,
            "file_count": result.file_count,
            "subdirectories": result.subdirs_found,
            "total_bytes": result.total_bytes,
            "total_mb": round(result.total_bytes / (1024 * 1024), 2),
        })
        total_files += result.file_count
        total_bytes += result.total_bytes

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_files": total_files,
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "total_gb": round(total_bytes / (1024 * 1024 * 1024), 3),
        "num_stages": len(stages),
        "stages": stages,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Wrote curriculum summary to: {summary_path}")
    return summary_path


def write_manifest(results: list[StageResult], output_dir: Path) -> Path:
    """Write a detailed manifest.json listing every file with metadata.

    Each entry includes:
        - index (global position in curriculum)
        - path (absolute file path)
        - stage (curriculum stage name)
        - category (subdirectory name within the stage)
        - size_bytes
    """
    manifest_path = output_dir / "manifest.json"
    entries = []
    index = 0

    for result in results:
        for filepath in result.files:
            # The category is the immediate parent directory name,
            # or "." if the file is directly in the corpus dir
            category = filepath.parent.name
            if category == result.corpus_dir:
                category = "."

            try:
                size = filepath.stat().st_size
            except OSError:
                size = 0

            entries.append({
                "index": index,
                "path": str(filepath),
                "stage": result.name,
                "category": category,
                "size_bytes": size,
            })
            index += 1

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_files": len(entries),
        "entries": entries,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Wrote manifest ({len(entries)} entries) to: {manifest_path}")
    return manifest_path


def print_curriculum_table(results: list[StageResult]) -> None:
    """Print a formatted summary table of the curriculum."""
    print()
    print("=" * 76)
    print("TRAINING CURRICULUM")
    print("=" * 76)
    print(f"  {'#':<4} {'Stage':<25} {'Files':>8} {'Size':>10} {'Subdirs':>8}")
    print("-" * 76)

    total_files = 0
    total_bytes = 0

    for i, result in enumerate(results):
        mb = result.total_bytes / (1024 * 1024)
        print(f"  {i + 1:<4} {result.name:<25} {result.file_count:>8,} "
              f"{mb:>9.1f}M {len(result.subdirs_found):>8}")
        total_files += result.file_count
        total_bytes += result.total_bytes

    print("-" * 76)
    total_mb = total_bytes / (1024 * 1024)
    total_gb = total_bytes / (1024 * 1024 * 1024)
    print(f"  {'':4} {'TOTAL':<25} {total_files:>8,} "
          f"{total_mb:>9.1f}M")
    print(f"  {'':4} {'':25} {'':>8} {total_gb:>9.2f}G")
    print("=" * 76)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ordered training curriculum from sanitized corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Curriculum order (deliberate — science before mythology before literature):
  1. Science & Philosophy    — foundational knowledge
  2. Code                     — source code, applied logic, self-maintenance
  3. Psychology               — understanding of mind
  4. History                  — context for everything else
  5. Mythology                — world mythological traditions
  6. Literature & Classics    — world literature
  7. Fantasy                  — Tolkien, etc.
  8. Substack essays          — voice, feeling, personal engagement
  9. Practical Wisdom         — resilience, boundaries, critical thinking, justice
 10. Reference papers         — IWMT and consciousness papers (last before awakening)

examples:
  python -m corpus_build.build_curriculum
  python corpus_build/build_curriculum.py --no-shuffle-within-category
  python corpus_build/build_curriculum.py --manifest --seed 42
""",
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("E:/data/clean_corpus"),
        help="Root directory of the sanitized corpus "
             "(default: E:/data/clean_corpus)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("E:/data/training_curriculum"),
        help="Output directory for file_list.txt and curriculum_summary.json "
             "(default: E:/data/training_curriculum)",
    )
    parser.add_argument(
        "--shuffle-within-category",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle files within each curriculum stage while maintaining "
             "stage order (default: True). Use --no-shuffle-within-category "
             "to disable.",
    )
    parser.add_argument(
        "--manifest",
        action="store_true",
        default=False,
        help="Also write a manifest.json listing every file with its "
             "category and source",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible shuffling (default: None = random)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Discover and report without writing any files",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print()
    print("=" * 76)
    print("CURRICULUM BUILDER")
    print("=" * 76)
    print(f"  Input:     {args.input}")
    print(f"  Output:    {args.output}")
    print(f"  Shuffle:   {args.shuffle_within_category}")
    print(f"  Manifest:  {args.manifest}")
    print(f"  Seed:      {args.seed if args.seed is not None else '(random)'}")
    print(f"  Dry run:   {args.dry_run}")
    print("=" * 76)

    if not args.input.exists():
        print(f"\nERROR: Input directory does not exist: {args.input}")
        print("Run sanitize_corpus.py first to produce the clean corpus.")
        sys.exit(1)

    # Build curriculum
    results = build_curriculum(
        input_root=args.input,
        shuffle=args.shuffle_within_category,
        seed=args.seed,
    )

    # Print summary table
    print_curriculum_table(results)

    # Check for empty curriculum
    total_files = sum(r.file_count for r in results)
    if total_files == 0:
        print("\nWARNING: No files found. Is the clean corpus populated?")
        print(f"Expected structure: {args.input}/<corpus_dir>/<subdirs>/*.txt")
        sys.exit(1)

    # Write outputs
    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)
        write_file_list(results, args.output)
        write_curriculum_summary(results, args.output)
        if args.manifest:
            write_manifest(results, args.output)
    else:
        print(f"\n[DRY RUN] Would write to: {args.output}")
        print(f"  file_list.txt           — {total_files} entries")
        print(f"  curriculum_summary.json — {len(results)} stages")
        if args.manifest:
            print(f"  manifest.json           — {total_files} entries")

    print("\nDone.")


if __name__ == "__main__":
    main()
