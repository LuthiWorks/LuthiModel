"""Batch corpus sanitization script.

Walks all .txt files across corpus directories, runs them through
text_sanitizer.process_file(), writes clean versions to a staging
directory, and reports per-corpus and overall statistics.

Usage:
    python -m corpus_build.sanitize_corpus
    python corpus_build/sanitize_corpus.py
    python corpus_build/sanitize_corpus.py --dry-run
    python corpus_build/sanitize_corpus.py --force --skip-academic
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

# Handle both invocation styles:
#   python -m corpus_build.sanitize_corpus  (package import)
#   python corpus_build/sanitize_corpus.py  (direct script)
try:
    from corpus_build.text_sanitizer import process_file
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from corpus_build.text_sanitizer import process_file


# ---------------------------------------------------------------------------
# Default corpus directories
# ---------------------------------------------------------------------------

DEFAULT_CORPORA = {
    "mythology_corpus": Path("E:/data/mythology_corpus"),
    "history_corpus": Path("E:/data/history_corpus"),
    "psychology_corpus": Path("E:/data/psychology_corpus"),
    "classics_corpus": Path("E:/data/classics_corpus"),
    "fantasy_corpus": Path("E:/data/fantasy_corpus"),
    "academic_corpus": Path("E:/data/academic_corpus"),
    "substack_corpus": Path("E:/data/substack_corpus"),
}

# Corpora that bypass text sanitization (code files would fail English checks)
# These are copied directly to clean_corpus with no modification.
PASSTHROUGH_CORPORA = {
    "code_corpus": Path("E:/data/code_corpus"),
}

DEFAULT_OUTPUT = Path("E:/data/clean_corpus")

# Encodings to try when reading files, in order of preference
ENCODINGS_TO_TRY = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]


# ---------------------------------------------------------------------------
# Statistics tracking
# ---------------------------------------------------------------------------

@dataclass
class CorpusStats:
    """Per-corpus statistics accumulator."""
    name: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    warnings: int = 0
    errors_list: list = field(default_factory=list)
    encoding_errors: int = 0
    total_chars_in: int = 0
    total_chars_out: int = 0
    total_words_out: int = 0
    elapsed: float = 0.0


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------

def read_with_fallback(filepath: Path) -> str | None:
    """Try multiple encodings to read a text file.

    Returns the decoded text on success, or None if every encoding fails.
    """
    for enc in ENCODINGS_TO_TRY:
        try:
            return filepath.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
        except OSError as e:
            print(f"  OS ERROR [{filepath}]: {e}")
            return None
    return None


def output_path_for(source_file: Path, corpus_root: Path,
                    corpus_name: str, output_dir: Path) -> Path:
    """Compute the output path that mirrors the source directory structure.

    E.g. E:/data/fantasy_corpus/tolkien/hobbit.txt
      -> E:/data/clean_corpus/fantasy_corpus/tolkien/hobbit.txt
    """
    rel = source_file.relative_to(corpus_root)
    return output_dir / corpus_name / rel


def process_single_file(
    filepath: Path,
    corpus_root: Path,
    corpus_name: str,
    output_dir: Path,
    force: bool,
    dry_run: bool,
) -> dict:
    """Process one file: read, sanitize, write, return result dict."""
    result = {
        "file": str(filepath),
        "corpus": corpus_name,
        "status": "unknown",
        "warnings": 0,
        "errors": [],
        "chars_in": 0,
        "chars_out": 0,
        "words_out": 0,
        "encoding_error": False,
    }

    out_path = output_path_for(filepath, corpus_root, corpus_name, output_dir)

    # Skip if already processed (unless --force)
    if not force and out_path.exists():
        result["status"] = "skipped"
        return result

    # Read source file
    raw_text = read_with_fallback(filepath)
    if raw_text is None:
        result["status"] = "encoding_error"
        result["encoding_error"] = True
        result["errors"] = [f"Could not decode {filepath} with any known encoding"]
        return result

    result["chars_in"] = len(raw_text)

    # Sanitize + validate
    source_label = f"{corpus_name}/{filepath.relative_to(corpus_root)}"
    clean_text, report = process_file(raw_text, source=source_label)

    result["warnings"] = len(report.get("warnings", []))
    result["errors"] = report.get("errors", [])

    if clean_text is None:
        result["status"] = "failed"
        return result

    result["chars_out"] = len(clean_text)
    result["words_out"] = report.get("stats", {}).get("word_count", 0)
    result["status"] = "passed"

    # Write output (unless --dry-run)
    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(clean_text, encoding="utf-8")

    return result


# ---------------------------------------------------------------------------
# Corpus-level processing
# ---------------------------------------------------------------------------

def discover_txt_files(corpus_dir: Path) -> list[Path]:
    """Recursively find all .txt files under a directory."""
    if not corpus_dir.exists():
        print(f"  WARNING: Directory does not exist: {corpus_dir}")
        return []
    return sorted(corpus_dir.rglob("*.txt"))


def process_corpus(
    corpus_name: str,
    corpus_dir: Path,
    output_dir: Path,
    force: bool,
    dry_run: bool,
    max_workers: int,
) -> CorpusStats:
    """Process all .txt files in a single corpus directory."""
    stats = CorpusStats(name=corpus_name)
    files = discover_txt_files(corpus_dir)
    stats.total = len(files)

    if not files:
        print(f"  No .txt files found in {corpus_dir}")
        return stats

    t0 = time.time()

    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for fp in files:
            fut = executor.submit(
                process_single_file,
                fp, corpus_dir, corpus_name, output_dir, force, dry_run,
            )
            futures[fut] = fp

        for fut in as_completed(futures):
            result = fut.result()
            if result["status"] == "passed":
                stats.passed += 1
                stats.total_chars_in += result["chars_in"]
                stats.total_chars_out += result["chars_out"]
                stats.total_words_out += result["words_out"]
                stats.warnings += result["warnings"]
            elif result["status"] == "failed":
                stats.failed += 1
                stats.errors_list.extend(result["errors"])
                stats.warnings += result["warnings"]
            elif result["status"] == "skipped":
                stats.skipped += 1
            elif result["status"] == "encoding_error":
                stats.encoding_errors += 1
                stats.errors_list.extend(result["errors"])

    stats.elapsed = time.time() - t0
    return stats


# ---------------------------------------------------------------------------
# Summary / reporting
# ---------------------------------------------------------------------------

def print_corpus_summary(stats: CorpusStats) -> None:
    """Print a formatted summary line for one corpus."""
    processed = stats.passed + stats.failed
    total_label = f"{stats.total} files"
    parts = []
    if stats.passed:
        parts.append(f"{stats.passed} passed")
    if stats.failed:
        parts.append(f"{stats.failed} failed")
    if stats.skipped:
        parts.append(f"{stats.skipped} skipped")
    if stats.encoding_errors:
        parts.append(f"{stats.encoding_errors} encoding errors")
    if stats.warnings:
        parts.append(f"{stats.warnings} warnings")

    status_str = ", ".join(parts) if parts else "nothing to process"
    time_str = f"{stats.elapsed:.1f}s" if stats.elapsed > 0 else "-"

    print(f"  {stats.name:<25} {total_label:<14} {status_str}  [{time_str}]")

    if stats.total_words_out > 0:
        mb = stats.total_chars_out / (1024 * 1024)
        print(f"  {'':25} {'':<14} "
              f"{stats.total_words_out:,} words, {mb:.1f} MB clean text")


def print_overall_summary(all_stats: list[CorpusStats]) -> None:
    """Print the grand total across all corpora."""
    total = sum(s.total for s in all_stats)
    passed = sum(s.passed for s in all_stats)
    failed = sum(s.failed for s in all_stats)
    skipped = sum(s.skipped for s in all_stats)
    enc_err = sum(s.encoding_errors for s in all_stats)
    warnings = sum(s.warnings for s in all_stats)
    chars_out = sum(s.total_chars_out for s in all_stats)
    words_out = sum(s.total_words_out for s in all_stats)
    elapsed = sum(s.elapsed for s in all_stats)

    print()
    print("=" * 72)
    print("OVERALL TOTALS")
    print("-" * 72)
    print(f"  Files found:      {total:>8,}")
    print(f"  Passed:           {passed:>8,}")
    print(f"  Failed:           {failed:>8,}")
    print(f"  Skipped (clean):  {skipped:>8,}")
    print(f"  Encoding errors:  {enc_err:>8,}")
    print(f"  Warnings:         {warnings:>8,}")
    print(f"  Clean output:     {words_out:>8,} words "
          f"({chars_out / (1024 * 1024):.1f} MB)")
    print(f"  Time:             {elapsed:>8.1f}s")
    print("=" * 72)


def build_json_report(all_stats: list[CorpusStats], args) -> dict:
    """Build a JSON-serializable report dict."""
    corpora = {}
    for s in all_stats:
        corpora[s.name] = {
            "total_files": s.total,
            "passed": s.passed,
            "failed": s.failed,
            "skipped": s.skipped,
            "encoding_errors": s.encoding_errors,
            "warnings": s.warnings,
            "errors": s.errors_list,
            "chars_in": s.total_chars_in,
            "chars_out": s.total_chars_out,
            "words_out": s.total_words_out,
            "elapsed_seconds": round(s.elapsed, 2),
        }

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "output_dir": str(args.output),
        "force": args.force,
        "dry_run": args.dry_run,
        "skip_academic": args.skip_academic,
        "workers": args.workers,
        "corpora": corpora,
        "totals": {
            "files": sum(s.total for s in all_stats),
            "passed": sum(s.passed for s in all_stats),
            "failed": sum(s.failed for s in all_stats),
            "skipped": sum(s.skipped for s in all_stats),
            "encoding_errors": sum(s.encoding_errors for s in all_stats),
            "warnings": sum(s.warnings for s in all_stats),
            "chars_out": sum(s.total_chars_out for s in all_stats),
            "words_out": sum(s.total_words_out for s in all_stats),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-sanitize text corpora for Luthi Model training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python -m corpus_build.sanitize_corpus
  python -m corpus_build.sanitize_corpus --dry-run
  python -m corpus_build.sanitize_corpus --force --skip-academic
  python -m corpus_build.sanitize_corpus --output E:/data/staging
""",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Root output directory for clean files (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-process files even if a clean version already exists",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Report what would happen without writing any files",
    )
    parser.add_argument(
        "--skip-academic",
        action="store_true",
        help="Exclude the academic corpus (already sanitized during download)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=8,
        help="Number of parallel threads (default: 8)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Build the list of corpora to process
    corpora = dict(DEFAULT_CORPORA)
    if args.skip_academic:
        corpora.pop("academic_corpus", None)

    print()
    print("=" * 72)
    print("CORPUS SANITIZATION")
    print("=" * 72)
    print(f"  Output:       {args.output}")
    print(f"  Force:        {args.force}")
    print(f"  Dry run:      {args.dry_run}")
    print(f"  Workers:      {args.workers}")
    print(f"  Corpora:      {len(corpora)}")
    if args.skip_academic:
        print(f"  (academic corpus skipped)")
    print("=" * 72)
    print()

    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)

    all_stats: list[CorpusStats] = []

    for corpus_name, corpus_dir in corpora.items():
        print(f"Processing: {corpus_name} ({corpus_dir})")

        stats = process_corpus(
            corpus_name=corpus_name,
            corpus_dir=corpus_dir,
            output_dir=args.output,
            force=args.force,
            dry_run=args.dry_run,
            max_workers=args.workers,
        )
        all_stats.append(stats)
        print_corpus_summary(stats)
        print()

    # Passthrough corpora (code, etc.) — copy directly without sanitization
    for corpus_name, corpus_dir in PASSTHROUGH_CORPORA.items():
        print(f"Passthrough: {corpus_name} ({corpus_dir})")
        stats = CorpusStats(name=corpus_name)
        if not corpus_dir.exists():
            print(f"  WARNING: Directory does not exist: {corpus_dir}")
            all_stats.append(stats)
            continue

        files = discover_txt_files(corpus_dir)
        # Also discover source code files by common extensions
        for ext in ("*.py", "*.rs", "*.go", "*.c", "*.h", "*.js", "*.mjs",
                    "*.ts", "*.tsx", "*.jsx", "*.md", "*.rst"):
            files.extend(sorted(corpus_dir.rglob(ext)))
        # Deduplicate (txt files may overlap with other patterns)
        files = sorted(set(files))
        stats.total = len(files)

        t0 = time.time()
        for fp in files:
            out_path = args.output / corpus_name / fp.relative_to(corpus_dir)
            if not args.force and out_path.exists():
                stats.skipped += 1
                continue
            try:
                content = fp.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                try:
                    content = fp.read_text(encoding="latin-1")
                except OSError:
                    stats.encoding_errors += 1
                    continue
            stats.passed += 1
            stats.total_chars_in += len(content)
            stats.total_chars_out += len(content)
            if not args.dry_run:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(content, encoding="utf-8")

        stats.elapsed = time.time() - t0
        all_stats.append(stats)
        print_corpus_summary(stats)
        print()

    # Overall summary
    print_overall_summary(all_stats)

    # Write JSON report
    report = build_json_report(all_stats, args)
    report_path = args.output / "sanitization_report.json"

    if not args.dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nReport written to: {report_path}")
    else:
        print(f"\n[DRY RUN] Report would be written to: {report_path}")

    # Exit code: 0 if no failures, 1 if any files failed
    total_failed = sum(s.failed + s.encoding_errors for s in all_stats)
    if total_failed > 0:
        print(f"\n{total_failed} file(s) failed sanitization.")
        sys.exit(1)


if __name__ == "__main__":
    main()
