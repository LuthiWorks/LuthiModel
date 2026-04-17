"""Download deep psychology text corpus from Internet Archive.

Uses the Archive.org Advanced Search API to find texts across psychology
specializations, downloads the pre-extracted plain text (_djvu.txt) files,
validates them, and saves to the corpus.

This goes deeper than the general "Psychology" subject in download_academic.py,
covering clinical, cognitive, developmental, social, behavioral, neuroscience,
emotion, learning, and applied psychology specializations.

Usage:
    python -m corpus_build.download_psychology
    python -m corpus_build.download_psychology --dry-run
    python -m corpus_build.download_psychology --subjects "Clinical Psychology" "Neuropsychology"
    python -m corpus_build.download_psychology --per-subject 100
"""

import argparse
import gzip
import json
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Force UTF-8 output on Windows to avoid UnicodeEncodeError on special chars
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from corpus_build.text_sanitizer import process_file

# Archive.org Advanced Search endpoint
SEARCH_URL = "https://archive.org/advancedsearch.php"
DOWNLOAD_BASE = "https://archive.org/download"
METADATA_BASE = "https://archive.org/metadata"

HEADERS = {
    "User-Agent": "LuthiModel-CorpusBuild/1.0 (personal research project)"
}

# Deep psychology specializations for the entity's education.
# Each tuple is (subject_query, display_name, target_count).
# target_count is per-subject — how many items to download.
SUBJECT_CATALOG = [
    # === Foundations ===
    ("Psychology AND textbook", "Psychology Textbooks", 500),
    ("\"History of psychology\"", "History of Psychology", 300),
    ("\"Experimental psychology\"", "Experimental Psychology", 300),

    # === Clinical & Abnormal ===
    ("\"Clinical psychology\"", "Clinical Psychology", 500),
    ("\"Abnormal psychology\"", "Abnormal Psychology", 300),
    ("Psychotherapy", "Psychotherapy", 300),
    ("\"Mental health\"", "Mental Health", 400),
    ("Psychiatric", "Psychiatric", 300),

    # === Cognitive ===
    ("\"Cognitive psychology\"", "Cognitive Psychology", 500),
    ("\"Cognitive science\"", "Cognitive Science", 400),
    ("Perception AND psychology", "Perception Psychology", 200),
    ("Memory AND psychology", "Memory Psychology", 200),
    ("Attention AND cognition", "Attention and Cognition", 200),
    ("\"Decision making\" AND psychology", "Decision Making Psychology", 200),

    # === Developmental ===
    ("\"Developmental psychology\"", "Developmental Psychology", 400),
    ("\"Child psychology\"", "Child Psychology", 300),
    ("\"Adolescent psychology\"", "Adolescent Psychology", 200),

    # === Social & Personality ===
    ("\"Social psychology\"", "Social Psychology", 400),
    ("Personality AND psychology", "Personality Psychology", 300),
    ("\"Group behavior\"", "Group Behavior", 200),
    ("\"Social cognition\"", "Social Cognition", 200),

    # === Behavioral ===
    ("\"Behavioral psychology\"", "Behavioral Psychology", 300),
    ("Behaviorism", "Behaviorism", 200),
    ("Conditioning AND psychology", "Conditioning Psychology", 100),

    # === Neuroscience of Mind ===
    ("Neuropsychology", "Neuropsychology", 400),
    ("\"Cognitive neuroscience\"", "Cognitive Neuroscience", 400),
    ("Brain AND behavior", "Brain and Behavior", 300),
    ("Consciousness AND psychology", "Consciousness Psychology", 300),

    # === Emotion & Motivation ===
    ("Emotion AND psychology", "Emotion Psychology", 300),
    ("Motivation AND psychology", "Motivation Psychology", 200),
    ("Affect AND psychology", "Affect Psychology", 200),

    # === Learning & Intelligence ===
    ("Learning AND psychology", "Learning Psychology", 300),
    ("Intelligence AND psychology", "Intelligence Psychology", 200),
    ("\"Educational psychology\"", "Educational Psychology", 300),

    # === Applied ===
    ("\"Positive psychology\"", "Positive Psychology", 200),
    ("\"Organizational psychology\"", "Organizational Psychology", 200),
    ("\"Forensic psychology\"", "Forensic Psychology", 100),
    ("\"Health psychology\"", "Health Psychology", 200),
]


def search_archive(
    subject: str,
    max_results: int = 500,
    page_size: int = 100,
) -> list[dict]:
    """Search Archive.org for text items matching a subject.

    Prioritizes books over periodicals by sorting by downloads (popular
    items tend to be books/textbooks rather than obscure periodical issues).

    Returns list of dicts with 'identifier' and 'title' keys.
    """
    results = []
    page = 1
    # Cap at API limit of 10,000
    max_results = min(max_results, 10000)

    while len(results) < max_results:
        rows = min(page_size, max_results - len(results))
        params = {
            "q": f"subject:{subject} AND mediatype:texts",
            "fl[]": ["identifier", "title"],
            "sort[]": "downloads desc",
            "rows": rows,
            "page": page,
            "output": "json",
        }
        url = SEARCH_URL + "?" + urllib.parse.urlencode(params, doseq=True)

        try:
            req = urllib.request.Request(url, headers=HEADERS)
            resp = urllib.request.urlopen(req, timeout=60)
            data = json.loads(resp.read())
        except Exception as e:
            print(f"    Search error (page {page}): {e}")
            break

        docs = data.get("response", {}).get("docs", [])
        if not docs:
            break

        results.extend(docs)
        total = data["response"]["numFound"]

        if len(docs) < rows or page * page_size >= total:
            break

        page += 1
        time.sleep(0.3)

    return results[:max_results]


def get_text_file_url(identifier: str) -> str | None:
    """Find the best plain text file for an item.

    Checks for (in order of preference):
        1. _djvu.txt — OCR-extracted plain text
        2. _hocr_searchtext.txt.gz — compressed OCR text
        3. .txt — any plain text file

    Returns download URL or None.
    """
    metadata_url = f"{METADATA_BASE}/{identifier}/files"

    try:
        req = urllib.request.Request(metadata_url, headers=HEADERS)
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())
    except Exception:
        return None

    files = data.get("result", [])
    if not files:
        return None

    # Preference order for text files
    djvu_txt = None
    hocr_gz = None
    plain_txt = None

    for f in files:
        name = f.get("name", "")
        fmt = f.get("format", "")

        if name.endswith("_djvu.txt"):
            djvu_txt = name
        elif name.endswith("_hocr_searchtext.txt.gz"):
            hocr_gz = name
        elif name.endswith(".txt") and "meta" not in name.lower():
            # Avoid metadata files like _meta.txt
            if fmt in ("Text", "DjVuTXT", "") or name.endswith(".txt"):
                plain_txt = name

    best = djvu_txt or hocr_gz or plain_txt
    if best:
        return f"{DOWNLOAD_BASE}/{identifier}/{urllib.parse.quote(best)}"
    return None


def fetch_text(url: str, retries: int = 3) -> str | None:
    """Download text content from a URL. Handles gzipped files."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            resp = urllib.request.urlopen(req, timeout=60)
            data = resp.read()

            if url.endswith(".gz"):
                data = gzip.decompress(data)

            return data.decode("utf-8", errors="replace")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
            else:
                return None


def download_subject(
    subject_query: str,
    display_name: str,
    target_count: int,
    output_dir: Path,
    seen_ids: set,
    dry_run: bool = False,
    delay: float = 0.3,
) -> dict:
    """Download texts for a single subject.

    Args:
        subject_query: Archive.org subject query string.
        display_name: Human-readable subject name.
        target_count: How many items to aim for.
        output_dir: Base output directory (subject subdirectory created).
        seen_ids: Set of already-processed identifiers (dedup across subjects).
        dry_run: If True, just count what's available.
        delay: Seconds between downloads.

    Returns:
        Stats dict for this subject.
    """
    stats = {
        "subject": display_name,
        "searched": 0,
        "downloaded": 0,
        "validated": 0,
        "skipped_dup": 0,
        "skipped_no_text": 0,
        "failed": 0,
        "total_words": 0,
    }

    subject_dir = output_dir / display_name.replace(" ", "_").replace("&", "and")

    print(f"\n  --- {display_name} ---")

    # Search for items
    # Request more than target since some won't have text files
    search_count = min(target_count * 3, 10000)
    items = search_archive(subject_query, max_results=search_count)
    stats["searched"] = len(items)
    print(f"  Found {len(items):,} items, targeting {target_count}")

    if dry_run:
        return stats

    saved = 0
    for item in items:
        if saved >= target_count:
            break

        identifier = item["identifier"]
        title = item.get("title", identifier)

        # Dedup across subjects
        if identifier in seen_ids:
            stats["skipped_dup"] += 1
            continue
        seen_ids.add(identifier)

        # Check if already downloaded
        safe_id = identifier.replace("/", "_").replace("\\", "_")
        txt_path = subject_dir / f"{safe_id}.txt"
        if txt_path.exists():
            saved += 1
            stats["validated"] += 1
            continue

        # Find text file URL
        time.sleep(delay)
        text_url = get_text_file_url(identifier)
        if text_url is None:
            stats["skipped_no_text"] += 1
            continue

        # Download text
        time.sleep(delay)
        raw_text = fetch_text(text_url)
        if raw_text is None:
            stats["failed"] += 1
            continue

        stats["downloaded"] += 1

        # Sanitize and validate
        clean_text, report = process_file(raw_text, source=f"{display_name}/{identifier}")

        if clean_text is None:
            stats["failed"] += 1
            continue

        # Save
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(clean_text, encoding="utf-8")

        word_count = report["stats"]["word_count"]
        stats["validated"] += 1
        stats["total_words"] += word_count
        saved += 1

        if saved % 25 == 0 or saved <= 3:
            print(f"    [{saved}/{target_count}] {title[:60]}... ({word_count:,} words)")

    print(f"  Result: {stats['validated']} saved, {stats['total_words']:,} words")
    return stats


def download_corpus(
    output_dir: Path,
    subjects: list[tuple[str, str, int]] | None = None,
    per_subject_override: int | None = None,
    dry_run: bool = False,
    delay: float = 0.3,
) -> dict:
    """Download the full psychology corpus across all subjects.

    Args:
        output_dir: Base output directory.
        subjects: List of (query, name, count) tuples. Defaults to SUBJECT_CATALOG.
        per_subject_override: If set, override all target counts.
        dry_run: Just report counts, don't download.
        delay: Seconds between API requests.

    Returns:
        Summary dict.
    """
    if subjects is None:
        subjects = SUBJECT_CATALOG

    output_dir.mkdir(parents=True, exist_ok=True)

    total_target = sum(
        (per_subject_override or count) for _, _, count in subjects
    )

    print(f"\n{'='*60}")
    print(f"  Deep Psychology Corpus Download")
    print(f"{'='*60}")
    print(f"  Output: {output_dir}")
    print(f"  Subjects: {len(subjects)}")
    print(f"  Target items: ~{total_target:,}")
    if dry_run:
        print(f"  MODE: DRY RUN")
    print()

    seen_ids: set[str] = set()
    all_stats = []

    for subject_query, display_name, target_count in subjects:
        if per_subject_override:
            target_count = per_subject_override

        stats = download_subject(
            subject_query=subject_query,
            display_name=display_name,
            target_count=target_count,
            output_dir=output_dir,
            seen_ids=seen_ids,
            dry_run=dry_run,
            delay=delay,
        )
        all_stats.append(stats)

    # Summary
    total_validated = sum(s["validated"] for s in all_stats)
    total_words = sum(s["total_words"] for s in all_stats)
    total_downloaded = sum(s["downloaded"] for s in all_stats)
    total_failed = sum(s["failed"] for s in all_stats)
    total_no_text = sum(s["skipped_no_text"] for s in all_stats)
    total_dup = sum(s["skipped_dup"] for s in all_stats)

    print(f"\n{'='*60}")
    print(f"  Deep Psychology Corpus Download Complete")
    print(f"{'='*60}")
    print(f"  Subjects processed:   {len(all_stats)}")
    print(f"  Items downloaded:     {total_downloaded:,}")
    print(f"  Items validated:      {total_validated:,}")
    print(f"  Skipped (duplicate):  {total_dup:,}")
    print(f"  Skipped (no text):    {total_no_text:,}")
    print(f"  Failed:               {total_failed:,}")
    print(f"  Total words:          {total_words:,}")
    print()

    # Per-subject breakdown
    print(f"  Per-subject breakdown:")
    for s in all_stats:
        if s["validated"] > 0 or s["searched"] > 0:
            print(
                f"    {s['subject']:30s} | "
                f"{s['validated']:5d} items | "
                f"{s['total_words']:>12,} words"
            )

    summary = {
        "subjects": all_stats,
        "totals": {
            "validated": total_validated,
            "downloaded": total_downloaded,
            "words": total_words,
            "failed": total_failed,
            "duplicates_skipped": total_dup,
            "no_text_skipped": total_no_text,
        },
    }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Download deep psychology text corpus from Internet Archive"
    )
    parser.add_argument(
        "--output", type=str,
        default="E:/data/psychology_corpus",
        help="Output directory for processed text files",
    )
    parser.add_argument(
        "--per-subject", type=int, default=None,
        help="Override per-subject item count (e.g., 100 for testing)",
    )
    parser.add_argument(
        "--subjects", type=str, nargs="+", default=None,
        help="Only download specific subjects (by display name, e.g., 'Clinical Psychology' 'Neuropsychology')",
    )
    parser.add_argument(
        "--delay", type=float, default=0.3,
        help="Delay between API requests in seconds (default: 0.3)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Just count available items, don't download",
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    # Filter subjects if specified
    subjects = SUBJECT_CATALOG
    if args.subjects:
        filter_names = {s.lower() for s in args.subjects}
        subjects = [
            (q, name, count) for q, name, count in SUBJECT_CATALOG
            if name.lower() in filter_names
        ]
        if not subjects:
            print(f"No matching subjects found. Available:")
            for _, name, _ in SUBJECT_CATALOG:
                print(f"  {name}")
            return

    summary = download_corpus(
        output_dir=output_dir,
        subjects=subjects,
        per_subject_override=args.per_subject,
        dry_run=args.dry_run,
        delay=args.delay,
    )

    # Save summary
    summary_path = output_dir / "download_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
