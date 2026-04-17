"""Download fantasy/fiction collection from Internet Archive.

Fetches EPUBs and PDFs from the FantasyFictionebookcollection, converts
to clean text, validates for data poisoning, and saves to corpus.

Usage:
    python -m corpus_build.download_fantasy
    python -m corpus_build.download_fantasy --output E:/data/fantasy_corpus
    python -m corpus_build.download_fantasy --dry-run
"""

import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
from html.parser import HTMLParser
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from corpus_build.epub_converter import epub_to_text, pdf_to_text
from corpus_build.text_sanitizer import process_file


BASE_URL = "https://archive.org/download/FantasyFictionebookcollection"

# User-agent to be polite to Internet Archive
HEADERS = {
    "User-Agent": "LuthiModel-CorpusBuild/1.0 (personal research project)"
}

# Folders to exclude from download — content not appropriate for the
# entity's foundational education
EXCLUDED_FOLDERS = {
    "Song of Fire and Ice-Game of thrones",  # excessive graphic violence
    "Fifty Shades of Grey",  # not relevant to curriculum
}


class DirectoryParser(HTMLParser):
    """Parse Internet Archive directory listing to find links."""

    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value and not value.startswith("?"):
                    self.links.append(value)


def fetch_url(url: str, retries: int = 3, delay: float = 1.0) -> bytes:
    """Fetch URL content with retries and polite delays."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt < retries - 1:
                wait = delay * (attempt + 1)
                print(f"    Retry {attempt + 1}/{retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def list_directory(url: str) -> list[str]:
    """List files/subdirectories at an Internet Archive URL."""
    try:
        html = fetch_url(url).decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  Error listing {url}: {e}")
        return []

    parser = DirectoryParser()
    parser.feed(html)

    # Filter out parent directory and metadata links
    links = []
    for link in parser.links:
        # Skip parent dir, sort controls, and IA metadata
        if link in ("../", "/") or link.startswith("?") or link.startswith("/"):
            continue
        # Skip IA metadata files
        if link.endswith(("_meta.xml", "_files.xml", "_reviews.xml")):
            continue
        links.append(link)

    return links


def convert_file(file_path: Path) -> str:
    """Convert an EPUB or PDF to text based on extension."""
    ext = file_path.suffix.lower()
    if ext == ".epub":
        return epub_to_text(file_path)
    elif ext == ".pdf":
        return pdf_to_text(file_path)
    else:
        raise ValueError(f"Unsupported format: {ext}")


def download_collection(
    output_dir: Path,
    cache_dir: Path,
    dry_run: bool = False,
    delay: float = 0.5,
) -> dict:
    """Download and process the full fantasy fiction collection.

    Downloads both EPUBs and PDFs. Prefers EPUBs when both are
    available for a title (better text extraction). Falls back to
    PDF when no EPUB exists.

    Args:
        output_dir: Where to save processed .txt files.
        cache_dir: Where to cache downloaded files.
        dry_run: If True, just list what would be downloaded.
        delay: Seconds between requests (be polite to IA).

    Returns:
        Summary dict with counts and any failures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "total_found": 0,
        "downloaded": 0,
        "converted": 0,
        "validated": 0,
        "skipped_existing": 0,
        "skipped_excluded": 0,
        "failed": [],
        "books": [],
    }

    print(f"\n{'='*60}")
    print(f"  Fantasy Fiction Corpus Download")
    print(f"{'='*60}")
    print(f"  Source: {BASE_URL}")
    print(f"  Output: {output_dir}")
    print(f"  Cache:  {cache_dir}")
    if EXCLUDED_FOLDERS:
        print(f"  Excluding: {', '.join(sorted(EXCLUDED_FOLDERS))}")
    if dry_run:
        print(f"  MODE: DRY RUN (no downloads)")
    print()

    # List top-level directories (series/author folders)
    print("Scanning collection...")
    top_links = list_directory(BASE_URL)
    folders = [l for l in top_links if l.endswith("/")]
    print(f"Found {len(folders)} series/author folders\n")

    for folder in folders:
        folder_name = urllib.parse.unquote(folder.rstrip("/"))
        folder_url = f"{BASE_URL}/{folder}"

        # Skip non-folder links (external URLs from IA page)
        if folder_name.startswith("http"):
            continue

        # Check exclusion list
        if folder_name in EXCLUDED_FOLDERS:
            print(f"\n--- {folder_name} --- EXCLUDED")
            summary["skipped_excluded"] += 1
            continue

        print(f"\n--- {folder_name} ---")

        # List files in this folder
        time.sleep(delay)
        files = list_directory(folder_url)

        # Find downloadable book files — prefer EPUB, fall back to PDF
        epub_files = [f for f in files if f.lower().endswith(".epub")]
        pdf_files = [f for f in files if f.lower().endswith(".pdf")]

        # Build download list: use EPUB when available, PDF otherwise
        # Track titles we've already covered via EPUB to avoid duplicates
        download_files = []
        epub_titles = set()

        for f in epub_files:
            download_files.append(f)
            # Extract title stem for dedup
            title = urllib.parse.unquote(f).rsplit(".", 1)[0].lower().strip()
            epub_titles.add(title)

        for f in pdf_files:
            title = urllib.parse.unquote(f).rsplit(".", 1)[0].lower().strip()
            # Only add PDF if we don't already have an EPUB for this title
            if title not in epub_titles:
                download_files.append(f)

        if not download_files:
            print(f"  No books found (no EPUB or PDF), skipping")
            continue

        epub_count = len(epub_files)
        pdf_only_count = len(download_files) - epub_count
        print(f"  {len(download_files)} books ({epub_count} EPUB, {pdf_only_count} PDF-only)")
        summary["total_found"] += len(download_files)

        for file_name in download_files:
            decoded_name = urllib.parse.unquote(file_name)
            safe_name = re.sub(r'[<>:"/\\|?*]', '_', decoded_name)
            txt_name = Path(safe_name).with_suffix(".txt").name
            txt_path = output_dir / folder_name / txt_name
            cached_path = cache_dir / folder_name / safe_name

            # Skip if already processed
            if txt_path.exists():
                summary["skipped_existing"] += 1
                print(f"  SKIP (exists): {decoded_name}")
                continue

            if dry_run:
                print(f"  WOULD DOWNLOAD: {decoded_name}")
                continue

            # Download file
            file_url = f"{folder_url}{file_name}"
            print(f"  Downloading: {decoded_name}...", end=" ", flush=True)

            try:
                time.sleep(delay)
                file_data = fetch_url(file_url)
                cached_path.parent.mkdir(parents=True, exist_ok=True)
                cached_path.write_bytes(file_data)
                size_mb = len(file_data) / (1024 * 1024)
                print(f"({size_mb:.1f} MB)")
                summary["downloaded"] += 1
            except Exception as e:
                print(f"FAILED: {e}")
                summary["failed"].append({"file": decoded_name, "error": str(e)})
                continue

            # Convert to text
            try:
                raw_text = convert_file(cached_path)
                summary["converted"] += 1
            except Exception as e:
                print(f"  CONVERT FAILED: {decoded_name}: {e}")
                summary["failed"].append(
                    {"file": decoded_name, "error": f"Convert: {e}"}
                )
                continue

            # Sanitize and validate
            clean_text, report = process_file(raw_text, source=decoded_name)

            if clean_text is None:
                summary["failed"].append(
                    {"file": decoded_name, "error": f"Validation: {report['errors']}"}
                )
                continue

            # Save clean text
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            txt_path.write_text(clean_text, encoding="utf-8")

            word_count = report["stats"]["word_count"]
            fmt = cached_path.suffix.upper().lstrip(".")
            print(f"  SAVED: {txt_name} ({word_count:,} words) [{fmt}]")
            summary["validated"] += 1
            summary["books"].append({
                "title": decoded_name,
                "series": folder_name,
                "format": fmt,
                "words": word_count,
                "entropy": report["stats"]["entropy"],
                "english_score": report["stats"]["english_score"],
            })

    # Summary
    print(f"\n{'='*60}")
    print(f"  Download Complete")
    print(f"{'='*60}")
    print(f"  Total books found:  {summary['total_found']}")
    print(f"  Downloaded:         {summary['downloaded']}")
    print(f"  Converted:          {summary['converted']}")
    print(f"  Validated & saved:  {summary['validated']}")
    print(f"  Skipped (existing): {summary['skipped_existing']}")
    print(f"  Skipped (excluded): {summary['skipped_excluded']}")
    print(f"  Failed:             {len(summary['failed'])}")

    if summary["books"]:
        total_words = sum(b["words"] for b in summary["books"])
        print(f"  Total words:        {total_words:,}")

    if summary["failed"]:
        print(f"\n  Failures:")
        for f in summary["failed"]:
            print(f"    {f['file']}: {f['error']}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Download fantasy fiction corpus from Internet Archive"
    )
    parser.add_argument(
        "--output", type=str,
        default="E:/data/fantasy_corpus",
        help="Output directory for processed text files",
    )
    parser.add_argument(
        "--cache", type=str,
        default="E:/data/fantasy_cache",
        help="Cache directory for downloaded EPUBs/PDFs",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay between requests in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List what would be downloaded without downloading",
    )

    args = parser.parse_args()
    output_dir = Path(args.output)
    cache = Path(args.cache)

    summary = download_collection(
        output_dir=output_dir,
        cache_dir=cache,
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
