"""Download filtered English non-religious texts from Project Gutenberg.

Reads SPGC metadata, filters for English text-type books excluding
religious content, downloads raw text, strips Gutenberg boilerplate,
and saves clean text files.

Resumable: skips already-downloaded files on restart.
"""

import urllib.request
import time
import os
import sys
import pandas as pd

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "gutenberg")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "metadata.csv")
LOG_PATH = os.path.join(os.path.dirname(__file__), "download_log.txt")

RELIGIOUS_KEYWORDS = [
    "bible", "religion", "church", "theology", "christian", "sermon",
    "prayer", "gospel", "scripture", "worship", "catholic", "protestant",
    "baptist", "methodist", "presbyterian", "episcopal", "mormon",
    "devotional", "hymn", "psalm", "liturgy", "religious",
]

# Gutenberg exists to distribute this data freely — no need to crawl slowly
DELAY_SECONDS = 0.05


def strip_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header and footer boilerplate."""
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "***START OF THE PROJECT GUTENBERG EBOOK",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "***END OF THE PROJECT GUTENBERG EBOOK",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]

    # Find start (skip past the marker line)
    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            newline = text.find("\n", idx)
            if newline != -1:
                start_idx = newline + 1
            break

    # Find end
    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    return text[start_idx:end_idx].strip()


def log(msg: str) -> None:
    """Print and append to log file."""
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main():
    os.makedirs(CORPUS_DIR, exist_ok=True)

    # Load and filter metadata
    log("Loading metadata...")
    df = pd.read_csv(METADATA_PATH)
    en = df[df["language"].str.contains("'en'", na=False)]
    en_text = en[en["type"] == "Text"]

    subjects = en_text["subjects"].fillna("").str.lower()
    religious = subjects.apply(
        lambda s: any(k in s for k in RELIGIOUS_KEYWORDS)
    )
    books = en_text[~religious].copy()
    log(f"Target: {len(books):,} English non-religious texts")

    # Count already downloaded
    existing = set()
    for f in os.listdir(CORPUS_DIR):
        if f.startswith("PG") and f.endswith(".txt"):
            existing.add(f)

    log(f"Already downloaded: {len(existing):,}")
    log(f"Remaining: {len(books) - len(existing):,}")
    log("")

    downloaded = 0
    failed = 0
    skipped = len(existing)
    total = len(books)
    t_start = time.time()

    for i, (_, row) in enumerate(books.iterrows()):
        raw_id = row["id"]
        # Metadata IDs have "PG" prefix (e.g., "PG1342") — strip for URL
        numeric_id = str(raw_id).replace("PG", "")
        filename = f"PG{numeric_id}.txt"
        outpath = os.path.join(CORPUS_DIR, filename)

        if filename in existing:
            continue

        url = f"https://www.gutenberg.org/cache/epub/{numeric_id}/pg{numeric_id}.txt"
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "LuthiModel/1.0 (AI research corpus build)"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()

            # Try UTF-8 first, fall back to latin-1
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")

            clean = strip_boilerplate(text)

            if len(clean) > 1000:
                with open(outpath, "w", encoding="utf-8") as f:
                    f.write(clean)
                downloaded += 1
            else:
                failed += 1

        except Exception as e:
            failed += 1

        done = downloaded + failed + skipped
        if (downloaded + failed) % 200 == 0 and (downloaded + failed) > 0:
            elapsed = time.time() - t_start
            rate = (downloaded + failed) / elapsed
            remaining = (total - done) / rate if rate > 0 else 0
            log(
                f"  [{done:,}/{total:,}] "
                f"downloaded={downloaded:,} failed={failed:,} "
                f"rate={rate:.1f}/s ETA={remaining/3600:.1f}h"
            )

        time.sleep(DELAY_SECONDS)

    elapsed = time.time() - t_start
    log("")
    log(f"=== COMPLETE ===")
    log(f"Downloaded: {downloaded:,}")
    log(f"Failed: {failed:,}")
    log(f"Skipped (existing): {skipped:,}")
    log(f"Time: {elapsed/3600:.1f} hours")

    # Report total corpus size
    total_bytes = sum(
        os.path.getsize(os.path.join(CORPUS_DIR, f))
        for f in os.listdir(CORPUS_DIR)
        if f.endswith(".txt")
    )
    log(f"Total corpus size: {total_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
