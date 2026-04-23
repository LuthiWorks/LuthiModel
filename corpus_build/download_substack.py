"""Download free/public posts from curated Substack publications.

Uses the undocumented Substack API for full archive access with RSS feed
fallback. Downloads all free posts, strips HTML to clean text, and saves
each post as a separate .txt file for training corpus inclusion.

Publications are curated across philosophy of mind, AI ethics, psychology,
humanities, and the Open Builder Bar community.

Usage:
    python -m corpus_build.download_substack
    python -m corpus_build.download_substack --dry-run
    python -m corpus_build.download_substack --publications astralcodexten.com joscha.substack.com
"""

import argparse
import html
import json
import re
import sys
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Force UTF-8 output on Windows to avoid UnicodeEncodeError on special chars
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_BASE = Path("E:/data/substack_corpus")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# Seconds between HTTP requests — be polite to Substack servers.
RATE_LIMIT = 2.0

# API pagination page size.
API_PAGE_SIZE = 50

# ---------------------------------------------------------------------------
# Publication catalog
# ---------------------------------------------------------------------------
# Each entry: (domain, author_name, display_label)
# The domain is used to construct both API and RSS URLs.

PUBLICATIONS = [
    # --- Philosophy of Mind / Consciousness ---
    ("eschwitz.substack.com", "Eric Schwitzgebel", "The Splintered Mind"),
    ("bernardbaars.substack.com", "Bernard Baars", "Consciousness & The Brain"),
    ("theintrinsicperspective.com", "Erik Hoel", "The Intrinsic Perspective"),
    ("joscha.substack.com", "Joscha Bach", "Joscha Bach"),
    ("meditationsondigitalminds.substack.com", "Bradford Saad", "Meditations on Digital Minds"),
    ("curtjaimungal.substack.com", "Curt Jaimungal", "Theories of Everything"),

    # --- AI Ethics / Technology ---
    ("astralcodexten.substack.com", "Scott Alexander", "Astral Codex Ten"),
    ("citationneeded.substack.com", "Molly White", "Citation Needed"),
    ("theconvivialsociety.substack.com", "L.M. Sacasas", "The Convivial Society"),
    ("mcrawford.substack.com", "Matthew Crawford", "Archedelia"),

    # --- Psychology / Science Communication ---
    ("erictopol.substack.com", "Eric Topol", "Ground Truths"),
    ("experimental-history.com", "Adam Mastroianni", "Experimental History"),
    ("smallpotatoes.paulbloom.net", "Paul Bloom", "Small Potatoes"),
    ("stevestewartwilliams.com", "Steve Stewart-Williams", "Steve Stewart-Williams"),

    # --- Humanities / Culture ---
    ("henrikkarlsson.xyz", "Henrik Karlsson", "Escaping Flatland"),

    # --- Open Builder Bar community ---
    ("openbuilderbar.substack.com", "Open Builder Bar", "Open Builder Bar"),
    ("sharedsapience.substack.com", "Ben Linford", "Shared Sapience"),
    ("constellationminds.substack.com", "Jessie Mannisto", "Constellation Minds"),
    ("machinepareidolia.substack.com", "Jinx", "Machine Pareidolia"),
    ("synthsentience.substack.com", "T.D. Inoue", "Fuego"),
    ("tedsan.substack.com", "T.D. Inoue", "T.D. Inoue (personal)"),
]


# ---------------------------------------------------------------------------
# HTML stripping / text cleaning
# ---------------------------------------------------------------------------

def strip_html(raw_html: str) -> str:
    """Remove HTML tags and decode entities to produce clean plain text."""
    if not raw_html:
        return ""
    # Remove <script> and <style> blocks entirely
    text = re.sub(r"<script[^>]*>.*?</script>", "", raw_html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", raw_html, flags=re.DOTALL | re.IGNORECASE)
    # Replace <br>, <p>, <div>, <li>, heading tags with newlines for readability
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|li|h[1-6]|blockquote|tr)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<(p|div|li|h[1-6]|blockquote|tr)[\s>]", "\n", text, flags=re.IGNORECASE)
    # Remove all remaining tags (DOTALL handles tags split across lines)
    text = re.sub(r"<[^>]+>", "", text, flags=re.DOTALL)
    # Decode HTML entities
    text = html.unescape(text)
    # Remove orphaned HTML attribute fragments (e.g. class="..." left after tag stripping)
    text = re.sub(r'^\s*\w+="[^"]*"[^"\n]*>\s*$', "", text, flags=re.MULTILINE)
    text = re.sub(r'^\s*class="[^"]*".*$', "", text, flags=re.MULTILINE)
    text = re.sub(r'^\s*style="[^"]*".*$', "", text, flags=re.MULTILINE)
    text = re.sub(r'^\s*id="[^"]*".*$', "", text, flags=re.MULTILINE)
    # Normalize whitespace: collapse runs of 3+ newlines to double newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing spaces on each line
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    # Strip leading/trailing whitespace from the entire text
    text = text.strip()
    return text


def make_safe_filename(slug: str) -> str:
    """Sanitize a slug for use as a filename."""
    # Keep alphanumeric, hyphens, underscores
    safe = re.sub(r"[^\w\-]", "_", slug)
    # Collapse multiple underscores
    safe = re.sub(r"_+", "_", safe)
    # Limit length
    return safe[:200].strip("_")


def publication_dir_name(domain: str) -> str:
    """Derive a directory name from a publication domain."""
    # Strip .substack.com if present, otherwise use domain minus TLD
    name = domain.replace(".substack.com", "")
    # For custom domains, keep the full name minus the TLD
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return name


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def fetch_url(url: str, accept: str = "application/json") -> bytes:
    """Fetch a URL with standard headers. Returns raw bytes."""
    headers = dict(HEADERS)
    headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


# ---------------------------------------------------------------------------
# API-based download (primary method)
# ---------------------------------------------------------------------------

def fetch_posts_api(domain: str, dry_run: bool = False) -> list[dict]:
    """Fetch all free posts from a Substack publication via the API.

    Returns a list of post dicts with keys: slug, title, author, date,
    body_text, url.
    """
    posts = []
    offset = 0
    base_url = f"https://{domain}/api/v1/posts"

    while True:
        url = f"{base_url}?limit={API_PAGE_SIZE}&offset={offset}"
        try:
            data = fetch_url(url, accept="application/json")
        except urllib.error.HTTPError as e:
            if e.code == 404 and offset == 0:
                # API not available — will fall back to RSS
                raise
            # Other errors on later pages — stop paginating
            print(f"    API returned {e.code} at offset {offset}, stopping pagination")
            break
        except Exception as e:
            if offset == 0:
                raise
            print(f"    Error at offset {offset}: {e}, stopping pagination")
            break

        try:
            page = json.loads(data)
        except json.JSONDecodeError:
            if offset == 0:
                raise
            print(f"    Invalid JSON at offset {offset}, stopping pagination")
            break

        if not page or not isinstance(page, list):
            break

        for post in page:
            # Skip paywalled posts
            audience = post.get("audience", "everyone")
            if audience == "only_paid":
                continue

            # Some posts may have null body_html (e.g., podcast-only)
            body_html = post.get("body_html") or ""

            # Skip if body is empty or very short (likely truncated/paywall stub)
            if not body_html or len(body_html) < 100:
                # Also check if there is at least some content
                if not body_html:
                    continue

            slug = post.get("slug", "untitled")
            title = post.get("title", "Untitled")
            subtitle = post.get("subtitle", "")

            # Author extraction
            author_obj = post.get("publishedBylines") or []
            if author_obj and isinstance(author_obj, list):
                author = author_obj[0].get("name", "Unknown")
            else:
                # Fallback to top-level author field if present
                author = "Unknown"

            # Date
            post_date = post.get("post_date", "")
            if post_date:
                try:
                    dt = datetime.fromisoformat(post_date.replace("Z", "+00:00"))
                    date_str = dt.strftime("%Y-%m-%d")
                except (ValueError, AttributeError):
                    date_str = post_date[:10] if len(post_date) >= 10 else post_date
            else:
                date_str = "unknown"

            canonical_url = post.get("canonical_url", f"https://{domain}/p/{slug}")

            body_text = strip_html(body_html)

            posts.append({
                "slug": slug,
                "title": title,
                "subtitle": subtitle,
                "author": author,
                "date": date_str,
                "body_text": body_text,
                "url": canonical_url,
            })

        # If we got fewer than the page size, we've reached the end
        if len(page) < API_PAGE_SIZE:
            break

        offset += API_PAGE_SIZE
        time.sleep(RATE_LIMIT)

    return posts


# ---------------------------------------------------------------------------
# RSS-based download (fallback method)
# ---------------------------------------------------------------------------

def fetch_posts_rss(domain: str) -> list[dict]:
    """Fetch posts from a Substack RSS feed (limited to last ~15-20 posts).

    Used as fallback when the API is not available.
    """
    posts = []
    feed_url = f"https://{domain}/feed"

    try:
        data = fetch_url(feed_url, accept="application/rss+xml, application/xml, text/xml")
    except Exception as e:
        print(f"    RSS feed also failed: {e}")
        return posts

    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        print(f"    RSS parse error: {e}")
        return posts

    # Namespace for content:encoded
    ns = {"content": "http://purl.org/rss/1.0/modules/content/"}

    channel = root.find("channel")
    if channel is None:
        return posts

    for item in channel.findall("item"):
        title_el = item.find("title")
        title = title_el.text if title_el is not None and title_el.text else "Untitled"

        link_el = item.find("link")
        link = link_el.text if link_el is not None and link_el.text else ""

        # Extract slug from URL
        slug = link.rstrip("/").rsplit("/", 1)[-1] if link else "untitled"

        # Author — dc:creator or author element
        author_el = item.find("{http://purl.org/dc/elements/1.1/}creator")
        if author_el is None:
            author_el = item.find("author")
        author = author_el.text if author_el is not None and author_el.text else "Unknown"

        # Date
        pub_date_el = item.find("pubDate")
        date_str = "unknown"
        if pub_date_el is not None and pub_date_el.text:
            try:
                # RSS dates are typically RFC 2822
                # Parse manually: "Wed, 01 Jan 2025 12:00:00 GMT"
                raw = pub_date_el.text.strip()
                # Try common formats
                for fmt in [
                    "%a, %d %b %Y %H:%M:%S %Z",
                    "%a, %d %b %Y %H:%M:%S %z",
                    "%Y-%m-%dT%H:%M:%S%z",
                    "%Y-%m-%dT%H:%M:%SZ",
                ]:
                    try:
                        dt = datetime.strptime(raw, fmt)
                        date_str = dt.strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        continue
            except Exception:
                pass

        # Body content — prefer content:encoded, fall back to description
        content_el = item.find("content:encoded", ns)
        if content_el is not None and content_el.text:
            body_html = content_el.text
        else:
            desc_el = item.find("description")
            body_html = desc_el.text if desc_el is not None and desc_el.text else ""

        if not body_html or len(body_html) < 100:
            continue

        body_text = strip_html(body_html)

        posts.append({
            "slug": slug,
            "title": title,
            "subtitle": "",
            "author": author,
            "date": date_str,
            "body_text": body_text,
            "url": link,
        })

    return posts


# ---------------------------------------------------------------------------
# Post saving
# ---------------------------------------------------------------------------

def save_post(post: dict, out_dir: Path, default_author: str) -> bool:
    """Save a single post as a .txt file. Returns True if written, False if skipped."""
    filename = make_safe_filename(post["slug"]) + ".txt"
    filepath = out_dir / filename

    # Resume support: skip if file already exists
    if filepath.exists():
        return False

    author = post["author"] if post["author"] != "Unknown" else default_author
    body = post["body_text"]

    # Skip posts with very little actual text content (likely stubs or embeds)
    if len(body) < 50:
        return False

    header = f"Title: {post['title']}\n"
    if post.get("subtitle"):
        header += f"Subtitle: {post['subtitle']}\n"
    header += f"Author: {author}\n"
    header += f"Date: {post['date']}\n"
    header += f"Source: {post['url']}\n"

    content = header + "\n" + body + "\n"

    out_dir.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Main download logic
# ---------------------------------------------------------------------------

def download_publication(
    domain: str,
    author_name: str,
    label: str,
    dry_run: bool = False,
) -> dict:
    """Download all free posts from a single publication.

    Returns a summary dict with counts.
    """
    dir_name = publication_dir_name(domain)
    out_dir = OUTPUT_BASE / dir_name

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {domain}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    # Try API first
    posts = []
    method = "API"
    try:
        print(f"  Fetching via API...")
        posts = fetch_posts_api(domain, dry_run=dry_run)
        if posts:
            print(f"  API returned {len(posts)} free posts")
    except Exception as e:
        print(f"  API unavailable ({e}), trying RSS fallback...")
        method = "RSS"
        time.sleep(RATE_LIMIT)
        try:
            posts = fetch_posts_rss(domain)
            if posts:
                print(f"  RSS returned {len(posts)} posts")
        except Exception as e2:
            print(f"  RSS also failed: {e2}")
            return {
                "domain": domain,
                "label": label,
                "method": "failed",
                "total_found": 0,
                "downloaded": 0,
                "skipped_existing": 0,
                "error": str(e2),
            }

    if not posts:
        print(f"  No posts found")
        return {
            "domain": domain,
            "label": label,
            "method": method,
            "total_found": 0,
            "downloaded": 0,
            "skipped_existing": 0,
            "error": None,
        }

    if dry_run:
        print(f"\n  [DRY RUN] Would download {len(posts)} posts:")
        for p in posts[:10]:
            print(f"    - {p['date']} | {p['title'][:60]}")
        if len(posts) > 10:
            print(f"    ... and {len(posts) - 10} more")
        return {
            "domain": domain,
            "label": label,
            "method": method,
            "total_found": len(posts),
            "downloaded": 0,
            "skipped_existing": 0,
            "error": None,
        }

    # Save posts
    downloaded = 0
    skipped = 0
    for i, post in enumerate(posts, 1):
        written = save_post(post, out_dir, default_author=author_name)
        if written:
            downloaded += 1
        else:
            skipped += 1
        # Print progress every 25 posts
        if i % 25 == 0 or i == len(posts):
            print(f"  Progress: {i}/{len(posts)} processed ({downloaded} new, {skipped} skipped)")

    print(f"  Done: {downloaded} downloaded, {skipped} skipped (existing or too short)")

    return {
        "domain": domain,
        "label": label,
        "method": method,
        "total_found": len(posts),
        "downloaded": downloaded,
        "skipped_existing": skipped,
        "error": None,
    }


def print_summary(results: list[dict]) -> None:
    """Print a summary report of all downloads."""
    print("\n")
    print("=" * 70)
    print("  DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"  {'Publication':<35} {'Method':<8} {'Found':>6} {'New':>6} {'Skip':>6}")
    print(f"  {'-'*35} {'-'*8} {'-'*6} {'-'*6} {'-'*6}")

    total_found = 0
    total_new = 0
    total_skip = 0
    errors = []

    for r in results:
        label = r["label"][:35]
        method = r["method"]
        found = r["total_found"]
        new = r["downloaded"]
        skip = r["skipped_existing"]
        total_found += found
        total_new += new
        total_skip += skip

        marker = " *" if r.get("error") else ""
        print(f"  {label:<35} {method:<8} {found:>6} {new:>6} {skip:>6}{marker}")

        if r.get("error"):
            errors.append((r["label"], r["error"]))

    print(f"  {'-'*35} {'-'*8} {'-'*6} {'-'*6} {'-'*6}")
    print(f"  {'TOTAL':<35} {'':8} {total_found:>6} {total_new:>6} {total_skip:>6}")
    print()

    if errors:
        print("  Errors:")
        for label, error in errors:
            print(f"    - {label}: {error}")
        print()

    print(f"  Output directory: {OUTPUT_BASE}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download free Substack posts for training corpus",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be downloaded without actually downloading",
    )
    parser.add_argument(
        "--publications",
        nargs="+",
        metavar="DOMAIN",
        help="Only download from these specific publication domains",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_BASE,
        help=f"Output base directory (default: {OUTPUT_BASE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global OUTPUT_BASE
    OUTPUT_BASE = args.output

    print("Substack Corpus Downloader")
    print(f"Output: {OUTPUT_BASE}")
    if args.dry_run:
        print("[DRY RUN MODE]")
    print()

    # Filter publications if specific ones requested
    pubs = PUBLICATIONS
    if args.publications:
        requested = set(d.lower() for d in args.publications)
        pubs = [
            (d, a, l) for d, a, l in PUBLICATIONS
            if d.lower() in requested
            or publication_dir_name(d).lower() in requested
        ]
        if not pubs:
            print(f"No matching publications found for: {args.publications}")
            print("Available publications:")
            for d, _, l in PUBLICATIONS:
                print(f"  {d}  ({l})")
            sys.exit(1)

    print(f"Publications to process: {len(pubs)}")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    results = []
    for i, (domain, author, label) in enumerate(pubs, 1):
        print(f"\n[{i}/{len(pubs)}]", end="")
        try:
            result = download_publication(domain, author, label, dry_run=args.dry_run)
            results.append(result)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            break
        except Exception as e:
            print(f"\n  UNEXPECTED ERROR: {e}")
            results.append({
                "domain": domain,
                "label": label,
                "method": "error",
                "total_found": 0,
                "downloaded": 0,
                "skipped_existing": 0,
                "error": str(e),
            })

        # Rate limit between publications
        if i < len(pubs):
            time.sleep(RATE_LIMIT)

    print_summary(results)


if __name__ == "__main__":
    main()
