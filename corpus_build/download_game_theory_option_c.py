"""Download Wikipedia + Stanford Encyclopedia of Philosophy supplements
for the cooperative game theory canon.

Option C from the 2026-05-19 corpus audit: when the canonical books
(Axelrod, Maynard Smith, Schelling, de Waal, Trivers) couldn't be
obtained from Archive.org (lending DRM) or shadow libraries
(aggregator availability issues), fall back to encyclopedia-quality
secondary material covering the same concepts.

What this provides:
  - Lower fidelity than the original books, but rigorous and curated
  - Conceptual coverage of all the missing canonical works
  - Plus broader coverage (game theory itself, prisoner's dilemma,
    Nash equilibrium, evolutionary stable strategies) that the canon
    only partially addressed

Output: academic_corpus/Game_Theory_and_Cooperation/

Usage:
    python -m corpus_build.download_game_theory_option_c
"""

import argparse
import sys
import urllib.request
import urllib.parse
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from corpus_build.text_sanitizer import process_file

HEADERS = {
    "User-Agent": "LuthiModel-CorpusBuild/1.0 (personal research project; "
                  "wikipedia-and-sep supplement for cooperative game theory canon)"
}

# Wikipedia articles to fetch. Each entry: (title, attribution_note).
# Titles are passed to the Wikipedia API; redirects are followed.
# Articles chosen to cover:
#   (1) The canonical books we couldn't get (Axelrod, Maynard Smith,
#       Schelling, de Waal, Trivers's reciprocal altruism paper)
#   (2) The foundational concepts (game theory, prisoner's dilemma,
#       Nash equilibrium, ESS, tit-for-tat, Schelling point)
#   (3) Adjacent and applied topics (cooperation evolution, altruism,
#       kin selection, public goods game)
WIKIPEDIA_ARTICLES = [
    # --- The canonical books we couldn't get ---
    ("The Evolution of Cooperation",
     "Axelrod's 1984 book — the foundational empirical study of cooperation "
     "emerging from iterated prisoner's dilemma. Tit-for-tat as a robust "
     "strategy."),
    ("Evolutionary game theory",
     "The theoretical framework developed by Maynard Smith (Evolution and "
     "the Theory of Games, 1982) — game theory applied to natural selection."),
    ("Evolutionarily stable strategy",
     "Maynard Smith's core concept — the biological grounding for why "
     "cooperative behavior persists in evolved systems."),
    ("The Strategy of Conflict",
     "Schelling's 1960 book on game theory applied to negotiation, "
     "commitment, focal points, and credible threats. Direct intellectual "
     "ancestor of the costly-signal framework."),
    ("Chimpanzee Politics",
     "De Waal's 1982 observational study of primate cooperation, "
     "coalitions, deception, and reconciliation — biological ground for "
     "game theory's predictions."),
    ("Reciprocal altruism",
     "Trivers's 1971 paper concept — explains how genuine altruistic "
     "behavior can evolve in non-kin under repeated interaction."),

    # --- Foundational concepts ---
    ("Game theory",
     "The broader mathematical field. Background for everything else."),
    ("Prisoner's dilemma",
     "The canonical cooperation-vs-defection game. The matrix and the "
     "dilemma every other concept in this corpus references."),
    ("Iterated prisoner's dilemma",
     "Repeated-game version. Where tit-for-tat actually beats defection."),
    ("Nash equilibrium",
     "The fundamental solution concept of non-cooperative game theory."),
    ("Tit for tat",
     "The strategy Axelrod's tournaments identified as robust. Conceptually "
     "central to the corpus."),
    ("Focal point (game theory)",
     "Schelling point — focal points emerge from shared context, not "
     "explicit communication."),

    # --- Cooperation and altruism in biology ---
    ("Cooperation",
     "General biological and social-systems treatment."),
    ("Altruism (biology)",
     "Biological altruism — the puzzle that reciprocal altruism and kin "
     "selection solve."),
    ("Kin selection",
     "Hamilton's 1964 framework — explains altruism toward relatives via "
     "shared genes."),
    ("Inclusive fitness",
     "Hamilton's rule (rB > C). The mathematical framework for understanding "
     "when altruism evolves."),

    # --- Commons and collective action ---
    ("Tragedy of the commons",
     "Hardin's framing. Pairs with Ostrom's response (which is already in "
     "the corpus as the actual book)."),
    ("Elinor Ostrom",
     "Biographical + intellectual overview. Context for Governing the Commons."),
    ("Commons",
     "The broader institutional context Ostrom studied."),
    ("Public goods game",
     "Generalization of the cooperation problem beyond two players."),

    # --- Costly signaling (relevant to Brian's framing) ---
    ("Signalling theory",
     "Honest-signal theory — overlaps with Schelling's commitment work and "
     "with the costly-signal trust framework Brian invoked from his Gemini "
     "conversation."),
    ("Handicap principle",
     "Zahavi's biological version of costly signaling. Signals are credible "
     "because they're expensive."),
]


def fetch_wikipedia_article(title: str, max_retries: int = 3) -> str | None:
    """Fetch a Wikipedia article as plain text via the official API."""
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
    }
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            resp = urllib.request.urlopen(req, timeout=30)
            data = json.loads(resp.read())
            pages = data.get("query", {}).get("pages", {})
            for _, p in pages.items():
                extract = p.get("extract", "")
                if extract and len(extract) > 500:
                    actual_title = p.get("title", title)
                    return extract, actual_title
            return None, None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1.0 * (attempt + 1))
            else:
                print(f"    ERROR after {max_retries} retries: {e}")
                return None, None


def make_wikipedia_header(title: str, actual_title: str, note: str) -> str:
    """Create attribution header for Wikipedia content."""
    lines = [
        "=" * 72,
        "ATTRIBUTION",
        "=" * 72,
        f"Title:    {actual_title}",
        f"Source:   Wikipedia (en.wikipedia.org)",
        f"Note:     {note}",
        "License:  CC BY-SA 4.0",
        "",
        "=" * 72,
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Download Wikipedia supplements for game theory canon."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("E:/data/clean_corpus/academic_corpus/Game_Theory_and_Cooperation"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay", type=float, default=0.4,
                        help="Seconds between Wikipedia API calls (be polite)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Cooperative Game Theory Canon — Option C (Wikipedia supplements)")
    print("=" * 72)
    print(f"  Output: {output_dir}")
    print(f"  Articles: {len(WIKIPEDIA_ARTICLES)}")
    print(f"  Dry-run: {args.dry_run}")
    print()

    downloaded = 0
    skipped = 0
    failed = 0
    total_words = 0

    for i, (title, note) in enumerate(WIKIPEDIA_ARTICLES, start=1):
        print(f"[{i:2d}/{len(WIKIPEDIA_ARTICLES)}] {title}")
        safe_title = (
            title.replace("/", "-").replace("\\", "-")
            .replace(":", " -").replace("?", "").replace('"', "").strip()
        )
        filename = f"wikipedia - {safe_title}.txt"
        path = output_dir / filename

        if path.exists():
            print("    already exists, skipping")
            skipped += 1
            continue

        if args.dry_run:
            print("    [dry-run] would fetch")
            continue

        result = fetch_wikipedia_article(title)
        if not result or result[0] is None:
            print("    fetch failed or article too short")
            failed += 1
            continue

        extract, actual_title = result
        header = make_wikipedia_header(title, actual_title, note)
        full_text = header + extract

        # Sanitize through the standard pipeline
        clean_text, report = process_file(full_text, source=f"wikipedia/{title}")
        if clean_text is None:
            print("    failed sanitization")
            failed += 1
            continue

        path.write_text(clean_text, encoding="utf-8")
        word_count = report["stats"]["word_count"]
        print(f"    OK: {word_count:,} words")
        downloaded += 1
        total_words += word_count

        time.sleep(args.delay)

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total words added: {total_words:,}")


if __name__ == "__main__":
    main()
