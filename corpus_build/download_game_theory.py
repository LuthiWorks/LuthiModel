"""Download canonical cooperative game theory works from Internet Archive.

Targeted, hand-curated catalog of foundational texts on cooperation,
reciprocity, and game theory. Companion to download_wisdom.py — same
pattern, narrower theme. Imports the helper functions
(search_archive, get_text_file_url, fetch_text, make_attribution_header,
download_specific_work) from download_wisdom.py to avoid duplication.

Added 2026-05-19 in response to a corpus audit (see
docs/research/2026-05-19_corpus-audit-gemini-suggestions.md) that
found Gemini's "Proportional Reciprocity Frameworks" suggestion was
only thinly covered by passing mentions in economics textbooks. This
script adds the canonical works that anchor the theme.

Output: academic_corpus/Game_Theory_and_Cooperation/

Usage:
    python -m corpus_build.download_game_theory
    python -m corpus_build.download_game_theory --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
import urllib.request
import urllib.parse
from corpus_build.download_wisdom import (
    HEADERS,
    DOWNLOAD_BASE,
    METADATA_BASE,
    make_attribution_header,
    fetch_text,
)
from corpus_build.text_sanitizer import process_file


# ============================================================================
# CURATED CATALOG — canonical cooperative game theory works.
#
# Theme directory chosen as "Game_Theory_and_Cooperation" (single subdir;
# wisdom-script's "theme_dir" field reused for parent dir under
# academic_corpus). All works share the same theme_dir so they cluster
# coherently in the academic_corpus structure.
# ============================================================================

# Direct-identifier approach: each entry specifies the verified
# archive.org identifier and the corresponding _djvu.txt filename.
# This avoids the search-noise problem where title-only queries
# returned unrelated DTIC reports and short derivative papers
# instead of the actual books.
#
# All identifiers verified 2026-05-19 to exist with _djvu.txt files
# via the archive.org metadata API.

SPECIFIC_WORKS = [
    {
        "identifier": "evolutionofcoop000axel",
        "txt_file": "evolutionofcoop000axel_djvu.txt",
        "title": "The Evolution of Cooperation",
        "author": "Robert Axelrod (b. 1943)",
        "year": "1984",
        "theme_dir": "Game_Theory_and_Cooperation",
        "note": (
            "Empirically demonstrated that 'tit-for-tat' — start cooperative, "
            "then mirror your opponent — outperforms more sophisticated "
            "strategies in iterated prisoner's dilemma tournaments. The "
            "foundational study of how cooperation emerges and stabilizes "
            "from purely self-interested agents."
        ),
    },
    {
        "identifier": "evolutiontheoryo0000mayn",
        "txt_file": "evolutiontheoryo0000mayn_djvu.txt",
        "title": "Evolution and the Theory of Games",
        "author": "John Maynard Smith (1920-2004)",
        "year": "1982",
        "theme_dir": "Game_Theory_and_Cooperation",
        "note": (
            "Introduced the Evolutionary Stable Strategy (ESS) — the "
            "biological grounding for why cooperative behavior persists. "
            "Game theory applied to natural selection. Shows that "
            "cooperation isn't a moral choice but a stable equilibrium "
            "that biology converges on."
        ),
    },
    {
        "identifier": "governingthecommons",
        "txt_file": "Governing the Commons_djvu.txt",
        "title": "Governing the Commons",
        "author": "Elinor Ostrom (1933-2012)",
        "year": "1990",
        "theme_dir": "Game_Theory_and_Cooperation",
        "note": (
            "Nobel Prize-winning empirical response to Hardin's 'Tragedy of "
            "the Commons.' Studied real communities managing shared "
            "resources — fisheries, forests, irrigation systems — and "
            "documented the design principles that let cooperation "
            "self-organize without top-down control."
        ),
    },
    {
        "identifier": "strategyofconfli00sche",
        "txt_file": "strategyofconfli00sche_djvu.txt",
        "title": "The Strategy of Conflict",
        "author": "Thomas Schelling (1921-2016)",
        "year": "1960",
        "theme_dir": "Game_Theory_and_Cooperation",
        "note": (
            "Game theory applied to strategy, bargaining, and negotiation. "
            "Introduced focal points (Schelling points) and the role of "
            "commitment, credible threats, and costly signaling in "
            "establishing trust between rational actors. The intellectual "
            "ancestor of the costly-signal framework for building trust "
            "across asymmetric power."
        ),
    },
    {
        "identifier": "chimpanzeepoliti00waal",
        "txt_file": "chimpanzeepoliti00waal_djvu.txt",
        "title": "Chimpanzee Politics: Power and Sex Among Apes",
        "author": "Frans de Waal (1948-2024)",
        "year": "1982",
        "theme_dir": "Game_Theory_and_Cooperation",
        "note": (
            "Years of close observation of a chimpanzee colony showing "
            "that primates engage in coalitions, reciprocal favors, "
            "deception, and reconciliation — the empirical biological "
            "ground for cooperative game theory's predictions. The "
            "theoretical structures Maynard Smith mathematizes are "
            "visible here in primate social life."
        ),
    },
    {
        "identifier": "hardin-garret-the-tragedy-of-the-commons",
        "txt_file": "HARDIN, Garret - The Tragedy of the Commons_djvu.txt",
        "title": "The Tragedy of the Commons",
        "author": "Garrett Hardin (1915-2003)",
        "year": "1968",
        "theme_dir": "Game_Theory_and_Cooperation",
        "note": (
            "Short, foundational essay (originally in Science) on how "
            "individually rational decisions can produce collectively "
            "ruinous outcomes when shared resources are involved. The "
            "problem statement that Ostrom's 'Governing the Commons' "
            "later showed could be solved by community self-organization. "
            "Read this first, then Ostrom."
        ),
    },
    {
        # Trivers' "Evolution of Reciprocal Altruism" (1971) is included
        # as part of this collected-papers volume rather than as the
        # original Quarterly Review of Biology journal article (which
        # isn't on archive.org as a standalone). The volume contains
        # the 1971 paper plus other Trivers works.
        "identifier": "naturalselection0000triv",
        "txt_file": "naturalselection0000triv_djvu.txt",
        "title": "Natural Selection and Social Theory: Selected Papers of Robert Trivers (includes 'The Evolution of Reciprocal Altruism', 1971)",
        "author": "Robert Trivers (b. 1943)",
        "year": "2002 (collection); 1971 (original paper)",
        "theme_dir": "Game_Theory_and_Cooperation",
        "note": (
            "Trivers' collected papers, including the 1971 Quarterly Review "
            "of Biology paper 'The Evolution of Reciprocal Altruism' — the "
            "biological mechanism behind tit-for-tat's success in human "
            "and animal cooperation. Reciprocal altruism explains how "
            "genuine altruistic behavior can evolve in non-kin under "
            "conditions of repeated interaction."
        ),
    },
]


def download_by_identifier(
    work: dict,
    output_dir,
    dry_run: bool = False,
    delay: float = 0.5,
) -> dict:
    """Direct-identifier download. Skips search; goes straight to the
    known _djvu.txt URL for the work. Same sanitization + attribution
    pipeline as download_specific_work, but with no search-result
    ambiguity.
    """
    import time as _time
    stats = {
        "title": work["title"],
        "author": work["author"],
        "identifier": work["identifier"],
        "status": "pending",
        "words": 0,
    }

    theme_dir = output_dir / work["theme_dir"]
    safe_title = (
        work["title"][:80]
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", " -")
        .replace("?", "")
        .replace('"', "")
        .strip()
    )
    existing = list(theme_dir.glob(f"{safe_title}*"))
    if existing:
        stats["status"] = "exists"
        return stats

    if dry_run:
        stats["status"] = "dry_run"
        return stats

    txt_url = f"{DOWNLOAD_BASE}/{work['identifier']}/{urllib.parse.quote(work['txt_file'])}"
    print(f"  Fetching: {work['identifier']}")
    raw_text = fetch_text(txt_url)
    _time.sleep(delay)

    if raw_text is None or len(raw_text.strip()) < 1000:
        stats["status"] = "fetch_failed_or_too_short"
        return stats

    header = make_attribution_header(work, archive_id=work["identifier"])
    full_text = header + raw_text
    clean_text, report = process_file(full_text, source=f"game_theory/{work['title']}")

    if clean_text is None:
        stats["status"] = "failed_validation"
        return stats

    theme_dir.mkdir(parents=True, exist_ok=True)
    txt_path = theme_dir / f"{safe_title}.txt"
    txt_path.write_text(clean_text, encoding="utf-8")

    word_count = report["stats"]["word_count"]
    stats["status"] = "downloaded"
    stats["words"] = word_count
    print(f"    OK: {word_count:,} words")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download canonical cooperative game theory works."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("E:/data/clean_corpus/academic_corpus"),
        help="Output directory. Defaults to E:/data/clean_corpus/academic_corpus "
        "so works land in the science_philosophy curriculum stage.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds between archive.org requests (rate-limit polite).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Cooperative Game Theory Canon Download")
    print("=" * 72)
    print(f"  Output: {output_dir}")
    print(f"  Works to fetch: {len(SPECIFIC_WORKS)}")
    print(f"  Dry-run: {args.dry_run}")
    print()

    results = []
    for i, work in enumerate(SPECIFIC_WORKS, start=1):
        print(f"[{i}/{len(SPECIFIC_WORKS)}] {work['title']}")
        stats = download_by_identifier(
            work=work,
            output_dir=output_dir,
            dry_run=args.dry_run,
            delay=args.delay,
        )
        results.append(stats)

    # Summary
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    statuses: dict[str, int] = {}
    total_words = 0
    for r in results:
        status = r.get("status", "unknown")
        statuses[status] = statuses.get(status, 0) + 1
        total_words += r.get("words", 0)

    for status, count in sorted(statuses.items()):
        print(f"  {status:20s} {count}")
    print()
    print(f"  Total words downloaded: {total_words:,}")

    # Note any not-found items so Brian can decide how to proceed
    not_found = [r for r in results if r.get("status") in
                 ("fetch_failed_or_too_short", "failed_validation")]
    if not_found:
        print()
        print("Items that need follow-up:")
        for r in not_found:
            print(f"  - {r['title']} ({r['status']})")


if __name__ == "__main__":
    main()
