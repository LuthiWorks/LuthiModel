"""Text sanitization and validation for training corpus.

Catches data poisoning, encoding issues, and quality problems before
text enters the training pipeline. Every file that goes into the corpus
passes through these checks.

Usage:
    from corpus_build.text_sanitizer import sanitize_text, validate_text

    clean = sanitize_text(raw_text)
    report = validate_text(clean)
    if report["passed"]:
        # safe to use
"""

import math
import re
import unicodedata
from collections import Counter


# Characters that should never appear in training text
FORBIDDEN_CHARS = set(
    "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f"
    "\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f"
    "\x7f"  # DEL
    "\ufeff"  # BOM
    "\ufff9\ufffa\ufffb"  # interlinear annotation
    "\u200b\u200c\u200d\u200e\u200f"  # zero-width chars
    "\u202a\u202b\u202c\u202d\u202e"  # bidi overrides
    "\u2060\u2061\u2062\u2063\u2064"  # invisible operators
    "\u2066\u2067\u2068\u2069"  # bidi isolates
    "\ufffe\uffff"  # non-characters
)

# Regex for HTML/XML tags
HTML_TAG_RE = re.compile(r"<[^>]+>")

# Regex for excessive whitespace
MULTI_SPACE_RE = re.compile(r" {3,}")
MULTI_NEWLINE_RE = re.compile(r"\n{4,}")


def sanitize_text(text: str) -> str:
    """Clean text for safe inclusion in the training corpus.

    Steps:
        1. Strip forbidden/invisible Unicode characters
        2. Normalize Unicode to NFC form
        3. Remove any residual HTML/XML tags
        4. Normalize whitespace
        5. Strip leading/trailing whitespace

    Args:
        text: Raw text from any source.

    Returns:
        Sanitized text safe for training.
    """
    # 1. Remove forbidden characters
    text = "".join(ch for ch in text if ch not in FORBIDDEN_CHARS)

    # 2. Remove Unicode control category (Cc) except \t \n \r
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) != "Cc" or ch in "\t\n\r"
    )

    # 3. Normalize to NFC (canonical decomposition + composition)
    text = unicodedata.normalize("NFC", text)

    # 4. Strip HTML/XML tags
    text = HTML_TAG_RE.sub("", text)

    # 5. Normalize whitespace
    text = MULTI_SPACE_RE.sub("  ", text)
    text = MULTI_NEWLINE_RE.sub("\n\n\n", text)

    # 6. Replace tabs with spaces
    text = text.replace("\t", "    ")

    # 7. Strip
    text = text.strip()

    return text


def compute_entropy(text: str) -> float:
    """Compute Shannon entropy of character distribution.

    Normal English prose: ~4.0-4.5 bits/char
    Random garbage: ~7-8 bits/char
    Repetitive spam: ~1-2 bits/char
    """
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def detect_repetition(text: str, window: int = 100) -> float:
    """Detect repetitive text patterns.

    Slides a window across the text and measures how often windows
    repeat. Returns fraction of windows that are duplicates (0.0 = no
    repetition, 1.0 = completely repetitive).
    """
    if len(text) < window * 2:
        return 0.0

    windows = []
    step = window // 2
    for i in range(0, len(text) - window, step):
        windows.append(text[i:i + window])

    if not windows:
        return 0.0

    unique = len(set(windows))
    return 1.0 - (unique / len(windows))


def check_language(text: str, sample_size: int = 2000) -> float:
    """Quick English detection via common word frequency.

    Returns fraction of words that are common English words (0.0-1.0).
    English prose typically scores > 0.4.
    """
    common_english = {
        "the", "and", "of", "to", "a", "in", "is", "it", "that", "was",
        "for", "on", "are", "with", "as", "his", "they", "be", "at", "one",
        "have", "this", "from", "or", "had", "by", "but", "not", "what",
        "all", "were", "we", "when", "your", "can", "said", "there", "each",
        "which", "she", "do", "how", "their", "if", "will", "up", "other",
        "about", "out", "many", "then", "them", "these", "so", "some",
        "her", "would", "make", "him", "into", "has", "two", "more", "no",
        "could", "he", "been", "who", "its", "did", "my", "than", "an",
        "you", "i", "me", "her", "he", "she", "his", "our", "us", "them",
    }

    sample = text[:sample_size].lower()
    words = re.findall(r"[a-z]+", sample)
    if not words:
        return 0.0

    english_count = sum(1 for w in words if w in common_english)
    return english_count / len(words)


def validate_text(text: str) -> dict:
    """Run all validation checks on sanitized text.

    Returns a report dict with:
        - passed: bool — True if all checks pass
        - checks: dict of individual check results
        - warnings: list of warning messages
        - errors: list of error messages (any error = failed)
    """
    report = {
        "passed": True,
        "checks": {},
        "warnings": [],
        "errors": [],
        "stats": {},
    }

    # Length check
    char_count = len(text)
    report["stats"]["char_count"] = char_count
    report["stats"]["word_count"] = len(text.split())

    if char_count < 100:
        report["errors"].append(f"Too short: {char_count} chars (min 100)")
        report["passed"] = False
    elif char_count < 500:
        report["warnings"].append(f"Very short: {char_count} chars")

    # Entropy check
    entropy = compute_entropy(text[:10000])  # Sample first 10K chars
    report["checks"]["entropy"] = entropy
    report["stats"]["entropy"] = entropy

    if entropy < 2.0:
        report["errors"].append(
            f"Entropy too low: {entropy:.2f} bits/char (likely repetitive/spam)"
        )
        report["passed"] = False
    elif entropy > 6.5:
        report["errors"].append(
            f"Entropy too high: {entropy:.2f} bits/char (likely garbage/binary)"
        )
        report["passed"] = False
    elif entropy < 3.0:
        report["warnings"].append(
            f"Low entropy: {entropy:.2f} bits/char (may be repetitive)"
        )

    # Repetition check
    repetition = detect_repetition(text[:50000])
    report["checks"]["repetition"] = repetition
    report["stats"]["repetition"] = repetition

    if repetition > 0.5:
        report["errors"].append(
            f"High repetition: {repetition:.2f} (>50% of text windows repeat)"
        )
        report["passed"] = False
    elif repetition > 0.3:
        report["warnings"].append(
            f"Moderate repetition: {repetition:.2f}"
        )

    # Language check
    english_score = check_language(text)
    report["checks"]["english_score"] = english_score
    report["stats"]["english_score"] = english_score

    if english_score < 0.15:
        report["errors"].append(
            f"Low English score: {english_score:.2f} (may not be English text)"
        )
        report["passed"] = False
    elif english_score < 0.25:
        report["warnings"].append(
            f"Low English score: {english_score:.2f} (unusual for prose)"
        )

    # Remaining HTML check
    html_matches = HTML_TAG_RE.findall(text[:5000])
    if html_matches:
        report["warnings"].append(
            f"Residual HTML tags found: {len(html_matches)} "
            f"(first: {html_matches[0][:50]})"
        )

    # Forbidden character check (should be caught by sanitize, but verify)
    forbidden_found = [ch for ch in text[:10000] if ch in FORBIDDEN_CHARS]
    if forbidden_found:
        report["errors"].append(
            f"Forbidden characters found: {len(forbidden_found)}"
        )
        report["passed"] = False

    return report


def process_file(text: str, source: str = "unknown") -> tuple[str | None, dict]:
    """Full pipeline: sanitize, validate, return clean text or None.

    Args:
        text: Raw text content.
        source: Source identifier for logging.

    Returns:
        (clean_text, report) — clean_text is None if validation failed.
    """
    clean = sanitize_text(text)
    report = validate_text(clean)
    report["source"] = source

    if report["warnings"]:
        for w in report["warnings"]:
            print(f"  WARNING [{source}]: {w}")

    if not report["passed"]:
        for e in report["errors"]:
            print(f"  FAILED [{source}]: {e}")
        return None, report

    return clean, report
