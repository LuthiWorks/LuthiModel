"""EPUB and PDF to plain text converter.

EPUBs are ZIP archives containing XHTML content files. PDFs are
extracted using pypdf. Both produce clean plain text suitable for
the training corpus.

Usage:
    from corpus_build.epub_converter import epub_to_text, pdf_to_text

    text = epub_to_text("path/to/book.epub")
    text = pdf_to_text("path/to/book.pdf")
"""

import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET


class HTMLTextExtractor(HTMLParser):
    """Extract visible text from HTML, preserving paragraph structure."""

    # Tags that should insert line breaks
    BLOCK_TAGS = {
        "p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "tr", "blockquote", "pre", "section", "article",
        "header", "footer", "figcaption",
    }

    # Tags whose content should be skipped entirely
    SKIP_TAGS = {"script", "style", "svg", "math"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        elif tag in self.BLOCK_TAGS:
            self.parts.append("\n")
        elif tag == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self.parts.append(data)

    def get_text(self) -> str:
        return "".join(self.parts)


def html_to_text(html: str) -> str:
    """Convert HTML to plain text, preserving paragraph structure."""
    extractor = HTMLTextExtractor()
    extractor.feed(html)
    text = extractor.get_text()

    # Clean up whitespace
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            # Collapse internal whitespace
            line = re.sub(r"\s+", " ", line)
            lines.append(line)

    return "\n\n".join(lines)


def _get_spine_order(epub_zip: zipfile.ZipFile) -> list[str]:
    """Parse the OPF file to get content files in reading order.

    The EPUB spine defines the order chapters should be read in.
    """
    # Find the OPF file (rootfile in container.xml)
    try:
        container = epub_zip.read("META-INF/container.xml").decode("utf-8")
        root = ET.fromstring(container)
        # Namespace handling
        ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
        rootfile = root.find(".//c:rootfile", ns)
        if rootfile is None:
            # Try without namespace
            rootfile = root.find(".//{*}rootfile")
        if rootfile is None:
            return []
        opf_path = rootfile.get("full-path", "")
    except (KeyError, ET.ParseError):
        # Fall back to finding .opf file
        opf_files = [n for n in epub_zip.namelist() if n.endswith(".opf")]
        if not opf_files:
            return []
        opf_path = opf_files[0]

    # Parse OPF
    try:
        opf_content = epub_zip.read(opf_path).decode("utf-8")
        opf_root = ET.fromstring(opf_content)
    except (KeyError, ET.ParseError):
        return []

    # Get the base directory of the OPF file
    opf_dir = "/".join(opf_path.split("/")[:-1])

    # Build manifest: id -> href
    manifest = {}
    for item in opf_root.iter():
        if item.tag.endswith("}item") or item.tag == "item":
            item_id = item.get("id", "")
            href = item.get("href", "")
            media_type = item.get("media-type", "")
            if "html" in media_type or "xml" in media_type:
                # Resolve relative path
                if opf_dir:
                    full_path = f"{opf_dir}/{href}"
                else:
                    full_path = href
                manifest[item_id] = full_path

    # Get spine order
    spine_ids = []
    for itemref in opf_root.iter():
        if itemref.tag.endswith("}itemref") or itemref.tag == "itemref":
            idref = itemref.get("idref", "")
            if idref in manifest:
                spine_ids.append(manifest[idref])

    return spine_ids


def epub_to_text(epub_path: str | Path) -> str:
    """Extract plain text from an EPUB file in reading order.

    Args:
        epub_path: Path to the .epub file.

    Returns:
        Full text content of the book.

    Raises:
        ValueError: If the file is not a valid EPUB.
    """
    epub_path = Path(epub_path)

    if not epub_path.exists():
        raise FileNotFoundError(f"EPUB not found: {epub_path}")

    if not zipfile.is_zipfile(epub_path):
        raise ValueError(f"Not a valid ZIP/EPUB file: {epub_path}")

    with zipfile.ZipFile(epub_path, "r") as z:
        # Get content files in spine order
        spine_files = _get_spine_order(z)

        if not spine_files:
            # Fallback: read all HTML/XHTML files in sorted order
            spine_files = sorted(
                n for n in z.namelist()
                if n.endswith((".html", ".xhtml", ".htm"))
                and "nav" not in n.lower()
                and "toc" not in n.lower()
            )

        if not spine_files:
            raise ValueError(f"No content files found in EPUB: {epub_path}")

        # Extract text from each content file
        chapters = []
        for filepath in spine_files:
            try:
                raw = z.read(filepath)
                # Try UTF-8 first, fall back to latin-1
                try:
                    html = raw.decode("utf-8")
                except UnicodeDecodeError:
                    html = raw.decode("latin-1")

                text = html_to_text(html)
                if text.strip():
                    chapters.append(text)
            except KeyError:
                # File listed in spine but not in ZIP
                continue

    # Join chapters with clear separation
    full_text = "\n\n\n".join(chapters)

    # Final cleanup
    full_text = re.sub(r"\n{4,}", "\n\n\n", full_text)

    return full_text


def pdf_to_text(pdf_path: str | Path) -> str:
    """Extract plain text from a PDF file.

    Args:
        pdf_path: Path to the .pdf file.

    Returns:
        Full text content of the PDF.

    Raises:
        ValueError: If the file cannot be read.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("PDF support requires pypdf. Install with: pip install pypdf")

    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            pages.append(text.strip())

    full_text = "\n\n\n".join(pages)

    # Clean up common PDF artifacts
    # Fix hyphenated line breaks (word- \n continuation)
    full_text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", full_text)
    # Collapse excessive whitespace
    full_text = re.sub(r"\n{4,}", "\n\n\n", full_text)

    return full_text


def convert_epub_file(
    epub_path: str | Path,
    output_dir: str | Path,
) -> Path | None:
    """Convert a single EPUB to a .txt file in the output directory.

    Args:
        epub_path: Path to .epub file.
        output_dir: Directory to write .txt file.

    Returns:
        Path to output .txt file, or None if conversion failed.
    """
    epub_path = Path(epub_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / epub_path.with_suffix(".txt").name

    try:
        text = epub_to_text(epub_path)
        if not text.strip():
            print(f"  SKIP (empty): {epub_path.name}")
            return None

        output_path.write_text(text, encoding="utf-8")
        word_count = len(text.split())
        print(f"  OK: {epub_path.name} -> {output_path.name} ({word_count:,} words)")
        return output_path

    except Exception as e:
        print(f"  ERROR: {epub_path.name}: {e}")
        return None
