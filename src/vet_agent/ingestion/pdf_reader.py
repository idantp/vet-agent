import re
from pathlib import Path

from pypdf import PdfReader

# Only de-hyphenate when the continuation starts with a lowercase letter — this joins
# soft-wrapped words ("metroni-\ndazole") while preserving numeric dose ranges that
# wrap at a hyphen (e.g. "(8.1-\n25 lb)" must NOT become "(8.125 lb)").
_HYPHEN_WRAP_RE = re.compile(r"-\n([a-z])")
_MULTISPACE_RE = re.compile(r"[ \t]+")
_MULTINEWLINE_RE = re.compile(r"\n{3,}")


def clean_page_text(raw: str) -> str:
    """Normalize extracted page text while preserving line structure.

    De-hyphenates wrapped words and collapses runs of spaces and excess blank
    lines, but keeps single newlines intact so the sectionizer and species-header
    parser can detect line-based headers (e.g. ``Uses/Indications``, ``DOGS:``).
    """
    if not raw:
        return ""
    text = _HYPHEN_WRAP_RE.sub(r"\1", raw)
    text = _MULTISPACE_RE.sub(" ", text)
    # Strip each line BEFORE collapsing blank lines, so whitespace-only lines
    # (reduced to a single space above) don't defeat the blank-line cap.
    lines = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = _MULTINEWLINE_RE.sub("\n\n", text)
    return text.strip()


_LEADING_NUM_RE = re.compile(r"^(\d+)\b")
_TRAILING_NUM_RE = re.compile(r"\b(\d+)$")


def detect_book_page(page_text: str) -> int | None:
    """Parse the printed book-page number from a monograph page's running header.

    Headers alternate by side: even (left) pages read ``"874 Metronidazole"`` (number
    first) and odd (right) pages read ``"Metronidazole 873"`` (number last). Returns the
    integer page number, or None for pages with no header number (front matter, etc.).
    """
    first = next((ln.strip() for ln in page_text.split("\n") if ln.strip()), "")
    m = _LEADING_NUM_RE.match(first) or _TRAILING_NUM_RE.search(first)
    return int(m.group(1)) if m else None


def extract_pages(pdf_path: Path) -> list[str]:
    """Return cleaned text for every page of the PDF (index 0 == first page)."""
    reader = PdfReader(pdf_path)
    return [clean_page_text(page.extract_text() or "") for page in reader.pages]
