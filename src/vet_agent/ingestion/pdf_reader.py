import re
from pathlib import Path

from pypdf import PdfReader

_HYPHEN_WRAP_RE = re.compile(r"-\n")
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
    text = _HYPHEN_WRAP_RE.sub("", raw)
    text = _MULTISPACE_RE.sub(" ", text)
    text = _MULTINEWLINE_RE.sub("\n\n", text)
    lines = [ln.strip() for ln in text.split("\n")]
    return "\n".join(lines).strip()


def extract_pages(pdf_path: Path) -> list[str]:
    """Return cleaned text for every page of the PDF (index 0 == first page)."""
    reader = PdfReader(str(pdf_path))
    return [clean_page_text(page.extract_text() or "") for page in reader.pages]
