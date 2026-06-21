import re

from pydantic import BaseModel

from vet_agent.ingestion.models import TocEntry
from vet_agent.ingestion.pdf_reader import detect_book_page

# How many pages past a drug's anchor page we still accept a heading match within.
# A monograph starts on its TOC book page, so its title is on the anchor page; the
# window allows a little slack for running-header detection noise.
_HEADING_WINDOW_PAGES = 2


class MonographBlock(BaseModel):
    drug_name: str
    book_page: int
    body: str


class SegmentationResult(BaseModel):
    blocks: list[MonographBlock]
    missing: list[TocEntry]


def _heading_index(text: str, drug_name: str, start: int, end: int) -> int | None:
    """Earliest offset of a heading line for *drug_name* within ``text[start:end]``.

    Matches the three real heading-line formats: bare (``Acarbose``), page-number
    prefix (``90 Acarbose``, even pages) and suffix (``Acarbose 90``, odd pages).
    """
    escaped = re.escape(drug_name)
    patterns = (
        re.compile(rf"^{escaped}\s*$", re.MULTILINE),
        re.compile(rf"^\d+\s+{escaped}\s*$", re.MULTILINE),
        re.compile(rf"^{escaped}\s+\d+\s*$", re.MULTILINE),
    )
    candidates = [m.start() for p in patterns if (m := p.search(text, start, end))]
    return min(candidates) if candidates else None


def segment_monographs(pages: list[str], toc: list[TocEntry]) -> SegmentationResult:
    """Slice the book into per-drug blocks, anchoring each drug at its TOC book page.

    Each page's printed book-page number is read from its running header, giving a
    ``book_page -> page-index`` map. For each TOC drug we search for its heading line
    near its own book-page anchor (not a shared forward cursor), so a missing or false
    match for one drug cannot cascade into the next; same-page drugs separate naturally
    because each search targets that drug's own name. Drugs whose book page is absent
    from the text are returned in `missing` (errors-as-values).
    """
    body = "\n".join(pages)
    page_offsets: list[int] = []
    acc = 0
    for page in pages:
        page_offsets.append(acc)
        acc += len(page) + 1  # +1 for the "\n" join separator

    book_to_index: dict[int, int] = {}
    for i, page in enumerate(pages):
        book_page = detect_book_page(page)
        if book_page is not None and book_page not in book_to_index:
            book_to_index[book_page] = i

    located: list[tuple[TocEntry, int]] = []
    missing: list[TocEntry] = []
    for entry in toc:
        idx = book_to_index.get(entry.book_page)
        if idx is None:
            missing.append(entry)
            continue
        anchor = page_offsets[idx]
        window_idx = min(idx + _HEADING_WINDOW_PAGES, len(pages) - 1)
        window_end = page_offsets[window_idx] + len(pages[window_idx])
        pos = _heading_index(body, entry.drug_name, anchor, window_end)
        # If the title line isn't cleanly matched near the anchor, fall back to the
        # page start — the monograph still begins on this page.
        located.append((entry, pos if pos is not None else anchor))

    located.sort(key=lambda pair: pair[1])
    blocks: list[MonographBlock] = []
    for i, (entry, start) in enumerate(located):
        end = located[i + 1][1] if i + 1 < len(located) else len(body)
        block_body = body[start:end].strip()
        blocks.append(
            MonographBlock(drug_name=entry.drug_name, book_page=entry.book_page, body=block_body)
        )
    return SegmentationResult(blocks=blocks, missing=missing)
