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


def _flexible_name_pattern(drug_name: str) -> str:
    """Regex for a drug name tolerant of whitespace/hyphen differences.

    The TOC and the body sometimes disagree on spacing around hyphens (e.g. the TOC
    reads ``L- Theanine`` while the body reads ``L-Theanine``). Splitting on any run of
    spaces/hyphens and rejoining with ``[\\s-]+`` matches all such variants, so a
    boundary drug whose name has this quirk is still located (and doesn't fall back to
    its page-top anchor, which would truncate the preceding drug's tail).
    """
    parts = [re.escape(p) for p in re.split(r"[\s-]+", drug_name.strip()) if p]
    return r"[\s-]+".join(parts)


def _heading_index(text: str, drug_name: str, start: int, end: int) -> int | None:
    """Offset of the monograph-title line for *drug_name* within ``text[start:end]``.

    The true title is the *bare* name on its own line (``Acetazolamide``); the
    ``Name 9`` / ``9 Name`` forms are running headers that repeat on every page of the
    monograph (and on a shared page can sit above the *previous* drug's tail). So we
    prefer the bare title and fall back to the numbered running-header forms only when
    no bare title is found — otherwise a shared-page running header would truncate the
    preceding drug before its Dosages section.
    """
    name = _flexible_name_pattern(drug_name)
    bare = re.compile(rf"^{name}\s*$", re.MULTILINE)
    m = bare.search(text, start, end)
    if m:
        return m.start()
    numbered = (
        re.compile(rf"^\d+\s+{name}\s*$", re.MULTILINE),
        re.compile(rf"^{name}\s+\d+\s*$", re.MULTILINE),
    )
    candidates = [mm.start() for p in numbered if (mm := p.search(text, start, end))]
    return min(candidates) if candidates else None


def _interpolate_index(book_page: int, book_to_index: dict[int, int], n_pages: int) -> int | None:
    """Estimate a page index for a book page with no detected running-header number.

    Uses the nearest known book page and assumes the local page-index-to-book-page
    offset is constant (it is, within a run of normal pages). Returns None if the map
    is empty or the estimate falls outside the page range.
    """
    if not book_to_index:
        return None
    nearest = min(book_to_index, key=lambda known: abs(known - book_page))
    est = book_to_index[nearest] - (nearest - book_page)
    return est if 0 <= est < n_pages else None


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
        interpolated = idx is None
        if interpolated:
            # A few pages carry no running-header number (notably the very first
            # monograph, book page 1). Estimate the anchor from the nearest known
            # book page, but only accept the drug if its title is actually found there
            # — so non-monograph TOC lines don't grab arbitrary content.
            idx = _interpolate_index(entry.book_page, book_to_index, len(pages))
        if idx is None:
            missing.append(entry)
            continue
        anchor = page_offsets[idx]
        window_idx = min(idx + _HEADING_WINDOW_PAGES, len(pages) - 1)
        window_end = page_offsets[window_idx] + len(pages[window_idx])
        pos = _heading_index(body, entry.drug_name, anchor, window_end)
        if pos is None:
            if interpolated:
                missing.append(entry)
                continue
            # Heading not cleanly matched near a detected anchor — the monograph still
            # begins on this page, so fall back to the page start.
            pos = anchor
        located.append((entry, pos))

    located.sort(key=lambda pair: pair[1])
    blocks: list[MonographBlock] = []
    for i, (entry, start) in enumerate(located):
        end = located[i + 1][1] if i + 1 < len(located) else len(body)
        block_body = body[start:end].strip()
        blocks.append(
            MonographBlock(drug_name=entry.drug_name, book_page=entry.book_page, body=block_body)
        )
    return SegmentationResult(blocks=blocks, missing=missing)
