import re

from pydantic import BaseModel

from vet_agent.ingestion.models import TocEntry


class MonographBlock(BaseModel):
    drug_name: str
    book_page: int
    body: str


class SegmentationResult(BaseModel):
    blocks: list[MonographBlock]
    missing: list[TocEntry]


def _heading_index(text: str, drug_name: str, start: int) -> int | None:
    """Find the offset of a monograph-heading line for *drug_name*, at/after *start*.

    The PDF uses three heading-line formats depending on which column the running
    header falls on:

    * Bare:          ``Acarbose``
    * Number-prefix: ``90 Antivenom, North American Coral Snake``  (even book pages)
    * Number-suffix: ``Apomorphine 93``                            (odd book pages)

    We try the bare form first (cheapest, most specific).  If that fails — or if the
    first bare match is implausibly far ahead (i.e. it sits in a back-of-book index
    rather than the actual monograph section) — we also search for the two numbered
    variants and return the earliest match from any of the three patterns.
    """
    escaped = re.escape(drug_name)
    bare_pat = re.compile(rf"^{escaped}\s*$", re.MULTILINE)
    prefix_pat = re.compile(rf"^\d+\s+{escaped}\s*$", re.MULTILINE)
    suffix_pat = re.compile(rf"^{escaped}\s+\d+\s*$", re.MULTILINE)

    candidates: list[int] = []
    for pat in (bare_pat, prefix_pat, suffix_pat):
        m = pat.search(text, start)
        if m:
            candidates.append(m.start())

    return min(candidates) if candidates else None


def segment_monographs(text: str, toc: list[TocEntry]) -> SegmentationResult:
    """Slice the full book text into per-drug blocks following TOC order.

    Each drug's block runs from its heading line up to the next found heading. This is
    a pure transformation: drugs whose heading cannot be located are returned in
    `missing` (errors-as-values), leaving coverage policy to the orchestration layer.
    """
    located: list[tuple[TocEntry, int]] = []
    missing: list[TocEntry] = []
    cursor = 0
    for entry in toc:
        idx = _heading_index(text, entry.drug_name, cursor)
        if idx is None:
            missing.append(entry)
            continue
        located.append((entry, idx))
        cursor = idx + len(entry.drug_name)

    blocks: list[MonographBlock] = []
    for i, (entry, start) in enumerate(located):
        end = located[i + 1][1] if i + 1 < len(located) else len(text)
        body = text[start + len(entry.drug_name) : end].strip()
        blocks.append(
            MonographBlock(drug_name=entry.drug_name, book_page=entry.book_page, body=body)
        )
    return SegmentationResult(blocks=blocks, missing=missing)
