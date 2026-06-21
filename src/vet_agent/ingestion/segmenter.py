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
    """Find the offset of a line that is exactly the drug name, at/after `start`."""
    pattern = re.compile(rf"^{re.escape(drug_name)}\s*$", re.MULTILINE)
    m = pattern.search(text, start)
    return m.start() if m else None


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
