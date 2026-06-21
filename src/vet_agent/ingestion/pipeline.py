import logging

from vet_agent.ingestion.builder import build_monograph
from vet_agent.ingestion.chunker import chunk_monograph
from vet_agent.ingestion.models import Chunk, Monograph, ParseReport
from vet_agent.ingestion.report import build_parse_report
from vet_agent.ingestion.segmenter import segment_monographs
from vet_agent.ingestion.toc import parse_toc_lines

logger = logging.getLogger(__name__)


def run_ingestion(
    pages: list[str], toc_page_range: tuple[int, int]
) -> tuple[list[Monograph], list[Chunk], ParseReport]:
    """Run the full ingestion chain over already-extracted page text.

    toc_page_range is an inclusive (start, end) range of page indices holding the TOC.
    Unlocated drug headings are logged (WARNING) and recorded in the report; the
    coverage policy (fail-or-not) is enforced by the caller (CLI), not here.
    """
    start, end = toc_page_range
    if start > end:
        raise ValueError(f"toc_page_range start ({start}) must not exceed end ({end})")
    toc_lines: list[str] = []
    for page in pages[start : end + 1]:
        toc_lines.extend(page.split("\n"))
    toc = parse_toc_lines(toc_lines)

    body_text = "\n".join(pages[end + 1 :])
    segmentation = segment_monographs(body_text, toc)
    for entry in segmentation.missing:
        logger.warning(
            "Could not locate heading for TOC drug: %s (p.%d)",
            entry.drug_name,
            entry.book_page,
        )
    monographs = [build_monograph(b) for b in segmentation.blocks]

    chunks: list[Chunk] = []
    for mono in monographs:
        chunks.extend(chunk_monograph(mono))

    report = build_parse_report(monographs, toc=toc, missing=segmentation.missing)
    logger.info(
        "Ingestion: located %d/%d TOC drugs, %d missing, %d chunks",
        report.drugs_parsed,
        report.toc_entries,
        len(report.missing_headings),
        len(chunks),
    )
    return monographs, chunks, report
