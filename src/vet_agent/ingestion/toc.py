import logging
import re

from vet_agent.ingestion.models import TocEntry

logger = logging.getLogger(__name__)

# "<Drug Name> <page>" — name may contain letters, spaces, slashes, hyphens, parens.
_TOC_LINE_RE = re.compile(r"^(?P<name>[A-Za-z][A-Za-z0-9 ,/()'+-]+?)\s+(?P<page>\d{1,4})$")


def parse_toc_lines(lines: list[str]) -> list[TocEntry]:
    """Parse '<Drug> <page>' table-of-contents lines into TocEntry objects.

    Blank lines are ignored silently. Every non-blank line that fails to parse is
    logged at DEBUG (with the offending text); a single INFO summary reports how many
    entries parsed and how many lines were skipped.

    Duplicate drug names (e.g. from physically repeated TOC pages in the PDF) are
    deduplicated: the first occurrence wins and subsequent ones are silently dropped.
    """
    entries: list[TocEntry] = []
    seen_names: set[str] = set()
    skipped = 0
    duplicates = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        m = _TOC_LINE_RE.match(stripped)
        if not m:
            skipped += 1
            logger.debug("Skipping non-entry TOC line: %r", stripped)
            continue
        name = m.group("name").strip()
        if name in seen_names:
            duplicates += 1
            logger.debug("Skipping duplicate TOC entry: %r", name)
            continue
        seen_names.add(name)
        entries.append(TocEntry(drug_name=name, book_page=int(m.group("page"))))
    if duplicates:
        logger.info(
            "Parsed %d TOC entries (%d non-blank lines skipped, %d duplicates dropped)",
            len(entries),
            skipped,
            duplicates,
        )
    else:
        logger.info("Parsed %d TOC entries (%d non-blank lines skipped)", len(entries), skipped)
    return entries
