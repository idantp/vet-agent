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
    """
    entries: list[TocEntry] = []
    skipped = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        m = _TOC_LINE_RE.match(stripped)
        if not m:
            skipped += 1
            logger.debug("Skipping non-entry TOC line: %r", stripped)
            continue
        entries.append(
            TocEntry(drug_name=m.group("name").strip(), book_page=int(m.group("page")))
        )
    logger.info("Parsed %d TOC entries (%d non-blank lines skipped)", len(entries), skipped)
    return entries
