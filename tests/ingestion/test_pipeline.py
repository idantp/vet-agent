import logging

import pytest

from vet_agent.ingestion.models import SectionType
from vet_agent.ingestion.pipeline import run_ingestion


def test_run_ingestion_end_to_end():
    # Page index 0 = TOC; page indices 1-2 = monograph bodies.
    pages = [
        "Metronidazole 873\nMidazolam 880",
        "Metronidazole\nUses/Indications\nTreats Giardia in dogs.\nDoses\nDOGS:\n25 mg/kg PO q12h",
        "Midazolam\nUses/Indications\nA benzodiazepine used in cats.",
    ]
    monographs, chunks, report = run_ingestion(pages, toc_page_range=(0, 0))

    assert [m.drug_name for m in monographs] == ["Metronidazole", "Midazolam"]
    assert report.toc_entries == 2
    assert report.drugs_parsed == 2
    assert report.missing_headings == []

    dose_chunks = [c for c in chunks if c.section_type == SectionType.DOSES]
    assert dose_chunks[0].species == ["dog"]
    assert "25 mg/kg" in dose_chunks[0].text


def test_missing_heading_is_reported_and_warned(caplog):
    # "Ghostazole" is in the TOC but never appears as a heading in the body pages.
    pages = [
        "Metronidazole 873\nGhostazole 999",
        "Metronidazole\nUses/Indications\nTreats Giardia in dogs.",
    ]
    with caplog.at_level(logging.WARNING, logger="vet_agent.ingestion.pipeline"):
        monographs, _chunks, report = run_ingestion(pages, toc_page_range=(0, 0))

    assert [m.drug_name for m in monographs] == ["Metronidazole"]
    assert report.toc_entries == 2
    assert report.missing_headings == ["Ghostazole"]
    assert "Ghostazole" in caplog.text
    assert "Could not locate heading" in caplog.text


def test_inverted_toc_page_range_raises():
    with pytest.raises(ValueError, match="must not exceed end"):
        run_ingestion(["a", "b"], toc_page_range=(2, 0))
