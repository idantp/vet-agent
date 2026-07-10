import logging

from vet_agent.ingestion.toc import parse_toc_lines


def test_parses_drug_and_page():
    lines = [
        "Metronidazole 873",
        "Midazolam 880",
        "Moxidectin/Moxidectin Combination Products 912",
    ]
    entries = parse_toc_lines(lines)
    assert entries[0].drug_name == "Metronidazole"
    assert entries[0].book_page == 873
    assert entries[2].drug_name == "Moxidectin/Moxidectin Combination Products"
    assert entries[2].book_page == 912


def test_ignores_non_entry_lines():
    lines = ["Table of Contents", "Preface vii", "Metronidazole 873", ""]
    entries = parse_toc_lines(lines)
    # "Preface vii" has no integer page -> skipped; heading skipped
    assert [e.drug_name for e in entries] == ["Metronidazole"]


def test_logs_success_count_and_skipped_lines(caplog):
    lines = ["Table of Contents", "Preface vii", "Metronidazole 873", ""]
    with caplog.at_level(logging.DEBUG, logger="vet_agent.ingestion.toc"):
        parse_toc_lines(lines)
    # INFO summary with the success count, and a DEBUG line per failed parse.
    assert "Parsed 1 TOC entries" in caplog.text
    assert "Skipping non-entry TOC line" in caplog.text
    assert "Preface vii" in caplog.text  # the failing line is named


def test_deduplicates_repeated_entries():
    """Duplicate TOC pages (physically repeated in the PDF) must not create
    duplicate TocEntry objects — the first occurrence wins."""
    lines = [
        "Metronidazole 873",
        "Midazolam 880",
        # Same two entries again (simulating a duplicated TOC page)
        "Metronidazole 873",
        "Midazolam 880",
    ]
    entries = parse_toc_lines(lines)
    assert len(entries) == 2
    assert entries[0].drug_name == "Metronidazole"
    assert entries[1].drug_name == "Midazolam"


def test_deduplication_logs_duplicates_at_debug(caplog):
    lines = ["Metronidazole 873", "Metronidazole 873"]
    with caplog.at_level(logging.DEBUG, logger="vet_agent.ingestion.toc"):
        entries = parse_toc_lines(lines)
    assert len(entries) == 1
    assert "duplicate" in caplog.text.lower()
