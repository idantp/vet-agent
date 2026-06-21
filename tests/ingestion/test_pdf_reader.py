from vet_agent.ingestion.pdf_reader import clean_page_text


def test_dehyphenates_line_wrapped_words():
    raw = "metroni-\ndazole is an anti-\nbacterial agent"
    assert clean_page_text(raw) == "metronidazole is an antibacterial agent"


def test_collapses_spaces_and_preserves_line_structure():
    # Single newlines are preserved (headers/species sub-headers rely on line breaks);
    # only runs of spaces/tabs collapse and 3+ blank lines reduce to one.
    raw = "Adverse Effects\n\nIn   dogs,  vomiting\noccurs."
    assert clean_page_text(raw) == "Adverse Effects\n\nIn dogs, vomiting\noccurs."


def test_strips_empty_input():
    assert clean_page_text("") == ""
