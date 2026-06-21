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


def test_preserves_numeric_dose_range_at_hyphen_wrap():
    # A weight range that wraps at a hyphen must NOT have its digits fused.
    assert clean_page_text("(8.1-\n25 lb)") == "(8.1-\n25 lb)"
    assert "8.125" not in clean_page_text("(8.1-\n25 lb)")


def test_collapses_excess_blank_lines_including_whitespace_only():
    assert clean_page_text("A\n\n\n\nB") == "A\n\nB"
    assert clean_page_text("section\n  \n  \nmore") == "section\n\nmore"
