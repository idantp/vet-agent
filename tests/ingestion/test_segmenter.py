from vet_agent.ingestion.models import TocEntry
from vet_agent.ingestion.segmenter import segment_monographs


def test_segments_two_drugs_in_order():
    text = "\n".join(
        [
            "Metronidazole",
            "Uses/Indications",
            "Treats Giardia.",
            "Midazolam",
            "Uses/Indications",
            "A benzodiazepine.",
        ]
    )
    toc = [
        TocEntry(drug_name="Metronidazole", book_page=873),
        TocEntry(drug_name="Midazolam", book_page=880),
    ]
    result = segment_monographs(text, toc)
    blocks = result.blocks
    assert [b.drug_name for b in blocks] == ["Metronidazole", "Midazolam"]
    assert "Treats Giardia." in blocks[0].body
    assert "Treats Giardia." not in blocks[1].body
    assert "A benzodiazepine." in blocks[1].body
    assert blocks[0].book_page == 873
    assert result.missing == []


def test_missing_drug_heading_is_reported():
    text = "Metronidazole\nUses/Indications\nTreats Giardia."
    toc = [
        TocEntry(drug_name="Metronidazole", book_page=873),
        TocEntry(drug_name="Nonexistent Drug", book_page=999),
    ]
    result = segment_monographs(text, toc)
    # Located drugs are returned as blocks; unlocated ones are surfaced in `missing`
    # (returned as data, NOT silently dropped) so the caller can enforce a policy.
    assert [b.drug_name for b in result.blocks] == ["Metronidazole"]
    assert [e.drug_name for e in result.missing] == ["Nonexistent Drug"]
