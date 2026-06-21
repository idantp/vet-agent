from vet_agent.ingestion.models import TocEntry
from vet_agent.ingestion.segmenter import segment_monographs


def test_segments_two_drugs_in_order():
    # Each page's first line is the running header "<Drug> <book_page>".
    pages = [
        "Metronidazole 873\nUses/Indications\nTreats Giardia.",
        "Midazolam 880\nUses/Indications\nA benzodiazepine.",
    ]
    toc = [
        TocEntry(drug_name="Metronidazole", book_page=873),
        TocEntry(drug_name="Midazolam", book_page=880),
    ]
    result = segment_monographs(pages, toc)
    blocks = result.blocks
    assert [b.drug_name for b in blocks] == ["Metronidazole", "Midazolam"]
    assert "Treats Giardia." in blocks[0].body
    assert "Treats Giardia." not in blocks[1].body
    assert "A benzodiazepine." in blocks[1].body
    assert blocks[0].book_page == 873
    assert result.missing == []


def test_drug_whose_book_page_is_absent_is_reported_missing():
    pages = ["Metronidazole 873\nUses/Indications\nTreats Giardia."]
    toc = [
        TocEntry(drug_name="Metronidazole", book_page=873),
        TocEntry(drug_name="Nonexistent Drug", book_page=999),
    ]
    result = segment_monographs(pages, toc)
    # Book page 999 appears on no page -> the drug is surfaced in `missing`, not
    # silently dropped, so the caller can enforce a coverage policy.
    assert [b.drug_name for b in result.blocks] == ["Metronidazole"]
    assert [e.drug_name for e in result.missing] == ["Nonexistent Drug"]


def test_two_drugs_on_same_page_are_split_by_heading():
    # Both monographs share book page 100. The running header names the first drug;
    # the second drug's title appears lower on the same page. Anchoring + per-drug-name
    # search separates them.
    pages = [
        "Acarbose 100\nUses/Indications\nAlpha-glucosidase inhibitor.\n"
        "Acebutolol\nUses/Indications\nA beta-blocker."
    ]
    toc = [
        TocEntry(drug_name="Acarbose", book_page=100),
        TocEntry(drug_name="Acebutolol", book_page=100),
    ]
    result = segment_monographs(pages, toc)
    assert [b.drug_name for b in result.blocks] == ["Acarbose", "Acebutolol"]
    assert "Alpha-glucosidase" in result.blocks[0].body
    assert "Alpha-glucosidase" not in result.blocks[1].body
    assert "beta-blocker" in result.blocks[1].body
