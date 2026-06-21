from vet_agent.ingestion.models import (
    Chunk,
    Monograph,
    ParseReport,
    Section,
    SectionType,
    TocEntry,  # noqa: F401
)


def test_section_type_has_doses_and_other():
    assert SectionType.DOSES.value == "doses"
    assert SectionType.OTHER.value == "other"


def test_monograph_roundtrip():
    mono = Monograph(
        drug_name="Metronidazole",
        book_page=873,
        sections=[
            Section(section_type=SectionType.INDICATIONS, text="Used for Giardia."),
            Section(section_type=SectionType.DOSES, text="DOGS: 25 mg/kg PO q12h"),
        ],
    )
    assert mono.section_text(SectionType.DOSES) == "DOGS: 25 mg/kg PO q12h"
    assert mono.section_text(SectionType.MONITORING) is None


def test_chunk_defaults():
    c = Chunk(
        drug_name="Metronidazole",
        section_type=SectionType.DOSES,
        species=["dog"],
        book_page=875,
        text="DOGS: 25 mg/kg PO q12h",
        ordinal=0,
    )
    assert c.species == ["dog"]
    assert c.ordinal == 0


def test_parse_report_counts():
    r = ParseReport(
        toc_entries=3,
        drugs_parsed=2,
        missing_headings=["Lost Drug"],
        anomalies=[{"drug": "X", "issue": "no sections"}],
    )
    assert r.toc_entries == 3
    assert r.drugs_parsed == 2
    assert r.missing_headings == ["Lost Drug"]
    assert r.anomalies[0]["drug"] == "X"
