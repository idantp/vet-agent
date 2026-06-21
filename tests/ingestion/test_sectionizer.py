from vet_agent.ingestion.models import SectionType
from vet_agent.ingestion.sectionizer import normalize_header, split_sections


def test_normalize_header_collapses_spaces():
    assert normalize_header("Dosage Forms/ Regulatory Status") == "Dosage Forms/Regulatory Status"


def test_split_sections_maps_known_headers():
    body = "\n".join(
        [
            "Uses/Indications",
            "Used for Giardia in dogs and cats.",
            "Adverse Effects",
            "Vomiting and lethargy.",
            "Doses",
            "DOGS: 25 mg/kg PO q12h",
        ]
    )
    sections = split_sections(body)
    by_type = {s.section_type: s.text for s in sections}
    assert by_type[SectionType.INDICATIONS] == "Used for Giardia in dogs and cats."
    assert by_type[SectionType.ADVERSE_EFFECTS] == "Vomiting and lethargy."
    assert by_type[SectionType.DOSES] == "DOGS: 25 mg/kg PO q12h"


def test_overdosage_variant_maps_to_overdose_toxicity():
    body = "Overdosage/Acute Toxicity\nSupportive care."
    sections = split_sections(body)
    assert sections[0].section_type == SectionType.OVERDOSE_TOXICITY


def test_unknown_header_is_not_treated_as_section():
    # Text before the first known header is ignored (it's the drug intro block).
    body = "Some intro prose.\nUses/Indications\nReal content."
    sections = split_sections(body)
    assert [s.section_type for s in sections] == [SectionType.INDICATIONS]
    assert sections[0].text == "Real content."
