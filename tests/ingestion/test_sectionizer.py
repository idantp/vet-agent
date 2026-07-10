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
            "Dosages",
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


def test_consecutive_headers_with_no_body_emit_no_empty_section():
    # Two recognized headers back-to-back must not produce a Section(text="").
    body = "Monitoring\nClient Information\nSome real text."
    sections = split_sections(body)
    assert [s.section_type for s in sections] == [SectionType.CLIENT_INFORMATION]
    assert sections[0].text == "Some real text."


def test_alternate_template_header_variants_map_correctly():
    # The older/biologic monograph template uses different header wording.
    body = (
        "Indications/Actions\nUsed in birds.\n"
        "Suggested Dosages/Uses\nBIRDS: 1 mg/kg\n"
        "Contraindications/Precautions\nAvoid in renal disease."
    )
    by = {s.section_type: s.text for s in split_sections(body)}
    assert by[SectionType.INDICATIONS] == "Used in birds."
    assert by[SectionType.DOSES] == "BIRDS: 1 mg/kg"
    assert by[SectionType.CONTRAINDICATIONS] == "Avoid in renal disease."
