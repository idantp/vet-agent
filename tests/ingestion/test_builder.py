from vet_agent.ingestion.builder import build_monograph
from vet_agent.ingestion.models import SectionType
from vet_agent.ingestion.segmenter import MonographBlock


def test_build_monograph_from_block():
    block = MonographBlock(
        drug_name="Metronidazole",
        book_page=873,
        body="Uses/Indications\nTreats Giardia.\nDoses\nDOGS: 25 mg/kg PO q12h",
    )
    mono = build_monograph(block)
    assert mono.drug_name == "Metronidazole"
    assert mono.book_page == 873
    assert mono.section_text(SectionType.INDICATIONS) == "Treats Giardia."
    assert mono.section_text(SectionType.DOSES) == "DOGS: 25 mg/kg PO q12h"
