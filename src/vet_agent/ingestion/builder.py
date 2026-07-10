from vet_agent.ingestion.models import Monograph
from vet_agent.ingestion.sectionizer import split_sections
from vet_agent.ingestion.segmenter import MonographBlock


def build_monograph(block: MonographBlock) -> Monograph:
    """Assemble a typed Monograph from a raw per-drug block."""
    return Monograph(
        drug_name=block.drug_name,
        book_page=block.book_page,
        sections=split_sections(block.body),
    )
