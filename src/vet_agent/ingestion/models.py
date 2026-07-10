from enum import StrEnum

from pydantic import BaseModel, Field


class SectionType(StrEnum):
    """Canonical monograph section types (Plumb's standard set)."""

    PRESCRIBER_HIGHLIGHTS = "prescriber_highlights"
    INDICATIONS = "indications"
    CONTRAINDICATIONS = "contraindications"
    ADVERSE_EFFECTS = "adverse_effects"
    REPRODUCTIVE_SAFETY = "reproductive_safety"
    OVERDOSE_TOXICITY = "overdose_toxicity"
    DRUG_INTERACTIONS = "drug_interactions"
    LABORATORY_CONSIDERATIONS = "laboratory_considerations"
    PHARMACOLOGY = "pharmacology"
    PHARMACOKINETICS = "pharmacokinetics"
    MONITORING = "monitoring"
    CLIENT_INFORMATION = "client_information"
    CHEMISTRY = "chemistry"
    STORAGE = "storage"
    COMPOUNDING = "compounding"
    DOSAGE_FORMS = "dosage_forms"
    DOSES = "doses"
    OTHER = "other"


class TocEntry(BaseModel):
    drug_name: str
    book_page: int


class Section(BaseModel):
    section_type: SectionType
    text: str


class Monograph(BaseModel):
    drug_name: str
    book_page: int
    sections: list[Section] = Field(default_factory=list)

    def section_text(self, section_type: SectionType) -> str | None:
        for s in self.sections:
            if s.section_type == section_type:
                return s.text
        return None


class Chunk(BaseModel):
    drug_name: str
    section_type: SectionType
    species: list[str]
    book_page: int
    text: str
    ordinal: int


class ParseReport(BaseModel):
    toc_entries: int = 0
    drugs_parsed: int = 0
    missing_headings: list[str] = Field(default_factory=list)
    anomalies: list[dict[str, str]] = Field(default_factory=list)
