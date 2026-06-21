import re

from vet_agent.ingestion.models import Section, SectionType

# Canonical header string (normalized) -> SectionType.
HEADER_TO_SECTION: dict[str, SectionType] = {
    "Prescriber Highlights": SectionType.PRESCRIBER_HIGHLIGHTS,
    "Uses/Indications": SectionType.INDICATIONS,
    "Contraindications/Precautions/Warnings": SectionType.CONTRAINDICATIONS,
    "Adverse Effects": SectionType.ADVERSE_EFFECTS,
    "Reproductive/Nursing Safety": SectionType.REPRODUCTIVE_SAFETY,
    "Overdose/Acute Toxicity": SectionType.OVERDOSE_TOXICITY,
    "Overdosage/Acute Toxicity": SectionType.OVERDOSE_TOXICITY,
    "Drug Interactions": SectionType.DRUG_INTERACTIONS,
    "Pharmacology/Actions": SectionType.PHARMACOLOGY,
    "Pharmacokinetics": SectionType.PHARMACOKINETICS,
    "Monitoring": SectionType.MONITORING,
    "Client Information": SectionType.CLIENT_INFORMATION,
    "Chemistry/Synonyms": SectionType.CHEMISTRY,
    "Storage/Stability": SectionType.STORAGE,
    "Compatibility/Compounding Considerations": SectionType.COMPOUNDING,
    "Dosage Forms/Regulatory Status": SectionType.DOSAGE_FORMS,
    "Dose Forms/Regulatory Status": SectionType.DOSAGE_FORMS,
    "Doses": SectionType.DOSES,
}

_SLASH_SPACE_RE = re.compile(r"\s*/\s*")
_MULTISPACE_RE = re.compile(r"\s+")


def normalize_header(line: str) -> str:
    """Normalize spacing around slashes and runs of whitespace in a header line."""
    text = _SLASH_SPACE_RE.sub("/", line.strip())
    return _MULTISPACE_RE.sub(" ", text)


def split_sections(body: str) -> list[Section]:
    """Split a monograph body into labeled sections at known header lines.

    Text appearing before the first recognized header (the drug intro) is dropped.
    """
    sections: list[Section] = []
    current_type: SectionType | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current_type is not None:
            ct = current_type
            sections.append(Section(section_type=ct, text="\n".join(buffer).strip()))

    for line in body.split("\n"):
        header_type = HEADER_TO_SECTION.get(normalize_header(line))
        if header_type is not None:
            flush()
            current_type = header_type
            buffer = []
        elif current_type is not None:
            buffer.append(line)
    flush()
    return sections
