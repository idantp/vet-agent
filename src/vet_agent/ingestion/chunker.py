import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from vet_agent.ingestion.models import Chunk, Monograph, Section, SectionType
from vet_agent.ingestion.species import (
    detect_species_mentions,
    is_species_header,
    parse_species_header,
)

DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP = 0


def _size_split(text: str, max_chars: int, overlap: int = DEFAULT_OVERLAP) -> list[str]:
    """Split overlong text on natural separators (paragraph -> line -> word).

    Delegates to RecursiveCharacterTextSplitter, which prefers paragraph then line
    boundaries before falling back to spaces, so a fallback split never cuts mid-word.
    Overlap defaults to 0: our chunks are already structurally bounded and duplicating
    dose lines across chunks is undesirable, but the knob is exposed so retrieval eval
    (Phase 6) can introduce overlap later if it measurably helps.
    """
    if not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


def _doses_species_groups(text: str) -> list[tuple[list[str], str]]:
    """Group dose lines under their species sub-headers.

    Returns (species, text) pairs; lines before any header get species ['unspecified'].
    """
    groups: list[tuple[list[str], list[str]]] = []
    current_species: list[str] = ["unspecified"]
    current_lines: list[str] = []

    def flush() -> None:
        if current_lines:
            groups.append((current_species, current_lines))

    for line in text.split("\n"):
        species = parse_species_header(line)
        if species:
            flush()
            current_species = species
            current_lines = []
        elif is_species_header(line):
            # Header-shaped but no recognized species (e.g. "AVIANS:") — start a new
            # 'unspecified' group rather than misattributing its doses to the prior species.
            flush()
            current_species = ["unspecified"]
            current_lines = []
        else:
            current_lines.append(line)
    flush()
    return [(sp, "\n".join(lines).strip()) for sp, lines in groups]


def _chunk_section(
    drug_name: str, book_page: int, section: Section, max_chars: int, overlap: int
) -> list[Chunk]:
    if section.section_type == SectionType.DOSES:
        chunks: list[Chunk] = []
        ordinal = 0
        for species, text in _doses_species_groups(section.text):
            for piece in _size_split(text, max_chars, overlap):
                chunks.append(
                    Chunk(
                        drug_name=drug_name,
                        section_type=SectionType.DOSES,
                        species=species,
                        book_page=book_page,
                        text=piece,
                        ordinal=ordinal,
                    )
                )
                ordinal += 1
        return chunks

    species = detect_species_mentions(section.text) or ["all"]
    return [
        Chunk(
            drug_name=drug_name,
            section_type=section.section_type,
            species=species,
            book_page=book_page,
            text=piece,
            ordinal=ordinal,
        )
        for ordinal, piece in enumerate(_size_split(section.text, max_chars, overlap))
    ]


def chunk_monograph(
    mono: Monograph, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP
) -> list[Chunk]:
    """Produce structure-aware chunks for one monograph."""
    chunks: list[Chunk] = []
    for section in mono.sections:
        chunks.extend(_chunk_section(mono.drug_name, mono.book_page, section, max_chars, overlap))
    return chunks


def logical_key(chunk: Chunk) -> str:
    """Stable identity for a chunk (drug|section|species|ordinal)."""
    species = "+".join(sorted(chunk.species))
    return f"{chunk.drug_name.lower()}|{chunk.section_type.value}|{species}|{chunk.ordinal}"


def content_hash(chunk: Chunk) -> str:
    """SHA-256 of chunk text + identity, for idempotent re-indexing (Phase 2)."""
    payload = f"{logical_key(chunk)}::{chunk.text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
