from vet_agent.ingestion.chunker import chunk_monograph, content_hash, logical_key
from vet_agent.ingestion.models import Monograph, Section, SectionType


def _mono() -> Monograph:
    # Species sub-headers sit on their own line (matching Plumb's formatting and
    # the structure preserved by clean_page_text).
    doses = "DOGS:\n25 mg/kg PO q12h\nCATS:\n25 mg/kg PO q24h\nDOGS & CATS:\nbonus line"
    return Monograph(
        drug_name="Metronidazole",
        book_page=873,
        sections=[
            Section(
                section_type=SectionType.INDICATIONS,
                text="Used for Giardia in dogs and cats.",
            ),
            Section(section_type=SectionType.DOSES, text=doses),
        ],
    )


def test_doses_split_per_species():
    chunks = chunk_monograph(_mono())
    dose_chunks = [c for c in chunks if c.section_type == SectionType.DOSES]
    species_sets = sorted(tuple(c.species) for c in dose_chunks)
    assert ("cat",) in species_sets
    assert ("dog",) in species_sets
    assert ("cat", "dog") in species_sets  # "DOGS & CATS:" -> combined
    cat_chunk = next(c for c in dose_chunks if c.species == ["cat"])
    assert "25 mg/kg PO q24h" in cat_chunk.text


def test_prose_section_uses_soft_species_mentions():
    chunks = chunk_monograph(_mono())
    ind = next(c for c in chunks if c.section_type == SectionType.INDICATIONS)
    assert ind.species == ["cat", "dog"]
    assert ind.ordinal == 0


def test_prose_section_with_no_species_tagged_all():
    mono = Monograph(
        drug_name="X",
        book_page=1,
        sections=[Section(section_type=SectionType.STORAGE, text="Store at 25 C.")],
    )
    chunk = chunk_monograph(mono)[0]
    assert chunk.species == ["all"]


def test_long_section_is_size_split_with_ordinals():
    long_text = " ".join(f"word{i}" for i in range(1000))
    mono = Monograph(
        drug_name="X",
        book_page=1,
        sections=[Section(section_type=SectionType.PHARMACOLOGY, text=long_text)],
    )
    chunks = chunk_monograph(mono, max_chars=200)
    assert len(chunks) > 1
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_logical_key_and_content_hash_are_deterministic_and_distinct():
    chunks = chunk_monograph(_mono())
    first = chunks[0]
    # deterministic
    assert logical_key(first) == logical_key(first)
    assert content_hash(first) == content_hash(first)
    assert "metronidazole" in logical_key(first).lower()
    # distinct chunks within a monograph have distinct keys
    keys = [logical_key(c) for c in chunks]
    assert len(keys) == len(set(keys))
    # changing text changes the hash (key held constant)
    mutated = first.model_copy(update={"text": first.text + " EXTRA"})
    assert content_hash(mutated) != content_hash(first)


def test_unknown_species_header_not_misattributed_to_prior_species():
    doses = "DOGS:\n25 mg/kg PO q12h\nAVIANS:\n10 mg/kg PO q24h"
    mono = Monograph(
        drug_name="X",
        book_page=1,
        sections=[Section(section_type=SectionType.DOSES, text=doses)],
    )
    chunks = chunk_monograph(mono)
    dog = next(c for c in chunks if c.species == ["dog"])
    assert "10 mg/kg PO q24h" not in dog.text  # avian dose must NOT become a dog dose
    assert any(c.species == ["unspecified"] and "10 mg/kg PO q24h" in c.text for c in chunks)


def test_empty_section_yields_no_chunks():
    mono = Monograph(
        drug_name="X",
        book_page=1,
        sections=[Section(section_type=SectionType.STORAGE, text="   ")],
    )
    assert chunk_monograph(mono) == []
