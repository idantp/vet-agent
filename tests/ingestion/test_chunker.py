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
    # GERBILS is species-shaped but not in the vocabulary -> its dose becomes its own
    # 'unspecified' group rather than being attributed to the preceding DOGS.
    doses = "DOGS:\n25 mg/kg PO q12h\nGERBILS:\n10 mg/kg PO q24h"
    mono = Monograph(
        drug_name="X",
        book_page=1,
        sections=[Section(section_type=SectionType.DOSES, text=doses)],
    )
    chunks = chunk_monograph(mono)
    dog = next(c for c in chunks if c.species == ["dog"])
    assert "10 mg/kg PO q24h" not in dog.text  # gerbil dose must NOT become a dog dose
    assert any(c.species == ["unspecified"] and "10 mg/kg PO q24h" in c.text for c in chunks)


def test_note_header_in_doses_stays_with_current_species():
    # A 'NOTES:' line is content within the current species, not a new species group,
    # so the dose under DOGS keeps species=['dog'] (not 'unspecified').
    doses = "DOGS:\n25 mg/kg PO q12h\nNOTES:\nGive with food."
    mono = Monograph(
        drug_name="X",
        book_page=1,
        sections=[Section(section_type=SectionType.DOSES, text=doses)],
    )
    chunks = chunk_monograph(mono)
    assert all(c.species == ["dog"] for c in chunks)
    assert any("Give with food." in c.text for c in chunks)


def test_empty_section_yields_no_chunks():
    mono = Monograph(
        drug_name="X",
        book_page=1,
        sections=[Section(section_type=SectionType.STORAGE, text="   ")],
    )
    assert chunk_monograph(mono) == []


def test_duplicate_section_type_gets_unique_logical_keys():
    # Two sections mapping to the same SectionType (e.g. headers 'Dosages' and 'Doses')
    # must not produce colliding logical_keys, or a Phase-2 upsert would drop one.
    mono = Monograph(
        drug_name="X",
        book_page=1,
        sections=[
            Section(section_type=SectionType.DOSES, text="DOGS:\n10 mg/kg"),
            Section(section_type=SectionType.DOSES, text="DOGS:\n20 mg/kg"),
        ],
    )
    chunks = chunk_monograph(mono)
    keys = [logical_key(c) for c in chunks]
    assert len(keys) == len(set(keys))
    assert content_hash(chunks[0]) != content_hash(chunks[1])
