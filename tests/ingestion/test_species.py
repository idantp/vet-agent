from vet_agent.ingestion.species import (
    detect_species_mentions,
    is_species_header,
    parse_species_header,
)


def test_resolves_exotic_and_farm_species():
    # Real Plumb's dose sub-headers beyond the common species (incl. the space-before-
    # colon form the PDF emits).
    assert parse_species_header("REPTILES :") == ["reptile"]
    assert parse_species_header("CAMELIDS:") == ["camelid"]
    assert parse_species_header("EQUINE:") == ["horse"]
    assert parse_species_header("SMALL MAMMALS :") == ["small mammal"]


def test_is_species_header_excludes_notes_and_routes():
    assert is_species_header("REPTILES:") is True
    assert is_species_header("GERBILS:") is True  # unknown species, still a boundary
    assert is_species_header("NOTES:") is False  # a note, not a species
    assert is_species_header("IM:") is False  # a route, not a species
    assert is_species_header("25 mg/kg PO q12h") is False


def test_parse_single_species_header():
    assert parse_species_header("DOGS:") == ["dog"]
    assert parse_species_header("CATS:") == ["cat"]
    assert parse_species_header("HORSES:") == ["horse"]
    assert parse_species_header("CATTLE:") == ["cattle"]
    assert parse_species_header("SWINE:") == ["swine"]


def test_parse_combined_species_header():
    assert parse_species_header("DOGS & CATS:") == ["cat", "dog"]
    assert parse_species_header("DOGS/CATS:") == ["cat", "dog"]


def test_non_header_returns_empty():
    assert parse_species_header("Giardiasis (extra-label):") == []
    assert parse_species_header("25 mg/kg PO q12h") == []


def test_detect_species_mentions_in_prose():
    text = "Used extensively in dogs and cats; in horses it may cause ataxia."
    assert detect_species_mentions(text) == ["cat", "dog", "horse"]


def test_detect_species_mentions_none():
    assert detect_species_mentions("No specific species discussed.") == []


def test_is_species_header_recognizes_shape_even_for_unknown_species():
    from vet_agent.ingestion.species import is_species_header

    assert is_species_header("AVIANS:") is True
    assert is_species_header("DOGS:") is True
    assert is_species_header("Giardiasis (extra-label):") is False
    assert is_species_header("25 mg/kg PO q12h") is False
