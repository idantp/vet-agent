from vet_agent.ingestion.species import (
    detect_species_mentions,
    parse_species_header,
)


def test_parse_single_species_header():
    assert parse_species_header("DOGS:") == ["dog"]
    assert parse_species_header("CATS:") == ["cat"]
    assert parse_species_header("HORSES:") == ["horse"]


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
