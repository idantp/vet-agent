from vet_agent.tools.drug_index import DrugIndex, ResolvedDrug
from vet_agent.tools.models import DrugNotFound

NAMES = ["Metronidazole", "Metoclopramide", "Meloxicam", "Carprofen"]


def _index() -> DrugIndex:
    return DrugIndex(NAMES)


def test_exact_case_insensitive_match():
    r = _index().resolve("metronidazole")
    assert isinstance(r, ResolvedDrug)
    assert r.canonical == "Metronidazole"
    assert r.exact is True


def test_whitespace_is_normalized():
    r = _index().resolve("  Metronidazole  ")
    assert isinstance(r, ResolvedDrug)
    assert r.exact is True


def test_fuzzy_single_close_match_is_visible_correction():
    r = _index().resolve("metronidazol")  # missing final 'e'
    assert isinstance(r, ResolvedDrug)
    assert r.canonical == "Metronidazole"
    assert r.exact is False


def test_miss_returns_suggestions():
    r = _index().resolve("metoclopramid3e")  # garbled but close-ish
    if isinstance(r, DrugNotFound):
        assert "Metoclopramide" in r.suggestions
    else:  # a high-cutoff fuzzy hit is also acceptable for this input
        assert r.canonical == "Metoclopramide"


def test_garbage_returns_empty_suggestions():
    r = _index().resolve("xyzzyplugh")
    assert isinstance(r, DrugNotFound)
    assert r.query == "xyzzyplugh"
    assert r.suggestions == []


def test_from_chunks_builds_from_distinct_drug_names(tmp_path):
    import json

    chunks = [
        {
            "drug_name": "Metronidazole",
            "section_type": "doses",
            "species": ["dog"],
            "book_page": 873,
            "text": "t",
            "ordinal": 0,
        },
        {
            "drug_name": "Metronidazole",
            "section_type": "indications",
            "species": ["all"],
            "book_page": 873,
            "text": "t",
            "ordinal": 0,
        },
    ]
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps(chunks), encoding="utf-8")
    index = DrugIndex.from_chunks(path)
    r = index.resolve("METRONIDAZOLE")
    assert isinstance(r, ResolvedDrug) and r.canonical == "Metronidazole"
