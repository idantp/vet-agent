from pathlib import Path

from vet_agent.eval.eval_set import EvalCase, derive_relevant_keys, load_eval_set
from vet_agent.ingestion.models import Chunk, SectionType

FIXTURE = Path("tests/eval/fixtures/retrieval_eval.yaml")


def _chunk(section, species, ordinal, drug="Metronidazole") -> Chunk:
    return Chunk(
        drug_name=drug,
        section_type=section,
        species=species,
        book_page=873,
        text="t",
        ordinal=ordinal,
    )


def test_load_eval_set_parses_cases():
    cases = load_eval_set(FIXTURE)
    assert len(cases) == 2
    assert isinstance(cases[0], EvalCase)
    assert cases[0].section is SectionType.DOSES
    assert cases[1].species == []


def test_derive_relevant_keys_matches_drug_section_species():
    chunks = [
        _chunk(SectionType.DOSES, ["dog"], 0),
        _chunk(SectionType.DOSES, ["cat"], 1),
        _chunk(SectionType.DOSES, ["cat", "dog"], 2),
        _chunk(SectionType.INDICATIONS, ["all"], 0),
        _chunk(SectionType.DOSES, ["dog"], 0, drug="Other"),
    ]
    keys = derive_relevant_keys(
        chunks, drug="metronidazole", section=SectionType.DOSES, species=["dog"]
    )
    assert set(keys) == {"metronidazole|doses|dog|0", "metronidazole|doses|cat+dog|2"}


def test_derive_relevant_keys_empty_species_matches_any():
    chunks = [_chunk(SectionType.INDICATIONS, ["all"], 0)]
    keys = derive_relevant_keys(
        chunks, drug="Metronidazole", section=SectionType.INDICATIONS, species=[]
    )
    assert keys == ["metronidazole|indications|all|0"]
