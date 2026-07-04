from vet_agent.eval.eval_set import load_eval_set
from vet_agent.eval.eval_set_builder import (
    FLOW_SECTIONS,
    build_eval_set,
    promote_eval_set,
    write_eval_set,
)
from vet_agent.ingestion.models import Chunk, SectionType


class FakeQueryPhraser:
    def phrase(self, drug, section, species, sample_text):
        return f"Q: {drug}/{section.value}/{'+'.join(species) or 'any'}"


def _chunks():
    return [
        Chunk(
            drug_name="Metronidazole",
            section_type=SectionType.DOSES,
            species=["dog"],
            book_page=1,
            text="dog dose",
            ordinal=0,
        ),
        Chunk(
            drug_name="Metronidazole",
            section_type=SectionType.INDICATIONS,
            species=["all"],
            book_page=1,
            text="treats giardia",
            ordinal=0,
        ),
    ]


def test_flow_sections_cover_expected_flows():
    assert len(FLOW_SECTIONS) == 8
    assert FLOW_SECTIONS["dose"] is SectionType.DOSES
    assert FLOW_SECTIONS["indication"] is SectionType.INDICATIONS
    assert FLOW_SECTIONS["contraindication"] is SectionType.CONTRAINDICATIONS
    assert FLOW_SECTIONS["reproductive_safety"] is SectionType.REPRODUCTIVE_SAFETY
    assert FLOW_SECTIONS["interaction"] is SectionType.DRUG_INTERACTIONS
    assert FLOW_SECTIONS["adverse_effects"] is SectionType.ADVERSE_EFFECTS
    assert FLOW_SECTIONS["monitoring"] is SectionType.MONITORING
    assert FLOW_SECTIONS["administration"] is SectionType.CLIENT_INFORMATION


def test_build_eval_set_labels_and_phrases_deterministically():
    cases = build_eval_set(_chunks(), FakeQueryPhraser(), per_flow=5, seed=0)
    flows = {c.flow for c in cases}
    assert "dose" in flows and "indication" in flows
    dose = next(c for c in cases if c.flow == "dose")
    assert dose.relevant_logical_keys == ["metronidazole|doses|dog|0"]
    assert dose.query.startswith("Q: Metronidazole/doses/")
    again = build_eval_set(_chunks(), FakeQueryPhraser(), per_flow=5, seed=0)
    assert [c.model_dump() for c in cases] == [c.model_dump() for c in again]


def test_write_and_reload_roundtrip(tmp_path):
    cases = build_eval_set(_chunks(), FakeQueryPhraser(), per_flow=5, seed=0)
    out = tmp_path / "retrieval_eval.yaml"
    write_eval_set(cases, out)
    assert [c.model_dump() for c in load_eval_set(out)] == [c.model_dump() for c in cases]


def test_promote_validates_and_preserves_hand_edited_phrasing(tmp_path):
    cases = build_eval_set(_chunks(), FakeQueryPhraser(), per_flow=5, seed=0)
    draft = tmp_path / "retrieval_eval.draft.yaml"
    write_eval_set(cases, draft)
    edited = draft.read_text(encoding="utf-8").replace(
        "Q: Metronidazole/doses/", "Edited dog dose question? "
    )
    draft.write_text(edited, encoding="utf-8")

    final = tmp_path / "retrieval_eval.yaml"
    count = promote_eval_set(draft, final)
    promoted = load_eval_set(final)
    assert count == len(promoted)
    dose = next(c for c in promoted if c.flow == "dose")
    assert dose.query.startswith("Edited dog dose question?")
    assert dose.relevant_logical_keys == ["metronidazole|doses|dog|0"]


def _dose_chunk(drug: str, species: list[str]) -> Chunk:
    return Chunk(
        drug_name=drug,
        section_type=SectionType.DOSES,
        species=species,
        book_page=1,
        text="d",
        ordinal=0,
    )


def test_companion_species_dominate_sampling():
    # 30 dog-dose drugs + 30 exotic/food-animal drugs (varied). other_fraction is passed
    # explicitly (0.2) so the test verifies the quota mechanism independent of the default.
    exotics = ["horse", "cattle", "swine", "rabbit", "llama", "alpaca"]
    chunks = [_dose_chunk(f"Comp{i}", ["dog"]) for i in range(30)]
    chunks += [_dose_chunk(f"Exotic{i}", [exotics[i % len(exotics)]]) for i in range(30)]
    dose = [
        c
        for c in build_eval_set(chunks, FakeQueryPhraser(), per_flow=10, seed=0, other_fraction=0.2)
        if c.flow == "dose"
    ]
    companion = [c for c in dose if set(c.species) <= {"dog", "cat"}]
    assert len(companion) == 8  # ~80% companion at other_fraction=0.2
    assert len(dose) - len(companion) == 2  # only a few exotics


def test_a_few_other_species_appear_when_other_fraction_positive():
    # With other_fraction>0, even a tiny per_flow keeps at least one 'other' species.
    chunks = [_dose_chunk(f"Drug{i}", ["dog"]) for i in range(10)]
    chunks.append(_dose_chunk("RareDrug", ["cattle"]))
    cases = build_eval_set(chunks, FakeQueryPhraser(), per_flow=2, seed=0, other_fraction=0.1)
    dose_species = {tuple(c.species) for c in cases if c.flow == "dose"}
    assert ("cattle",) in dose_species  # the small 'other' quota (min 1) still includes it


def test_default_is_dog_cat_only():
    # Default other_fraction is 0 -> exotic/food-animal targets are excluded by default.
    chunks = [_dose_chunk(f"Drug{i}", ["dog"]) for i in range(10)]
    chunks.append(_dose_chunk("RareDrug", ["cattle"]))
    cases = build_eval_set(chunks, FakeQueryPhraser(), per_flow=5, seed=0)
    dose_species = {tuple(c.species) for c in cases if c.flow == "dose"}
    assert dose_species == {("dog",)}  # no exotics by default


def test_other_fraction_zero_excludes_exotics():
    # other_fraction=0 with a sufficient dog/cat pool -> strictly companion (no backfill needed).
    chunks = [_dose_chunk(f"Comp{i}", ["dog"]) for i in range(5)]
    chunks.append(_dose_chunk("RareDrug", ["cattle"]))
    cases = build_eval_set(chunks, FakeQueryPhraser(), per_flow=5, seed=0, other_fraction=0.0)
    dose_species = {tuple(c.species) for c in cases if c.flow == "dose"}
    assert dose_species == {("dog",)}  # no exotics


def test_zero_per_flow_returns_no_cases():
    chunks = [_dose_chunk("D1", ["dog"]), _dose_chunk("D2", ["cattle"])]
    cases = build_eval_set(chunks, FakeQueryPhraser(), per_flow=0, seed=0)
    assert cases == []
