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


def test_flow_sections_cover_three_flows():
    assert FLOW_SECTIONS["dose"] is SectionType.DOSES
    assert FLOW_SECTIONS["indication"] is SectionType.INDICATIONS
    assert FLOW_SECTIONS["contraindication"] is SectionType.CONTRAINDICATIONS


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
