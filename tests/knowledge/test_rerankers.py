import pytest

from vet_agent.ingestion.models import SectionType
from vet_agent.knowledge.interfaces import Passage
from vet_agent.knowledge.rerankers import CrossEncoderReranker, _apply_scores


def _p(key: str, text: str) -> Passage:
    return Passage(
        drug_name="X",
        section_type=SectionType.DOSES,
        species=["dog"],
        book_page=1,
        text=text,
        logical_key=key,
    )


def test_apply_scores_sorts_desc_and_truncates():
    passages = [_p("a", "aa"), _p("b", "bb"), _p("c", "cc")]
    out = _apply_scores(passages, [0.1, 0.9, 0.5], top_k=2)
    assert [p.logical_key for p in out] == ["b", "c"]
    assert out[0].score == 0.9


@pytest.mark.slow
def test_real_bge_reranker_reorders_by_relevance():
    rr = CrossEncoderReranker("BAAI/bge-reranker-v2-m3")
    passages = [
        _p("off", "Storage: keep refrigerated."),
        _p("on", "DOGS: metronidazole 25 mg/kg PO q12h for giardia."),
    ]
    out = rr.rerank("metronidazole dose for a dog with giardia", passages, top_k=2)
    assert out[0].logical_key == "on"
