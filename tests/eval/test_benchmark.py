from tests.knowledge.fakes import FakeEmbedder
from vet_agent.eval.benchmark import benchmark_model, choose_default, render_scorecard
from vet_agent.eval.eval_set import EvalCase
from vet_agent.ingestion.models import Chunk, SectionType


def _chunk(text, ordinal) -> Chunk:
    return Chunk(
        drug_name="Metronidazole",
        section_type=SectionType.DOSES,
        species=["dog"],
        book_page=873,
        text=text,
        ordinal=ordinal,
    )


def _cases():
    # FakeEmbedder maps identical text -> identical vector, so a query equal to a
    # chunk's text ranks that chunk #1 deterministically.
    return [
        EvalCase(
            query="alpha dose",
            flow="dose",
            drug="Metronidazole",
            section=SectionType.DOSES,
            species=["dog"],
            relevant_logical_keys=["metronidazole|doses|dog|0"],
        )
    ]


def test_benchmark_model_scores_a_perfect_match():
    chunks = [_chunk("alpha dose", 0), _chunk("unrelated text", 1)]
    score = benchmark_model("fake", FakeEmbedder(dim=8), chunks, _cases(), ks=[1, 3])
    assert score.model == "fake"
    assert score.metrics["recall@1"] == 1.0
    assert score.metrics["mrr"] == 1.0


def test_choose_default_picks_best_recall_at_5():
    chunks = [_chunk("alpha dose", 0)]
    good = benchmark_model("good", FakeEmbedder(dim=8), chunks, _cases(), ks=[5])
    # A model that never matches: empty corpus -> recall 0.
    bad = benchmark_model("bad", FakeEmbedder(dim=8), [], _cases(), ks=[5])
    assert choose_default([good, bad]) == "good"


def test_render_scorecard_lists_models_and_winner():
    chunks = [_chunk("alpha dose", 0)]
    score = benchmark_model("fake", FakeEmbedder(dim=8), chunks, _cases(), ks=[1, 5])
    md = render_scorecard([score], ks=[1, 5])
    assert "fake" in md
    assert "Winner" in md
