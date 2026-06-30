from qdrant_client import QdrantClient

from tests.knowledge.fakes import FakeEmbedder, FakeReranker
from vet_agent.ingestion.models import Chunk, SectionType
from vet_agent.knowledge.loader import load_chunks
from vet_agent.knowledge.retrieval import Retriever
from vet_agent.knowledge.vector_store import QdrantVectorStore


def _chunk(text, *, species, ordinal) -> Chunk:
    return Chunk(
        drug_name="Metronidazole",
        section_type=SectionType.DOSES,
        species=species,
        book_page=873,
        text=text,
        ordinal=ordinal,
    )


def _loaded_retriever(reranker=None) -> Retriever:
    store = QdrantVectorStore(QdrantClient(location=":memory:"), "vet_test")
    emb = FakeEmbedder(dim=8)
    chunks = [
        _chunk("dog dose text", species=["dog"], ordinal=0),
        _chunk("cat dose text", species=["cat"], ordinal=1),
    ]
    load_chunks(chunks, emb, store)
    return Retriever(emb, store, reranker=reranker)


def test_semantic_query_matches_identical_text_first():
    r = _loaded_retriever()
    hits = r.retrieve("dog dose text", top_k=1)
    assert hits[0].text == "dog dose text"  # FakeEmbedder: identical text -> cosine 1.0


def test_species_filter_excludes_other_species():
    r = _loaded_retriever()
    hits = r.retrieve("dose", species="dog", top_k=10)
    assert {h.species[0] for h in hits} == {"dog"}


def test_rerank_path_invokes_reranker():
    r = _loaded_retriever(reranker=FakeReranker())
    plain = r.retrieve("dose", top_k=2)
    reranked = r.retrieve("dose", top_k=2, rerank=True)
    assert [h.logical_key for h in reranked] == [h.logical_key for h in reversed(plain)]
