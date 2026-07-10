from vet_agent.ingestion.models import SectionType
from vet_agent.knowledge.interfaces import Embedder, Passage, Reranker, VectorStore

RERANK_FETCH_K = 20  # over-fetch this many candidates before reranking


class Retriever:
    """Filtered semantic search over a VectorStore, with optional reranking."""

    def __init__(
        self, embedder: Embedder, store: VectorStore, reranker: Reranker | None = None
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._reranker = reranker

    def retrieve(
        self,
        query: str,
        *,
        drug: str | None = None,
        section: SectionType | None = None,
        species: str | None = None,
        top_k: int = 5,
        rerank: bool = False,
    ) -> list[Passage]:
        vector = self._embedder.embed_query(query)
        fetch_k = max(top_k, RERANK_FETCH_K) if rerank else top_k
        hits = self._store.search(
            vector, drug=drug, section=section, species=species, top_k=fetch_k
        )
        if rerank and self._reranker is not None:
            hits = self._reranker.rerank(query, hits, top_k)
        return hits[:top_k]
