"""Deterministic, offline test doubles for the knowledge interfaces."""

import hashlib
import math

from vet_agent.knowledge.interfaces import Passage


class FakeEmbedder:
    """Maps text -> unit vector via SHA-256 bytes.

    Identical text yields an identical vector (so a query equal to a chunk's text
    scores cosine 1.0); distinct text yields a different vector. No semantics — it
    exercises wiring (idempotency, filtering, ranking), not retrieval quality.
    """

    name = "fake"

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [float(digest[i % len(digest)]) for i in range(self.dim)]
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeReranker:
    """Reverses the candidate order (so tests can prove rerank ran), returns top_k."""

    name = "fake-reranker"

    def rerank(self, query: str, passages: list[Passage], top_k: int) -> list[Passage]:
        return list(reversed(passages))[:top_k]
