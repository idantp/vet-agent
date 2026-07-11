from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from vet_agent.ingestion.models import SectionType


class Passage(BaseModel):
    """A retrieved chunk with its citation fields and similarity score."""

    drug_name: str
    section_type: SectionType
    species: list[str]
    book_page: int
    text: str
    logical_key: str
    score: float | None = None


class PointPayload(BaseModel):
    """A fully-formed Qdrant point: id, vector, and stored payload."""

    point_id: str
    vector: list[float]
    drug_name: str
    section_type: SectionType
    species: list[str]
    book_page: int
    text: str
    logical_key: str
    content_hash: str


@runtime_checkable
class Embedder(Protocol):
    """Turns text into L2-normalized vectors. Implementations must normalize."""

    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class Reranker(Protocol):
    """Reorders candidate passages by query relevance and returns the top_k."""

    name: str

    def rerank(self, query: str, passages: list[Passage], top_k: int) -> list[Passage]: ...


class VectorStore(Protocol):
    """Idempotent vector storage with metadata-filtered search."""

    def ensure_collection(self, dim: int) -> None: ...

    def existing_hashes(self) -> dict[str, str]: ...  # point_id -> content_hash

    def upsert(self, points: list[PointPayload]) -> None: ...

    def delete(self, ids: list[str]) -> None: ...

    def search(
        self,
        vector: list[float],
        *,
        drug: str | None = None,
        section: SectionType | None = None,
        species: str | None = None,
        top_k: int = 5,
    ) -> list[Passage]: ...

    def fetch_section(self, drug: str, section: SectionType) -> list[Passage]: ...
