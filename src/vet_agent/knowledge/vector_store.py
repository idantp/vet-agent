import re
from uuid import UUID

from qdrant_client import QdrantClient, models

from vet_agent.ingestion.models import SectionType
from vet_agent.knowledge.interfaces import Passage, PointPayload


def collection_name(prefix: str, model_key: str) -> str:
    """Derive a per-model collection name: '<prefix>__<sanitized model key>'."""
    safe = re.sub(r"[^a-z0-9]+", "_", model_key.lower()).strip("_")
    return f"{prefix}__{safe}"


def _to_passage(point: models.ScoredPoint) -> Passage:
    payload = point.payload or {}
    return Passage(
        drug_name=payload["drug_name"],
        section_type=SectionType(payload["section_type"]),
        species=list(payload["species"]),
        book_page=int(payload["book_page"]),
        text=payload["text"],
        logical_key=payload["logical_key"],
        score=point.score,
    )


class QdrantVectorStore:
    """VectorStore backed by Qdrant (server URL or `:memory:` local mode)."""

    def __init__(self, client: QdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    def ensure_collection(self, dim: int) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            self._collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        for field in ("drug_name", "section_type", "species"):
            self._client.create_payload_index(
                self._collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def existing_hashes(self) -> dict[str, str]:
        if not self._client.collection_exists(self._collection):
            return {}
        result: dict[str, str] = {}
        offset: int | str | UUID | None = None
        while True:
            points, offset = self._client.scroll(
                self._collection,
                with_payload=["content_hash"],
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            for p in points:
                ch = (p.payload or {}).get("content_hash")
                if isinstance(ch, str):
                    result[str(p.id)] = ch
            if offset is None:
                break
        return result

    def upsert(self, points: list[PointPayload]) -> None:
        if not points:
            return
        self._client.upsert(
            self._collection,
            points=[
                models.PointStruct(
                    id=p.point_id,
                    vector=p.vector,
                    payload={
                        "drug_name": p.drug_name,
                        "section_type": p.section_type.value,
                        "species": p.species,
                        "book_page": p.book_page,
                        "text": p.text,
                        "logical_key": p.logical_key,
                        "content_hash": p.content_hash,
                    },
                )
                for p in points
            ],
        )

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        id_list: list[int | str | UUID] = list(ids)
        self._client.delete(self._collection, points_selector=models.PointIdsList(points=id_list))

    def search(
        self,
        vector: list[float],
        *,
        drug: str | None = None,
        section: SectionType | None = None,
        species: str | None = None,
        top_k: int = 5,
    ) -> list[Passage]:
        must: list[models.Condition] = []
        if drug is not None:
            must.append(models.FieldCondition(key="drug_name", match=models.MatchValue(value=drug)))
        if section is not None:
            must.append(
                models.FieldCondition(
                    key="section_type", match=models.MatchValue(value=section.value)
                )
            )
        if species is not None:
            must.append(
                models.FieldCondition(key="species", match=models.MatchValue(value=species))
            )
        flt = models.Filter(must=must) if must else None
        response = self._client.query_points(
            self._collection,
            query=vector,
            query_filter=flt,
            limit=top_k,
            with_payload=True,
        )
        return [_to_passage(point) for point in response.points]
