import uuid
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

from vet_agent.ingestion.chunker import content_hash, logical_key
from vet_agent.ingestion.models import Chunk
from vet_agent.knowledge.interfaces import Embedder, PointPayload, VectorStore

# Fixed namespace so uuid5(point ids) are stable across runs and machines.
NAMESPACE_VET = uuid.UUID("5b3d2c1a-9e8f-4a7b-8c6d-1f2e3a4b5c6d")

_CHUNKS_ADAPTER = TypeAdapter(list[Chunk])


def point_id(logical_key_value: str) -> str:
    """Deterministic Qdrant point id (UUIDv5) derived from a chunk's logical key."""
    return str(uuid.uuid5(NAMESPACE_VET, logical_key_value))


def read_chunks(path: Path) -> list[Chunk]:
    """Load chunks.json (the Phase-1 artifact) into typed Chunk objects."""
    return _CHUNKS_ADAPTER.validate_json(path.read_text(encoding="utf-8"))


class LoadReport(BaseModel):
    upserted: int
    skipped: int
    pruned: int


def load_chunks(
    chunks: list[Chunk],
    embedder: Embedder,
    store: VectorStore,
    *,
    prune: bool = True,
    batch_size: int = 64,
) -> LoadReport:
    """Idempotently load chunks into the store.

    Skips chunks whose stored content_hash is unchanged, embeds+upserts changed/new
    ones, and (unless prune=False) deletes points whose logical key has disappeared.
    """
    store.ensure_collection(embedder.dim)

    desired: dict[str, Chunk] = {}
    hashes: dict[str, str] = {}
    for chunk in chunks:
        lk = logical_key(chunk)
        pid = point_id(lk)
        desired[pid] = chunk
        hashes[pid] = content_hash(chunk)

    existing = store.existing_hashes()
    to_embed = [pid for pid in desired if existing.get(pid) != hashes[pid]]

    points: list[PointPayload] = []
    for start in range(0, len(to_embed), batch_size):
        batch = to_embed[start : start + batch_size]
        vectors = embedder.embed_documents([desired[pid].text for pid in batch])
        for pid, vector in zip(batch, vectors, strict=True):
            chunk = desired[pid]
            points.append(
                PointPayload(
                    point_id=pid,
                    vector=vector,
                    drug_name=chunk.drug_name,
                    section_type=chunk.section_type,
                    species=chunk.species,
                    book_page=chunk.book_page,
                    text=chunk.text,
                    logical_key=logical_key(chunk),
                    content_hash=hashes[pid],
                )
            )
    store.upsert(points)

    orphans = [pid for pid in existing if pid not in desired]
    if prune and orphans:
        store.delete(orphans)

    return LoadReport(
        upserted=len(points),
        skipped=len(desired) - len(points),
        pruned=len(orphans) if prune else 0,
    )
