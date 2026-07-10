import pytest
from qdrant_client import QdrantClient

from tests.knowledge.fakes import FakeEmbedder
from vet_agent.ingestion.models import Chunk, SectionType
from vet_agent.knowledge.loader import load_chunks, point_id, read_chunks
from vet_agent.knowledge.vector_store import QdrantVectorStore


def _chunk(text: str, *, drug="Metronidazole", ordinal=0) -> Chunk:
    return Chunk(
        drug_name=drug,
        section_type=SectionType.DOSES,
        species=["dog"],
        book_page=873,
        text=text,
        ordinal=ordinal,
    )


def _store() -> QdrantVectorStore:
    return QdrantVectorStore(QdrantClient(location=":memory:"), "vet_test")


def test_first_load_embeds_all_then_second_load_skips_all():
    chunks = [_chunk("a", ordinal=0), _chunk("b", ordinal=1)]
    store, emb = _store(), FakeEmbedder(dim=8)
    r1 = load_chunks(chunks, emb, store)
    assert (r1.upserted, r1.skipped, r1.pruned) == (2, 0, 0)
    r2 = load_chunks(chunks, emb, store)
    assert (r2.upserted, r2.skipped, r2.pruned) == (0, 2, 0)


def test_changed_chunk_is_reembedded_only():
    chunks = [_chunk("a", ordinal=0), _chunk("b", ordinal=1)]
    store, emb = _store(), FakeEmbedder(dim=8)
    load_chunks(chunks, emb, store)
    chunks[1] = _chunk("b-edited", ordinal=1)  # same logical_key, new content_hash
    r = load_chunks(chunks, emb, store)
    assert (r.upserted, r.skipped, r.pruned) == (1, 1, 0)


def test_orphan_is_pruned():
    chunks = [_chunk("a", ordinal=0), _chunk("b", ordinal=1)]
    store, emb = _store(), FakeEmbedder(dim=8)
    load_chunks(chunks, emb, store)
    r = load_chunks([chunks[0]], emb, store)  # second chunk disappeared
    assert (r.upserted, r.skipped, r.pruned) == (0, 1, 1)
    assert point_id("metronidazole|doses|dog|1") not in store.existing_hashes()


def test_no_prune_keeps_orphans():
    chunks = [_chunk("a", ordinal=0), _chunk("b", ordinal=1)]
    store, emb = _store(), FakeEmbedder(dim=8)
    load_chunks(chunks, emb, store)
    r = load_chunks([chunks[0]], emb, store, prune=False)
    assert r.pruned == 0
    assert len(store.existing_hashes()) == 2


def test_read_chunks_roundtrips(tmp_path):
    import json

    chunks = [_chunk("a", ordinal=0)]
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps([c.model_dump() for c in chunks]), encoding="utf-8")
    loaded = read_chunks(path)
    assert loaded == chunks


def test_duplicate_logical_key_input_raises():
    # Same drug/section/species/ordinal -> identical logical_key -> identical point_id.
    dup = [_chunk("a", ordinal=0), _chunk("b", ordinal=0)]
    store, emb = _store(), FakeEmbedder(dim=8)
    with pytest.raises(ValueError, match="Duplicate logical_key"):
        load_chunks(dup, emb, store)


def test_empty_input_with_prune_raises_when_collection_nonempty():
    chunks = [_chunk("a", ordinal=0)]
    store, emb = _store(), FakeEmbedder(dim=8)
    load_chunks(chunks, emb, store)
    with pytest.raises(ValueError, match="prune=True"):
        load_chunks([], emb, store)


def test_empty_input_with_prune_false_is_noop():
    chunks = [_chunk("a", ordinal=0)]
    store, emb = _store(), FakeEmbedder(dim=8)
    load_chunks(chunks, emb, store)
    r = load_chunks([], emb, store, prune=False)
    assert (r.upserted, r.skipped, r.pruned) == (0, 0, 0)
    assert len(store.existing_hashes()) == 1
