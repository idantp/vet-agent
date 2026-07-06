# NOTE: qdrant-client :memory: mode requires point ids to be unsigned ints or valid UUIDs.
# Short strings like "p1", "dog", "cat" are rejected with:
#   ValueError: Point id p1 is not a valid UUID
# Minimal fix per task spec: use UUID strings in tests. The production loader (Task 2.7)
# will pass real UUIDs, so this is consistent. See CONCERNS in the task report.
from qdrant_client import QdrantClient

from vet_agent.ingestion.models import SectionType
from vet_agent.knowledge.interfaces import PointPayload
from vet_agent.knowledge.vector_store import QdrantVectorStore, collection_name

# Stable UUIDs for deterministic tests
_UUID_P1 = "00000000-0000-0000-0000-000000000001"
_UUID_DOG = "00000000-0000-0000-0000-000000000002"
_UUID_CAT = "00000000-0000-0000-0000-000000000003"
_UUID_BOTH = "00000000-0000-0000-0000-000000000004"


def _store() -> QdrantVectorStore:
    return QdrantVectorStore(QdrantClient(location=":memory:"), "vet_test")


def _point(
    pid: str, *, species, vector, drug="Metronidazole", text="t", ch="h", key: str | None = None
) -> PointPayload:
    return PointPayload(
        point_id=pid,
        vector=vector,
        drug_name=drug,
        section_type=SectionType.DOSES,
        species=species,
        book_page=873,
        text=text,
        logical_key=key if key is not None else pid,
        content_hash=ch,
    )


def test_collection_name_is_model_suffixed_and_sanitized():
    assert collection_name("vet_chunks", "medembed-base") == "vet_chunks__medembed_base"
    assert collection_name("vet_chunks", "bge-base") == "vet_chunks__bge_base"


def test_upsert_then_existing_hashes_roundtrip():
    store = _store()
    store.ensure_collection(dim=2)
    store.upsert([_point(_UUID_P1, species=["dog"], vector=[1.0, 0.0], ch="h1", key="p1")])
    assert store.existing_hashes() == {_UUID_P1: "h1"}


def test_existing_hashes_empty_when_no_collection():
    assert _store().existing_hashes() == {}


def test_upsert_splits_large_input_across_multiple_client_calls():
    # Regression: a single upsert of the whole corpus produced a ~262 MB request that
    # Qdrant rejects (32 MB limit). Points must be sent in bounded per-request batches.
    store = _store()
    store.ensure_collection(dim=2)

    calls = 0
    real_upsert = store._client.upsert

    def counting_upsert(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_upsert(*args, **kwargs)

    store._client.upsert = counting_upsert  # type: ignore[method-assign]

    n = QdrantVectorStore._UPSERT_BATCH * 2 + 1
    points = [
        _point(
            f"00000000-0000-0000-0000-{i:012d}",
            species=["dog"],
            vector=[1.0, 0.0],
            ch=f"h{i}",
            key=f"p{i}",
        )
        for i in range(n)
    ]
    store.upsert(points)

    assert calls == 3  # ceil(n / batch) with batch=256 -> 3 requests
    assert len(store.existing_hashes()) == n  # every point landed


def test_search_respects_species_filter_including_list_membership():
    store = _store()
    store.ensure_collection(dim=2)
    store.upsert(
        [
            _point(_UUID_DOG, species=["dog"], vector=[1.0, 0.0], text="dog dose", key="dog"),
            _point(_UUID_CAT, species=["cat"], vector=[1.0, 0.0], text="cat dose", key="cat"),
            _point(
                _UUID_BOTH,
                species=["cat", "dog"],
                vector=[1.0, 0.0],
                text="shared dose",
                key="both",
            ),
        ]
    )
    hits = store.search([1.0, 0.0], species="dog", top_k=10)
    keys = {h.logical_key for h in hits}
    assert keys == {"dog", "both"}  # "cat" excluded; list-valued "both" matches
    assert all(h.section_type is SectionType.DOSES for h in hits)


def test_delete_removes_points():
    store = _store()
    store.ensure_collection(dim=2)
    store.upsert([_point(_UUID_P1, species=["dog"], vector=[1.0, 0.0], key="p1")])
    store.delete([_UUID_P1])
    assert store.existing_hashes() == {}
