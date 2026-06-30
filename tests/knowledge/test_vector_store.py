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
    assert collection_name("vet_chunks", "qwen3-0.6b") == "vet_chunks__qwen3_0_6b"
    assert collection_name("vet_chunks", "medembed-base") == "vet_chunks__medembed_base"


def test_upsert_then_existing_hashes_roundtrip():
    store = _store()
    store.ensure_collection(dim=2)
    store.upsert([_point(_UUID_P1, species=["dog"], vector=[1.0, 0.0], ch="h1", key="p1")])
    assert store.existing_hashes() == {_UUID_P1: "h1"}


def test_existing_hashes_empty_when_no_collection():
    assert _store().existing_hashes() == {}


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
