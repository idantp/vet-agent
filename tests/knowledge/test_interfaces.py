from tests.knowledge.fakes import FakeEmbedder
from vet_agent.ingestion.models import SectionType
from vet_agent.knowledge.interfaces import Embedder, Passage, PointPayload


def test_passage_carries_citation_fields():
    p = Passage(
        drug_name="Metronidazole",
        section_type=SectionType.DOSES,
        species=["dog"],
        book_page=873,
        text="DOGS: 25 mg/kg PO q12h",
        logical_key="metronidazole|doses|dog|0",
    )
    assert p.score is None
    assert p.section_type is SectionType.DOSES


def test_point_payload_holds_vector_and_hash():
    pp = PointPayload(
        point_id="abc",
        vector=[0.1, 0.2],
        drug_name="X",
        section_type=SectionType.STORAGE,
        species=["all"],
        book_page=1,
        text="Store cool.",
        logical_key="x|storage|all|0",
        content_hash="deadbeef",
    )
    assert pp.vector == [0.1, 0.2]


def test_fake_embedder_satisfies_protocol_and_is_deterministic():
    emb = FakeEmbedder(dim=8)
    assert isinstance(emb, Embedder)  # runtime_checkable structural check
    assert emb.dim == 8
    assert emb.embed_query("giardia") == emb.embed_query("giardia")
    assert emb.embed_query("a") != emb.embed_query("b")
    vec = emb.embed_query("giardia")
    assert abs(sum(v * v for v in vec) - 1.0) < 1e-9  # unit length
