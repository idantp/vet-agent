from qdrant_client import QdrantClient

from tests.knowledge.fakes import FakeEmbedder
from vet_agent.ingestion.models import Chunk, SectionType
from vet_agent.knowledge.loader import load_chunks
from vet_agent.knowledge.vector_store import QdrantVectorStore
from vet_agent.tools.drug_index import DrugIndex
from vet_agent.tools.indications import ListIndications, ListIndicationsInput
from vet_agent.tools.models import DrugNotFound, IndicationReport


def _tool() -> ListIndications:
    store = QdrantVectorStore(QdrantClient(location=":memory:"), "vet_test")
    chunks = [
        Chunk(
            drug_name="Metronidazole",
            section_type=SectionType.INDICATIONS,
            species=["horse"],
            book_page=873,
            text="equine indications",
            ordinal=0,
        ),
        Chunk(
            drug_name="Metronidazole",
            section_type=SectionType.INDICATIONS,
            species=["all"],
            book_page=873,
            text="general indications",
            ordinal=1,
        ),
        Chunk(
            drug_name="Metronidazole",
            section_type=SectionType.INDICATIONS,
            species=["cat", "dog"],
            book_page=873,
            text="small animal indications",
            ordinal=2,
        ),
    ]
    load_chunks(chunks, FakeEmbedder(dim=8), store)
    return ListIndications(store, DrugIndex(["Metronidazole"]))


def test_all_passages_returned_without_species():
    out = _tool()(ListIndicationsInput(drug="metronidazole"))
    assert isinstance(out, IndicationReport)
    assert out.species is None
    assert len(out.passages) == 3


def test_species_reorders_but_never_excludes():
    out = _tool()(ListIndicationsInput(drug="metronidazole", species="Dogs"))
    assert isinstance(out, IndicationReport)
    assert out.species == "dog"
    assert len(out.passages) == 3  # soft signal: nothing dropped
    # dog-tagged and 'all'-tagged sort before horse-only
    assert {out.passages[0].text, out.passages[1].text} == {
        "general indications",
        "small animal indications",
    }
    assert out.passages[2].text == "equine indications"


def test_unknown_drug_returns_drug_not_found():
    out = _tool()(ListIndicationsInput(drug="xyzzy"))
    assert isinstance(out, DrugNotFound)


def test_fuzzy_drug_is_surfaced():
    out = _tool()(ListIndicationsInput(drug="metronidazol"))
    assert isinstance(out, IndicationReport)
    assert out.resolved_from == "metronidazol"
