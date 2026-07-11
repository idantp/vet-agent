import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient

from tests.knowledge.fakes import FakeEmbedder
from vet_agent.ingestion.models import Chunk, SectionType
from vet_agent.knowledge.loader import load_chunks
from vet_agent.knowledge.retrieval import Retriever
from vet_agent.knowledge.vector_store import QdrantVectorStore
from vet_agent.tools.drug_index import DrugIndex
from vet_agent.tools.models import DrugNotFound, NoPassagesFound, RetrievedPassages
from vet_agent.tools.retrieve import RetrieveMonograph, RetrieveMonographInput


def _tool() -> RetrieveMonograph:
    store = QdrantVectorStore(QdrantClient(location=":memory:"), "vet_test")
    emb = FakeEmbedder(dim=8)
    chunks = [
        Chunk(
            drug_name="Metronidazole",
            section_type=SectionType.DOSES,
            species=["dog"],
            book_page=873,
            text="dog dose text",
            ordinal=0,
        ),
        Chunk(
            drug_name="Metronidazole",
            section_type=SectionType.DOSES,
            species=["cat"],
            book_page=873,
            text="cat dose text",
            ordinal=0,
        ),
    ]
    load_chunks(chunks, emb, store)
    return RetrieveMonograph(Retriever(emb, store), DrugIndex(["Metronidazole"]))


def test_happy_path_resolves_drug_and_filters_species():
    out = _tool()(
        RetrieveMonographInput(
            query="dose", drug="metronidazole", section=SectionType.DOSES, species="Dogs"
        )
    )
    assert isinstance(out, RetrievedPassages)
    assert out.drug_name == "Metronidazole"  # canonical echoed back
    assert {p.species[0] for p in out.passages} == {"dog"}  # "Dogs" canonicalized + filtered


def test_unknown_drug_short_circuits_to_drug_not_found():
    out = _tool()(RetrieveMonographInput(query="dose", drug="xyzzyplugh"))
    assert isinstance(out, DrugNotFound)


def test_zero_hits_echo_the_filters():
    out = _tool()(
        RetrieveMonographInput(
            query="dose",
            drug="metronidazole",
            section=SectionType.DOSES,
            species="ferret",  # loaded corpus has no ferret chunk
        )
    )
    assert isinstance(out, NoPassagesFound)
    assert out.filters == {
        "drug": "Metronidazole",
        "section": "doses",
        "species": "ferret",
    }


def test_unrecognized_species_passes_through_lowercased():
    out = _tool()(RetrieveMonographInput(query="dose", species="Axolotl"))
    assert isinstance(out, NoPassagesFound)
    assert out.filters == {"species": "axolotl"}


def test_blank_query_is_rejected():
    with pytest.raises(ValidationError):
        RetrieveMonographInput(query="   ")


def test_top_k_bounds():
    with pytest.raises(ValidationError):
        RetrieveMonographInput(query="q", top_k=0)
    with pytest.raises(ValidationError):
        RetrieveMonographInput(query="q", top_k=21)


def test_fuzzy_resolution_is_surfaced_not_silent():
    out = _tool()(RetrieveMonographInput(query="dose", drug="metronidazol"))  # typo
    assert isinstance(out, RetrievedPassages)
    assert out.drug_name == "Metronidazole"
    assert out.resolved_from == "metronidazol"


def test_exact_resolution_has_no_resolved_from():
    out = _tool()(RetrieveMonographInput(query="dose", drug="Metronidazole"))
    assert isinstance(out, RetrievedPassages)
    assert out.resolved_from is None
