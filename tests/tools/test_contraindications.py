from qdrant_client import QdrantClient

from tests.knowledge.fakes import FakeEmbedder
from vet_agent.ingestion.models import Chunk, SectionType
from vet_agent.knowledge.loader import load_chunks
from vet_agent.knowledge.vector_store import QdrantVectorStore
from vet_agent.tools.contraindications import FindContraindications, FindContraindicationsInput
from vet_agent.tools.drug_index import DrugIndex
from vet_agent.tools.models import ContraindicationReport, DrugNotFound


def _chunk(section: SectionType, text: str, ordinal: int = 0) -> Chunk:
    return Chunk(
        drug_name="Metronidazole",
        section_type=section,
        species=["all"],
        book_page=873,
        text=text,
        ordinal=ordinal,
    )


def _tool() -> FindContraindications:
    store = QdrantVectorStore(QdrantClient(location=":memory:"), "vet_test")
    chunks = [
        _chunk(SectionType.CONTRAINDICATIONS, "Contraindicated in hepatic dysfunction."),
        _chunk(SectionType.DRUG_INTERACTIONS, "Cimetidine: may decrease metabolism.", 0),
        _chunk(SectionType.DRUG_INTERACTIONS, "Cyclosporine: may increase levels.", 1),
        _chunk(SectionType.DOSES, "DOGS: 25 mg/kg PO q12h."),  # must not leak into the report
    ]
    load_chunks(chunks, FakeEmbedder(dim=8), store)
    return FindContraindications(store, DrugIndex(["Metronidazole", "Cimetidine", "Carprofen"]))


def test_report_carries_both_sections_and_nothing_else():
    out = _tool()(FindContraindicationsInput(drug="metronidazole"))
    assert isinstance(out, ContraindicationReport)
    assert out.drug_name == "Metronidazole"
    assert len(out.contraindications) == 1
    assert len(out.interactions) == 2
    assert out.flagged == [] and out.unresolved_other_drugs == []


def test_other_drug_is_flagged_when_mentioned():
    out = _tool()(FindContraindicationsInput(drug="metronidazole", other_drugs=["cimetidine"]))
    assert isinstance(out, ContraindicationReport)
    assert len(out.flagged) == 1
    assert out.flagged[0].other_drug == "Cimetidine"  # canonical
    assert "Cimetidine" in out.flagged[0].passages[0].text


def test_resolved_but_unmentioned_other_drug_is_not_flagged():
    out = _tool()(FindContraindicationsInput(drug="metronidazole", other_drugs=["carprofen"]))
    assert isinstance(out, ContraindicationReport)
    assert out.flagged == []
    assert out.unresolved_other_drugs == []


def test_unresolvable_other_drug_is_surfaced_not_dropped():
    out = _tool()(FindContraindicationsInput(drug="metronidazole", other_drugs=["xyzzy"]))
    assert isinstance(out, ContraindicationReport)
    assert out.unresolved_other_drugs == ["xyzzy"]


def test_unknown_primary_drug_returns_drug_not_found():
    out = _tool()(FindContraindicationsInput(drug="xyzzy"))
    assert isinstance(out, DrugNotFound)
