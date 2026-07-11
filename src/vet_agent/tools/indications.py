from pydantic import BaseModel

from vet_agent.ingestion.models import SectionType
from vet_agent.ingestion.species import canonical_species
from vet_agent.knowledge.interfaces import VectorStore
from vet_agent.tools.drug_index import DrugIndex
from vet_agent.tools.models import DrugNotFound, IndicationReport


class ListIndicationsInput(BaseModel):
    drug: str
    species: str | None = None


class ListIndications:
    """The whole Uses/Indications section; species is a soft ordering signal."""

    def __init__(self, store: VectorStore, drugs: DrugIndex) -> None:
        self._store = store
        self._drugs = drugs

    def __call__(self, inp: ListIndicationsInput) -> IndicationReport | DrugNotFound:
        resolved = self._drugs.resolve(inp.drug)
        if isinstance(resolved, DrugNotFound):
            return resolved

        passages = self._store.fetch_section(resolved.canonical, SectionType.INDICATIONS)

        species: str | None = None
        if inp.species is not None:
            species = canonical_species(inp.species) or inp.species.strip().lower()
            sp = species
            # Prose species tags are best-effort mentions - reorder, never exclude
            # (sorted() is stable, so within each bucket logical_key order is kept).
            passages = sorted(
                passages,
                key=lambda p: 0 if (sp in p.species or "all" in p.species) else 1,
            )

        return IndicationReport(drug_name=resolved.canonical, species=species, passages=passages)
