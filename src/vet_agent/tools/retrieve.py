from pydantic import BaseModel, Field, field_validator

from vet_agent.ingestion.models import SectionType
from vet_agent.ingestion.species import canonical_species
from vet_agent.knowledge.retrieval import Retriever
from vet_agent.tools.drug_index import DrugIndex
from vet_agent.tools.models import DrugNotFound, NoPassagesFound, RetrievedPassages


class RetrieveMonographInput(BaseModel):
    query: str
    drug: str | None = None
    section: SectionType | None = None
    species: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


class RetrieveMonograph:
    """Filtered semantic retrieval with drug-name resolution and cited passages."""

    def __init__(self, retriever: Retriever, drugs: DrugIndex, *, rerank: bool = False) -> None:
        self._retriever = retriever
        self._drugs = drugs
        self._rerank = rerank

    def __call__(
        self, inp: RetrieveMonographInput
    ) -> RetrievedPassages | DrugNotFound | NoPassagesFound:
        canonical: str | None = None
        resolved_from: str | None = None
        if inp.drug is not None:
            resolved = self._drugs.resolve(inp.drug)
            if isinstance(resolved, DrugNotFound):
                return resolved
            canonical = resolved.canonical
            resolved_from = None if resolved.exact else inp.drug

        species: str | None = None
        if inp.species is not None:
            # Unrecognized species pass through lowercased: the filter then matches
            # nothing, which surfaces legibly as NoPassagesFound rather than an error.
            species = canonical_species(inp.species) or inp.species.strip().lower()

        hits = self._retriever.retrieve(
            inp.query,
            drug=canonical,
            section=inp.section,
            species=species,
            top_k=inp.top_k,
            rerank=self._rerank,
        )
        if not hits:
            filters: dict[str, str] = {}
            if canonical is not None:
                filters["drug"] = canonical
            if inp.section is not None:
                filters["section"] = inp.section.value
            if species is not None:
                filters["species"] = species
            return NoPassagesFound(query=inp.query, filters=filters)
        return RetrievedPassages(drug_name=canonical, resolved_from=resolved_from, passages=hits)
