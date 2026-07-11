from pydantic import BaseModel, Field

from vet_agent.ingestion.models import SectionType
from vet_agent.knowledge.interfaces import VectorStore
from vet_agent.tools.drug_index import DrugIndex
from vet_agent.tools.models import (
    ContraindicationReport,
    DrugNotFound,
    FlaggedInteraction,
)


class FindContraindicationsInput(BaseModel):
    drug: str
    other_drugs: list[str] = Field(default_factory=list)


class FindContraindications:
    """Complete contraindications + drug-interactions report via exact section fetch."""

    def __init__(self, store: VectorStore, drugs: DrugIndex) -> None:
        self._store = store
        self._drugs = drugs

    def __call__(self, inp: FindContraindicationsInput) -> ContraindicationReport | DrugNotFound:
        resolved = self._drugs.resolve(inp.drug)
        if isinstance(resolved, DrugNotFound):
            return resolved

        contraindications = self._store.fetch_section(
            resolved.canonical, SectionType.CONTRAINDICATIONS
        )
        interactions = self._store.fetch_section(resolved.canonical, SectionType.DRUG_INTERACTIONS)

        flagged: list[FlaggedInteraction] = []
        unresolved: list[str] = []
        for other in inp.other_drugs:
            other_resolved = self._drugs.resolve(other)
            if isinstance(other_resolved, DrugNotFound):
                unresolved.append(other)
                continue
            # Known limitation (spec §9): canonical-name match only; brand names in
            # prose won't match. Plumb's interaction lists use generic names.
            needle = other_resolved.canonical.lower()
            mentions = [p for p in contraindications + interactions if needle in p.text.lower()]
            if mentions:
                flagged.append(
                    FlaggedInteraction(other_drug=other_resolved.canonical, passages=mentions)
                )

        return ContraindicationReport(
            drug_name=resolved.canonical,
            resolved_from=None if resolved.exact else inp.drug,
            contraindications=contraindications,
            interactions=interactions,
            flagged=flagged,
            unresolved_other_drugs=unresolved,
        )
