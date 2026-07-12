from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vet_agent.knowledge.interfaces import Passage

# Loose on purpose: activated charcoal is dosed ~1-4 g/kg, so a tight bound would
# reject real regimens; the bound only guards against absurd transcriptions.
MAX_MG_PER_KG = Decimal(10_000)


class DoseRule(BaseModel):
    """One dosing regimen, always traceable to a cited Doses passage."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["dose_rule"] = "dose_rule"
    drug_name: str
    # Mirrors the source chunk, e.g. ["dog"] or ["cat", "dog"].
    species: list[str] = Field(min_length=1)
    indication: str
    mg_per_kg_low: Decimal = Field(gt=0, le=MAX_MG_PER_KG)
    mg_per_kg_high: Decimal | None = Field(default=None, gt=0, le=MAX_MG_PER_KG)
    route: str  # verbatim from the text: "PO", "IV over 30 min", ...
    frequency: str  # verbatim: "q12h", "once daily for 5 days", ...
    notes: str | None = None  # combination therapy, duration caveats
    source_logical_key: str
    book_page: int

    @model_validator(mode="after")
    def _range_is_ordered(self) -> "DoseRule":
        if self.mg_per_kg_high is not None and self.mg_per_kg_high <= self.mg_per_kg_low:
            raise ValueError("mg_per_kg_high must be greater than mg_per_kg_low")
        return self


class DoseRuleSet(BaseModel):
    """Every grounded regimen in a passage (extract_dose_rule list-all mode)."""

    kind: Literal["dose_rule_set"] = "dose_rule_set"
    rules: list[DoseRule] = Field(min_length=1)


class NeedsClarification(BaseModel):
    """Ambiguous outcome; candidates are structured options the agent can relay."""

    kind: Literal["needs_clarification"] = "needs_clarification"
    reason: str
    candidates: list[DoseRule] = Field(default_factory=list)


class DrugNotFound(BaseModel):
    kind: Literal["drug_not_found"] = "drug_not_found"
    query: str
    suggestions: list[str] = Field(default_factory=list)


class NoPassagesFound(BaseModel):
    """Zero retrieval hits; echoes the filters so the agent sees why."""

    kind: Literal["no_passages_found"] = "no_passages_found"
    query: str
    filters: dict[str, str]


class RetrievedPassages(BaseModel):
    kind: Literal["retrieved_passages"] = "retrieved_passages"
    drug_name: str | None  # canonical, post-resolution; None when no drug filter
    resolved_from: str | None = None  # original query when drug resolution was fuzzy; None if exact
    passages: list[Passage]


class DoseResult(BaseModel):
    """A computed dose; embeds the full rule for provenance (rule -> logical_key -> page)."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["dose_result"] = "dose_result"
    drug_name: str
    species: list[str] = Field(min_length=1)
    indication: str
    weight_kg: Decimal
    dose_mg_low: Decimal
    dose_mg_high: Decimal | None = None
    route: str
    frequency: str
    notes: str | None = None
    rule: DoseRule


class FlaggedInteraction(BaseModel):
    other_drug: str  # canonical
    passages: list[Passage]  # interaction/contraindication passages mentioning it


class ContraindicationReport(BaseModel):
    kind: Literal["contraindication_report"] = "contraindication_report"
    drug_name: str
    resolved_from: str | None = None  # original query when drug resolution was fuzzy; None if exact
    contraindications: list[Passage]
    interactions: list[Passage]
    flagged: list[FlaggedInteraction] = Field(default_factory=list)
    unresolved_other_drugs: list[str] = Field(default_factory=list)


class IndicationReport(BaseModel):
    kind: Literal["indication_report"] = "indication_report"
    drug_name: str
    resolved_from: str | None = None  # original query when drug resolution was fuzzy; None if exact
    species: str | None
    passages: list[Passage]  # species-matching first; never excluded (soft signal)
