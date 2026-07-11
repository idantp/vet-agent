import re
from decimal import Decimal
from typing import Protocol

from pydantic import BaseModel, Field

from vet_agent.knowledge.interfaces import Passage
from vet_agent.tools.models import MAX_MG_PER_KG, DoseRule, DoseRuleSet, NeedsClarification


class ExtractedRegimen(BaseModel):
    """One regimen as transcribed by the LLM - untrusted until grounded.

    Dose bounds mirror DoseRule's so an absurd transcription fails here, at the
    extractor's parse boundary, rather than mid-tool when building the rule.
    """

    indication: str
    mg_per_kg_low: Decimal = Field(gt=0, le=MAX_MG_PER_KG)
    mg_per_kg_high: Decimal | None = Field(default=None, gt=0, le=MAX_MG_PER_KG)
    route: str
    frequency: str
    notes: str | None = None


class RegimenExtractor(Protocol):
    """The LLM seam: transcribe a Doses passage into raw candidate regimens."""

    def extract_regimens(self, passage_text: str) -> list[ExtractedRegimen]: ...


class ExtractDoseRuleInput(BaseModel):
    passage: Passage
    indication: str | None = None
    all_regimens: bool = False  # list-all mode: return every grounded regimen


def _decimal_forms(value: Decimal) -> set[str]:
    """String forms a number may take in the text ("50", "50.0" -> both match)."""
    normalized = value.normalize()
    # normalize() turns 50 into 5E+1; undo. (exponent is int for finite values --
    # the isinstance guard is for mypy, whose stubs allow NaN/Inf sentinels.)
    exponent = normalized.as_tuple().exponent
    if isinstance(exponent, int) and exponent > 0:
        normalized = normalized.quantize(Decimal(1))
    return {str(value), str(normalized)}


def _number_in_text(value: Decimal, text: str) -> bool:
    """True if the dose number appears verbatim (digit-boundary-aware) in the text."""
    return any(
        re.search(rf"(?<![\d.]){re.escape(form)}(?![\d.])", text) for form in _decimal_forms(value)
    )


def _is_valid(regimen: ExtractedRegimen, passage_text: str) -> bool:
    """Grounding + range sanity. A regimen failing either is discarded, never used."""
    if regimen.mg_per_kg_high is not None and regimen.mg_per_kg_high <= regimen.mg_per_kg_low:
        return False
    if not _number_in_text(regimen.mg_per_kg_low, passage_text):
        return False
    if regimen.mg_per_kg_high is not None and not _number_in_text(
        regimen.mg_per_kg_high, passage_text
    ):
        return False
    return True


def _to_rule(regimen: ExtractedRegimen, passage: Passage) -> DoseRule:
    return DoseRule(
        drug_name=passage.drug_name,
        species=passage.species,
        indication=regimen.indication,
        mg_per_kg_low=regimen.mg_per_kg_low,
        mg_per_kg_high=regimen.mg_per_kg_high,
        route=regimen.route,
        frequency=regimen.frequency,
        notes=regimen.notes,
        source_logical_key=passage.logical_key,
        book_page=passage.book_page,
    )


def _indication_matches(query: str, indication: str) -> bool:
    a, b = query.strip().lower(), indication.strip().lower()
    return bool(a) and (a in b or b in a)


class ExtractDoseRule:
    """Turn a cited Doses passage into structured dose rules.

    The injected extractor (the only LLM call in the tools layer) merely
    transcribes; this class grounds every dose number against the passage,
    then selects deterministically.
    """

    def __init__(self, extractor: RegimenExtractor) -> None:
        self._extractor = extractor

    def __call__(self, inp: ExtractDoseRuleInput) -> DoseRule | DoseRuleSet | NeedsClarification:
        raw = self._extractor.extract_regimens(inp.passage.text)
        valid = [r for r in raw if _is_valid(r, inp.passage.text)]
        discarded = len(raw) - len(valid)

        if not valid:
            if discarded:
                reason = (
                    f"{discarded} extracted regimen(s) failed the grounding check "
                    "(dose numbers not found verbatim in the cited passage)"
                )
            else:
                reason = "no dose regimens could be extracted from the passage"
            return NeedsClarification(reason=reason)

        rules = [_to_rule(r, inp.passage) for r in valid]

        if inp.all_regimens:
            return DoseRuleSet(rules=rules)

        if inp.indication is not None:
            matches = [r for r in rules if _indication_matches(inp.indication, r.indication)]
            if len(matches) == 1:
                return matches[0]
            if not matches:
                return NeedsClarification(
                    reason=f"no regimen matches indication '{inp.indication}'",
                    candidates=rules,
                )
            return NeedsClarification(
                reason=f"multiple regimens match indication '{inp.indication}'",
                candidates=matches,
            )

        if len(rules) == 1:
            return rules[0]
        return NeedsClarification(
            reason="multiple regimens in passage; provide an indication or set all_regimens",
            candidates=rules,
        )
