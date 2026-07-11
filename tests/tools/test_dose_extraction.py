from decimal import Decimal

from tests.tools.fakes import FakeRegimenExtractor
from vet_agent.ingestion.models import SectionType
from vet_agent.knowledge.interfaces import Passage
from vet_agent.tools.dose_extraction import (
    ExtractDoseRule,
    ExtractDoseRuleInput,
    ExtractedRegimen,
)
from vet_agent.tools.models import DoseRule, DoseRuleSet, NeedsClarification

# Condensed from the real Metronidazole dog chunk (p.873): two giardiasis regimens
# plus one for other protozoal infections.
PASSAGE_TEXT = (
    "Giardiasis (extra-label): a) 25 mg/kg PO twice daily in combination with "
    "fenbendazole 50 mg/kg PO once daily for 5 days. b) 50 mg/kg PO once daily "
    "for 5 to 7 days. Other protozoal infections (extra-label): 25 mg/kg PO "
    "every 12 hours for 8 days."
)


def _passage(text: str = PASSAGE_TEXT) -> Passage:
    return Passage(
        drug_name="Metronidazole",
        section_type=SectionType.DOSES,
        species=["dog"],
        book_page=873,
        text=text,
        logical_key="metronidazole|doses|dog|0",
    )


def _reg(indication: str, low: str, high: str | None = None, **overrides) -> ExtractedRegimen:
    base = dict(
        indication=indication,
        mg_per_kg_low=Decimal(low),
        mg_per_kg_high=Decimal(high) if high is not None else None,
        route="PO",
        frequency="q12h",
        notes=None,
    )
    base.update(overrides)
    return ExtractedRegimen(**base)


GIARDIA_A = _reg("giardiasis, with fenbendazole", "25")
GIARDIA_B = _reg("giardiasis, monotherapy", "50", frequency="once daily for 5 to 7 days")
PROTOZOAL = _reg("other protozoal infections", "25", frequency="q12h for 8 days")


def _tool(*regimens: ExtractedRegimen) -> ExtractDoseRule:
    return ExtractDoseRule(FakeRegimenExtractor(list(regimens)))


def test_single_regimen_returns_dose_rule_with_citation():
    out = _tool(PROTOZOAL)(ExtractDoseRuleInput(passage=_passage()))
    assert isinstance(out, DoseRule)
    assert out.mg_per_kg_low == Decimal("25")
    assert out.source_logical_key == "metronidazole|doses|dog|0"
    assert out.book_page == 873
    assert out.drug_name == "Metronidazole"
    assert out.species == ["dog"]


def test_multiple_regimens_without_indication_needs_clarification():
    out = _tool(GIARDIA_A, GIARDIA_B, PROTOZOAL)(ExtractDoseRuleInput(passage=_passage()))
    assert isinstance(out, NeedsClarification)
    assert len(out.candidates) == 3  # structured options the agent can relay


def test_indication_narrowing_to_exactly_one_returns_it():
    out = _tool(GIARDIA_A, PROTOZOAL)(
        ExtractDoseRuleInput(passage=_passage(), indication="giardia")
    )
    assert isinstance(out, DoseRule)
    assert out.indication == "giardiasis, with fenbendazole"


def test_indication_matching_is_substring_both_ways():
    # query "monotherapy" is a substring of regimen indication "giardiasis, monotherapy"
    out = _tool(GIARDIA_A, GIARDIA_B)(
        ExtractDoseRuleInput(passage=_passage(), indication="monotherapy")
    )
    assert isinstance(out, DoseRule)
    assert out.mg_per_kg_low == Decimal("50")


def test_indication_matching_multiple_needs_clarification_with_candidates():
    out = _tool(GIARDIA_A, GIARDIA_B, PROTOZOAL)(
        ExtractDoseRuleInput(passage=_passage(), indication="giardia")
    )
    assert isinstance(out, NeedsClarification)
    assert {c.mg_per_kg_low for c in out.candidates} == {Decimal("25"), Decimal("50")}


def test_indication_matching_zero_offers_all_grounded_candidates():
    out = _tool(GIARDIA_A)(ExtractDoseRuleInput(passage=_passage(), indication="pyoderma"))
    assert isinstance(out, NeedsClarification)
    assert len(out.candidates) == 1


def test_all_regimens_mode_returns_every_grounded_rule():
    out = _tool(GIARDIA_A, GIARDIA_B, PROTOZOAL)(
        ExtractDoseRuleInput(passage=_passage(), all_regimens=True)
    )
    assert isinstance(out, DoseRuleSet)
    assert len(out.rules) == 3


def test_grounding_discards_hallucinated_numbers_in_every_mode():
    hallucinated = _reg("giardiasis", "30")  # 30 appears nowhere in the passage
    out = _tool(GIARDIA_A, hallucinated)(
        ExtractDoseRuleInput(passage=_passage(), all_regimens=True)
    )
    assert isinstance(out, DoseRuleSet)
    assert [r.mg_per_kg_low for r in out.rules] == [Decimal("25")]

    out2 = _tool(hallucinated)(ExtractDoseRuleInput(passage=_passage()))
    assert isinstance(out2, NeedsClarification)
    assert out2.candidates == []
    assert "grounding" in out2.reason


def test_grounding_is_number_boundary_aware():
    # "25" must not be grounded by "125" or "254".
    reg = _reg("x", "25")
    out = _tool(reg)(ExtractDoseRuleInput(passage=_passage(text="give 125 mg/kg or 254 mg")))
    assert isinstance(out, NeedsClarification)


def test_grounding_accepts_trailing_zero_variants():
    # LLM returns "50.0" for a passage that says "50 mg/kg" - same number, grounded.
    reg = _reg("x", "50.0")
    out = _tool(reg)(ExtractDoseRuleInput(passage=_passage(text="give 50 mg/kg PO")))
    assert isinstance(out, DoseRule)


def test_inverted_range_is_discarded_as_invalid():
    bad = _reg("x", "50", high="25")  # high < low: transcription error, never a rule
    out = _tool(bad)(ExtractDoseRuleInput(passage=_passage(text="25 to 50 mg/kg")))
    assert isinstance(out, NeedsClarification)


def test_empty_extraction_needs_clarification():
    out = _tool()(ExtractDoseRuleInput(passage=_passage()))
    assert isinstance(out, NeedsClarification)
    assert out.candidates == []
