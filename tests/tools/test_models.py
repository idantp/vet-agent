from decimal import Decimal

import pytest
from pydantic import ValidationError

from vet_agent.ingestion.models import SectionType
from vet_agent.knowledge.interfaces import Passage
from vet_agent.tools.models import (
    ContraindicationReport,
    DoseResult,
    DoseRule,
    DoseRuleSet,
    DrugNotFound,
    IndicationReport,
    NeedsClarification,
    NoPassagesFound,
    RetrievedPassages,
)


def _rule(**overrides) -> DoseRule:
    base = dict(
        drug_name="Metronidazole",
        species=["dog"],
        indication="giardiasis",
        mg_per_kg_low=Decimal("25"),
        route="PO",
        frequency="q12h",
        source_logical_key="metronidazole|doses|dog|0",
        book_page=873,
    )
    base.update(overrides)
    return DoseRule(**base)


def test_dose_rule_defaults_and_kind():
    rule = _rule()
    assert rule.kind == "dose_rule"
    assert rule.mg_per_kg_high is None
    assert rule.notes is None
    with pytest.raises(ValidationError):
        rule.mg_per_kg_low = Decimal("1")  # frozen: mutation must fail


def test_dose_rule_rejects_nonpositive_and_absurd_doses():
    with pytest.raises(ValidationError):
        _rule(mg_per_kg_low=Decimal("0"))
    with pytest.raises(ValidationError):
        _rule(mg_per_kg_low=Decimal("-5"))
    with pytest.raises(ValidationError):
        _rule(mg_per_kg_low=Decimal("20000"))  # > 10_000 mg/kg bound


def test_dose_rule_rejects_inverted_range():
    with pytest.raises(ValidationError):
        _rule(mg_per_kg_low=Decimal("50"), mg_per_kg_high=Decimal("25"))
    # a proper range is fine
    rule = _rule(mg_per_kg_low=Decimal("25"), mg_per_kg_high=Decimal("50"))
    assert rule.mg_per_kg_high == Decimal("50")


def test_dose_rule_set_requires_at_least_one_rule():
    with pytest.raises(ValidationError):
        DoseRuleSet(rules=[])
    assert DoseRuleSet(rules=[_rule()]).kind == "dose_rule_set"


def test_decimal_survives_json_roundtrip_exactly():
    rule = _rule(mg_per_kg_low=Decimal("0.1"))
    restored = DoseRule.model_validate_json(rule.model_dump_json())
    assert restored.mg_per_kg_low == Decimal("0.1")  # not 0.1000000000000000055...


def test_result_unions_have_distinct_kind_discriminators():
    passage = Passage(
        drug_name="Metronidazole",
        section_type=SectionType.DOSES,
        species=["dog"],
        book_page=873,
        text="DOGS: 25 mg/kg PO q12h",
        logical_key="metronidazole|doses|dog|0",
    )
    kinds = {
        NeedsClarification(reason="x").kind,
        DrugNotFound(query="x").kind,
        NoPassagesFound(query="x", filters={}).kind,
        RetrievedPassages(drug_name=None, passages=[passage]).kind,
        DoseRuleSet(rules=[_rule()]).kind,
        _rule().kind,
        DoseResult(
            drug_name="Metronidazole",
            species=["dog"],
            indication="giardiasis",
            weight_kg=Decimal("12"),
            dose_mg_low=Decimal("300"),
            route="PO",
            frequency="q12h",
            rule=_rule(),
        ).kind,
        ContraindicationReport(
            drug_name="Metronidazole",
            contraindications=[],
            interactions=[],
            flagged=[],
            unresolved_other_drugs=[],
        ).kind,
        IndicationReport(drug_name="x", species=None, passages=[]).kind,
    }
    assert len(kinds) == 9  # every kind-bearing model is distinguishable
