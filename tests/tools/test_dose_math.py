from decimal import Decimal

import pytest
from pydantic import ValidationError

from vet_agent.tools.dose_math import CalculateDoseInput, calculate_dose
from vet_agent.tools.models import DoseRule


def _rule(low="25", high=None, **overrides) -> DoseRule:
    base = dict(
        drug_name="Metronidazole",
        species=["dog"],
        indication="giardiasis",
        mg_per_kg_low=Decimal(low),
        mg_per_kg_high=Decimal(high) if high is not None else None,
        route="PO",
        frequency="q12h",
        source_logical_key="metronidazole|doses|dog|0",
        book_page=873,
    )
    base.update(overrides)
    return DoseRule(**base)


def test_single_dose_worked_example():
    # The spec's worked example: 25 mg/kg x 12 kg dog = 300 mg.
    result = calculate_dose(CalculateDoseInput(weight=Decimal("12"), rule=_rule("25")))
    assert result.kind == "dose_result"
    assert result.dose_mg_low == Decimal("300")
    assert result.dose_mg_high is None
    assert result.weight_kg == Decimal("12")


def test_range_dose_computes_both_ends():
    result = calculate_dose(CalculateDoseInput(weight=Decimal("12"), rule=_rule("25", high="50")))
    assert result.dose_mg_low == Decimal("300")
    assert result.dose_mg_high == Decimal("600")


def test_pounds_convert_exactly():
    # 1 lb = 0.45359237 kg exactly; 22 lb x 10 mg/kg = 99.79... mg, exact Decimal.
    result = calculate_dose(
        CalculateDoseInput(weight=Decimal("22"), weight_unit="lb", rule=_rule("10"))
    )
    assert result.weight_kg == Decimal("22") * Decimal("0.45359237")
    assert result.dose_mg_low == Decimal("22") * Decimal("0.45359237") * Decimal("10")


def test_microdose_stays_exact():
    # dexmedetomidine-scale dosing: 0.0025 mg/kg x 4.2 kg cat.
    result = calculate_dose(
        CalculateDoseInput(weight=Decimal("4.2"), rule=_rule("0.0025", species=["cat"]))
    )
    assert result.dose_mg_low == Decimal("0.01050")


def test_decimal_exactness_where_floats_fail():
    # 0.1 x 3 == 0.3 exactly in Decimal (0.30000000000000004 in float).
    result = calculate_dose(CalculateDoseInput(weight=Decimal("3"), rule=_rule("0.1")))
    assert result.dose_mg_low == Decimal("0.3")


def test_result_carries_rule_and_citation_provenance():
    rule = _rule("25")
    result = calculate_dose(CalculateDoseInput(weight=Decimal("12"), rule=rule))
    assert result.rule == rule
    assert result.rule.source_logical_key == "metronidazole|doses|dog|0"
    assert (result.drug_name, result.species, result.indication) == (
        "Metronidazole",
        ["dog"],
        "giardiasis",
    )
    assert (result.route, result.frequency) == ("PO", "q12h")


def test_weight_bounds_reject_nonsense():
    for bad in (Decimal("0"), Decimal("-4"), Decimal("5001")):
        with pytest.raises(ValidationError):
            CalculateDoseInput(weight=bad, rule=_rule("25"))


def test_weight_unit_is_a_closed_enum():
    with pytest.raises(ValidationError):
        CalculateDoseInput(weight=Decimal("12"), weight_unit="stone", rule=_rule("25"))
