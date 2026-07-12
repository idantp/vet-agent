from decimal import Decimal
from typing import Literal

import pint
from pydantic import BaseModel, Field

from vet_agent.tools.models import DoseResult, DoseRule

# Decimal-native unit registry: conversion factors parse as Decimal, so lb -> kg
# (1 lb = 0.45359237 kg, an exact defined constant) introduces no float error.
_UREG: pint.UnitRegistry[Decimal] = pint.UnitRegistry(non_int_type=Decimal)


class CalculateDoseInput(BaseModel):
    """Validated inputs for the fixed dose computation. No LLM-derived strings."""

    weight: Decimal = Field(gt=0, le=5000)  # sane physiological bound, cattle-inclusive
    weight_unit: Literal["kg", "lb"] = "kg"
    rule: DoseRule


def calculate_dose(inp: CalculateDoseInput) -> DoseResult:
    """Fixed arithmetic (weight_kg x mg_per_kg) on a validated DoseRule.

    Pure Python: no LLM, no eval of any kind, no rounding (presentation is the
    agent's job). Exact Decimal end-to-end; pint owns the unit conversion.
    """
    weight_kg: Decimal = _UREG.Quantity(inp.weight, inp.weight_unit).to("kg").magnitude
    dose_low = weight_kg * inp.rule.mg_per_kg_low
    dose_high = weight_kg * inp.rule.mg_per_kg_high if inp.rule.mg_per_kg_high is not None else None
    return DoseResult(
        drug_name=inp.rule.drug_name,
        species=inp.rule.species,
        indication=inp.rule.indication,
        weight_kg=weight_kg,
        dose_mg_low=dose_low,
        dose_mg_high=dose_high,
        route=inp.rule.route,
        frequency=inp.rule.frequency,
        notes=inp.rule.notes,
        rule=inp.rule,
    )
