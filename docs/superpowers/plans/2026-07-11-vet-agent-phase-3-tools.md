# Vet-Agent Phase 3 — Tools Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the five framework-agnostic tools (`retrieve_monograph`, `extract_dose_rule`, `calculate_dose`, `find_contraindications`, `list_indications`) with typed pydantic I/O in a new `src/vet_agent/tools/` package, plus the `VectorStore.fetch_section()` and `canonical_species()` support they need, and a `vet-agent dose` CLI demo.

**Architecture:** Result unions discriminated on a literal `kind` field (expected outcomes are return types, exceptions are for infra failures). Tools with dependencies are callable classes with constructor DI; pure tools are plain functions. The one LLM call (`extract_dose_rule`) hides behind a `RegimenExtractor` Protocol — the LLM only *transcribes* prose into candidate structures; deterministic code grounds every dose number against the cited passage, selects among regimens, and does all arithmetic (`Decimal` + `pint`, no rounding, no expression evaluation). Report tools bypass ANN entirely via an exact filtered Qdrant scroll (`fetch_section`), because HNSW gives no completeness guarantee.

**Tech Stack:** Python 3.12+, uv, pydantic v2, `pint` (new), `anthropic` (promoted dev → main), qdrant-client `:memory:` mode for tests, pytest, ruff, mypy (strict, package-scoped).

**Spec:** `docs/superpowers/specs/2026-07-11-vet-agent-phase-3-tools-design.md`. Read it first — this plan implements it exactly.

---

## File Structure

**New package `src/vet_agent/tools/`:**

- `__init__.py` — package docstring.
- `models.py` — all tool I/O models + result unions (`DoseRule`, `DoseRuleSet`, `NeedsClarification`, `DrugNotFound`, `NoPassagesFound`, `RetrievedPassages`, `DoseResult`, `FlaggedInteraction`, `ContraindicationReport`, `IndicationReport`). The contract.
- `drug_index.py` — `ResolvedDrug`, `DrugIndex` (case-insensitive + fuzzy resolution over canonical monograph names).
- `dose_math.py` — `CalculateDoseInput`, `calculate_dose` (pure `Decimal` × `pint`).
- `retrieve.py` — `RetrieveMonographInput`, `RetrieveMonograph` (wraps `knowledge.Retriever`).
- `dose_extraction.py` — `ExtractedRegimen`, `RegimenExtractor` Protocol, `ExtractDoseRuleInput`, `ExtractDoseRule`, `AnthropicRegimenExtractor`.
- `contraindications.py` — `FindContraindicationsInput`, `FindContraindications`.
- `indications.py` — `ListIndicationsInput`, `ListIndications`.

**Modified:**

- `pyproject.toml` — add `pint` + `anthropic` to main deps; drop `anthropic` from dev group.
- `src/vet_agent/ingestion/species.py` — add public `canonical_species()` (the synonym table is already there, private).
- `src/vet_agent/knowledge/interfaces.py` — add `fetch_section()` to the `VectorStore` Protocol.
- `src/vet_agent/knowledge/vector_store.py` — implement `fetch_section()` (exact filtered scroll).
- `src/vet_agent/cli/main.py` — new `dose` command (the phase demo).
- `README.md` — document the `dose` command.

**Tests (mirror layout):**

- `tests/tools/__init__.py`, `tests/tools/fakes.py` (`FakeRegimenExtractor`).
- `tests/tools/test_models.py`, `test_drug_index.py`, `test_dose_math.py`, `test_retrieve.py`, `test_dose_extraction.py`, `test_anthropic_extractor.py`, `test_contraindications.py`, `test_indications.py`.
- Extend: `tests/ingestion/test_species.py`, `tests/knowledge/test_vector_store.py`, `tests/test_cli.py`.

No config changes — `reasoning_model`, `anthropic_api_key`, `retrieval_top_k`, `rerank_enabled` already cover Phase 3.

---

## Conventions for every task

- Use `uv run pytest <path> -v` to run tests; `make check` runs ruff + mypy + the **fast** suite (`-m "not slow"`).
- ruff line-length is 100; sort imports; new CLI options need `# noqa: B008` (matching `cli/main.py`).
- pydantic v2 models; plain `def test_x():` (mypy does not type-check tests).
- All tests offline and deterministic: `FakeEmbedder`/`FakeReranker` from `tests/knowledge/fakes.py`, Qdrant `:memory:`, `FakeRegimenExtractor` for the LLM seam. No network, no API keys.
- Commit after each task with the message shown.

---

## PHASE 3 — TOOLS LAYER

### Task 3.1: Dependencies + tools package + shared I/O models

**Files:**

- Modify: `pyproject.toml` (via uv)
- Create: `src/vet_agent/tools/__init__.py`
- Create: `src/vet_agent/tools/models.py`
- Create: `tests/tools/__init__.py`
- Test: `tests/tools/test_models.py`

- [ ] **Step 1: Move `anthropic` to main deps and add `pint`**

uv resolves the latest compatible versions (honors the project's "prefer latest" rule — do not hand-pin stale floors):

```bash
uv remove --dev anthropic
uv add anthropic pint
```

Expected: `pyproject.toml` `dependencies` gains `anthropic` and `pint`; `dependency-groups.dev` no longer lists `anthropic`; `uv.lock` updates.

Note: `pint` ships type information; if `make check` later fails with `import-untyped` for `pint`, append to `pyproject.toml`:

```toml
[[tool.mypy.overrides]]
module = ["pint.*"]
ignore_missing_imports = true
```

- [ ] **Step 2: Write the failing test**

`tests/tools/__init__.py`: (empty file)

`tests/tools/test_models.py`:

```python
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
    }
    assert len(kinds) == 8  # every union member is distinguishable
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/tools/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vet_agent.tools'`.

- [ ] **Step 4: Create the package + models**

`src/vet_agent/tools/__init__.py`:

```python
"""Tools layer: five pure, framework-agnostic tools with typed pydantic I/O (Phase 3)."""
```

`src/vet_agent/tools/models.py`:

```python
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from vet_agent.knowledge.interfaces import Passage

# Loose on purpose: activated charcoal is dosed ~1-4 g/kg, so a tight bound would
# reject real regimens; the bound only guards against absurd transcriptions.
MAX_MG_PER_KG = Decimal(10_000)


class DoseRule(BaseModel):
    """One dosing regimen, always traceable to a cited Doses passage."""

    kind: Literal["dose_rule"] = "dose_rule"
    drug_name: str
    species: list[str]  # mirrors the source chunk, e.g. ["dog"] or ["cat", "dog"]
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
    passages: list[Passage]


class DoseResult(BaseModel):
    """A computed dose; embeds the full rule for provenance (rule -> logical_key -> page)."""

    kind: Literal["dose_result"] = "dose_result"
    drug_name: str
    species: list[str]
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
    contraindications: list[Passage]
    interactions: list[Passage]
    flagged: list[FlaggedInteraction] = Field(default_factory=list)
    unresolved_other_drugs: list[str] = Field(default_factory=list)


class IndicationReport(BaseModel):
    kind: Literal["indication_report"] = "indication_report"
    drug_name: str
    species: str | None
    passages: list[Passage]  # species-matching first; never excluded (soft signal)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/tools/test_models.py -v`
Expected: PASS (6 passed).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/vet_agent/tools/ tests/tools/
git commit -m "feat(tools): add tools package with typed I/O models and result unions"
```

---

### Task 3.2: Public `canonical_species()` helper

**Files:**

- Modify: `src/vet_agent/ingestion/species.py`
- Test: `tests/ingestion/test_species.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/ingestion/test_species.py`:

```python
from vet_agent.ingestion.species import canonical_species


def test_canonical_species_normalizes_case_and_plural():
    assert canonical_species("Dogs") == "dog"
    assert canonical_species("  CAT ") == "cat"
    assert canonical_species("equine") == "horse"


def test_canonical_species_rejects_unknown_and_ambiguous():
    assert canonical_species("axolotl") is None
    assert canonical_species("dogs and cats") is None  # two species -> not a single filter value
    assert canonical_species("") is None
```

(Adjust the import line to merge with the file's existing `from vet_agent.ingestion.species import ...` import.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_species.py -v`
Expected: FAIL with `ImportError: cannot import name 'canonical_species'`.

- [ ] **Step 3: Add the helper**

Append to `src/vet_agent/ingestion/species.py`:

```python
def canonical_species(text: str) -> str | None:
    """Canonicalize a single species string ("Dogs" -> "dog"); None if not exactly one."""
    tokens = _canonical_tokens(text)
    return tokens[0] if len(tokens) == 1 else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_species.py -v`
Expected: PASS (all existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/vet_agent/ingestion/species.py tests/ingestion/test_species.py
git commit -m "feat(ingestion): expose canonical_species() for tool-input normalization"
```

---

### Task 3.3: `DrugIndex` — canonical-name resolution

**Files:**

- Create: `src/vet_agent/tools/drug_index.py`
- Test: `tests/tools/test_drug_index.py`

- [ ] **Step 1: Write the failing test**

`tests/tools/test_drug_index.py`:

```python
from vet_agent.tools.drug_index import DrugIndex, ResolvedDrug
from vet_agent.tools.models import DrugNotFound

NAMES = ["Metronidazole", "Metoclopramide", "Meloxicam", "Carprofen"]


def _index() -> DrugIndex:
    return DrugIndex(NAMES)


def test_exact_case_insensitive_match():
    r = _index().resolve("metronidazole")
    assert isinstance(r, ResolvedDrug)
    assert r.canonical == "Metronidazole"
    assert r.exact is True


def test_whitespace_is_normalized():
    r = _index().resolve("  Metronidazole  ")
    assert isinstance(r, ResolvedDrug)
    assert r.exact is True


def test_fuzzy_single_close_match_is_visible_correction():
    r = _index().resolve("metronidazol")  # missing final 'e'
    assert isinstance(r, ResolvedDrug)
    assert r.canonical == "Metronidazole"
    assert r.exact is False


def test_miss_returns_suggestions():
    r = _index().resolve("metoclopramid3e")  # garbled but close-ish
    if isinstance(r, DrugNotFound):
        assert "Metoclopramide" in r.suggestions
    else:  # a high-cutoff fuzzy hit is also acceptable for this input
        assert r.canonical == "Metoclopramide"


def test_garbage_returns_empty_suggestions():
    r = _index().resolve("xyzzyplugh")
    assert isinstance(r, DrugNotFound)
    assert r.query == "xyzzyplugh"
    assert r.suggestions == []


def test_from_chunks_builds_from_distinct_drug_names(tmp_path):
    import json

    chunks = [
        {
            "drug_name": "Metronidazole",
            "section_type": "doses",
            "species": ["dog"],
            "book_page": 873,
            "text": "t",
            "ordinal": 0,
        },
        {
            "drug_name": "Metronidazole",
            "section_type": "indications",
            "species": ["all"],
            "book_page": 873,
            "text": "t",
            "ordinal": 0,
        },
    ]
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps(chunks), encoding="utf-8")
    index = DrugIndex.from_chunks(path)
    r = index.resolve("METRONIDAZOLE")
    assert isinstance(r, ResolvedDrug) and r.canonical == "Metronidazole"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tools/test_drug_index.py -v`
Expected: FAIL with `ModuleNotFoundError` for `vet_agent.tools.drug_index`.

- [ ] **Step 3: Write the implementation**

`src/vet_agent/tools/drug_index.py`:

```python
import difflib
from pathlib import Path

from pydantic import BaseModel

from vet_agent.tools.models import DrugNotFound

# High cutoff: a fuzzy hit is used to *filter retrieval*, so it must be near-certain.
# Suggestions use a lower cutoff — they are shown to the user, never acted on.
_MATCH_CUTOFF = 0.85
_SUGGESTION_CUTOFF = 0.6


def _norm(name: str) -> str:
    return " ".join(name.strip().lower().split())


class ResolvedDrug(BaseModel):
    """Internal resolution result — consumed by tools, not part of the result unions."""

    canonical: str
    exact: bool


class DrugIndex:
    """Resolves free-form drug names against the canonical monograph names."""

    def __init__(self, names: list[str]) -> None:
        self._by_norm = {_norm(n): n for n in names}

    @classmethod
    def from_chunks(cls, path: Path) -> "DrugIndex":
        from vet_agent.knowledge.loader import read_chunks  # lazy: avoids qdrant import cost

        return cls(sorted({c.drug_name for c in read_chunks(path)}))

    def resolve(self, query: str) -> ResolvedDrug | DrugNotFound:
        key = _norm(query)
        if key in self._by_norm:
            return ResolvedDrug(canonical=self._by_norm[key], exact=True)
        close = difflib.get_close_matches(key, self._by_norm, n=1, cutoff=_MATCH_CUTOFF)
        if close:
            return ResolvedDrug(canonical=self._by_norm[close[0]], exact=False)
        near = difflib.get_close_matches(key, self._by_norm, n=3, cutoff=_SUGGESTION_CUTOFF)
        return DrugNotFound(query=query, suggestions=[self._by_norm[n] for n in near])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tools/test_drug_index.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vet_agent/tools/drug_index.py tests/tools/test_drug_index.py
git commit -m "feat(tools): add DrugIndex with case-insensitive + fuzzy resolution"
```

---

### Task 3.4: `calculate_dose` — pure Decimal + pint arithmetic

**Files:**

- Create: `src/vet_agent/tools/dose_math.py`
- Test: `tests/tools/test_dose_math.py`

Notes (spec §8): fixed arithmetic only (`weight_kg × mg_per_kg`), **never** any expression evaluation; `Decimal` end-to-end with no rounding; `pint` does lb→kg so a unit mistake is structurally impossible; pydantic bounds reject nonsense before arithmetic runs. This is the exhaustively-tested crown jewel.

- [ ] **Step 1: Write the failing test**

`tests/tools/test_dose_math.py`:

```python
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
    result = calculate_dose(
        CalculateDoseInput(weight=Decimal("12"), rule=_rule("25", high="50"))
    )
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tools/test_dose_math.py -v`
Expected: FAIL with `ModuleNotFoundError` for `vet_agent.tools.dose_math`.

- [ ] **Step 3: Write the implementation**

`src/vet_agent/tools/dose_math.py`:

```python
from decimal import Decimal
from typing import Literal

import pint
from pydantic import BaseModel, Field

from vet_agent.tools.models import DoseResult, DoseRule

# Decimal-native unit registry: conversion factors parse as Decimal, so lb -> kg
# (1 lb = 0.45359237 kg, an exact defined constant) introduces no float error.
_UREG = pint.UnitRegistry(non_int_type=Decimal)


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
    dose_high = (
        weight_kg * inp.rule.mg_per_kg_high if inp.rule.mg_per_kg_high is not None else None
    )
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tools/test_dose_math.py -v`
Expected: PASS (8 passed). If pint's Decimal registry surprises (e.g. `.magnitude` not Decimal), fix the conversion — do not weaken the exactness assertions.

- [ ] **Step 5: Commit**

```bash
git add src/vet_agent/tools/dose_math.py tests/tools/test_dose_math.py
git commit -m "feat(tools): add calculate_dose - pure Decimal+pint, exhaustively tested"
```

---

### Task 3.5: `VectorStore.fetch_section()` — exact filtered scroll

**Files:**

- Modify: `src/vet_agent/knowledge/interfaces.py`
- Modify: `src/vet_agent/knowledge/vector_store.py`
- Test: `tests/knowledge/test_vector_store.py` (extend)

Notes (spec §9): the report tools mean "the WHOLE section for this drug"; filtered ANN has no completeness guarantee, so this is a Qdrant **scroll** with an exact filter — no vector, deterministic, ordered by `logical_key`.

- [ ] **Step 1: Extend the `_point` helper, then write the failing test**

The existing helper in `tests/knowledge/test_vector_store.py` takes a UUID-string `pid`
(qdrant `:memory:` rejects non-UUID ids), a separate `key=` for the logical key, and
hardcodes `section_type=SectionType.DOSES`. Add one defaulted `section` parameter
(existing tests untouched):

```python
def _point(
    pid: str,
    *,
    species,
    vector,
    drug="Metronidazole",
    text="t",
    ch="h",
    key: str | None = None,
    section=SectionType.DOSES,
) -> PointPayload:
    return PointPayload(
        point_id=pid,
        vector=vector,
        drug_name=drug,
        section_type=section,
        species=species,
        book_page=873,
        text=text,
        logical_key=key if key is not None else pid,
        content_hash=ch,
    )
```

Then append the tests (note the UUID-constant pattern used by the existing tests):

```python
_UUID_C0 = "00000000-0000-0000-0000-000000000010"
_UUID_C1 = "00000000-0000-0000-0000-000000000011"
_UUID_D0 = "00000000-0000-0000-0000-000000000012"
_UUID_OTHER = "00000000-0000-0000-0000-000000000013"


def test_fetch_section_returns_all_matching_chunks_ordered():
    store = _store()
    store.ensure_collection(dim=2)
    contra = SectionType.CONTRAINDICATIONS
    store.upsert(
        [
            _point(_UUID_C1, species=["all"], vector=[1.0, 0.0], key="m|c|all|1", section=contra),
            _point(_UUID_C0, species=["all"], vector=[0.0, 1.0], key="m|c|all|0", section=contra),
            _point(_UUID_D0, species=["dog"], vector=[1.0, 0.0], key="m|doses|dog|0"),
            _point(
                _UUID_OTHER,
                species=["all"],
                vector=[1.0, 0.0],
                drug="Carprofen",
                key="carprofen|c|all|0",
                section=contra,
            ),
        ]
    )
    hits = store.fetch_section("Metronidazole", contra)
    # only the requested drug+section, in stable logical_key order
    assert [h.logical_key for h in hits] == ["m|c|all|0", "m|c|all|1"]
    assert all(h.score is None for h in hits)  # no vector search, no score


def test_fetch_section_empty_when_no_match_or_no_collection():
    assert _store().fetch_section("Metronidazole", SectionType.DOSES) == []
    store = _store()
    store.ensure_collection(dim=2)
    assert store.fetch_section("Nope", SectionType.DOSES) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/knowledge/test_vector_store.py -v`
Expected: existing tests PASS; the two new tests FAIL with `AttributeError: ... 'fetch_section'`.

- [ ] **Step 3: Add to the Protocol**

In `src/vet_agent/knowledge/interfaces.py`, add to the `VectorStore` Protocol (after `search`):

```python
    def fetch_section(self, drug: str, section: SectionType) -> list[Passage]: ...
```

- [ ] **Step 4: Implement in `QdrantVectorStore`**

In `src/vet_agent/knowledge/vector_store.py`:

First, refactor `_to_passage` so the payload→Passage mapping is reusable without a `ScoredPoint` (scroll returns `Record`s, which have no score):

```python
def _payload_to_passage(payload: dict[str, object], score: float | None) -> Passage:
    return Passage(
        drug_name=str(payload["drug_name"]),
        section_type=SectionType(str(payload["section_type"])),
        species=[str(s) for s in payload["species"]],  # type: ignore[union-attr]
        book_page=int(payload["book_page"]),  # type: ignore[arg-type]
        text=str(payload["text"]),
        logical_key=str(payload["logical_key"]),
        score=score,
    )
```

(Adapt the exact typing to keep mypy strict green — the current `_to_passage` body shows the working pattern; the refactor is only "split payload mapping from score". Update `search()`'s result mapping to `_payload_to_passage(point.payload or {}, point.score)` and delete `_to_passage`.)

Then add the method to `QdrantVectorStore`:

```python
    def fetch_section(self, drug: str, section: SectionType) -> list[Passage]:
        """All chunks for (drug, section) via exact filtered scroll - complete, no ANN."""
        if not self._client.collection_exists(self._collection):
            return []
        flt = models.Filter(
            must=[
                models.FieldCondition(key="drug_name", match=models.MatchValue(value=drug)),
                models.FieldCondition(
                    key="section_type", match=models.MatchValue(value=section.value)
                ),
            ]
        )
        passages: list[Passage] = []
        offset: str | int | None = None
        while True:
            points, offset = self._client.scroll(
                self._collection,
                scroll_filter=flt,
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            passages.extend(_payload_to_passage(p.payload or {}, None) for p in points)
            if offset is None:
                break
        return sorted(passages, key=lambda p: p.logical_key)
```

- [ ] **Step 5: Run the full knowledge suite**

Run: `uv run pytest tests/knowledge/ -v`
Expected: PASS (all existing + 2 new — the `_to_passage` refactor must not break `search`).

- [ ] **Step 6: Commit**

```bash
git add src/vet_agent/knowledge/interfaces.py src/vet_agent/knowledge/vector_store.py \
        tests/knowledge/test_vector_store.py
git commit -m "feat(knowledge): add VectorStore.fetch_section - exact filtered scroll"
```

---

### Task 3.6: `RetrieveMonograph`

**Files:**

- Create: `src/vet_agent/tools/retrieve.py`
- Test: `tests/tools/test_retrieve.py`

- [ ] **Step 1: Write the failing test**

`tests/tools/test_retrieve.py`:

```python
import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient

from vet_agent.ingestion.models import Chunk, SectionType
from vet_agent.knowledge.loader import load_chunks
from vet_agent.knowledge.retrieval import Retriever
from vet_agent.knowledge.vector_store import QdrantVectorStore
from vet_agent.tools.drug_index import DrugIndex
from vet_agent.tools.models import DrugNotFound, NoPassagesFound, RetrievedPassages
from vet_agent.tools.retrieve import RetrieveMonograph, RetrieveMonographInput
from tests.knowledge.fakes import FakeEmbedder


def _tool() -> RetrieveMonograph:
    store = QdrantVectorStore(QdrantClient(location=":memory:"), "vet_test")
    emb = FakeEmbedder(dim=8)
    chunks = [
        Chunk(
            drug_name="Metronidazole",
            section_type=SectionType.DOSES,
            species=["dog"],
            book_page=873,
            text="dog dose text",
            ordinal=0,
        ),
        Chunk(
            drug_name="Metronidazole",
            section_type=SectionType.DOSES,
            species=["cat"],
            book_page=873,
            text="cat dose text",
            ordinal=0,
        ),
    ]
    load_chunks(chunks, emb, store)
    return RetrieveMonograph(Retriever(emb, store), DrugIndex(["Metronidazole"]))


def test_happy_path_resolves_drug_and_filters_species():
    out = _tool()(
        RetrieveMonographInput(
            query="dose", drug="metronidazole", section=SectionType.DOSES, species="Dogs"
        )
    )
    assert isinstance(out, RetrievedPassages)
    assert out.drug_name == "Metronidazole"  # canonical echoed back
    assert {p.species[0] for p in out.passages} == {"dog"}  # "Dogs" canonicalized + filtered


def test_unknown_drug_short_circuits_to_drug_not_found():
    out = _tool()(RetrieveMonographInput(query="dose", drug="xyzzyplugh"))
    assert isinstance(out, DrugNotFound)


def test_zero_hits_echo_the_filters():
    out = _tool()(
        RetrieveMonographInput(
            query="dose",
            drug="metronidazole",
            section=SectionType.DOSES,
            species="ferret",  # loaded corpus has no ferret chunk
        )
    )
    assert isinstance(out, NoPassagesFound)
    assert out.filters == {
        "drug": "Metronidazole",
        "section": "doses",
        "species": "ferret",
    }


def test_unrecognized_species_passes_through_lowercased():
    out = _tool()(RetrieveMonographInput(query="dose", species="Axolotl"))
    assert isinstance(out, NoPassagesFound)
    assert out.filters == {"species": "axolotl"}


def test_blank_query_is_rejected():
    with pytest.raises(ValidationError):
        RetrieveMonographInput(query="   ")


def test_top_k_bounds():
    with pytest.raises(ValidationError):
        RetrieveMonographInput(query="q", top_k=0)
    with pytest.raises(ValidationError):
        RetrieveMonographInput(query="q", top_k=21)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tools/test_retrieve.py -v`
Expected: FAIL with `ModuleNotFoundError` for `vet_agent.tools.retrieve`.

- [ ] **Step 3: Write the implementation**

`src/vet_agent/tools/retrieve.py`:

```python
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
        if inp.drug is not None:
            resolved = self._drugs.resolve(inp.drug)
            if isinstance(resolved, DrugNotFound):
                return resolved
            canonical = resolved.canonical

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
        return RetrievedPassages(drug_name=canonical, passages=hits)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tools/test_retrieve.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vet_agent/tools/retrieve.py tests/tools/test_retrieve.py
git commit -m "feat(tools): add RetrieveMonograph with drug resolution and typed misses"
```

---

### Task 3.7: `ExtractDoseRule` — grounding, selection, list-all (fake extractor)

**Files:**

- Create: `src/vet_agent/tools/dose_extraction.py` (Protocol + pure tool; the Anthropic impl is Task 3.8)
- Create: `tests/tools/fakes.py`
- Test: `tests/tools/test_dose_extraction.py`

Notes (spec §7): the LLM only transcribes; everything decision-like is pure code. Grounding check applies to the mg/kg numbers in **every** mode; free-text fields are not number-checked. Dose values cross the LLM boundary as **strings** so `Decimal` parses them exactly (no float artifacts).

- [ ] **Step 1: Write the failing test**

`tests/tools/fakes.py`:

```python
"""Deterministic test double for the RegimenExtractor LLM seam."""

from vet_agent.tools.dose_extraction import ExtractedRegimen


class FakeRegimenExtractor:
    """Returns a canned regimen list, ignoring the passage text."""

    def __init__(self, regimens: list[ExtractedRegimen]) -> None:
        self._regimens = regimens

    def extract_regimens(self, passage_text: str) -> list[ExtractedRegimen]:
        return list(self._regimens)
```

`tests/tools/test_dose_extraction.py`:

```python
from decimal import Decimal

from vet_agent.ingestion.models import SectionType
from vet_agent.knowledge.interfaces import Passage
from vet_agent.tools.dose_extraction import (
    ExtractDoseRule,
    ExtractDoseRuleInput,
    ExtractedRegimen,
)
from vet_agent.tools.models import DoseRule, DoseRuleSet, NeedsClarification
from tests.tools.fakes import FakeRegimenExtractor

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
    # query "giardiasis in dogs" contains regimen indication substring? No - but
    # regimen "giardiasis, monotherapy" vs query "monotherapy" matches the other way.
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tools/test_dose_extraction.py -v`
Expected: FAIL with `ModuleNotFoundError` for `vet_agent.tools.dose_extraction`.

- [ ] **Step 3: Write the implementation (Protocol + pure tool)**

`src/vet_agent/tools/dose_extraction.py`:

```python
import re
from decimal import Decimal
from typing import Protocol

from pydantic import BaseModel, Field

from vet_agent.knowledge.interfaces import Passage
from vet_agent.tools.models import DoseRule, DoseRuleSet, NeedsClarification


class ExtractedRegimen(BaseModel):
    """One regimen as transcribed by the LLM - untrusted until grounded."""

    indication: str
    mg_per_kg_low: Decimal = Field(gt=0)
    mg_per_kg_high: Decimal | None = Field(default=None, gt=0)
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
        re.search(rf"(?<![\d.]){re.escape(form)}(?![\d.])", text)
        for form in _decimal_forms(value)
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

    def __call__(
        self, inp: ExtractDoseRuleInput
    ) -> DoseRule | DoseRuleSet | NeedsClarification:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tools/test_dose_extraction.py -v`
Expected: PASS (13 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vet_agent/tools/dose_extraction.py tests/tools/fakes.py \
        tests/tools/test_dose_extraction.py
git commit -m "feat(tools): add ExtractDoseRule - grounded, indication-aware, list-all mode"
```

---

### Task 3.8: `AnthropicRegimenExtractor` — the one real LLM impl

**Files:**

- Modify: `src/vet_agent/tools/dose_extraction.py` (append)
- Test: `tests/tools/test_anthropic_extractor.py`

Notes: forced tool use (`tool_choice={"type": "tool", ...}`) with `strict: True` guarantees the reply is a schema-valid regimen list — never free text to parse. Dose values are **strings** in the schema so `Decimal` parses exactly. CI never calls the API: tests inject a stubbed client object. Model comes from `config.reasoning_model`; key from the existing `SecretStr` setting — both wired by the caller (CLI/Phase 4), not read here.

- [ ] **Step 1: Write the failing test**

`tests/tools/test_anthropic_extractor.py`:

```python
from decimal import Decimal
from types import SimpleNamespace

from vet_agent.tools.dose_extraction import AnthropicRegimenExtractor

RAW_REGIMENS = [
    {
        "indication": "giardiasis",
        "mg_per_kg_low": "25",
        "mg_per_kg_high": None,
        "route": "PO",
        "frequency": "q12h",
        "notes": "with fenbendazole",
    },
    {  # malformed: non-numeric dose -> must be dropped, not crash
        "indication": "bad",
        "mg_per_kg_low": "twenty-five",
        "mg_per_kg_high": None,
        "route": "PO",
        "frequency": "q12h",
        "notes": None,
    },
]


class _FakeMessages:
    def __init__(self, content):
        self._content = content
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(content=self._content)


def _fake_client(content):
    return SimpleNamespace(messages=_FakeMessages(content))


def _tool_use_block(regimens):
    return SimpleNamespace(type="tool_use", input={"regimens": regimens})


def test_parses_forced_tool_use_and_drops_malformed_regimens():
    client = _fake_client([_tool_use_block(RAW_REGIMENS)])
    extractor = AnthropicRegimenExtractor("claude-sonnet-5", client=client)
    regimens = extractor.extract_regimens("passage text")
    assert len(regimens) == 1
    assert regimens[0].mg_per_kg_low == Decimal("25")
    assert regimens[0].notes == "with fenbendazole"


def test_request_forces_the_extraction_tool():
    client = _fake_client([_tool_use_block([])])
    extractor = AnthropicRegimenExtractor("claude-sonnet-5", client=client)
    extractor.extract_regimens("passage text")
    kwargs = client.messages.last_kwargs
    assert kwargs["model"] == "claude-sonnet-5"
    assert kwargs["tool_choice"] == {"type": "tool", "name": "record_regimens"}
    assert kwargs["tools"][0]["name"] == "record_regimens"
    assert kwargs["tools"][0]["strict"] is True
    assert "passage text" in kwargs["messages"][0]["content"]


def test_no_tool_use_block_returns_empty():
    client = _fake_client([SimpleNamespace(type="text", text="cannot comply")])
    extractor = AnthropicRegimenExtractor("claude-sonnet-5", client=client)
    assert extractor.extract_regimens("passage text") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tools/test_anthropic_extractor.py -v`
Expected: FAIL with `ImportError: cannot import name 'AnthropicRegimenExtractor'`.

- [ ] **Step 3: Append the implementation**

Append to `src/vet_agent/tools/dose_extraction.py` (add `Any` to the `typing` import and `ValidationError` to the pydantic import):

```python
_EXTRACTION_TOOL_NAME = "record_regimens"

# Dose values are strings so Decimal parses them exactly (JSON floats would
# smuggle in binary-float artifacts before validation could see them).
_EXTRACTION_TOOL: dict[str, Any] = {
    "name": _EXTRACTION_TOOL_NAME,
    "description": (
        "Record every mg/kg dosing regimen found in the veterinary handbook passage, "
        "one entry per distinct regimen (per indication, and per lettered alternative "
        "such as 'a)' / 'b)')."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "regimens": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "indication": {
                            "type": "string",
                            "description": "What this regimen treats, as stated in the text.",
                        },
                        "mg_per_kg_low": {
                            "type": "string",
                            "description": (
                                "The mg/kg dose (or low end of a range), copied verbatim "
                                "as a plain number string, e.g. '25' or '0.5'."
                            ),
                        },
                        "mg_per_kg_high": {
                            "type": ["string", "null"],
                            "description": "High end of a range, verbatim; null if no range.",
                        },
                        "route": {
                            "type": "string",
                            "description": "Route verbatim, e.g. 'PO', 'IV over 30 min'.",
                        },
                        "frequency": {
                            "type": "string",
                            "description": "Frequency/duration verbatim, e.g. 'q12h for 8 days'.",
                        },
                        "notes": {
                            "type": ["string", "null"],
                            "description": "Combination therapy or caveats; null if none.",
                        },
                    },
                    "required": [
                        "indication",
                        "mg_per_kg_low",
                        "mg_per_kg_high",
                        "route",
                        "frequency",
                        "notes",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["regimens"],
        "additionalProperties": False,
    },
}

_EXTRACTION_SYSTEM = (
    "You transcribe dosing regimens from a veterinary drug handbook passage into "
    "structured records. Copy every number exactly as written - never compute, "
    "convert, round, or infer a value that is not literally in the text. Record "
    "only regimens expressed in mg/kg; skip per-animal, mg/m2, or otherwise "
    "non-mg/kg doses. Doses belonging to OTHER drugs mentioned in passing (e.g. a "
    "combination-therapy partner) go in 'notes', never in the dose fields."
)


class AnthropicRegimenExtractor:
    """RegimenExtractor backed by Claude with forced, strict tool-use output."""

    def __init__(self, model: str, *, api_key: str | None = None, client: Any = None) -> None:
        if client is None:
            from anthropic import Anthropic  # lazy: keeps import cost out of pure paths

            client = Anthropic(api_key=api_key)
        self._client = client
        self._model = model

    def extract_regimens(self, passage_text: str) -> list[ExtractedRegimen]:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=_EXTRACTION_SYSTEM,
            tools=[_EXTRACTION_TOOL],
            tool_choice={"type": "tool", "name": _EXTRACTION_TOOL_NAME},
            messages=[{"role": "user", "content": f"Passage:\n\n{passage_text}"}],
        )
        for block in message.content:
            if getattr(block, "type", None) == "tool_use":
                raw = block.input.get("regimens", [])
                return [r for r in map(_parse_regimen, raw) if r is not None]
        return []


def _parse_regimen(raw: dict[str, Any]) -> ExtractedRegimen | None:
    """Validate one raw regimen; malformed entries are dropped (grounding is separate)."""
    try:
        return ExtractedRegimen.model_validate(raw)
    except ValidationError:
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tools/test_anthropic_extractor.py tests/tools/test_dose_extraction.py -v`
Expected: PASS (16 passed) — the pure-tool tests must still pass untouched.

- [ ] **Step 5: Commit**

```bash
git add src/vet_agent/tools/dose_extraction.py tests/tools/test_anthropic_extractor.py
git commit -m "feat(tools): add AnthropicRegimenExtractor with forced strict tool-use"
```

---

### Task 3.9: `FindContraindications` + `ListIndications`

**Files:**

- Create: `src/vet_agent/tools/contraindications.py`
- Create: `src/vet_agent/tools/indications.py`
- Test: `tests/tools/test_contraindications.py`, `tests/tools/test_indications.py`

Notes (spec §9): both fetch **whole sections** via `fetch_section` (no ANN, no embedding). `find_contraindications` pulls `contraindications` **and** `drug_interactions`; other-drug flagging is a case-insensitive canonical-name match in passage text; unresolvable names are surfaced, never dropped. `list_indications` treats species as a soft signal: matching (or `all`) passages sort first, nothing is excluded.

- [ ] **Step 1: Write the failing tests**

`tests/tools/test_contraindications.py`:

```python
from qdrant_client import QdrantClient

from vet_agent.ingestion.models import Chunk, SectionType
from vet_agent.knowledge.loader import load_chunks
from vet_agent.knowledge.vector_store import QdrantVectorStore
from vet_agent.tools.contraindications import FindContraindications, FindContraindicationsInput
from vet_agent.tools.drug_index import DrugIndex
from vet_agent.tools.models import ContraindicationReport, DrugNotFound
from tests.knowledge.fakes import FakeEmbedder


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
    out = _tool()(
        FindContraindicationsInput(drug="metronidazole", other_drugs=["cimetidine"])
    )
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
```

`tests/tools/test_indications.py`:

```python
from qdrant_client import QdrantClient

from vet_agent.ingestion.models import Chunk, SectionType
from vet_agent.knowledge.loader import load_chunks
from vet_agent.knowledge.vector_store import QdrantVectorStore
from vet_agent.tools.drug_index import DrugIndex
from vet_agent.tools.indications import ListIndications, ListIndicationsInput
from vet_agent.tools.models import DrugNotFound, IndicationReport
from tests.knowledge.fakes import FakeEmbedder


def _tool() -> ListIndications:
    store = QdrantVectorStore(QdrantClient(location=":memory:"), "vet_test")
    chunks = [
        Chunk(
            drug_name="Metronidazole",
            section_type=SectionType.INDICATIONS,
            species=["horse"],
            book_page=873,
            text="equine indications",
            ordinal=0,
        ),
        Chunk(
            drug_name="Metronidazole",
            section_type=SectionType.INDICATIONS,
            species=["all"],
            book_page=873,
            text="general indications",
            ordinal=1,
        ),
        Chunk(
            drug_name="Metronidazole",
            section_type=SectionType.INDICATIONS,
            species=["cat", "dog"],
            book_page=873,
            text="small animal indications",
            ordinal=2,
        ),
    ]
    load_chunks(chunks, FakeEmbedder(dim=8), store)
    return ListIndications(store, DrugIndex(["Metronidazole"]))


def test_all_passages_returned_without_species():
    out = _tool()(ListIndicationsInput(drug="metronidazole"))
    assert isinstance(out, IndicationReport)
    assert out.species is None
    assert len(out.passages) == 3


def test_species_reorders_but_never_excludes():
    out = _tool()(ListIndicationsInput(drug="metronidazole", species="Dogs"))
    assert isinstance(out, IndicationReport)
    assert out.species == "dog"
    assert len(out.passages) == 3  # soft signal: nothing dropped
    # dog-tagged and 'all'-tagged sort before horse-only
    assert {out.passages[0].text, out.passages[1].text} == {
        "general indications",
        "small animal indications",
    }
    assert out.passages[2].text == "equine indications"


def test_unknown_drug_returns_drug_not_found():
    out = _tool()(ListIndicationsInput(drug="xyzzy"))
    assert isinstance(out, DrugNotFound)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tools/test_contraindications.py tests/tools/test_indications.py -v`
Expected: FAIL with `ModuleNotFoundError` for both new modules.

- [ ] **Step 3: Write the implementations**

`src/vet_agent/tools/contraindications.py`:

```python
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

    def __call__(
        self, inp: FindContraindicationsInput
    ) -> ContraindicationReport | DrugNotFound:
        resolved = self._drugs.resolve(inp.drug)
        if isinstance(resolved, DrugNotFound):
            return resolved

        contraindications = self._store.fetch_section(
            resolved.canonical, SectionType.CONTRAINDICATIONS
        )
        interactions = self._store.fetch_section(
            resolved.canonical, SectionType.DRUG_INTERACTIONS
        )

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
            mentions = [
                p for p in contraindications + interactions if needle in p.text.lower()
            ]
            if mentions:
                flagged.append(
                    FlaggedInteraction(other_drug=other_resolved.canonical, passages=mentions)
                )

        return ContraindicationReport(
            drug_name=resolved.canonical,
            contraindications=contraindications,
            interactions=interactions,
            flagged=flagged,
            unresolved_other_drugs=unresolved,
        )
```

`src/vet_agent/tools/indications.py`:

```python
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
            # Prose species tags are best-effort mentions - reorder, never exclude
            # (sorted() is stable, so within each bucket logical_key order is kept).
            passages = sorted(
                passages,
                key=lambda p: 0 if (species in p.species or "all" in p.species) else 1,
            )

        return IndicationReport(
            drug_name=resolved.canonical, species=species, passages=passages
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/tools/test_contraindications.py tests/tools/test_indications.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vet_agent/tools/contraindications.py src/vet_agent/tools/indications.py \
        tests/tools/test_contraindications.py tests/tools/test_indications.py
git commit -m "feat(tools): add FindContraindications and ListIndications report tools"
```

---

### Task 3.10: CLI `dose` command — the phase demo

**Files:**

- Modify: `src/vet_agent/cli/main.py`
- Modify: `README.md`
- Test: `tests/test_cli.py` (extend)

Notes: chains retrieve → extract → calculate against live Qdrant + the Anthropic API. This is the only code path that needs credentials; tests only exercise argument validation and help (no network).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
def test_help_lists_dose_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "dose" in result.stdout


def test_dose_requires_existing_chunks(tmp_path):
    missing = tmp_path / "nope.json"
    result = runner.invoke(
        app,
        [
            "dose",
            "metronidazole dose for a dog with giardia",
            "--drug",
            "metronidazole",
            "--species",
            "dog",
            "--weight",
            "12",
            "--chunks",
            str(missing),
        ],
    )
    assert result.exit_code != 0
    assert "not found" in result.stdout.lower()


def test_dose_rejects_non_numeric_weight(tmp_path):
    chunks = tmp_path / "chunks.json"
    chunks.write_text("[]", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "dose",
            "q",
            "--drug",
            "metronidazole",
            "--species",
            "dog",
            "--weight",
            "twelve",
            "--chunks",
            str(chunks),
        ],
    )
    assert result.exit_code != 0
    assert "weight" in result.stdout.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: existing tests PASS; the three new ones FAIL (`dose` not in help / exit code 2 usage errors with different messages).

- [ ] **Step 3: Add the command**

In `src/vet_agent/cli/main.py`, add imports:

```python
from decimal import Decimal, InvalidOperation

from vet_agent.tools.dose_extraction import (
    AnthropicRegimenExtractor,
    ExtractDoseRule,
    ExtractDoseRuleInput,
)
from vet_agent.tools.dose_math import CalculateDoseInput, calculate_dose
from vet_agent.tools.drug_index import DrugIndex
from vet_agent.tools.models import DoseRule, DoseRuleSet, NeedsClarification
from vet_agent.tools.retrieve import RetrieveMonograph, RetrieveMonographInput
```

Add the command:

```python
@app.command()
def dose(
    question: str = typer.Argument(..., help="Natural-language dose question"),  # noqa: B008
    drug: str = typer.Option(..., help="Drug name (resolved against the monographs)"),  # noqa: B008
    species: str = typer.Option(..., help="Patient species, e.g. dog"),  # noqa: B008
    weight: str = typer.Option(..., help="Patient weight (number)"),  # noqa: B008
    weight_unit: str = typer.Option("kg", help="kg or lb"),  # noqa: B008
    indication: str = typer.Option("", help="Indication hint, e.g. giardia"),  # noqa: B008
    all_regimens: bool = typer.Option(  # noqa: B008
        False, "--all-regimens", help="List every grounded regimen instead of picking one"
    ),
    chunks: Path = typer.Option(Path("data/ingest/chunks.json")),  # noqa: B008
) -> None:
    """Phase 3 demo: retrieve -> extract_dose_rule -> calculate_dose, with citations."""
    if not chunks.is_file():
        typer.echo(f"Error: chunks file not found at {chunks}")
        raise typer.Exit(code=1)
    try:
        weight_value = Decimal(weight)
    except InvalidOperation as exc:
        typer.echo(f"Error: weight must be a number, got '{weight}'")
        raise typer.Exit(code=1) from exc
    if weight_unit not in ("kg", "lb"):
        typer.echo(f"Error: weight-unit must be kg or lb, got '{weight_unit}'")
        raise typer.Exit(code=1)
    settings = Settings()
    if settings.anthropic_api_key is None:
        typer.echo("Error: VET_ANTHROPIC_API_KEY is required for dose extraction.")
        raise typer.Exit(code=1)

    model_key = settings.embedding_model
    embedder = get_embedder(model_key)
    client = QdrantClient(url=settings.qdrant_url)
    store = QdrantVectorStore(client, collection_name(settings.qdrant_collection_prefix, model_key))
    drug_index = DrugIndex.from_chunks(chunks)

    retrieve_tool = RetrieveMonograph(Retriever(embedder, store), drug_index)
    retrieved = retrieve_tool(
        RetrieveMonographInput(
            query=question, drug=drug, section=SectionType.DOSES, species=species
        )
    )
    if retrieved.kind == "drug_not_found":
        hint = f" Did you mean: {', '.join(retrieved.suggestions)}?" if retrieved.suggestions else ""
        typer.echo(f"Drug not found: '{retrieved.query}'.{hint}")
        raise typer.Exit(code=1)
    if retrieved.kind == "no_passages_found":
        typer.echo(f"No dose passages found for filters {retrieved.filters}.")
        raise typer.Exit(code=1)

    passage = retrieved.passages[0]
    typer.echo(f"Passage: {passage.drug_name} / doses / p.{passage.book_page}")

    extractor = AnthropicRegimenExtractor(
        settings.reasoning_model,
        api_key=settings.anthropic_api_key.get_secret_value(),
    )
    extracted = ExtractDoseRule(extractor)(
        ExtractDoseRuleInput(
            passage=passage, indication=indication or None, all_regimens=all_regimens
        )
    )

    if isinstance(extracted, NeedsClarification):
        typer.echo(f"Needs clarification: {extracted.reason}")
        for c in extracted.candidates:
            typer.echo(f"  - {_describe_rule(c)}")
        return
    if isinstance(extracted, DoseRuleSet):
        typer.echo(f"{len(extracted.rules)} grounded regimen(s):")
        for rule in extracted.rules:
            typer.echo(f"  - {_describe_rule(rule)}")
        return

    result = calculate_dose(
        CalculateDoseInput(weight=weight_value, weight_unit=weight_unit, rule=extracted)  # type: ignore[arg-type]
    )
    dose_range = f"{result.dose_mg_low} mg"
    if result.dose_mg_high is not None:
        dose_range += f" - {result.dose_mg_high} mg"
    typer.echo(
        f"Dose: {dose_range} {result.route} {result.frequency} "
        f"({result.rule.mg_per_kg_low} mg/kg x {result.weight_kg} kg)"
    )
    typer.echo(f"Indication: {result.indication}")
    if result.notes:
        typer.echo(f"Notes: {result.notes}")
    typer.echo(
        f"Source: {result.drug_name}, Doses, p.{result.rule.book_page} "
        f"[{result.rule.source_logical_key}]"
    )
    typer.echo("Decision support only - consult a licensed veterinarian.")


def _describe_rule(rule: DoseRule) -> str:
    dose_str = f"{rule.mg_per_kg_low}"
    if rule.mg_per_kg_high is not None:
        dose_str += f"-{rule.mg_per_kg_high}"
    text = f"{rule.indication}: {dose_str} mg/kg {rule.route} {rule.frequency}"
    if rule.notes:
        text += f" ({rule.notes})"
    return text
```

Note on `weight_unit`: it is validated by hand above, then passed into `CalculateDoseInput` where the `Literal["kg", "lb"]` annotation is authoritative — the `# type: ignore` is only for typer's `str` type. If mypy complains differently, cast with `typing.cast(Literal["kg", "lb"], weight_unit)` instead of loosening the model.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (all existing + 3 new).

- [ ] **Step 5: Document the command in the README**

In `README.md`, after the `retrieve` step of the workflow section, add:

```markdown
### 6. Ask a dose question (Phase 3 demo)

Chains the Phase 3 tools end to end — filtered retrieval → LLM regimen extraction
(grounded against the cited passage) → pure-Python dose arithmetic:

```bash
uv run vet-agent dose "metronidazole dose for a dog with giardia?" \
  --drug metronidazole --species dog --weight 12 --indication giardia
# or list every grounded regimen in the passage:
uv run vet-agent dose "metronidazole dosing options for dogs?" \
  --drug metronidazole --species dog --weight 12 --all-regimens
```

Requires a loaded Qdrant collection (step 5) and `VET_ANTHROPIC_API_KEY`.
```

(Match the README's existing numbering/formatting; renumber if needed.)

- [ ] **Step 6: Commit**

```bash
git add src/vet_agent/cli/main.py tests/test_cli.py README.md
git commit -m "feat(cli): add dose command - retrieve -> extract -> calculate demo"
```

---

### Task 3.11: Full gate + manual live verification (Definition of Done)

**Files:** none (verification only)

- [ ] **Step 1: Full offline gate**

Run: `make check`
Expected: ruff clean, mypy `Success`, all fast tests pass (existing 110 + ~45 new), slow tests deselected. Fix anything that fails before proceeding.

- [ ] **Step 2: Bring up Qdrant and confirm the collection**

```bash
docker compose up -d qdrant
uv run vet-agent load data/ingest/chunks.json
```

Expected: `Loaded into 'vet_chunks__bge_base': upserted=0 skipped=15292 pruned=0` (idempotent no-op if already loaded).

- [ ] **Step 3: Run the dose demo (selection mode)**

Requires `VET_ANTHROPIC_API_KEY` in `.env`.

```bash
uv run vet-agent dose "What is the metronidazole dose for a 12 kg dog with giardia?" \
  --drug metronidazole --species dog --weight 12 --indication giardia
```

Expected (either is a **correct** outcome — the real passage lists two giardiasis regimens):

- `Needs clarification: multiple regimens match indication 'giardia'` with the 25 mg/kg and 50 mg/kg candidates listed, **or**
- a single cited dose if the retrieved chunk happens to contain only one matching regimen.

A wrong outcome would be: a silently picked regimen when several match, a dose number not present in the passage, or a cat/horse passage being used.

- [ ] **Step 4: Run the dose demo (list-all mode)**

```bash
uv run vet-agent dose "metronidazole dosing options for dogs" \
  --drug metronidazole --species dog --weight 12 --all-regimens
```

Expected: every grounded regimen from the top dog Doses passage, each with indication, mg/kg, route, frequency.

- [ ] **Step 5: Spot-check the typed misses**

```bash
uv run vet-agent dose "q" --drug metronidazol --species dog --weight 12   # fuzzy: resolves
uv run vet-agent dose "q" --drug xyzzy --species dog --weight 12          # DrugNotFound + suggestions
```

- [ ] **Step 6: Record results + commit any fixes**

Note the actual outputs in the plan's Status section (mirroring Phase 2's Task 2.13). If fixes were needed, commit them with descriptive messages.

---

## Definition of Done (from spec §12)

- [ ] `make check` green (ruff + mypy strict + pytest), all new tests offline and fast.
- [ ] All five tools implemented with the spec'd contracts; `calculate_dose` exhaustively tested.
- [ ] `VectorStore.fetch_section()` + `canonical_species()` shipped with tests.
- [ ] `anthropic` promoted to main deps; `pint` added.
- [ ] Manual verification: `vet-agent dose` answers a real dose question end to end against live
      Qdrant (cited dose or candidate regimens), and `--all-regimens` lists every grounded option.
