# Vet-Agent Phase 0–1 (Scaffold + Ingestion) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the project scaffold, then build an ingestion pipeline that turns *Plumb's Veterinary Drug Handbook* (PDF) into typed `Monograph` objects and structure-aware `Chunk`s, plus an auditable `parse_report.json` — all without any embedding or vector-DB dependency yet.

**Architecture:** A `uv`-managed, `src`-layout Python package. Ingestion is a chain of small, pure, independently testable stages: `pdf_reader` (PDF → page text) → `toc` (table of contents → drug list) → `segmenter` (full text → per-drug blocks) → `sectionizer` (block → labeled sections) → `chunker` (monograph → section/species chunks) → `report` (anomalies → parse_report.json), wired together in `pipeline.py` and exposed via a Typer CLI. Domain logic is TDD'd against small synthetic text fixtures; one final task runs the whole chain against the real PDF for manual verification.

**Tech Stack:** Python 3.12+, uv, pydantic v2 + pydantic-settings, pypdf, Typer, pytest, ruff, mypy. (LangGraph, Qdrant, Anthropic, embeddings arrive in later-phase plans.)

**Scope:** This plan implements **Phase 0 (Scaffold)** and **Phase 1 (Ingestion)** from the design spec (`docs/superpowers/specs/2026-06-13-agentic-rag-vet-drug-assistant-design.md`). Phase 1 stops at in-memory/serialized chunks + parse report; embedding and loading into Qdrant are Phase 2 (separate plan).

---

## File Structure

**Phase 0 (scaffold):**

- `pyproject.toml` — project metadata, deps, ruff/mypy/pytest config
- `Makefile` — `install`, `lint`, `typecheck`, `test`, `check` targets
- `docker-compose.yml` — Qdrant service (config only; used in Phase 2)
- `.env.example` — documented env vars
- `src/vet_agent/__init__.py`
- `src/vet_agent/config.py` — `Settings` (pydantic-settings)
- `tests/test_config.py`
- `tests/test_smoke.py`

**Phase 1 (ingestion) — all under `src/vet_agent/ingestion/`:**

- `models.py` — `SectionType` enum, `TocEntry`, `Section`, `Monograph`, `Chunk`, `ParseReport`
- `species.py` — canonical species vocab, `parse_species_header`, `detect_species_mentions`
- `pdf_reader.py` — `clean_page_text`, `extract_pages`
- `toc.py` — `parse_toc_lines`
- `segmenter.py` — `segment_monographs`
- `sectionizer.py` — `HEADER_TO_SECTION` map, `split_sections`
- `builder.py` — `build_monograph`
- `chunker.py` — `chunk_monograph` (+ `logical_key`, `content_hash`)
- `report.py` — `build_parse_report`, `write_artifacts`
- `pipeline.py` — `run_ingestion`
- `src/vet_agent/cli/__init__.py`, `src/vet_agent/cli/main.py` — Typer app, `ingest` command
- Tests mirror each module under `tests/ingestion/`

---

## PHASE 0 — SCAFFOLD

### Task 0.1: Initialize uv project and package skeleton

**Files:**

- Create: `pyproject.toml`
- Create: `src/vet_agent/__init__.py`
- Create: `src/vet_agent/py.typed` (empty marker — required because `[tool.mypy] packages = ["vet_agent"]` type-checks the installed package)
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "vet-agent"
version = "0.1.0"
description = "Agentic RAG assistant over Plumb's Veterinary Drug Handbook"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "pypdf>=4.2",
    "typer>=0.12",
    "langchain-text-splitters>=1.1.2",
]

[dependency-groups]
dev = [
    "pytest>=8.2",
    "ruff>=0.5",
    "mypy>=1.10",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/vet_agent"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.mypy]
python_version = "3.12"
packages = ["vet_agent"]
strict = true

[tool.pytest.ini_options]
addopts = "-q"
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: Create the package marker**

`src/vet_agent/__init__.py`:

```python
"""Vet-Agent: agentic RAG over Plumb's Veterinary Drug Handbook."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Write a smoke test**

`tests/test_smoke.py`:

```python
import vet_agent


def test_package_imports_and_has_version():
    assert vet_agent.__version__ == "0.1.0"
```

- [ ] **Step 4: Sync the environment**

Run: `uv sync`
Expected: creates `.venv`, resolves and installs deps, no errors.

- [ ] **Step 5: Run the smoke test to verify it passes**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: PASS (1 passed).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/vet_agent/__init__.py tests/test_smoke.py uv.lock
git commit -m "chore: scaffold uv project with src layout and smoke test"
```

---

### Task 0.2: Settings via pydantic-settings

**Files:**

- Create: `src/vet_agent/config.py`
- Create: `.env.example`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:

```python
from pathlib import Path

from vet_agent.config import Settings


def test_defaults():
    s = Settings()
    assert s.reasoning_model == "claude-sonnet-4-6"
    assert s.qdrant_url == "http://localhost:6333"
    assert s.data_dir == Path("data")
    assert s.anthropic_api_key is None


def test_env_override(monkeypatch):
    monkeypatch.setenv("VET_REASONING_MODEL", "claude-3-5-sonnet-latest")
    monkeypatch.setenv("VET_ANTHROPIC_API_KEY", "sk-test")
    s = Settings()
    assert s.reasoning_model == "claude-3-5-sonnet-latest"
    assert s.anthropic_api_key == "sk-test"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vet_agent.config'`.

- [ ] **Step 3: Write the implementation**

`src/vet_agent/config.py`:

```python
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, overridable via VET_-prefixed env vars or .env."""

    model_config = SettingsConfigDict(
        env_prefix="VET_", env_file=".env", extra="ignore"
    )

    # LLM
    anthropic_api_key: str | None = None
    reasoning_model: str = "claude-sonnet-4-6"

    # Vector DB (used from Phase 2 onward)
    qdrant_url: str = "http://localhost:6333"

    # Paths
    data_dir: Path = Path("data")
```

- [ ] **Step 4: Create `.env.example`**

`.env.example`:

```bash
# Copy to .env and fill in. All vars are prefixed VET_.
VET_ANTHROPIC_API_KEY=
VET_REASONING_MODEL=claude-sonnet-4-6
VET_QDRANT_URL=http://localhost:6333
VET_DATA_DIR=data
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add src/vet_agent/config.py tests/test_config.py .env.example
git commit -m "feat: add pydantic-settings configuration"
```

---

### Task 0.3: Makefile and quality gates

**Files:**

- Create: `Makefile`

- [ ] **Step 1: Create the Makefile**

`Makefile` (recipe lines MUST be tab-indented):

```makefile
.PHONY: install lint typecheck test check

install:
	uv sync

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test
```

- [ ] **Step 2: Format the codebase so the lint gate passes**

Run: `uv run ruff format src tests`
Expected: reformats files (or "N files left unchanged").

- [ ] **Step 3: Run the full gate**

Run: `make check`
Expected: ruff clean, mypy `Success: no issues found`, pytest all passing.

- [ ] **Step 4: Commit**

```bash
git add Makefile src tests
git commit -m "chore: add Makefile quality gates (lint, typecheck, test)"
```

---

### Task 0.4: Qdrant docker-compose (config only, for Phase 2)

**Files:**

- Create: `docker-compose.yml`

- [ ] **Step 1: Create `docker-compose.yml`**

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  qdrant_data:
```

- [ ] **Step 2: Validate the compose file parses**

Run: `docker compose config`
Expected: prints the normalized config with no error. (If Docker is not installed, skip — this file is only consumed in Phase 2.)

- [ ] **Step 3: Commit**

```bash
git add docker-compose.yml
git commit -m "chore: add Qdrant docker-compose service for Phase 2"
```

---

## PHASE 1 — INGESTION

### Task 1.1: Ingestion domain models

**Files:**

- Create: `src/vet_agent/ingestion/__init__.py`
- Create: `src/vet_agent/ingestion/models.py`
- Test: `tests/ingestion/__init__.py`, `tests/ingestion/test_models.py`

- [ ] **Step 1: Write the failing test**

`tests/ingestion/test_models.py`:

```python
from vet_agent.ingestion.models import (
    Chunk,
    Monograph,
    ParseReport,
    Section,
    SectionType,
    TocEntry,
)


def test_section_type_has_doses_and_other():
    assert SectionType.DOSES.value == "doses"
    assert SectionType.OTHER.value == "other"


def test_monograph_roundtrip():
    mono = Monograph(
        drug_name="Metronidazole",
        book_page=873,
        sections=[
            Section(section_type=SectionType.INDICATIONS, text="Used for Giardia."),
            Section(section_type=SectionType.DOSES, text="DOGS: 25 mg/kg PO q12h"),
        ],
    )
    assert mono.section_text(SectionType.DOSES) == "DOGS: 25 mg/kg PO q12h"
    assert mono.section_text(SectionType.MONITORING) is None


def test_chunk_defaults():
    c = Chunk(
        drug_name="Metronidazole",
        section_type=SectionType.DOSES,
        species=["dog"],
        book_page=875,
        text="DOGS: 25 mg/kg PO q12h",
        ordinal=0,
    )
    assert c.species == ["dog"]
    assert c.ordinal == 0


def test_parse_report_counts():
    r = ParseReport(
        toc_entries=3,
        drugs_parsed=2,
        missing_headings=["Lost Drug"],
        anomalies=[{"drug": "X", "issue": "no sections"}],
    )
    assert r.toc_entries == 3
    assert r.drugs_parsed == 2
    assert r.missing_headings == ["Lost Drug"]
    assert r.anomalies[0]["drug"] == "X"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vet_agent.ingestion'`.

- [ ] **Step 3: Write the implementation**

`src/vet_agent/ingestion/__init__.py`:

```python
"""Ingestion pipeline: PDF -> monographs -> chunks + parse report."""
```

`tests/ingestion/__init__.py`: (empty file)

`src/vet_agent/ingestion/models.py`:

```python
from enum import Enum

from pydantic import BaseModel, Field


class SectionType(str, Enum):
    """Canonical monograph section types (Plumb's standard set)."""

    PRESCRIBER_HIGHLIGHTS = "prescriber_highlights"
    INDICATIONS = "indications"
    CONTRAINDICATIONS = "contraindications"
    ADVERSE_EFFECTS = "adverse_effects"
    REPRODUCTIVE_SAFETY = "reproductive_safety"
    OVERDOSE_TOXICITY = "overdose_toxicity"
    DRUG_INTERACTIONS = "drug_interactions"
    PHARMACOLOGY = "pharmacology"
    PHARMACOKINETICS = "pharmacokinetics"
    MONITORING = "monitoring"
    CLIENT_INFORMATION = "client_information"
    CHEMISTRY = "chemistry"
    STORAGE = "storage"
    COMPOUNDING = "compounding"
    DOSAGE_FORMS = "dosage_forms"
    DOSES = "doses"
    OTHER = "other"


class TocEntry(BaseModel):
    drug_name: str
    book_page: int


class Section(BaseModel):
    section_type: SectionType
    text: str


class Monograph(BaseModel):
    drug_name: str
    book_page: int
    sections: list[Section] = Field(default_factory=list)

    def section_text(self, section_type: SectionType) -> str | None:
        for s in self.sections:
            if s.section_type == section_type:
                return s.text
        return None


class Chunk(BaseModel):
    drug_name: str
    section_type: SectionType
    species: list[str]
    book_page: int
    text: str
    ordinal: int


class ParseReport(BaseModel):
    toc_entries: int = 0
    drugs_parsed: int = 0
    missing_headings: list[str] = Field(default_factory=list)
    anomalies: list[dict[str, str]] = Field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_models.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vet_agent/ingestion/__init__.py src/vet_agent/ingestion/models.py tests/ingestion/
git commit -m "feat: add ingestion domain models"
```

---

### Task 1.2: Species vocabulary and parsing

**Files:**

- Create: `src/vet_agent/ingestion/species.py`
- Test: `tests/ingestion/test_species.py`

- [ ] **Step 1: Write the failing test**

`tests/ingestion/test_species.py`:

```python
from vet_agent.ingestion.species import (
    detect_species_mentions,
    parse_species_header,
)


def test_parse_single_species_header():
    assert parse_species_header("DOGS:") == ["dog"]
    assert parse_species_header("CATS:") == ["cat"]
    assert parse_species_header("HORSES:") == ["horse"]
    assert parse_species_header("CATTLE:") == ["cattle"]
    assert parse_species_header("SWINE:") == ["swine"]


def test_parse_combined_species_header():
    assert parse_species_header("DOGS & CATS:") == ["cat", "dog"]
    assert parse_species_header("DOGS/CATS:") == ["cat", "dog"]


def test_non_header_returns_empty():
    assert parse_species_header("Giardiasis (extra-label):") == []
    assert parse_species_header("25 mg/kg PO q12h") == []


def test_detect_species_mentions_in_prose():
    text = "Used extensively in dogs and cats; in horses it may cause ataxia."
    assert detect_species_mentions(text) == ["cat", "dog", "horse"]


def test_detect_species_mentions_none():
    assert detect_species_mentions("No specific species discussed.") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_species.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`src/vet_agent/ingestion/species.py`:

```python
import re

# Maps any surface token (singular/plural) to a canonical species name.
_SPECIES_SYNONYMS: dict[str, str] = {
    "dog": "dog",
    "dogs": "dog",
    "cat": "cat",
    "cats": "cat",
    "horse": "horse",
    "horses": "horse",
    "ferret": "ferret",
    "ferrets": "ferret",
    "rabbit": "rabbit",
    "rabbits": "rabbit",
    "bird": "bird",
    "birds": "bird",
    "cattle": "cattle",
    "cow": "cattle",
    "cows": "cattle",
    "swine": "swine",
    "pig": "swine",
    "pigs": "swine",
    "sheep": "sheep",
    "goat": "goat",
    "goats": "goat",
}

# A dose sub-header is short, mostly uppercase, and ends with a colon.
_HEADER_RE = re.compile(r"^[A-Z][A-Z &/]{0,40}:$")


def _canonical_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z]+", text.lower())
    found = {_SPECIES_SYNONYMS[t] for t in tokens if t in _SPECIES_SYNONYMS}
    return sorted(found)


def parse_species_header(line: str) -> list[str]:
    """Return canonical species for a Doses sub-header line, else []."""
    stripped = line.strip()
    if not _HEADER_RE.match(stripped):
        return []
    return _canonical_tokens(stripped)


def detect_species_mentions(text: str) -> list[str]:
    """Best-effort canonical species mentioned anywhere in prose text."""
    return _canonical_tokens(text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_species.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vet_agent/ingestion/species.py tests/ingestion/test_species.py
git commit -m "feat: add species vocabulary and header parsing"
```

---

### Task 1.3: PDF page-text cleaning

**Files:**

- Create: `src/vet_agent/ingestion/pdf_reader.py`
- Test: `tests/ingestion/test_pdf_reader.py`

Note: `clean_page_text` is a pure function (TDD'd here). `extract_pages` is a thin pypdf wrapper exercised against the real PDF in Task 1.10.

- [ ] **Step 1: Write the failing test**

`tests/ingestion/test_pdf_reader.py`:

```python
from vet_agent.ingestion.pdf_reader import clean_page_text


def test_dehyphenates_line_wrapped_words():
    raw = "metroni-\ndazole is an anti-\nbacterial agent"
    assert clean_page_text(raw) == "metronidazole is an antibacterial agent"


def test_collapses_spaces_and_preserves_line_structure():
    # Single newlines are preserved (headers/species sub-headers rely on line breaks);
    # only runs of spaces/tabs collapse and 3+ blank lines reduce to one.
    raw = "Adverse Effects\n\nIn   dogs,  vomiting\noccurs."
    assert clean_page_text(raw) == "Adverse Effects\n\nIn dogs, vomiting\noccurs."


def test_strips_empty_input():
    assert clean_page_text("") == ""


def test_preserves_numeric_dose_range_at_hyphen_wrap():
    # A weight range that wraps at a hyphen must NOT have its digits fused.
    assert clean_page_text("(8.1-\n25 lb)") == "(8.1-\n25 lb)"
    assert "8.125" not in clean_page_text("(8.1-\n25 lb)")


def test_collapses_excess_blank_lines_including_whitespace_only():
    assert clean_page_text("A\n\n\n\nB") == "A\n\nB"
    assert clean_page_text("section\n  \n  \nmore") == "section\n\nmore"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_pdf_reader.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`src/vet_agent/ingestion/pdf_reader.py`:

```python
import re
from pathlib import Path

from pypdf import PdfReader

# Only de-hyphenate when the continuation starts with a lowercase letter — this joins
# soft-wrapped words ("metroni-\ndazole") while preserving numeric dose ranges that
# wrap at a hyphen (e.g. "(8.1-\n25 lb)" must NOT become "(8.125 lb)").
_HYPHEN_WRAP_RE = re.compile(r"-\n([a-z])")
_MULTISPACE_RE = re.compile(r"[ \t]+")
_MULTINEWLINE_RE = re.compile(r"\n{3,}")


def clean_page_text(raw: str) -> str:
    """Normalize extracted page text while preserving line structure.

    De-hyphenates wrapped words and collapses runs of spaces and excess blank
    lines, but keeps single newlines intact so the sectionizer and species-header
    parser can detect line-based headers (e.g. ``Uses/Indications``, ``DOGS:``).
    """
    if not raw:
        return ""
    text = _HYPHEN_WRAP_RE.sub(r"\1", raw)
    text = _MULTISPACE_RE.sub(" ", text)
    # Strip each line BEFORE collapsing blank lines, so whitespace-only lines
    # (reduced to a single space above) don't defeat the blank-line cap.
    lines = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = _MULTINEWLINE_RE.sub("\n\n", text)
    return text.strip()


def extract_pages(pdf_path: Path) -> list[str]:
    """Return cleaned text for every page of the PDF (index 0 == first page)."""
    reader = PdfReader(pdf_path)
    return [clean_page_text(page.extract_text() or "") for page in reader.pages]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_pdf_reader.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vet_agent/ingestion/pdf_reader.py tests/ingestion/test_pdf_reader.py
git commit -m "feat: add PDF page-text cleaning"
```

---

### Task 1.4: Table-of-contents parsing

**Files:**

- Create: `src/vet_agent/ingestion/toc.py`
- Test: `tests/ingestion/test_toc.py`

- [ ] **Step 1: Write the failing test**

`tests/ingestion/test_toc.py`:

```python
import logging

from vet_agent.ingestion.toc import parse_toc_lines


def test_parses_drug_and_page():
    lines = [
        "Metronidazole 873",
        "Midazolam 880",
        "Moxidectin/Moxidectin Combination Products 912",
    ]
    entries = parse_toc_lines(lines)
    assert entries[0].drug_name == "Metronidazole"
    assert entries[0].book_page == 873
    assert entries[2].drug_name == "Moxidectin/Moxidectin Combination Products"
    assert entries[2].book_page == 912


def test_ignores_non_entry_lines():
    lines = ["Table of Contents", "Preface vii", "Metronidazole 873", ""]
    entries = parse_toc_lines(lines)
    # "Preface vii" has no integer page -> skipped; heading skipped
    assert [e.drug_name for e in entries] == ["Metronidazole"]


def test_logs_success_count_and_skipped_lines(caplog):
    lines = ["Table of Contents", "Preface vii", "Metronidazole 873", ""]
    with caplog.at_level(logging.DEBUG, logger="vet_agent.ingestion.toc"):
        parse_toc_lines(lines)
    # INFO summary with the success count, and a DEBUG line per failed parse.
    assert "Parsed 1 TOC entries" in caplog.text
    assert "Skipping non-entry TOC line" in caplog.text
    assert "Preface vii" in caplog.text  # the failing line is named
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_toc.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`src/vet_agent/ingestion/toc.py`:

```python
import logging
import re

from vet_agent.ingestion.models import TocEntry

logger = logging.getLogger(__name__)

# "<Drug Name> <page>" — name may contain letters, spaces, slashes, hyphens, parens.
_TOC_LINE_RE = re.compile(r"^(?P<name>[A-Za-z][A-Za-z0-9 ,/()'+-]+?)\s+(?P<page>\d{1,4})$")


def parse_toc_lines(lines: list[str]) -> list[TocEntry]:
    """Parse '<Drug> <page>' table-of-contents lines into TocEntry objects.

    Blank lines are ignored silently. Every non-blank line that fails to parse is
    logged at DEBUG (with the offending text); a single INFO summary reports how many
    entries parsed and how many lines were skipped.
    """
    entries: list[TocEntry] = []
    skipped = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        m = _TOC_LINE_RE.match(stripped)
        if not m:
            skipped += 1
            logger.debug("Skipping non-entry TOC line: %r", stripped)
            continue
        entries.append(
            TocEntry(drug_name=m.group("name").strip(), book_page=int(m.group("page")))
        )
    logger.info("Parsed %d TOC entries (%d non-blank lines skipped)", len(entries), skipped)
    return entries
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_toc.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vet_agent/ingestion/toc.py tests/ingestion/test_toc.py
git commit -m "feat: add table-of-contents parsing with diagnostic logging"
```

---

### Task 1.5: Section splitting (sectionizer)

**Files:**

- Create: `src/vet_agent/ingestion/sectionizer.py`
- Test: `tests/ingestion/test_sectionizer.py`

- [ ] **Step 1: Write the failing test**

`tests/ingestion/test_sectionizer.py`:

```python
from vet_agent.ingestion.models import SectionType
from vet_agent.ingestion.sectionizer import normalize_header, split_sections


def test_normalize_header_collapses_spaces():
    assert normalize_header("Dosage Forms/ Regulatory Status") == "Dosage Forms/Regulatory Status"


def test_split_sections_maps_known_headers():
    body = "\n".join(
        [
            "Uses/Indications",
            "Used for Giardia in dogs and cats.",
            "Adverse Effects",
            "Vomiting and lethargy.",
            "Doses",
            "DOGS: 25 mg/kg PO q12h",
        ]
    )
    sections = split_sections(body)
    by_type = {s.section_type: s.text for s in sections}
    assert by_type[SectionType.INDICATIONS] == "Used for Giardia in dogs and cats."
    assert by_type[SectionType.ADVERSE_EFFECTS] == "Vomiting and lethargy."
    assert by_type[SectionType.DOSES] == "DOGS: 25 mg/kg PO q12h"


def test_overdosage_variant_maps_to_overdose_toxicity():
    body = "Overdosage/Acute Toxicity\nSupportive care."
    sections = split_sections(body)
    assert sections[0].section_type == SectionType.OVERDOSE_TOXICITY


def test_unknown_header_is_not_treated_as_section():
    # Text before the first known header is ignored (it's the drug intro block).
    body = "Some intro prose.\nUses/Indications\nReal content."
    sections = split_sections(body)
    assert [s.section_type for s in sections] == [SectionType.INDICATIONS]
    assert sections[0].text == "Real content."


def test_consecutive_headers_with_no_body_emit_no_empty_section():
    # Two recognized headers back-to-back must not produce a Section(text="").
    body = "Monitoring\nClient Information\nSome real text."
    sections = split_sections(body)
    assert [s.section_type for s in sections] == [SectionType.CLIENT_INFORMATION]
    assert sections[0].text == "Some real text."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_sectionizer.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`src/vet_agent/ingestion/sectionizer.py`:

```python
import re

from vet_agent.ingestion.models import Section, SectionType

# Canonical header string (normalized) -> SectionType.
HEADER_TO_SECTION: dict[str, SectionType] = {
    "Prescriber Highlights": SectionType.PRESCRIBER_HIGHLIGHTS,
    "Uses/Indications": SectionType.INDICATIONS,
    "Contraindications/Precautions/Warnings": SectionType.CONTRAINDICATIONS,
    "Adverse Effects": SectionType.ADVERSE_EFFECTS,
    "Reproductive/Nursing Safety": SectionType.REPRODUCTIVE_SAFETY,
    "Overdose/Acute Toxicity": SectionType.OVERDOSE_TOXICITY,
    "Overdosage/Acute Toxicity": SectionType.OVERDOSE_TOXICITY,
    "Drug Interactions": SectionType.DRUG_INTERACTIONS,
    "Pharmacology/Actions": SectionType.PHARMACOLOGY,
    "Pharmacokinetics": SectionType.PHARMACOKINETICS,
    "Monitoring": SectionType.MONITORING,
    "Client Information": SectionType.CLIENT_INFORMATION,
    "Chemistry/Synonyms": SectionType.CHEMISTRY,
    "Storage/Stability": SectionType.STORAGE,
    "Compatibility/Compounding Considerations": SectionType.COMPOUNDING,
    "Dosage Forms/Regulatory Status": SectionType.DOSAGE_FORMS,
    "Dose Forms/Regulatory Status": SectionType.DOSAGE_FORMS,
    "Doses": SectionType.DOSES,
}

_SLASH_SPACE_RE = re.compile(r"\s*/\s*")
_MULTISPACE_RE = re.compile(r"\s+")


def normalize_header(line: str) -> str:
    """Normalize spacing around slashes and runs of whitespace in a header line."""
    text = _SLASH_SPACE_RE.sub("/", line.strip())
    return _MULTISPACE_RE.sub(" ", text)


def split_sections(body: str) -> list[Section]:
    """Split a monograph body into labeled sections at known header lines.

    Text appearing before the first recognized header (the drug intro) is dropped.
    Sections whose body is empty (e.g. two recognized headers back-to-back) are not
    emitted, so no empty Section reaches the chunker.

    Limitation: detection is purely by line text, so a body line that happens to
    normalize exactly to a known header string would be treated as a new section
    boundary. Real monograph bodies don't put a bare header string on its own line,
    but this is the trade-off of line-based detection without font/position signals.
    """
    sections: list[Section] = []
    current_type: SectionType | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current_type is not None:
            text = "\n".join(buffer).strip()
            if text:
                ct = current_type
                sections.append(Section(section_type=ct, text=text))

    for line in body.split("\n"):
        header_type = HEADER_TO_SECTION.get(normalize_header(line))
        if header_type is not None:
            flush()
            current_type = header_type
            buffer = []
        elif current_type is not None:
            buffer.append(line)
    flush()
    return sections
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_sectionizer.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vet_agent/ingestion/sectionizer.py tests/ingestion/test_sectionizer.py
git commit -m "feat: add section splitting with canonical header map"
```

---

### Task 1.6: Monograph segmentation

**Files:**

- Create: `src/vet_agent/ingestion/segmenter.py`
- Test: `tests/ingestion/test_segmenter.py`

`segment_monographs` takes the full cleaned book text (one big string with `\f` form-feeds between pages is NOT assumed; we pass the joined page list) plus TOC entries, and returns `(drug_name, book_page, raw_block)` spans. It locates each drug's monograph by finding the drug-name heading line that follows the TOC order.

- [ ] **Step 1: Write the failing test**

`tests/ingestion/test_segmenter.py`:

```python
from vet_agent.ingestion.models import TocEntry
from vet_agent.ingestion.segmenter import segment_monographs


def test_segments_two_drugs_in_order():
    text = "\n".join(
        [
            "Metronidazole",
            "Uses/Indications",
            "Treats Giardia.",
            "Midazolam",
            "Uses/Indications",
            "A benzodiazepine.",
        ]
    )
    toc = [
        TocEntry(drug_name="Metronidazole", book_page=873),
        TocEntry(drug_name="Midazolam", book_page=880),
    ]
    result = segment_monographs(text, toc)
    blocks = result.blocks
    assert [b.drug_name for b in blocks] == ["Metronidazole", "Midazolam"]
    assert "Treats Giardia." in blocks[0].body
    assert "Treats Giardia." not in blocks[1].body
    assert "A benzodiazepine." in blocks[1].body
    assert blocks[0].book_page == 873
    assert result.missing == []


def test_missing_drug_heading_is_reported():
    text = "Metronidazole\nUses/Indications\nTreats Giardia."
    toc = [
        TocEntry(drug_name="Metronidazole", book_page=873),
        TocEntry(drug_name="Nonexistent Drug", book_page=999),
    ]
    result = segment_monographs(text, toc)
    # Located drugs are returned as blocks; unlocated ones are surfaced in `missing`
    # (returned as data, NOT silently dropped) so the caller can enforce a policy.
    assert [b.drug_name for b in result.blocks] == ["Metronidazole"]
    assert [e.drug_name for e in result.missing] == ["Nonexistent Drug"]


def test_text_without_a_located_heading_is_absorbed_into_preceding_block():
    # Only TOC headings located in the text create boundaries. A drug in the TOC but
    # absent from the text goes to `missing`; text whose heading is not a boundary
    # (e.g. it has no TOC entry) is absorbed into the preceding block.
    text = "DrugA\nbody A\nDrugB\nbody B\nDrugC\nbody C"
    toc = [
        TocEntry(drug_name="DrugA", book_page=1),
        TocEntry(drug_name="MISSING", book_page=2),
        TocEntry(drug_name="DrugC", book_page=3),
    ]
    result = segment_monographs(text, toc)
    assert [e.drug_name for e in result.missing] == ["MISSING"]
    assert "body B" in result.blocks[0].body  # DrugB (no boundary) absorbed into DrugA
    assert result.blocks[1].drug_name == "DrugC"
    assert result.blocks[1].body == "body C"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_segmenter.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`src/vet_agent/ingestion/segmenter.py`:

```python
import re

from pydantic import BaseModel

from vet_agent.ingestion.models import TocEntry


class MonographBlock(BaseModel):
    drug_name: str
    book_page: int
    body: str


class SegmentationResult(BaseModel):
    blocks: list[MonographBlock]
    missing: list[TocEntry]


def _heading_index(text: str, drug_name: str, start: int) -> int | None:
    """Find the offset of a line that is exactly the drug name, at/after `start`."""
    pattern = re.compile(rf"^{re.escape(drug_name)}\s*$", re.MULTILINE)
    m = pattern.search(text, start)
    return m.start() if m else None


def segment_monographs(text: str, toc: list[TocEntry]) -> SegmentationResult:
    """Slice the full book text into per-drug blocks following TOC order.

    Each drug's block runs from its heading line up to the next found heading. This is
    a pure transformation: drugs whose heading cannot be located are returned in
    `missing` (errors-as-values), leaving coverage policy to the orchestration layer.
    """
    located: list[tuple[TocEntry, int]] = []
    missing: list[TocEntry] = []
    cursor = 0
    for entry in toc:
        idx = _heading_index(text, entry.drug_name, cursor)
        if idx is None:
            missing.append(entry)
            continue
        located.append((entry, idx))
        cursor = idx + len(entry.drug_name)

    blocks: list[MonographBlock] = []
    for i, (entry, start) in enumerate(located):
        end = located[i + 1][1] if i + 1 < len(located) else len(text)
        body = text[start + len(entry.drug_name) : end].strip()
        blocks.append(
            MonographBlock(drug_name=entry.drug_name, book_page=entry.book_page, body=body)
        )
    return SegmentationResult(blocks=blocks, missing=missing)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_segmenter.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vet_agent/ingestion/segmenter.py tests/ingestion/test_segmenter.py
git commit -m "feat: add monograph segmentation by TOC order"
```

---

### Task 1.7: Monograph builder

**Files:**

- Create: `src/vet_agent/ingestion/builder.py`
- Test: `tests/ingestion/test_builder.py`

- [ ] **Step 1: Write the failing test**

`tests/ingestion/test_builder.py`:

```python
from vet_agent.ingestion.builder import build_monograph
from vet_agent.ingestion.models import SectionType
from vet_agent.ingestion.segmenter import MonographBlock


def test_build_monograph_from_block():
    block = MonographBlock(
        drug_name="Metronidazole",
        book_page=873,
        body="Uses/Indications\nTreats Giardia.\nDoses\nDOGS: 25 mg/kg PO q12h",
    )
    mono = build_monograph(block)
    assert mono.drug_name == "Metronidazole"
    assert mono.book_page == 873
    assert mono.section_text(SectionType.INDICATIONS) == "Treats Giardia."
    assert mono.section_text(SectionType.DOSES) == "DOGS: 25 mg/kg PO q12h"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_builder.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`src/vet_agent/ingestion/builder.py`:

```python
from vet_agent.ingestion.models import Monograph
from vet_agent.ingestion.sectionizer import split_sections
from vet_agent.ingestion.segmenter import MonographBlock


def build_monograph(block: MonographBlock) -> Monograph:
    """Assemble a typed Monograph from a raw per-drug block."""
    return Monograph(
        drug_name=block.drug_name,
        book_page=block.book_page,
        sections=split_sections(block.body),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_builder.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vet_agent/ingestion/builder.py tests/ingestion/test_builder.py
git commit -m "feat: add monograph builder"
```

---

### Task 1.8: Chunker (section/species chunks)

**Files:**

- Create: `src/vet_agent/ingestion/chunker.py`
- Test: `tests/ingestion/test_chunker.py`

Rules implemented (from spec §6):

- Doses section → split into one chunk **per species sub-header**; `species` is a hard list (e.g. `["cat","dog"]`).
- Every other section → one chunk; `species` = best-effort mentions (soft signal), `["all"]` if none found.
- Long sections (> `DEFAULT_MAX_CHARS`) are size-split into multiple chunks with increasing `ordinal`, using `RecursiveCharacterTextSplitter` (separator hierarchy `\n\n` → `\n` → space, never mid-word). Overlap defaults to 0 (chunks are already structurally bounded; duplicating dose lines is undesirable) but is a configurable knob for later retrieval-eval tuning.
- `logical_key(chunk)` and `content_hash(chunk)` are deterministic helpers (used for idempotent re-indexing in Phase 2).

- [ ] **Step 1: Write the failing test**

`tests/ingestion/test_chunker.py`:

```python
from vet_agent.ingestion.chunker import chunk_monograph, content_hash, logical_key
from vet_agent.ingestion.models import Monograph, Section, SectionType


def _mono() -> Monograph:
    # Species sub-headers sit on their own line (matching Plumb's formatting and
    # the structure preserved by clean_page_text).
    doses = "DOGS:\n25 mg/kg PO q12h\nCATS:\n25 mg/kg PO q24h\nDOGS & CATS:\nbonus line"
    return Monograph(
        drug_name="Metronidazole",
        book_page=873,
        sections=[
            Section(
                section_type=SectionType.INDICATIONS,
                text="Used for Giardia in dogs and cats.",
            ),
            Section(section_type=SectionType.DOSES, text=doses),
        ],
    )


def test_doses_split_per_species():
    chunks = chunk_monograph(_mono())
    dose_chunks = [c for c in chunks if c.section_type == SectionType.DOSES]
    species_sets = sorted(tuple(c.species) for c in dose_chunks)
    assert ("cat",) in species_sets
    assert ("dog",) in species_sets
    assert ("cat", "dog") in species_sets  # "DOGS & CATS:" -> combined
    cat_chunk = next(c for c in dose_chunks if c.species == ["cat"])
    assert "25 mg/kg PO q24h" in cat_chunk.text


def test_prose_section_uses_soft_species_mentions():
    chunks = chunk_monograph(_mono())
    ind = next(c for c in chunks if c.section_type == SectionType.INDICATIONS)
    assert ind.species == ["cat", "dog"]
    assert ind.ordinal == 0


def test_prose_section_with_no_species_tagged_all():
    mono = Monograph(
        drug_name="X",
        book_page=1,
        sections=[Section(section_type=SectionType.STORAGE, text="Store at 25 C.")],
    )
    chunk = chunk_monograph(mono)[0]
    assert chunk.species == ["all"]


def test_long_section_is_size_split_with_ordinals():
    long_text = " ".join(f"word{i}" for i in range(1000))
    mono = Monograph(
        drug_name="X",
        book_page=1,
        sections=[Section(section_type=SectionType.PHARMACOLOGY, text=long_text)],
    )
    chunks = chunk_monograph(mono, max_chars=200)
    assert len(chunks) > 1
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_logical_key_and_content_hash_are_deterministic():
    chunk = chunk_monograph(_mono())[0]
    assert logical_key(chunk) == logical_key(chunk)
    assert content_hash(chunk) == content_hash(chunk)
    assert "metronidazole" in logical_key(chunk).lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_chunker.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`src/vet_agent/ingestion/chunker.py`:

```python
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from vet_agent.ingestion.models import Chunk, Monograph, Section, SectionType
from vet_agent.ingestion.species import detect_species_mentions, parse_species_header

DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP = 0


def _size_split(text: str, max_chars: int, overlap: int = DEFAULT_OVERLAP) -> list[str]:
    """Split overlong text on natural separators (paragraph -> line -> word).

    Delegates to RecursiveCharacterTextSplitter, which prefers paragraph then line
    boundaries before falling back to spaces, so a fallback split never cuts mid-word.
    Overlap defaults to 0: our chunks are already structurally bounded and duplicating
    dose lines across chunks is undesirable, but the knob is exposed so retrieval eval
    (Phase 6) can introduce overlap later if it measurably helps.
    """
    if not text:
        return [""]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text) or [""]


def _doses_species_groups(text: str) -> list[tuple[list[str], str]]:
    """Group dose lines under their species sub-headers.

    Returns (species, text) pairs; lines before any header get species ['unspecified'].
    """
    groups: list[tuple[list[str], list[str]]] = []
    current_species: list[str] = ["unspecified"]
    current_lines: list[str] = []

    def flush() -> None:
        if current_lines:
            groups.append((current_species, current_lines))

    for line in text.split("\n"):
        species = parse_species_header(line)
        if species:
            flush()
            current_species = species
            current_lines = []
        else:
            current_lines.append(line)
    flush()
    return [(sp, "\n".join(lines).strip()) for sp, lines in groups]


def _chunk_section(
    drug_name: str, book_page: int, section: Section, max_chars: int, overlap: int
) -> list[Chunk]:
    if section.section_type == SectionType.DOSES:
        chunks: list[Chunk] = []
        ordinal = 0
        for species, text in _doses_species_groups(section.text):
            for piece in _size_split(text, max_chars, overlap):
                chunks.append(
                    Chunk(
                        drug_name=drug_name,
                        section_type=SectionType.DOSES,
                        species=species,
                        book_page=book_page,
                        text=piece,
                        ordinal=ordinal,
                    )
                )
                ordinal += 1
        return chunks

    species = detect_species_mentions(section.text) or ["all"]
    return [
        Chunk(
            drug_name=drug_name,
            section_type=section.section_type,
            species=species,
            book_page=book_page,
            text=piece,
            ordinal=ordinal,
        )
        for ordinal, piece in enumerate(_size_split(section.text, max_chars, overlap))
    ]


def chunk_monograph(
    mono: Monograph, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP
) -> list[Chunk]:
    """Produce structure-aware chunks for one monograph."""
    chunks: list[Chunk] = []
    for section in mono.sections:
        chunks.extend(
            _chunk_section(mono.drug_name, mono.book_page, section, max_chars, overlap)
        )
    return chunks


def logical_key(chunk: Chunk) -> str:
    """Stable identity for a chunk (drug|section|species|ordinal)."""
    species = "+".join(sorted(chunk.species))
    return f"{chunk.drug_name.lower()}|{chunk.section_type.value}|{species}|{chunk.ordinal}"


def content_hash(chunk: Chunk) -> str:
    """SHA-256 of chunk text + identity, for idempotent re-indexing (Phase 2)."""
    payload = f"{logical_key(chunk)}::{chunk.text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_chunker.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vet_agent/ingestion/chunker.py tests/ingestion/test_chunker.py
git commit -m "feat: add structure-aware chunker with species rules"
```

---

### Task 1.9: Parse report + artifact writer

**Files:**

- Create: `src/vet_agent/ingestion/report.py`
- Test: `tests/ingestion/test_report.py`

- [ ] **Step 1: Write the failing test**

`tests/ingestion/test_report.py`:

```python
import json

from vet_agent.ingestion.models import Monograph, Section, SectionType, TocEntry
from vet_agent.ingestion.report import build_parse_report, write_artifacts


def _monos() -> list[Monograph]:
    return [
        Monograph(
            drug_name="Metronidazole",
            book_page=873,
            sections=[Section(section_type=SectionType.DOSES, text="DOGS: 25 mg/kg")],
        ),
        Monograph(drug_name="Empty Drug", book_page=900, sections=[]),
    ]


def _toc() -> list[TocEntry]:
    return [
        TocEntry(drug_name="Metronidazole", book_page=873),
        TocEntry(drug_name="Empty Drug", book_page=900),
        TocEntry(drug_name="Lost Drug", book_page=950),
    ]


def test_build_parse_report_records_coverage_empty_and_missing():
    report = build_parse_report(_monos(), toc=_toc(), missing=[_toc()[2]])
    assert report.toc_entries == 3
    assert report.drugs_parsed == 2
    assert report.missing_headings == ["Lost Drug"]
    assert any(a["drug"] == "Empty Drug" for a in report.anomalies)


def test_write_artifacts_writes_files(tmp_path):
    monos = _monos()
    report = build_parse_report(monos, toc=_toc(), missing=[])
    write_artifacts(monos, report, out_dir=tmp_path)

    report_data = json.loads((tmp_path / "parse_report.json").read_text())
    assert report_data["drugs_parsed"] == 2

    mono_data = json.loads((tmp_path / "monographs.json").read_text())
    assert mono_data[0]["drug_name"] == "Metronidazole"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`src/vet_agent/ingestion/report.py`:

```python
import json
from pathlib import Path

from vet_agent.ingestion.models import Monograph, ParseReport, TocEntry


def build_parse_report(
    monographs: list[Monograph], toc: list[TocEntry], missing: list[TocEntry]
) -> ParseReport:
    """Summarize an ingestion run: TOC coverage, unlocated headings, empty monographs."""
    anomalies: list[dict[str, str]] = [
        {"drug": m.drug_name, "issue": "no sections parsed"}
        for m in monographs
        if not m.sections
    ]
    return ParseReport(
        toc_entries=len(toc),
        drugs_parsed=len(monographs),
        missing_headings=[e.drug_name for e in missing],
        anomalies=anomalies,
    )


def write_artifacts(
    monographs: list[Monograph], report: ParseReport, out_dir: Path
) -> None:
    """Write monographs.json and parse_report.json to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "monographs.json").write_text(
        json.dumps([m.model_dump() for m in monographs], indent=2)
    )
    (out_dir / "parse_report.json").write_text(report.model_dump_json(indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_report.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vet_agent/ingestion/report.py tests/ingestion/test_report.py
git commit -m "feat: add parse report and artifact writer"
```

---

### Task 1.10: Pipeline orchestration

**Files:**

- Create: `src/vet_agent/ingestion/pipeline.py`
- Test: `tests/ingestion/test_pipeline.py`

`run_ingestion` wires the stages: it takes already-extracted pages (list of cleaned strings) plus the TOC page range, so it stays PDF-free and unit-testable. The CLI (Task 1.11) supplies real pages via `extract_pages`.

- [ ] **Step 1: Write the failing test**

`tests/ingestion/test_pipeline.py`:

```python
from vet_agent.ingestion.models import SectionType
from vet_agent.ingestion.pipeline import run_ingestion


def test_run_ingestion_end_to_end():
    # Page index 0 = TOC; page indices 1-2 = monograph bodies.
    pages = [
        "Metronidazole 873\nMidazolam 880",
        "Metronidazole\nUses/Indications\nTreats Giardia in dogs.\nDoses\nDOGS:\n25 mg/kg PO q12h",
        "Midazolam\nUses/Indications\nA benzodiazepine used in cats.",
    ]
    monographs, chunks, report = run_ingestion(pages, toc_page_range=(0, 0))

    assert [m.drug_name for m in monographs] == ["Metronidazole", "Midazolam"]
    assert report.toc_entries == 2
    assert report.drugs_parsed == 2
    assert report.missing_headings == []

    dose_chunks = [c for c in chunks if c.section_type == SectionType.DOSES]
    assert dose_chunks[0].species == ["dog"]
    assert "25 mg/kg" in dose_chunks[0].text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`src/vet_agent/ingestion/pipeline.py`:

```python
import logging

from vet_agent.ingestion.builder import build_monograph
from vet_agent.ingestion.chunker import chunk_monograph
from vet_agent.ingestion.models import Chunk, Monograph, ParseReport
from vet_agent.ingestion.report import build_parse_report
from vet_agent.ingestion.segmenter import segment_monographs
from vet_agent.ingestion.toc import parse_toc_lines

logger = logging.getLogger(__name__)


def run_ingestion(
    pages: list[str], toc_page_range: tuple[int, int]
) -> tuple[list[Monograph], list[Chunk], ParseReport]:
    """Run the full ingestion chain over already-extracted page text.

    toc_page_range is an inclusive (start, end) range of page indices holding the TOC.
    Unlocated drug headings are logged (WARNING) and recorded in the report; the
    coverage policy (fail-or-not) is enforced by the caller (CLI), not here.
    """
    start, end = toc_page_range
    toc_lines: list[str] = []
    for page in pages[start : end + 1]:
        toc_lines.extend(page.split("\n"))
    toc = parse_toc_lines(toc_lines)

    body_text = "\n".join(pages[end + 1 :])
    segmentation = segment_monographs(body_text, toc)
    for entry in segmentation.missing:
        logger.warning(
            "Could not locate heading for TOC drug: %s (p.%d)",
            entry.drug_name,
            entry.book_page,
        )
    monographs = [build_monograph(b) for b in segmentation.blocks]

    chunks: list[Chunk] = []
    for mono in monographs:
        chunks.extend(chunk_monograph(mono))

    report = build_parse_report(monographs, toc=toc, missing=segmentation.missing)
    logger.info(
        "Ingestion: located %d/%d TOC drugs, %d missing, %d chunks",
        report.drugs_parsed,
        report.toc_entries,
        len(report.missing_headings),
        len(chunks),
    )
    return monographs, chunks, report
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_pipeline.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vet_agent/ingestion/pipeline.py tests/ingestion/test_pipeline.py
git commit -m "feat: add ingestion pipeline orchestration"
```

---

### Task 1.11: Typer CLI `ingest` command + real-PDF verification

**Files:**

- Create: `src/vet_agent/cli/__init__.py`
- Create: `src/vet_agent/cli/main.py`
- Modify: `pyproject.toml` (add `[project.scripts]`)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:

```python
from typer.testing import CliRunner

from vet_agent.cli.main import app

runner = CliRunner()


def test_ingest_help_lists_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.stdout


def test_ingest_requires_existing_pdf(tmp_path):
    missing = tmp_path / "nope.pdf"
    result = runner.invoke(app, ["ingest", str(missing)])
    assert result.exit_code != 0
    assert "not found" in result.stdout.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vet_agent.cli'`.

- [ ] **Step 3: Write the implementation**

`src/vet_agent/cli/__init__.py`:

```python
"""Command-line interface for vet-agent."""
```

`src/vet_agent/cli/main.py`:

```python
import logging
from pathlib import Path

import typer

from vet_agent.ingestion.pdf_reader import extract_pages
from vet_agent.ingestion.pipeline import run_ingestion
from vet_agent.ingestion.report import write_artifacts

app = typer.Typer(help="Vet-Agent CLI")


@app.command()
def ingest(
    pdf: Path = typer.Argument(..., help="Path to the Plumb's handbook PDF"),
    toc_start: int = typer.Option(19, help="First page index (0-based) of the TOC"),
    toc_end: int = typer.Option(27, help="Last page index (0-based) of the TOC"),
    out_dir: Path = typer.Option(Path("data/ingest"), help="Output directory"),
    max_missing: int = typer.Option(
        0,
        help="Max TOC drugs allowed with no located heading before the run fails. "
        "Defaults to 0 (zero tolerance) — a medical reference must not lose drugs. "
        "Raise only for local iteration while fixing the parser.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show DEBUG logs"),
) -> None:
    """Parse the PDF into monographs + chunks and write artifacts."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not pdf.exists():
        typer.echo(f"Error: PDF not found at {pdf}")
        raise typer.Exit(code=1)

    typer.echo(f"Reading {pdf} ...")
    pages = extract_pages(pdf)
    monographs, chunks, report = run_ingestion(pages, toc_page_range=(toc_start, toc_end))

    # Always write artifacts first, so parse_report.json (with missing_headings) is
    # available for inspection even when the coverage gate below fails the run.
    write_artifacts(monographs, report, out_dir=out_dir)
    typer.echo(
        f"Parsed {report.drugs_parsed}/{report.toc_entries} TOC drugs, {len(chunks)} chunks, "
        f"{len(report.missing_headings)} missing headings, {len(report.anomalies)} anomalies. "
        f"Artifacts -> {out_dir}"
    )

    # Coverage gate: fail loudly if too many TOC drugs could not be located.
    if len(report.missing_headings) > max_missing:
        typer.echo(
            f"Error: {len(report.missing_headings)} TOC drugs had no located heading "
            f"(allowed: {max_missing}). See 'missing_headings' in "
            f"{out_dir / 'parse_report.json'}. Fix the parser, or pass --max-missing "
            f"to proceed during local iteration."
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Add the console script to `pyproject.toml`**

Insert after the `dependencies = [...]` block (before `[dependency-groups]`):

```toml
[project.scripts]
vet-agent = "vet_agent.cli.main:app"
```

- [ ] **Step 5: Re-sync so the script entry point installs**

Run: `uv sync`
Expected: no errors; `vet-agent` console script becomes available.

- [ ] **Step 6: Run the CLI test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (2 passed).

- [ ] **Step 7: Run the full quality gate**

Run: `make check`
Expected: ruff clean, mypy `Success`, all pytest tests passing.

- [ ] **Step 8: Real-PDF verification run (manual)**

First run (expect it to FAIL the coverage gate until the parser is tuned — that's intended):

Run: `uv run vet-agent ingest plumbs-veterinary-drug-handbook-10_compress.pdf --verbose`

The default zero-tolerance gate (`--max-missing 0`) will likely exit non-zero on the first run because some TOC drugs won't be located yet. Artifacts are still written first, so inspect them (this is the Phase-1 acceptance check):

- `data/ingest/parse_report.json` — `toc_entries` should be ~738; `drugs_parsed` should equal it; `missing_headings` lists exactly which drugs were not located; skim `anomalies` for monographs with no sections.
- `data/ingest/monographs.json` — find Metronidazole and confirm its `sections` include `indications`, `contraindications`, `doses`, etc., with sensible text.

Iterate until `missing_headings` is empty (so the gate passes with the default `--max-missing 0`):

- If `toc_entries` itself is wrong or `drugs_parsed` is far below it, the TOC page range is off — adjust `--toc-start/--toc-end` (locate the contents pages by skimming early pages of the PDF) and re-run. Record the working range as the new default in `cli/main.py`.
- Use the `--verbose` WARNING lines (`Could not locate heading for TOC drug: ...`) to see exactly which names failed and why (e.g. ligatures, trailing page numbers in the heading line, name punctuation differences), and refine `_heading_index` / the TOC name normalization accordingly. Add a regression test for each real-world heading quirk you fix.
- Known `_TOC_LINE_RE` limitation: it requires names to start with a letter (`[A-Za-z]`), so drugs whose name starts with a digit (e.g. `5-Fluorouracil`, `6-Mercaptopurine`) are skipped during TOC parsing. If any such drug shows up in `missing_headings`, broaden the `name` group's first character class (and add a regression test) — but keep the page group anchored so stray numeric lines aren't misparsed as entries.
- Use `--max-missing N` only as a temporary escape hatch while iterating; the committed default stays 0.

If the Metronidazole `doses` chunks are NOT split per species (e.g. one chunk tagged `["unspecified"]`), the species sub-headers in the real extraction are not landing on their own line as assumed. Print the raw Doses text for one drug and inspect: if headers appear inline (e.g. `DOGS: 25 mg/kg ...`), relax `_HEADER_RE` in `species.py` to match a leading uppercase species token followed by `:` and split the remainder as the first dose line. Add a regression test with the real-format string before changing the regex.

If the Metronidazole `doses` chunks are NOT split per species (e.g. one chunk tagged `["unspecified"]`), the species sub-headers in the real extraction are not landing on their own line as assumed. Print the raw Doses text for one drug and inspect: if headers appear inline (e.g. `DOGS: 25 mg/kg ...`), relax `_HEADER_RE` in `species.py` to match a leading uppercase species token followed by `:` and split the remainder as the first dose line. Add a regression test with the real-format string before changing the regex.

- [ ] **Step 9: Commit**

```bash
git add src/vet_agent/cli/ tests/test_cli.py pyproject.toml uv.lock
git commit -m "feat: add Typer CLI ingest command with TOC coverage gate"
```

---

## Definition of Done (Phases 0–1)

- `make check` is green (ruff + mypy strict + pytest).
- `uv run vet-agent ingest <pdf>` parses the real handbook and writes `monographs.json` + `parse_report.json`.
- The coverage gate passes at the default `--max-missing 0`: `parse_report.json` shows `drugs_parsed == toc_entries` (~738) and `missing_headings` is empty — no drug is silently lost.
- No systematic section-parsing failure (skim `anomalies`).
- Spot-check: Metronidazole's monograph has its expected sections, and its Doses chunks are split per species with a hard species list.

## Real-PDF verification outcome (Task 1.11b)

Achieved against the actual handbook (TOC pages 17–23, the "SYSTEMIC MONOGRAPHS" contents):

- **Coverage: 641/641 located (100%, 0 missing)**, ~15,300 chunks. Segmentation was rewritten to
  be **page-anchored** (each page's printed book-page number is read from its running header; each
  TOC drug is located at its own `book_page` anchor) after the original forward-cursor approach
  proved fragile on real text (a single false match cascaded — only 179/641).
- **Critical header fix:** the real dosing header is **`Dosages`** (not `Doses`), so dose sections
  were initially never parsed; corrected, plus a whole **alternate-template header set** mapped
  (`Indications/Actions`, `Suggested Dosages/Uses`, `Contraindications/Precautions`,
  `Precautions/Adverse Effects`, …). **635/641 monographs have a dose section**; **634/641 have the
  full core set** (indications + contraindications + adverse effects + dosages = **98.9%**).
  Metronidazole yields 11 dose chunks split per species (dog/cat/horse).
- **Robustness fixes that drove the numbers up:** (1) `_heading_index` prefers the **bare
  monograph title** over a page's running header, so a shared page (one drug's `Dosages` tail above
  the next drug's start) no longer truncates the preceding drug; (2) **anchor interpolation** for
  pages with no running-header number (book page 1 / `Acarbose`), accepted only when the title is
  found there; (3) **whitespace/hyphen-tolerant name matching** so a boundary drug whose TOC name
  differs from the body (`L- Theanine` vs `L-Theanine`) is still located and doesn't truncate the
  prior drug.
- **Coverage breakdown (641 systemic TOC entries):** **634 fully covered (98.9%)**, **2 partial**
  (`Vitamin A` doses, `Molybdates` contraindications — boundary drugs whose body name carries a
  special char like `±` that defeats name matching), **5 none** (all non-drug ophthalmic-appendix
  reference lines that legitimately have no monograph content). The gate passes at `--max-missing 0`.
  Extending the TOC to pages 17–25 also ingests the topical appendix (766/772) but adds ~97
  empty-section anomalies, so systemic-only (17–23) is the clean v1 scope.

### Deferred follow-ups (not blocking Phase 1)
- The 2 partial drugs (`Vitamin A`, `Molybdates`) need symbol-tolerant boundary matching
  (e.g. `±`); deferred to avoid over-fitting — confirm via the Phase 6 eval golden set.
- Recover a monograph's final section when it spills onto the next drug's first page
  (e.g. Metronidazole's `Dosage Forms`), which currently gets dropped.
- Trim the running-header line that the bare-title boundary leaves at the end of the preceding
  drug's last section (cosmetic noise in the section text).

## What's Next (later plans)

- **Phase 2 — Knowledge layer:** `Embedder`/`Reranker`/`VectorStore` interfaces, embed chunks, idempotent load into Qdrant using `logical_key`/`content_hash` (already defined here).
- **Phase 3 — Tools:** `retrieve_monograph`, `extract_dose_rule`, pure-Python `calculate_dose` (Decimal + pint), `find_contraindications`, `list_indications`.
- **Phases 4–8:** LangGraph agent + guardrails, FastAPI/CLI, eval harness, Langfuse observability, Docker/CI hardening.

## Known v1 limitations (deferred by decision)

- **Approximate page citations.** Every chunk is tagged with its monograph's TOC start `book_page`, so a citation can be off by a few pages within a long monograph (a dose on p.875 may cite p.873). Accepted for v1; exact per-chunk page provenance (carrying `list[(page_number, text)]` through `pdf_reader` → `segmenter` → `chunker`) is a planned later refinement, since precise "see p.N" citations matter for a clinical tool.

