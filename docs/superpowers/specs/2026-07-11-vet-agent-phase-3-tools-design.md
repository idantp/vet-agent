# Vet-Agent Phase 3 — Tools Layer (Design Spec)

**Date:** 2026-07-11
**Status:** Approved design — ready for implementation planning
**Depends on:** Phase 2 (knowledge layer) — `docs/superpowers/specs/2026-06-28-vet-agent-phase-2-knowledge-layer-design.md`
**Parent spec:** `docs/superpowers/specs/2026-06-13-agentic-rag-vet-drug-assistant-design.md` (§7 Tools, §6 chunking/species rules, §12 typed errors, §14 Phase 3)

## 1. Purpose & Scope

Build the **Tools layer**: five pure, framework-agnostic, individually unit-tested tools in a new
`src/vet_agent/tools/` package, with typed pydantic I/O. This is the layer Phase 4's LangGraph agent
binds; nothing in it imports LangGraph.

The design thesis (from the parent spec): **the LLM touches exactly one seam**. `extract_dose_rule`
is the only tool that calls a model, and even there the model only *transcribes* prose into
candidate structures — selection, validation, and all arithmetic are deterministic code.

```
retrieve_monograph ──▶ extract_dose_rule ──▶ calculate_dose
   (pure: filtered       (LLM transcribes;      (pure: Decimal × pint,
    retrieval, cited)     pure code grounds,     exhaustively tested)
                          selects, validates)
```

**In scope (Phase 3):**

- `tools/` package: shared I/O models + result unions, `DrugIndex` name resolution, and the five
  tools: `retrieve_monograph`, `extract_dose_rule`, `calculate_dose`, `find_contraindications`,
  `list_indications`.
- One knowledge-layer addition: `VectorStore.fetch_section()` (exact filtered scroll — see §8).
- One `ingestion/species.py` addition: a public `canonical_species()` helper over the existing
  (private) synonym table.
- A `vet-agent dose` CLI command as the phase demo (retrieve → extract → calculate, end to end).
- Dependencies: add `pint`; promote `anthropic` from dev to main.

**Out of scope (deferred):** the LangGraph agent, guardrail nodes, and prompts (Phase 4); a
precomputed structured dose table (parent spec §15); brand-name/synonym drug vocabulary beyond the
monograph titles; reverse lookup (condition → drugs).

## 2. Key Decisions (locked in brainstorm)

| Decision | Choice | Rationale |
|---|---|---|
| Multi-regimen doses | **Indication-aware pick + explicit list-all mode** | Real Doses passages nearly always hold several regimens (Metronidazole/dog lists two alternatives for giardiasis alone). Default mode returns a single `DoseRule` only when exactly one grounded regimen matches the `indication` hint; otherwise `NeedsClarification` carrying the candidates as structured options. `all_regimens=True` returns every grounded regimen as a `DoseRuleSet` for "what are all the options?" questions. The tool never silently picks among alternatives. |
| Drug-name lookup | **Case-insensitive + fuzzy resolver, suggestions on miss** | Qdrant's `drug_name` filter is exact keyword match, so "metronidazole" (lowercase) or a typo would surface as a misleading not-found. `DrugIndex` resolves against the ~636 canonical monograph names: exact ci match, then `difflib` close-match (high cutoff, correction echoed back visibly), else `DrugNotFound` with suggestions. Stdlib-only, deterministic. |
| Error style | **Result unions, discriminated on a literal `kind` field** | Expected domain outcomes (`DrugNotFound`, `NoPassagesFound`, `NeedsClarification`) are *return types*, matching the parent spec's `DoseRule \| NeedsClarification` sketch. Serializable for LangGraph state + Langfuse traces; forces tests to treat not-found paths as first-class. Exceptions are reserved for genuine infra failures (Qdrant down). |
| Tool shape | **Callable classes with constructor DI; pure tools stay functions** | Tools needing deps (`Retriever`, LLM client, `VectorStore`) take them in `__init__` and expose one typed `__call__`; `calculate_dose` and `DrugIndex.resolve` are pure. The LLM hides behind a small `RegimenExtractor` Protocol (the Phase 2 `QueryPhraser` pattern) so CI injects a fake. Phase 4 instantiates once and binds bound methods. No tool base class / registry — speculative framework for 5 single-use tools. |
| Report tools bypass ANN | **`fetch_section()` scroll, not vector search** | `find_contraindications` / `list_indications` mean "the whole section for this drug", and HNSW gives no completeness guarantee even with a hard filter and big top_k. An exact filtered scroll is deterministic, complete, needs no embedding round-trip, and is a small tested addition to the `VectorStore` protocol. |
| Dose-extraction safety | **LLM transcribes; pure code grounds + selects** | The model returns raw `ExtractedRegimen`s; deterministic code then (a) discards any regimen whose numbers don't appear verbatim in the cited passage (grounding check), and (b) does indication matching / mode handling. A hallucinated dose cannot survive the seam. |

## 3. Architecture & Module Layout

```
src/vet_agent/tools/
├── __init__.py            # package docstring
├── models.py              # all tool I/O models + result unions (the contract)
├── drug_index.py          # DrugIndex: canonical-name resolution (ci + fuzzy)
├── retrieve.py            # RetrieveMonograph (wraps knowledge.Retriever)
├── dose_extraction.py     # RegimenExtractor protocol + AnthropicRegimenExtractor + ExtractDoseRule
├── dose_math.py           # calculate_dose — pure Decimal + pint, no LLM anywhere near it
├── contraindications.py   # FindContraindications
└── indications.py         # ListIndications
```

Dependency direction is one-way: `tools → knowledge → ingestion`. No cycles, no LangGraph.

## 4. Shared models & result unions (`tools/models.py`)

Every expected outcome is a return type discriminated on `kind`. `Passage` is reused from
`knowledge/interfaces.py`. `Decimal` round-trips through pydantic JSON as a string, so exactness
survives LangGraph state and Langfuse traces.

```python
class DoseRule(BaseModel):
    kind: Literal["dose_rule"] = "dose_rule"
    drug_name: str
    species: list[str]              # mirrors the source chunk (e.g. ["dog"] or ["cat", "dog"])
    indication: str
    mg_per_kg_low: Decimal          # gt 0, le 10_000 (charcoal ≈ 4 g/kg — bound loose on purpose)
    mg_per_kg_high: Decimal | None = None   # set when the text gives a range
    route: str                      # "PO", "IV over 30 min" … verbatim from text
    frequency: str                  # "q12h", "once daily for 5 days" … verbatim
    notes: str | None = None        # combination therapy, duration caveats
    source_logical_key: str         # citation traceability — always
    book_page: int

class DoseRuleSet(BaseModel):
    kind: Literal["dose_rule_set"] = "dose_rule_set"
    rules: list[DoseRule]           # every grounded regimen (list-all mode); min_length=1

class NeedsClarification(BaseModel):
    kind: Literal["needs_clarification"] = "needs_clarification"
    reason: str
    candidates: list[DoseRule] = [] # structured options the agent can relay verbatim

class DrugNotFound(BaseModel):
    kind: Literal["drug_not_found"] = "drug_not_found"
    query: str
    suggestions: list[str] = []     # "did you mean Metronidazole?"

class NoPassagesFound(BaseModel):
    kind: Literal["no_passages_found"] = "no_passages_found"
    query: str
    filters: dict[str, str]         # the filters that produced zero hits

class RetrievedPassages(BaseModel):
    kind: Literal["retrieved_passages"] = "retrieved_passages"
    drug_name: str | None           # canonical, post-resolution
    passages: list[Passage]

class DoseResult(BaseModel):
    kind: Literal["dose_result"] = "dose_result"
    drug_name: str
    species: list[str]              # copied from the rule
    indication: str
    weight_kg: Decimal
    dose_mg_low: Decimal
    dose_mg_high: Decimal | None = None
    route: str
    frequency: str
    notes: str | None = None
    rule: DoseRule                  # provenance chain: result → rule → logical_key → page

class FlaggedInteraction(BaseModel):
    other_drug: str                 # canonical
    passages: list[Passage]         # interaction passages that mention it

class ContraindicationReport(BaseModel):
    kind: Literal["contraindication_report"] = "contraindication_report"
    drug_name: str
    contraindications: list[Passage]
    interactions: list[Passage]
    flagged: list[FlaggedInteraction]   # non-empty only when other_drugs was passed
    unresolved_other_drugs: list[str]   # surfaced, never silently dropped

class IndicationReport(BaseModel):
    kind: Literal["indication_report"] = "indication_report"
    drug_name: str
    species: str | None
    passages: list[Passage]         # species-matching first; never excluded (soft signal)
```

## 5. `DrugIndex` (`tools/drug_index.py`)

Built from the canonical monograph names: `DrugIndex(names: list[str])` plus a
`DrugIndex.from_chunks(path)` convenience. Which file to load is wiring — it stays in the CLI /
Phase 4, not in tools.

`resolve(query) → ResolvedDrug | DrugNotFound`:

1. **Exact** case-insensitive / whitespace-normalized match → `ResolvedDrug(canonical, exact=True)`.
2. Else **fuzzy**: `difflib.get_close_matches`, high cutoff (~0.85). A single close match →
   `ResolvedDrug(canonical, exact=False)` — used to filter, with the canonical name always echoed
   back in tool outputs so the correction is visible, never silent.
3. Else `DrugNotFound(suggestions=…)` — near-misses at a lower cutoff, top 3.

Pure, deterministic, stdlib-only. `ResolvedDrug` is a small internal model
(`canonical: str, exact: bool`) living in `drug_index.py` — it is consumed by the tools, not part
of the tool result unions in `models.py`.

## 6. `RetrieveMonograph` (`tools/retrieve.py`)

```python
class RetrieveMonographInput(BaseModel):
    query: str                    # min_length=1
    drug: str | None = None
    section: SectionType | None = None
    species: str | None = None    # canonicalized ("Dogs" → "dog")
    top_k: int = 5                # ge 1, le 20

class RetrieveMonograph:
    def __init__(self, retriever: Retriever, drugs: DrugIndex, *, rerank: bool = False) -> None: ...
    def __call__(self, inp: RetrieveMonographInput) -> RetrievedPassages | DrugNotFound | NoPassagesFound: ...
```

Flow: resolve `drug` if given (short-circuit `DrugNotFound`); canonicalize `species` via a new
public `canonical_species(text) -> str | None` helper in `ingestion/species.py` (the synonym table
already lives there, currently private); delegate to `Retriever.retrieve()` (Phase 2, unchanged).
A species that doesn't canonicalize (e.g. "axolotl") passes through lowercased as-is — the filter
then simply matches nothing, which surfaces legibly as `NoPassagesFound` rather than an error.
Zero hits → `NoPassagesFound` echoing the filters used, so the agent sees *why* ("no `doses` chunks
for species `ferret`") instead of an empty list.

## 7. `ExtractDoseRule` (`tools/dose_extraction.py`) — the LLM seam, made narrow

### Why an LLM here at all

Doses passages are free-form clinical prose, not tables. The real Metronidazole dog chunk reads
"a) 25 mg/kg PO twice daily in combination with fenbendazole 50 mg/kg PO once daily for 5 days…
b) 50 mg/kg PO once daily for 5 to 7 days; dose may be divided…" — regimens labeled a)/b), doses
for *other* drugs embedded mid-sentence, ranges, durations, and combination-therapy caveats, with
formatting that varies across 636 monographs. Turning that into structured numbers is a
natural-language *reading* task: a regex/rule parser would be brittle, silently wrong in exactly
the cases that matter, and effectively unmaintainable. Reading prose into structure is the one
thing an LLM is genuinely the right tool for here.

What makes it safe is how little the LLM is trusted to do: it only **transcribes** the passage into
candidate structures. It never does arithmetic (that is `calculate_dose`), never chooses among
regimens (that is the deterministic selection logic below), and every number it emits must appear
verbatim in the cited passage or the regimen is discarded. A hallucinated dose cannot survive the
seam.

### Contract

```python
class ExtractedRegimen(BaseModel):      # raw LLM output — untrusted
    indication: str
    mg_per_kg_low: Decimal
    mg_per_kg_high: Decimal | None = None
    route: str
    frequency: str
    notes: str | None = None

class RegimenExtractor(Protocol):       # the seam (Phase 2 QueryPhraser pattern)
    def extract_regimens(self, passage_text: str) -> list[ExtractedRegimen]: ...

class ExtractDoseRuleInput(BaseModel):
    passage: Passage
    indication: str | None = None
    all_regimens: bool = False          # list-all mode: return every grounded regimen

class ExtractDoseRule:
    def __init__(self, extractor: RegimenExtractor) -> None: ...
    def __call__(self, inp: ExtractDoseRuleInput) -> DoseRule | DoseRuleSet | NeedsClarification: ...
```

### Deterministic post-processing (all pure, all unit-tested)

- **Grounding check** (always, in every mode): each regimen's dose numbers — `mg_per_kg_low` and
  `mg_per_kg_high` — must appear verbatim in the passage text (string match of the number).
  A regimen with a dose number not found in the text is discarded as hallucinated (noted in
  `NeedsClarification.reason` when it changes the outcome). Free-text fields (`route`, `frequency`,
  `notes`) are transcription, not safety-critical arithmetic inputs, and are not number-checked.
- **List-all mode** (`all_regimens=True`): return every grounded regimen as a `DoseRuleSet` — for
  "what are all the dosing options?" questions. Nothing survives grounding → `NeedsClarification`.
  The safety contract is unchanged: `calculate_dose` still accepts exactly *one* `DoseRule`, so
  listing never skips the explicit selection step before any math.
- **Selection mode** (default) — indication matching: case-insensitive substring both ways
  ("giardia" ↔ "Giardiasis"). Exactly one surviving regimen matches → `DoseRule` (built from the
  regimen + the passage's citation fields). Zero or several → `NeedsClarification` carrying all
  grounded candidates. No indication given and multiple regimens → `NeedsClarification` with
  candidates.

### The real implementation

The LLM hides behind the `RegimenExtractor` protocol so the model call is swappable and fakeable.
`AnthropicRegimenExtractor` is the one real impl: Claude (`config.reasoning_model`, key via the
existing `SecretStr` setting) with **forced tool-use structured output** returning the regimen
list — forced tool use means the reply *must* be a schema-valid regimen list, never free text to
parse. CI never calls it: tool tests inject a fake extractor; the Anthropic impl's
request-building / response-parsing is tested against a faked client.

**Dependency promotion:** `anthropic` moves from dev-only to a main dependency in this phase. The
Phase 2 spec anticipated promotion "when the agent needs it at runtime" — that moment is
`extract_dose_rule`, not Phase 4.

## 8. `calculate_dose` (`tools/dose_math.py`) — pure, exhaustively tested

```python
class CalculateDoseInput(BaseModel):
    weight: Decimal                     # gt 0, le 5000 (sane physiological bound, cattle-inclusive)
    weight_unit: Literal["kg", "lb"] = "kg"
    rule: DoseRule

def calculate_dose(inp: CalculateDoseInput) -> DoseResult: ...
```

Per the parent spec's locked safe-arithmetic rules:

- Fixed arithmetic only (`weight_kg × mg_per_kg`, and `× mg_per_kg_high` when the rule has a
  range) — **no expression evaluation of any kind, ever** (no `eval`/`exec`/`sympify` on any
  LLM-derived string).
- `Decimal` end-to-end — exact; no rounding inside the tool (presentation is the agent's job).
- `pint` performs the lb→kg conversion, so a unit mistake is structurally impossible.
- pydantic bounds reject nonsense before arithmetic runs.

Species consistency is guaranteed *by construction*: the rule's species came from a hard-filtered
Doses passage, and the result embeds the full `rule` for provenance — a cat can never be dosed from
a dog's chunk.

## 9. Report tools — and why they don't use ANN

`find_contraindications` and `list_indications` mean "give me **the whole section** for this
drug," not "find the most similar chunks." Filtered ANN with a big `top_k` *usually* returns
everything, but HNSW gives no completeness guarantee — unacceptable when the contract is "list
*all* contraindications."

So this phase adds **`fetch_section(drug, section) → list[Passage]`** to the `VectorStore`
protocol and `QdrantVectorStore`, implemented as an exact filtered **scroll** — no vector,
deterministic, complete, ordered by `logical_key` for stable output. Small addition, tested
alongside the existing vector-store tests, and it spares both tools an embedding round-trip
entirely (no query text needed).

### `FindContraindications` (`tools/contraindications.py`)

`FindContraindications(store, drugs)` — `(drug, other_drugs: list[str] = []) →
ContraindicationReport | DrugNotFound`.

Fetches the `contraindications` **and** `drug_interactions` sections (per parent spec §6's explicit
note). Each `other_drug` is resolved via `DrugIndex`, then flagged by case-insensitive name match
within the fetched passage texts; unresolvable names land in `unresolved_other_drugs` — surfaced,
never silently dropped.

*Known limitation (accepted):* matching is by canonical monograph name — brand-name mentions in
prose won't match. Plumb's interaction lists use generic names, so impact is low.

### `ListIndications` (`tools/indications.py`)

`ListIndications(store, drugs)` — `(drug, species?) → IndicationReport | DrugNotFound`.

Fetches the `indications` section. When `species` is given, passages tagged with that species (or
`all`) sort first — **never excluded**. Prose species tags are best-effort mentions (verified in
the live corpus: 148 `indications` chunks are tagged `["all"]`); hard-filtering would wrongly drop
them.

## 10. Dependencies, config, CLI

- **Deps:** add `pint` (main); promote `anthropic` dev → main. Verify latest versions on PyPI
  before pinning (standing memo).
- **Config:** zero new settings — `reasoning_model`, `anthropic_api_key`, `retrieval_top_k`, and
  `rerank_enabled` already cover it.
- **CLI (the phase demo):** one new command —
  `vet-agent dose "<question>" --drug X --species dog --weight-kg 12 --indication giardia`
  (plus `--all-regimens`) — chaining retrieve → extract → calculate and printing the result or the
  clarification options, with citations. The Phase-3 equivalent of Phase 2's manual verification
  task: the only piece needing a live Qdrant + an API key; everything else is offline.

## 11. Testing strategy (offline, fast, deterministic — Phase 2's bar)

| Module | What's proven |
|---|---|
| `dose_math` | The exhaustive suite: single dose, range, lb conversion (pint), mcg-scale values (0.0025 mg/kg microdoses), Decimal exactness (the `0.1`-style cases that break floats), bound rejections (zero/negative/absurd weight, absurd mg/kg). |
| `drug_index` | Exact / case / fuzzy hit; fuzzy near-miss → suggestions; garbage → empty suggestions. |
| `dose_extraction` | `FakeRegimenExtractor` drives tool tests: happy path, indication disambiguation, multi-candidate → `NeedsClarification` with candidates, list-all mode → `DoseRuleSet` with every grounded regimen, grounding rejection (fake returns a hallucinated number → regimen discarded, in both modes). `AnthropicRegimenExtractor` parse/build tested with a faked client. No API calls in CI. |
| `retrieve` / `contraindications` / `indications` | In-memory Qdrant + the existing `FakeEmbedder`/`FakeReranker` from `tests/knowledge/fakes.py`. |
| `fetch_section` | Completeness + stable ordering, added to `tests/knowledge/test_vector_store.py`. |

`make check` stays green throughout: ruff, mypy strict (package-scoped), fast suite.

## 12. Definition of Done

- [ ] `make check` green (ruff + mypy strict + pytest), all new tests offline and fast.
- [ ] All five tools implemented with the contracts in §4–§9; `calculate_dose` exhaustively tested.
- [ ] `VectorStore.fetch_section()` + `canonical_species()` shipped with tests.
- [ ] `anthropic` promoted to main deps; `pint` added.
- [ ] Manual verification (the phase demo): `vet-agent dose` answers a real dose question end to
      end against live Qdrant — e.g. metronidazole, 12 kg dog, giardia → a cited dose (or the
      candidate regimens when ambiguous), and `--all-regimens` lists every grounded option.

## 13. What's Next (later plans)

- **Phase 4 — Agent:** LangGraph `StateGraph` binding these tools, scope/answer guardrail nodes,
  prompts, typed `AgentState`.
- **Phases 5–8:** FastAPI + Typer client, full eval (RAGAS + golden set — dose accuracy asserts
  exact `calculate_dose` outputs), Langfuse observability, Docker/CI hardening.
