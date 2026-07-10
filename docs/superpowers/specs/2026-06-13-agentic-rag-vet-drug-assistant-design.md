# Agentic RAG — Veterinary Drug Assistant (Design Spec)

**Date:** 2026-06-13
**Status:** Approved design — ready for implementation planning
**Data source:** *Plumb's Veterinary Drug Handbook, 10th Edition* (PDF, 738+ drug monographs)

## 1. Purpose & Goals

Build a production-grade **agentic RAG** system that answers veterinary drug questions
grounded in Plumb's Veterinary Drug Handbook. The project doubles as a learning vehicle for
modern, production-grade AI-engineering best practices (something concrete and impressive to
discuss in AI-engineer interviews).

**Three target flows:**

1. **Dosage calculator** — given animal species, weight, and treatment/indication, compute the dose.
2. **Contraindications & warnings** — including interactions with already-administered drugs.
3. **Indications** — what conditions a drug treats.

**Non-negotiable principle:** safety-critical answers (especially dosing) are **grounded in the
source, cited, and never the product of LLM free-form arithmetic.**

## 2. Key Decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| Safety model | **Deterministic tools + grounded RAG** | LLM never does dose arithmetic; a pure-Python tool computes it. Every answer cites the source; agent refuses/escalates when data is ambiguous or missing. |
| Deliverable | **FastAPI service + thin Typer CLI client** | Deployable, testable, async, clear API contracts; UI can be added later. |
| Reasoning LLM | **Claude Sonnet 4.6** (`claude-sonnet-4-6`), config-swappable (e.g. to Sonnet 3.5) | Strong tool-use + reasoning at a reasonable cost. Behind a config flag. |
| Embeddings | **Pluggable `Embedder` interface**; default a medical-domain model (MedEmbed family), with a general-purpose baseline for comparison | Claude has no first-party embeddings API. Domain-vs-general is chosen **empirically** via the retrieval eval. |
| Reranker | **Pluggable `Reranker`** (cross-encoder / `zerank`-style), optional | High-value retrieval-quality win; off by default until measured. |
| Vector store | **Qdrant** (Docker) | Purpose-built; clean **filtered ANN** (filterable HNSW), payload indexes, built-in hybrid search. Dataset is tiny (~5–15k chunks) so performance is irrelevant — chosen for filtering ergonomics + learning value. pgvector was the considered alternative (one fewer service). |
| Agent orchestration | **LangGraph — guarded tool-calling agent (Option A)** | Agentic tool-use core wrapped by **deterministic guardrail nodes**. Beat a router/multi-agent design (over-engineered for 3 flows over one KB) and a raw-SDK loop (slower to the real goals). |
| Eval | **RAGAS** (RAG metrics) + **pytest** (deterministic dose-accuracy) | Different concerns: RAG faithfulness/precision via RAGAS; dose math asserted exactly, never LLM-judged. |
| Observability | **Langfuse** (self-hosted via Docker) + structured logs | Trace every node/tool/retrieval/token/cost per `trace_id`. |
| Infra | **docker-compose** (Qdrant + Langfuse) + **Dockerfile** + **GitHub Actions CI** | Deployability + engineering hygiene; CI gates on lint/type/test/eval. |
| Tooling | Python 3.12, **uv**, ruff, mypy, pytest | Fast, typed, already partially set up. |

## 3. Architecture Overview

Two decoupled halves:

- **Ingestion pipeline (offline):** PDF → parsed monographs → structure-aware chunks → embeddings → Qdrant. Built and tested once, re-run on demand.
- **Agent service (online):** FastAPI app exposing a LangGraph agent that answers questions over the knowledge base.

```
query
  │
  ▼
[scope guardrail] ── off-topic / human-medicine ──▶ refuse + "consult a vet" disclaimer
  │ ok
  ▼
┌──────────────── AGENT NODE (Claude + tools) ────────────────┐
│  loops, calling tools until it has a grounded answer:        │
│   • retrieve_monograph(query, drug?, section?, species?)     │
│   • extract_dose_rule(passage) → DoseRule | NeedsClarification│
│   • calculate_dose(weight_kg, rule)        ← pure Python      │
│   • find_contraindications(drug, other_drugs?)               │
│   • list_indications(drug, species?)                         │
└──────────────────────────┬──────────────────────────────────┘
  │ draft answer + citations
  ▼
[answer guardrail] citations present? confidence ok?
  │ ok                              │ no
  ▼                                 ▼
final answer                  refuse / "consult a vet"
```

Guardrail nodes are **deterministic** — safety does not rely on the model "remembering" to behave.

## 4. Repository Layout (src layout)

```
vet-agent/
├── src/vet_agent/
│   ├── ingestion/      # pdf parse → Monograph model → chunk → embed → load
│   ├── knowledge/      # Embedder, Reranker, VectorStore interfaces + impls
│   ├── tools/          # deterministic tools (retrieve, extract_dose_rule, calculate_dose, ...)
│   ├── agent/          # LangGraph graph, nodes, guardrails, prompts, state
│   ├── api/            # FastAPI app, routes, pydantic schemas
│   ├── cli/            # Typer client (ask, ingest, eval)
│   ├── eval/           # golden set + eval runners + metrics
│   └── config.py       # pydantic-settings (models, keys, thresholds)
├── tests/
├── data/               # PDF (gitignored) + built artifacts
├── docker-compose.yml  # qdrant + langfuse
├── Dockerfile
└── docs/superpowers/specs/
```

Design intent: `tools/` and `knowledge/` are **framework-agnostic and independently testable**;
LangGraph only orchestrates them. This both demonstrates fundamentals and makes swapping/benchmarking trivial.

## 5. Ingestion Pipeline

Discrete, independently testable stages:

1. **Parse PDF → text** with page numbers preserved (pypdf — confirmed to extract cleanly).
2. **Segment into monographs** using the Table of Contents (canonical drug list + page numbers) to find boundaries, then split each monograph into **labeled sections** via the consistent headers (`Uses/Indications`, `Contraindications/Precautions/Warnings`, `Adverse Effects`, `Drug Interactions`, `Doses`, `Pharmacology/Pharmacokinetics`, etc.).
3. **Typed `Monograph` model** (pydantic): `drug_name`, `synonyms`, `drug_class`, `sections[section_type] → text`, `page_start`, `page_end`.
4. **Structure-aware chunking** (see §6).
5. **Embed** each chunk via the pluggable `Embedder`.
6. **Load** into Qdrant with metadata payload + payload indexes on `drug_name` / `section_type` / `species`.

**Idempotent, content-hash-based re-indexing:**
- Each chunk has a stable logical key (`drug + section + species + ordinal`); Qdrant point ID = `hash(logical_key)` → upserts overwrite, never duplicate. (`ordinal` = the sub-chunk index when one `(drug, section, species)` unit is too long for a single chunk and must be size-split; most units fit in one chunk → `ordinal=0`, ordered by reading order.)
- A `content_hash = sha256(text + metadata)` is stored on the point; on re-run, unchanged chunks are skipped (no re-embed), changed/new chunks are re-embedded + upserted, and points whose logical key disappears are pruned.
- Result: `vet-agent ingest` is one command; first run embeds everything, later runs after a parser tweak touch only affected chunks. (Optional future `ingest_manifest.json` for diffing without querying Qdrant.)

**Failure handling:** missed drugs / malformed headers are logged to `parse_report.json` (drugs parsed, sections per drug, anomalies) rather than failing silently — an auditable ingestion report.

## 6. Chunking & Retrieval Strategy

**Structure-aware, section-level chunking with metadata filtering.** A chunk = one section (or a
size-based sub-split of a long one), never crossing section/drug boundaries. Each chunk's Qdrant
payload: `drug_name`, `section_type`, `species` (list-valued), `page`.

**Species granularity — two different treatments by section type:**

- **Doses section → HARD species filter.** The Doses section is cleanly delimited by species
  sub-headers (`DOGS:`, `CATS:`, `DOGS & CATS:`, `HORSES:`, …). We split it into **one chunk per
  species**. `DOGS & CATS:` → list-valued `species:["dog","cat"]` (Qdrant matches if any value
  matches). Safety-critical: a cat must never retrieve a dog's dose, so species is enforced strictly.
  Unrecognized sub-headers → `species:["unspecified"]` + logged to `parse_report.json`.
- **Every other (prose) section → SOFT species signal.** This is the general rule for *all*
  non-Doses sections — not just a few. Cross-species prose, not cleanly splittable: keep the section
  as one chunk (size-split if long), tag `species` best-effort (mentioned species), and use it to
  **boost ranking, not hard-exclude** (indications/effects are often shared across species).

**Canonical `section_type` enum** (every chunk is tagged with exactly one; `doses` is the only
species-split, hard-filtered type — all others follow the prose rule):
`prescriber_highlights`, `indications`, `contraindications`, `adverse_effects`,
`reproductive_safety`, `overdose_toxicity`, `drug_interactions`, `pharmacology`,
`pharmacokinetics`, `monitoring`, `client_information`, `chemistry`, `storage`, `dosage_forms`,
`doses`. Unrecognized headers → `section_type="other"` + logged to `parse_report.json`.
Note: `find_contraindications` pulls both `contraindications` **and** `drug_interactions`.

**Retrieval** (`knowledge/`): `retrieve(query, filters)` → metadata-filtered ANN → top-k →
optional rerank → passages **with citations** (`drug + section + page`). The agent's tools pass
structured filters (e.g. `section_type="doses", species="dog"`) so retrieval is precise rather than
relying on semantic similarity alone.

**Worked example:** *"Metronidazole dose for a 12 kg dog with Giardia?"* →
`retrieve(filter: drug=metronidazole AND section=doses AND species∋dog)` → matches only the DOGS
dose chunk → `extract_dose_rule` → `{25 mg/kg, PO, q12h}` → `calculate_dose(12 kg)` → 300 mg.
Cat/horse dose chunks are never considered.

**Dose-rule extraction is query-time, not index-time:** the agent retrieves the cited Doses
passage, then `extract_dose_rule` turns *that specific passage* into a structured rule — so the
structured rule is always traceable to a citation. A precomputed structured dose table is explicit
future scope.

## 7. Tools (`tools/`)

Pure, framework-agnostic, individually unit-tested. Typed (pydantic) I/O.

| Tool | Signature (sketch) | Notes |
|---|---|---|
| `retrieve_monograph` | `(query, drug?, section?, species?) → list[Passage]` | Metadata-filtered ANN + optional rerank. `Passage` carries `drug, section, species, page, text`. |
| `extract_dose_rule` | `(passage) → DoseRule \| NeedsClarification` | LLM **structured output** over the retrieved passage → `{mg_per_kg, route, frequency, species, indication, dose_range?}`. Returns `NeedsClarification` on ambiguity / multiple regimens. |
| `calculate_dose` | `(weight_kg, DoseRule) → DoseResult` | **Pure Python, no LLM.** Total dose (+ range), unit-checked. Exhaustively unit-tested. See safe-arithmetic rules below. |
| `find_contraindications` | `(drug, other_drugs?) → ContraindicationReport` | Contraindications + Drug-Interactions sections; flags interactions with already-given drugs. |
| `list_indications` | `(drug, species?) → IndicationReport` | Uses/Indications section. |

The seam that makes dosing safe: **only `extract_dose_rule` does LLM reasoning; `calculate_dose`
is pure code.**

**Safe-arithmetic rules for `calculate_dose`:**
- It is **not an expression evaluator** — it does fixed arithmetic (`weight × mg_per_kg`) on a
  validated, structured `DoseRule`. **No `eval`/`exec`/`sympy.sympify` on any LLM-derived string,
  ever.** (If expression parsing were ever needed, use a restricted AST evaluator such as `asteval`,
  never raw `eval`.)
- **`decimal.Decimal`** for arithmetic — avoids float rounding error on medical doses.
- **`pint`** for units and conversions (kg↔lb, mg↔mcg) so a unit mismatch is structurally impossible.
- **pydantic** validation on all numeric inputs (non-negative weight, sane physiological bounds).

## 8. Agent (`agent/`)

- **LangGraph state graph**, Option A: `scope_guardrail → agent_node ⇄ tools → answer_guardrail`.
- **Agent type:** a **ReAct-style tool-calling agent** — the `agent_node ⇄ ToolNode` loop *is* the
  reason→act→observe pattern. With Claude's **native tool use** the model emits structured
  `tool_use` blocks (no free-text scratchpad parsing), the modern evolution of classic ReAct. We
  build a **custom `StateGraph`** (not the bare `create_react_agent` prebuilt) so we can attach the
  deterministic guardrail nodes and a typed `AgentState`.
- **`agent_node`**: Claude with tools bound, looping until grounded. System prompt enforces: always
  retrieve before answering; never compute dose math (call `calculate_dose`); always cite
  drug+section+page; say "I don't know" when retrieval is empty.
- **`AgentState`** (typed): messages, retrieved passages, tool results, citations, confidence flags —
  the object that is traced and evaluated.
- **Guardrail nodes (deterministic):**
  - *Scope (pre):* reject human-medicine / non-drug / off-topic → polite refusal + "consult a
    licensed veterinarian" disclaimer.
  - *Answer (post):* block answers lacking citations or flagged low-confidence; every dose answer
    ships with source citation + safety disclaimer.
- **Multi-drug contraindication queries** ("can I give X after Y?") are handled by the agent calling
  the relevant tools per drug and reasoning over both reports — no special routing.

## 9. API & CLI

- **FastAPI:** `POST /ask` (answer + citations + `trace_id`), `POST /ask/stream` (SSE — watch the
  agent reason), `GET /health`. Pydantic request/response schemas are the contract; per-request
  `trace_id` ties into Langfuse.
- **Typer CLI:** `vet-agent ask "..."`, `vet-agent ingest`, `vet-agent eval` — thin HTTP client over
  the API so CLI and any future UI share one backend.

## 10. Evaluation (`eval/`)

- **Golden set** (`golden_set.yaml`): ~30–50 hand-built cases across all three flows; each with
  expected drug/section/page (retrieval), and for dose questions the **exact expected number**.
- **Three layers, run in CI:**
  - *Retrieval* — recall@k / hit-rate; also the harness for **benchmarking MedEmbed vs a general
    embedder** (empirical embedder choice).
  - *Answer quality* — RAGAS faithfulness + context precision (grounded, no hallucination).
  - *Dose accuracy* — pytest **exact-match** on `calculate_dose`; deterministic, never LLM-judged.
- `vet-agent eval` prints a scorecard; CI fails on regression below thresholds.

## 11. Observability

Langfuse via the LangGraph callback: every run traces each node, tool call, retrieved passage,
token count, latency, and cost, keyed by `trace_id`. Used for debugging ("why that dose?") and demos.

## 12. Guardrails & Safety

- Scope guardrail (pre) and answer guardrail (post) as in §8.
- Typed tool errors (`drug-not-found`, `empty-retrieval`, ambiguous-dose → `NeedsClarification`)
  bubble up as structured states the agent handles gracefully — it asks a clarifying question or
  says "not found," never guesses.
- Every dose answer carries its citation + a "consult a licensed veterinarian" disclaimer.

## 13. Infrastructure

- `docker-compose.yml` brings up **Qdrant + Langfuse**; `Dockerfile` packages the API.
- **GitHub Actions CI:** ruff + mypy + pytest + the eval scorecard on every PR.

## 14. Phased Roadmap

Each phase ends in something runnable + tested, and becomes its own spec → plan → implementation cycle.

| Phase | Deliverable | Demo |
|---|---|---|
| **0. Scaffold** | uv project, config, ruff/mypy/pytest, docker-compose (Qdrant) | make targets, green CI |
| **1. Ingestion** | PDF → `Monograph` → section/species chunks + parse_report | inspect parsed monographs |
| **2. Knowledge layer** | Embedder/Reranker/VectorStore interfaces, load into Qdrant | filtered retrieval from a script |
| **3. Tools** | the 5 tools, esp. pure-Python `calculate_dose` (unit-tested) | dose math + retrieval in isolation |
| **4. Agent** | LangGraph graph + guardrails + prompts | end-to-end Q&A (CLI) |
| **5. API/CLI** | FastAPI (+streaming) + Typer client | HTTP `/ask`, the real product |
| **6. Eval** | golden set + 3 eval layers + embedder benchmark | scorecard; empirical embedder choice |
| **7. Observability** | Langfuse tracing + cost logging | full trace of a live query |
| **8. Harden** | guardrail polish, Dockerfile, CI eval gate | deployable, gated repo |

This brainstorm produces the overall design spec plus enough detail to start Phases 0–1.

## 15. Out of Scope (explicit)

- Precomputed structured dose table (query-time extraction instead, for v1).
- Fine-grained indication-within-species tagging for prose sections.
- **Reverse lookup** (condition → candidate drugs); v1 retrieval is drug-keyed. New question types
  are added later as eval cases and debugged if they fail — not pre-designed.
- **Independent physical scale-out** (Qdrant clustering, a standalone embedding-inference service,
  autoscaling). The design is decoupled via interfaces + a stateless API and *permits* this, but
  building it is out of v1 scope.
- Web/chat UI (FastAPI + CLI only; UI can be added on the existing backend later).
- Human medicine, non-drug veterinary questions (refused by the scope guardrail).
- Multi-tenant auth, rate limiting, and other operational concerns beyond the learning goal.

## 16. Disclaimer

This is a learning/portfolio project, not a certified medical device. All answers are decision
*support* grounded in Plumb's, carry citations + a "consult a licensed veterinarian" disclaimer, and
the system refuses rather than guesses when data is missing or ambiguous.
