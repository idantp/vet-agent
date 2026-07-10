# Vet-Agent Phase 2 — Knowledge Layer (Design Spec)

**Date:** 2026-06-28
**Status:** ✅ **Implemented (2026-07)** — plan `docs/superpowers/plans/2026-06-30-vet-agent-phase-2-knowledge-layer.md`. As-built deviations from this design: Qwen3-0.6B dropped (2 models benchmarked; **bge-base won**); species sampling made cat/dog-focused (not "companion-dominant with capped exotics"); added an `embed` command + MPS device selection + `SecretStr` API-key handling; reranker interface/impl shipped but lift not yet measured on the frozen set.
**Depends on:** Phase 0–1 (ingestion) — `docs/superpowers/plans/2026-06-19-vet-agent-phase-0-1-ingestion.md`
**Parent spec:** `docs/superpowers/specs/2026-06-13-agentic-rag-vet-drug-assistant-design.md` (§2, §5–6, §10, §14 Phase 2)

## 1. Purpose & Scope

Build the **Knowledge layer**: pluggable `Embedder` / `Reranker` / `VectorStore` interfaces with
concrete implementations, embed the Phase-1 chunks, and idempotently load them into Qdrant using the
existing `logical_key` / `content_hash` contract. The headline deliverable is an **empirical
benchmark** that picks the default embedding model **on our own corpus** (medical-domain vs. general)
rather than by assumption.

**In scope (Phase 2):**

- `knowledge/` package: `Embedder`, `Reranker`, `VectorStore` Protocols + a `Passage` model.
- Concrete impls: `SentenceTransformerEmbedder` (registry-driven), `CrossEncoderReranker`,
  `QdrantVectorStore`.
- A self-contained **retrieval eval set** (`retrieval_eval.yaml`) — metadata-labeled, LLM-phrased,
  committed, frozen.
- An **in-memory embedder benchmark** across the candidate models + a **reranker-lift** measurement →
  a committed scorecard, which chooses the default model.
- An **idempotent Qdrant loader** (embed-only-changed, prune orphans).
- A **filtered-retrieval demo** (`retrieve`) that validates the production path end-to-end.

**Out of scope (deferred):** hybrid sparse/BM25 retrieval (Qdrant supports it; defer to retrieval
tuning), vector quantization / GPU inference, the Phase-6 RAGAS + golden set, multi-collection
production serving / scale-out.

## 2. Key Decisions (locked in brainstorm)

| Decision | Choice | Rationale |
|---|---|---|
| Embedder lineup | **MedEmbed-base-v0.1** (768d, medical) vs **bge-base-en-v1.5** (768d, general) | MedEmbed-base is a *direct fine-tune* of bge-base, so the A/B isolates exactly the medical-fine-tuning benefit. **Deviation:** Qwen3-0.6B was in the original 3-model lineup as a modern-strong reference but was **dropped during execution** — the core question is medical-vs-general, and a 600M model was heavy on the target (fanless M3) machine. **Outcome (2026-07): bge-base won** (recall@5 0.764 vs 0.741, mrr 0.781 vs 0.746) — the human-medical fine-tune did not transfer to veterinary drug-handbook text, so `embedding_model` defaults to `bge-base`. |
| Inference | **Local via `sentence-transformers`** | No embeddings API dependency; matches the pluggable/local learning goal; corpus is tiny. |
| Benchmark path | **In-memory exact cosine** for the A/B, **Qdrant only for the chosen model** | ANN is lossy; routing the comparison through HNSW would blend embedder quality with index drop-rate. In-memory numpy is exact, fast (15k×768 ≈ 47 MB), deterministic. |
| Eval set | **Metadata-labeled, LLM-phrased, frozen YAML** | Chunk metadata (`drug`/`section`/`species`) gives free ground-truth labels; Claude writes natural phrasing. Generation is a one-time offline step → deterministic, key-free CI. |
| Reranker scope | **Interface + `bge-reranker-v2-m3` impl + measure lift** | The eval harness exists in this phase, so measuring reranker lift is near-free and yields a data-backed off/on default. Off unless it wins. |
| Idempotency source of truth | **Qdrant** (scroll existing `{id: content_hash}`) | No drift-prone sidecar manifest; Qdrant stays canonical. Skip unchanged, embed+upsert changed/new, prune orphans. |
| Collection naming | **Name-suffixed per model from day one** (`{prefix}__{model_key}`) | Makes querying a collection with the wrong model's vectors structurally impossible; re-benchmarking a new model never clobbers. |
| CLI surface | **Four commands**: `ingest` (exists) · `benchmark` · `load` · `retrieve` | Discrete, independently testable stages — the project's standing philosophy. |
| Eval-set generation | **Separate `scripts/build_eval_set.py`** (not a CLI command) | A one-time, Anthropic-key-requiring offline op; keeps the four CLI commands and CI deterministic + key-free. |

## 3. Architecture & Module Layout

```
src/vet_agent/knowledge/
├── interfaces.py     # Embedder, Reranker, VectorStore Protocols + Passage, PointPayload models
├── embedders.py      # SentenceTransformerEmbedder + MODEL_REGISTRY + get_embedder(key)
├── rerankers.py      # CrossEncoderReranker (bge-reranker-v2-m3)
├── vector_store.py   # QdrantVectorStore (impl of VectorStore)
├── loader.py         # idempotent load: chunks.json -> Qdrant
└── retrieval.py      # Retriever: filtered semantic search (+ optional rerank)

src/vet_agent/eval/
├── eval_set.py       # load/validate retrieval_eval.yaml; relevance-label derivation helpers
├── metrics.py        # recall@k, MRR, hit-rate@k (pure functions)
└── benchmark.py      # run candidate models in-memory -> scorecard

scripts/
└── build_eval_set.py # one-time: sample targets -> Claude phrasing -> retrieval_eval.yaml

data/eval/retrieval_eval.yaml          # committed, frozen eval set
data/eval/benchmark_scorecard.{json,md} # committed benchmark output
data/embeddings/<model_key>.npz        # gitignored embedding cache
```

Design intent (unchanged from the parent spec): `knowledge/` is framework-agnostic and independently
testable; nothing here imports LangGraph. The interfaces are the swap/benchmark seam.

## 4. Interfaces (`knowledge/interfaces.py`)

```python
class Passage(BaseModel):
    drug_name: str
    section_type: SectionType
    species: list[str]
    book_page: int
    text: str
    logical_key: str
    score: float | None = None

class PointPayload(BaseModel):
    point_id: str           # uuid5(NAMESPACE_VET, logical_key)
    vector: list[float]
    drug_name: str
    section_type: SectionType
    species: list[str]
    book_page: int
    text: str
    logical_key: str
    content_hash: str

class Embedder(Protocol):
    name: str
    dim: int
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...

class Reranker(Protocol):
    name: str
    def rerank(self, query: str, passages: list[Passage], top_k: int) -> list[Passage]: ...

class VectorStore(Protocol):
    def ensure_collection(self, dim: int) -> None: ...
    def existing_hashes(self) -> dict[str, str]: ...      # point_id -> content_hash
    def upsert(self, points: list[PointPayload]) -> None: ...
    def delete(self, ids: list[str]) -> None: ...
    def search(
        self, vector: list[float], *,
        drug: str | None = None,
        section: SectionType | None = None,
        species: str | None = None,
        top_k: int = 5,
    ) -> list[Passage]: ...
```

`embed_documents` vs. `embed_query` are split so a query-prefixed model could apply its instruction
prefix to queries only; for the two shipped models (BGE/MedEmbed) the paths are identical (no prefix). **All embedders
L2-normalize outputs**, so cosine == dot product everywhere.

## 5. Embedder impls + model registry (`knowledge/embedders.py`)

A single `SentenceTransformerEmbedder` parameterized by a registry entry — no per-model subclasses:

```python
@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    dim: int
    query_prefix: str | None = None   # applied in embed_query only
    matryoshka_dim: int | None = None # truncate+renormalize to this dim (kept for future models)

MODEL_REGISTRY: dict[str, ModelSpec] = {
    "medembed-base": ModelSpec("abhinand/MedEmbed-base-v0.1", dim=768),
    "bge-base":      ModelSpec("BAAI/bge-base-en-v1.5",       dim=768),
}

def get_embedder(key: str) -> Embedder: ...
```

**As-built:** only these two 768-d models ship (Qwen3-0.6B was dropped, see §2). `query_prefix` /
`matryoshka_dim` remain on `ModelSpec` as unused-but-ready knobs for adding a query-prefixed or
larger model later. `SentenceTransformerEmbedder` also selects the Apple **MPS** device when
available (falls back to CPU) to speed up on-device embedding.

Adding a model = one registry line. The default model key lives in `config.py` and is set to the
benchmark winner. Exact prefix strings and the current `sentence-transformers` API (`encode` vs.
`encode_query`/`encode_document`, available since ST 5.0) are confirmed against the live model cards
during implementation; **verify the latest releases on PyPI/HF before pinning** (standing memo).

## 6. Reranker impl (`knowledge/rerankers.py`)

`CrossEncoderReranker` wraps `sentence_transformers.CrossEncoder("BAAI/bge-reranker-v2-m3")`
(Apache-2.0, classic XLM-R cross-encoder — loads with plain `CrossEncoder(...)`, no
`trust_remote_code`, no ST-version risk). `rerank(query, passages, top_k)` scores each
`(query, passage.text)` pair, sorts descending, returns the top_k with updated `score`. Off by
default (`config.rerank_enabled = False`); the benchmark decides whether it's worth turning on.

## 7. Qdrant schema (`knowledge/vector_store.py`)

- **Collection name** is *derived*: `f"{qdrant_collection_prefix}__{embedding_model_key}"`
  (e.g. `vet_chunks__medembed_base`; non-alphanumerics in the key normalized to `_`). Both `load` and
  `retrieve` resolve the name from the configured model, so a collection is **always** queried with
  the same model that wrote it.
- **Vector params:** size = model `dim`, **distance = Cosine**.
- **Point ID:** `str(uuid5(NAMESPACE_VET, logical_key))` — deterministic, re-derivable, satisfies
  Qdrant's UUID/int ID requirement. `NAMESPACE_VET` is a fixed module-level UUID constant.
- **Payload:** `{drug_name, section_type, species[], book_page, text, logical_key, content_hash}`.
- **Payload indexes:** keyword indexes on `drug_name`, `section_type`, and `species` (list-valued →
  Qdrant "match any"), created in `ensure_collection`.
- **Filtered search:** `search()` builds a Qdrant `Filter` from the optional `drug`/`section`/
  `species` args and relies on **filterable HNSW** (filter applied *inside* graph traversal — the
  reason the parent spec chose Qdrant). Results map back to `Passage` (carrying citation fields +
  score).

A different-dim or different model simply lands in its own suffixed collection; nothing clobbers.

## 8. Idempotent loader (`knowledge/loader.py`)

```
chunks   = read_chunks(chunks_json)                       # list[Chunk]
desired  = { uuid5(NAMESPACE_VET, logical_key(c)): (c, content_hash(c)) for c in chunks }
store.ensure_collection(embedder.dim)
existing = store.existing_hashes()                        # one scroll: id -> content_hash
to_embed = [c for id,(c,h) in desired.items() if existing.get(id) != h]
vectors  = embedder.embed_documents([c.text for c in to_embed])   # only changed/new
store.upsert(build_points(to_embed, vectors, desired))
orphans  = set(existing) - set(desired)                   # logical keys that disappeared
if prune: store.delete(sorted(orphans))
return LoadReport(upserted=len(to_embed), skipped=len(desired)-len(to_embed), pruned=len(orphans))
```

First run embeds all ~15k chunks; a later run after a parser tweak re-embeds only affected chunks and
prunes vanished ones. `logical_key` / `content_hash` are reused verbatim from
`ingestion/chunker.py` — the contract defined in Phase 1.

## 9. Retrieval (`knowledge/retrieval.py`)

`Retriever(embedder, store, reranker=None)` exposes:

```python
def retrieve(self, query: str, *, drug=None, section=None, species=None,
             top_k=5, rerank=False) -> list[Passage]:
    vec = self.embedder.embed_query(query)
    # over-fetch when reranking so the cross-encoder has candidates to reorder
    fetch_k = top_k if not rerank else max(top_k, RERANK_FETCH_K)   # RERANK_FETCH_K = 20
    hits = self.store.search(vec, drug=drug, section=section, species=species, top_k=fetch_k)
    if rerank and self.reranker:
        hits = self.reranker.rerank(query, hits, top_k)
    return hits[:top_k]
```

This is the exact shape Phase 3's `retrieve_monograph` tool will call. Three usage modes fall out:
semantic-only, filtered-semantic (the safety-critical mode — e.g. dog-dose filter so a cat dose can
never surface), and (edge) filter-dominant.

## 10. Retrieval eval set (`scripts/build_eval_set.py` → `data/eval/retrieval_eval.yaml`)

**One-time, offline, committed.** Sampling across **eight question flows** that mirror the real
questions veterinarians ask at the point of care — each mapped to the monograph section that answers
it:

| Flow | Section | Flow | Section |
|---|---|---|---|
| `dose` | `doses` | `interaction` | `drug_interactions` |
| `indication` | `indications` | `adverse_effects` | `adverse_effects` |
| `contraindication` | `contraindications` | `monitoring` | `monitoring` |
| `reproductive_safety` | `reproductive_safety` | `administration` | `client_information` |

Sampling is **cat/dog-focused by default** to mirror small-animal practice: dog/cat targets (plus
deliberately species-agnostic `['all']` prose) fill the set, and exotic / food-animal species are
**excluded by default** (`other_fraction = 0`). Raising `--other-fraction` (e.g. 0.1) reintroduces a
small, capped, species-varied minority of exotics rather than an even spread. For each sampled
`(drug, section, species)` target, Claude writes one natural question from a **practicing-clinician
persona** (patient in front of them; known signalment/diagnosis/organ status), steered by
**flow-matched few-shot examples**, and grounded in a representative chunk's text. One question per
target, focused on that single section (so labels stay clean). Species-applicability ("meloxicam in
rabbits?") is covered by sampling exotic/food-animal species within the dose + contraindication
flows rather than a separate flow; dose adjustments live within the dose flow; food-animal
withdrawal times are deferred (no clean section in Plumb's). **Ground truth = every chunk whose
`(drug, section, species)` matches the target** (multi-chunk targets fully credited), captured as
`relevant_logical_keys`.

`retrieval_eval.yaml` entry schema:

```yaml
- query: "What's the metronidazole dose for a 12 kg dog with giardia?"
  flow: dose                       # one of the 8 flow keys above
  drug: Metronidazole
  section: doses
  species: [dog]
  relevant_logical_keys:
    - "metronidazole|doses|dog|0"
```

`eval/eval_set.py` loads + validates this file into typed objects and exposes the relevance-label
derivation helper (pure, unit-tested); `eval/eval_set_builder.py` holds the flow map, few-shots,
persona prompt, and stratified sampler. The Claude call lives only in the generation step and is
injected behind a small `QueryPhraser` interface so it can be faked in tests. Target size ≈ 8 flows
× ~25 ≈ ~200 queries (per-flow count is a CLI knob).

**Human review gate (draft → review → freeze).** The model-generated question phrasings are the one
non-deterministic, quality-sensitive input to the whole benchmark, so they are **human-approved
before they are frozen**, in a batch flow:

1. **Generate** writes a *draft* (`data/eval/retrieval_eval.draft.yaml`), not the committed file.
2. The author **reviews the draft in one pass** — editing weak/leading/off-target phrasings and
   deleting any bad cases directly in the YAML. Review scope is the `query` strings only; the
   `relevant_logical_keys` labels are derived deterministically from chunk metadata and are not
   hand-edited.
3. **Promote** validates the reviewed draft (parsing it back through `load_eval_set`, so a malformed
   hand-edit fails loudly) and freezes it to the committed `data/eval/retrieval_eval.yaml`.

The draft is gitignored; only the reviewed, frozen set is committed. The benchmark reads only the
frozen file, so CI stays deterministic and Anthropic-key-free.

## 11. Benchmark (`eval/benchmark.py` + `eval/metrics.py`)

For each candidate model key:

1. Embed the full corpus (cache to `data/embeddings/<key>.npz`, keyed by model + a hash of the
   corpus so the cache invalidates when `chunks.json` changes) and all eval queries.
2. For each query, rank **all** chunks by exact cosine (numpy) and locate the ranks of its
   `relevant_logical_keys`.
3. Compute **recall@k, MRR, hit-rate@k** for `k ∈ {1,3,5,10}`, aggregated overall and **per flow**.

Then a **reranker-lift** pass: take each model's top-`RERANK_FETCH_K` candidates, rerank with
`bge-reranker-v2-m3`, recompute metrics, and report the delta.

**Correctness note (tie-handling).** Ranking metrics sort candidates by score **then** by a
deterministic secondary key (`logical_key`), so ties produce reproducible ranks (the standard IR
convention breaks score ties by doc id). Exact cosine over distinct float vectors makes ties
practically impossible here, but we enforce the deterministic sort regardless and unit-test the
metric functions against hand-computed fixtures.

`metrics.py` holds pure ranking-metric functions (unit-tested against synthetic rankings).
Output: `benchmark_scorecard.md` (model × metric table + per-flow breakdown + reranker lift +
declared winner) and `benchmark_scorecard.json`. **The winner (best recall@5 / MRR) becomes
`config.embedding_model`.**

## 12. CLI (`cli/main.py`)

| Command | New? | Behavior |
|---|---|---|
| `ingest <pdf>` | exists | Unchanged (PDF → monographs/chunks/parse_report). |
| `embed` | new | `[--models medembed-base,bge-base] [--chunks] [--cache-dir]` → embed the corpus per model and cache vectors (the slow step); run one model at a time to spread load. |
| `benchmark` | new | `[--chunks] [--eval-set] [--models medembed-base,bge-base] [--k 1,3,5,10] [--cache-dir] [--out]` → reuses the `embed` cache, scores, writes scorecard. |
| `load` | new | `[--chunks] [--model <config default>] [--collection-prefix] [--batch-size 64] [--no-prune]` → idempotent Qdrant upsert; prints `LoadReport`. |
| `retrieve QUERY` | new | `QUERY` (required, natural language) `[--drug] [--section] [--species] [--top-k 5] [--rerank]` → filtered semantic search; prints passages with `drug / section / page` citations. |

`build_eval_set.py` stays a standalone script (one-time, needs an Anthropic key), not a CLI command.

## 13. Config additions (`config.py`)

```python
embedding_model: str = "bge-base"             # benchmark winner (general beat the medical model)
qdrant_collection_prefix: str = "vet_chunks"  # active collection = f"{prefix}__{model_key}"
rerank_enabled: bool = False
reranker_model: str = "bge-reranker-v2-m3"
embedding_batch_size: int = 64
retrieval_top_k: int = 5
```

All `VET_`-prefixed and env-overridable, consistent with Phase 0.

## 14. Testing strategy (fast, offline, deterministic)

- **`FakeEmbedder`** (deterministic hashed vectors, fixed `dim`) drives interface/loader/retrieval
  tests — no model downloads, no network.
- **Qdrant in `:memory:` mode** (`QdrantClient(location=":memory:")`) for `vector_store` and `loader`
  tests — no Docker required. Idempotency proven directly: load twice → all skipped; mutate one
  chunk → exactly that one re-embedded/upserted; drop a chunk → its point pruned.
- **`metrics.py`** unit-tested with synthetic rankings (known recall@k / MRR / hit-rate).
- **`eval_set.py`** label-derivation + YAML validation tested with fixtures; `build_eval_set.py`
  tested with an injected fake LLM client (no Claude call in CI).
- A few **real-model tests marked `slow`/opt-in** (skipped when offline or model not cached) so
  `make check` stays quick and hermetic.
- mypy stays package-scoped (tests excluded); add `ignore_missing_imports` overrides for any
  dependency lacking type stubs.

## 15. Dependencies

Add to `pyproject.toml` (verify latest on PyPI/HF before pinning — standing memo):

- **Main deps:** `sentence-transformers`, `qdrant-client`, `numpy`, `pyyaml`. `torch` (CPU) arrives
  transitively via `sentence-transformers`.
- **Dev group:** `anthropic` — used *only* by the one-time, offline `scripts/build_eval_set.py` to
  phrase eval queries. The package runtime and CI never call Claude (CI reads the committed YAML), so
  it stays out of main deps; Phase 4 promotes it to a main dependency when the agent needs it at
  runtime.

## 16. Definition of Done — ✅ met (2026-07)

- [x] `make check` green (ruff + mypy strict + pytest), tests offline and fast (110 passed, 2 slow-deselected).
- [x] `data/eval/retrieval_eval.yaml` committed (200 cat/dog-focused, human-reviewed cases); `benchmark`
  produced `data/eval/benchmark_scorecard.md` and the default model was chosen **on data** (bge-base).
  *Reranker lift is not yet measured on the frozen set — deferred (interface + impl ship; off by default).*
- [x] `load` proven idempotent by tests (skip-unchanged, re-embed-changed, prune-orphan) against
  in-memory Qdrant, **and** verified on live Qdrant (re-run: `upserted=0 skipped=15292 pruned=0`).
- [x] `retrieve "...12 kg dog..." --section doses --species dog` returns correctly filtered, cited
  passages from the loaded `vet_chunks__bge_base` collection (the Phase 2 demo).

## 17. What's Next (later plans)

- **Phase 3 — Tools:** `retrieve_monograph` (wraps §9 `Retriever`), `extract_dose_rule`,
  pure-Python `calculate_dose`, `find_contraindications`, `list_indications`.
- **Phases 4–8:** LangGraph agent + guardrails, FastAPI/CLI, full eval (RAGAS + golden set,
  reusing this benchmark harness), Langfuse observability, Docker/CI hardening.
