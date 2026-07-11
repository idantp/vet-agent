# vet-agent

Production-grade **agentic RAG** over *Plumb's Veterinary Drug Handbook* — answering veterinary
drug questions (dosing, contraindications, indications) grounded in the source, cited, and
safety-first. Dose math is done by deterministic Python, never free-form LLM arithmetic.

**Status:** Phases 0–2 complete — PDF ingestion, and the knowledge layer (pluggable
Embedder/Reranker/VectorStore, an empirical embedder benchmark, and idempotent Qdrant loading with
filtered semantic retrieval). The LangGraph agent, tools, and FastAPI service are later phases.

- Design: [`docs/superpowers/specs/2026-06-13-agentic-rag-vet-drug-assistant-design.md`](docs/superpowers/specs/2026-06-13-agentic-rag-vet-drug-assistant-design.md)
- Phase 2 spec: [`docs/superpowers/specs/2026-06-28-vet-agent-phase-2-knowledge-layer-design.md`](docs/superpowers/specs/2026-06-28-vet-agent-phase-2-knowledge-layer-design.md)

## Prerequisites

- **Python 3.12+** and **[uv](https://docs.astral.sh/uv/)**
- **Docker** (for Qdrant, at load/retrieve time)
- The Plumb's handbook **PDF** (not in the repo)
- An **Anthropic API key** — only to *regenerate* the retrieval eval set (a frozen copy is committed)

```bash
uv sync           # create the venv and install deps (pulls in torch CPU/MPS)
```

Configuration is via `VET_`-prefixed env vars or a `.env` file (see `.env.example`). Notable ones:
`VET_QDRANT_URL` (default `http://localhost:6333`), `VET_EMBEDDING_MODEL` (default `bge-base`, the
benchmark winner), `VET_ANTHROPIC_API_KEY` (eval-set generation only, stored as a masked secret).

## The workflow

The pipeline is a chain of small CLI commands. `ingest` is offline; `embed`/`benchmark` choose the
embedding model (one-time); `load`/`retrieve` build and query the vector index.

```
ingest ──▶ (embed ──▶ benchmark) ──▶ load ──▶ retrieve
 PDF→chunks   choose the embedder     into Qdrant   filtered search
```

### 1. `ingest` — PDF → typed monographs + chunks

Parses the handbook into structure-aware, species-split chunks with an auditable coverage gate.

```bash
uv run vet-agent ingest path/to/plumbs-handbook.pdf
# → data/ingest/{monographs.json, chunks.json, parse_report.json}   (~15,292 chunks)
```

### 2. `embed` — cache corpus vectors per model (the slow step)

Embeds every chunk with a model and caches the vectors to `data/embeddings/*.npz`. Run per model to
spread the load (fanless laptops throttle); it's a one-time cost and shows a live progress bar.

```bash
uv run vet-agent embed --models bge-base       # or: medembed-base ; comma-separate for several
```

### 3. `benchmark` — pick the embedder empirically

Scores each candidate model against the frozen eval set (`data/eval/retrieval_eval.yaml`) in memory
(reusing the `embed` cache) and writes a scorecard. This is how `bge-base` was chosen as the default.

```bash
uv run vet-agent benchmark                      # models: medembed-base,bge-base
# → data/eval/benchmark_scorecard.md  (recall@k / hit_rate@k / mrr per model + declared winner)
```

Pin the winner in config if it changes: set `VET_EMBEDDING_MODEL` (or `embedding_model` in
`src/vet_agent/config.py`).

### 4. `load` — embed + upsert into Qdrant (idempotent)

Bring up Qdrant, then load. Re-running only re-embeds changed chunks and prunes removed ones
(Qdrant is the source of truth; the collection is named `vet_chunks__<model>`).

```bash
docker compose up -d qdrant
uv run vet-agent load data/ingest/chunks.json
# first run:  upserted=15292 skipped=0 pruned=0
# re-run:     upserted=0 skipped=15292 pruned=0   (idempotent)
```

### 5. `retrieve` — filtered semantic search

Natural-language query with optional hard filters (the safety-critical part: a dog-dose query never
returns a cat dose). This is the shape the Phase-3 agent tool will call.

```bash
uv run vet-agent retrieve "metronidazole dose for a 12 kg dog with giardia" \
    --section doses --species dog
uv run vet-agent retrieve "what is enrofloxacin used for" --drug Enrofloxacin --section indications
# flags: --drug  --section  --species  --top-k 5  --rerank
```

Each hit prints as `[score] Drug / section / p.<page>` + a text snippet — always cited.

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

## Regenerating the eval set (optional)

The committed `data/eval/retrieval_eval.yaml` (200 cat/dog-focused, human-reviewed questions across 8
question flows) is what `benchmark` scores against. To rebuild it — an offline, human-gated step that
needs `VET_ANTHROPIC_API_KEY`:

```bash
uv run python scripts/build_eval_set.py generate     # → data/eval/retrieval_eval.draft.yaml
# review/edit the draft's `query` phrasings (labels are derived — leave them)
uv run python scripts/build_eval_set.py promote      # freeze → data/eval/retrieval_eval.yaml
```

`--other-fraction 0.1` reintroduces a small capped minority of exotic/food-animal species (default
`0` = dog/cat only).

## Development

```bash
make check        # ruff (lint + format) + mypy (strict) + pytest — all offline & fast
uv run pytest -m slow    # opt-in: exercises real models (network + downloads)
```

Tests run fully offline via a deterministic `FakeEmbedder` and Qdrant's in-memory (`:memory:`) mode —
no Docker, no model downloads, no network in the default suite.

## Layout

```
src/vet_agent/
├── ingestion/   # PDF → Monograph → section/species chunks + parse report
├── knowledge/   # Embedder / Reranker / VectorStore interfaces + impls, loader, retrieval
├── eval/        # retrieval metrics, eval-set schema/builder, embedder benchmark
├── cli/         # Typer CLI (ingest, embed, benchmark, load, retrieve)
└── config.py    # pydantic-settings (VET_-prefixed)
scripts/build_eval_set.py   # one-time eval-set generation (generate → review → promote)
docs/superpowers/           # specs + implementation plans
```
