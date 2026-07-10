# Vet-Agent Phase 2 — Knowledge Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pluggable Embedder/Reranker/VectorStore knowledge layer, embed the Phase-1 chunks idempotently into Qdrant, and empirically benchmark a medical embedding model against general ones to choose the default.

**Architecture:** Framework-agnostic interfaces (`Protocol`s) in `src/vet_agent/knowledge/`, with concrete impls: a registry-driven `SentenceTransformerEmbedder`, a `CrossEncoderReranker`, and a `QdrantVectorStore`. An idempotent `loader` reuses the existing `logical_key`/`content_hash` contract (Qdrant is the source of truth). An offline `eval` package (pure-Python metrics + numpy benchmark) scores candidate models against a frozen, metadata-labeled `retrieval_eval.yaml`. All automated tests run **offline** using a `FakeEmbedder` and Qdrant's in-memory mode; real models + a live Qdrant are exercised only in a final manual verification task.

**Tech Stack:** Python 3.12+, uv, pydantic v2, `sentence-transformers`, `qdrant-client` (incl. `:memory:` local mode), `numpy`, `pyyaml`, `anthropic` (dev-only, eval-set generation), pytest, ruff, mypy (strict, package-scoped).

**Spec:** `docs/superpowers/specs/2026-06-28-vet-agent-phase-2-knowledge-layer-design.md`. Read it first — this plan implements it exactly.

---

## Status — ✅ COMPLETE (2026-07-10)

All implementation tasks are done, reviewed (spec + code-quality), and `make check` is green
(110 passed, 2 slow-deselected). Task-level status:

- **Tasks 2.1–2.12** ✅ implemented + two-stage-reviewed.
- **Task 2.11b** ✅ (added mid-execution): expand eval to **8 flows** + clinician persona + few-shots.
- **Task 2.13** ✅ manual real-model + live-Qdrant verification (see that section for run results).
- Mid-execution additions: **security hardening** (`SecretStr` API key, `_env_file`-isolated config
  tests), a **gitignore fix** (track the committed eval artifacts under `data/`), and an **`embed`**
  CLI command + progress logging + Apple **MPS** device selection.

**Deviations from the plan/spec, as built:**
- **Qwen3-0.6B dropped** — only MedEmbed-base vs bge-base benchmarked (the medical-vs-general
  question; a 600M model was heavy on the fanless target machine).
- **Species sampling is cat/dog-focused by default** (`other_fraction=0`), not "companion-dominant
  with a capped exotic minority" — per user preference after reviewing a dry-run.
- **Reranker lift is not measured in the benchmark** — the reranker interface + impl ship and
  `retrieve --rerank` works, but quantifying lift on the frozen eval set is deferred.

**Empirical outcome:** **bge-base won** (recall@5 0.764 vs medembed 0.741; mrr 0.781 vs 0.746) →
`embedding_model` defaults to `bge-base`. Scorecard: `data/eval/benchmark_scorecard.md`.

**Remaining:** finish the development branch (merge/PR).

---

## File Structure

**New package `src/vet_agent/knowledge/`:**

- `__init__.py` — package docstring.
- `interfaces.py` — `Passage`, `PointPayload` (pydantic models) + `Embedder`, `Reranker`, `VectorStore` (`Protocol`s). The swap/benchmark seam.
- `embedders.py` — `ModelSpec`, `MODEL_REGISTRY`, `_postprocess` (pure: truncate+normalize), `SentenceTransformerEmbedder`, `get_embedder(key)`.
- `rerankers.py` — `_apply_scores` (pure helper) + `CrossEncoderReranker`.
- `vector_store.py` — `collection_name()`, `QdrantVectorStore` (impl of `VectorStore`).
- `loader.py` — `NAMESPACE_VET`, `point_id()`, `read_chunks()`, `LoadReport`, `load_chunks()`.
- `retrieval.py` — `RERANK_FETCH_K`, `Retriever`.

**New package `src/vet_agent/eval/`:**

- `__init__.py` — package docstring.
- `metrics.py` — `rank_by_score`, `recall_at_k`, `hit_rate_at_k`, `reciprocal_rank`, `evaluate_query`, `mean_metrics`. Pure functions.
- `eval_set.py` — `EvalCase` model, `load_eval_set()`, `derive_relevant_keys()`.
- `eval_set_builder.py` — `FLOW_SECTIONS`, `QueryPhraser` protocol, `AnthropicQueryPhraser`, `build_eval_set()`, `write_eval_set()`, `promote_eval_set()`. (In-package so it's mypy-checked and importable by tests without `sys.path` hacks.)
- `benchmark.py` — `embed_corpus`, `rank_for_query`, `ModelScore`, `benchmark_model`, `render_scorecard`, `write_scorecard`, `choose_default`.

**New script:**

- `scripts/build_eval_set.py` — thin Typer CLI with two commands: `generate` (writes a reviewable *draft*) and `promote` (freezes the reviewed draft to the committed eval set). Human review of the query phrasings happens between the two. (Lives outside ruff/mypy scope by design — keep it trivial.)

**Modified:**

- `pyproject.toml` — deps, dev deps, pytest markers/addopts, mypy override.
- `src/vet_agent/config.py` — Phase-2 settings.
- `src/vet_agent/cli/main.py` — `benchmark`, `load`, `retrieve` commands.

**Tests (mirror layout):**

- `tests/__init__.py`, `tests/knowledge/__init__.py`, `tests/eval/__init__.py` (make `tests` importable for shared fakes).
- `tests/knowledge/fakes.py` — `FakeEmbedder`, `FakeReranker`.
- `tests/knowledge/test_interfaces.py`, `test_embedders.py`, `test_rerankers.py`, `test_vector_store.py`, `test_loader.py`, `test_retrieval.py`.
- `tests/eval/test_metrics.py`, `test_eval_set.py`, `test_benchmark.py`, `test_build_eval_set.py`.
- `tests/eval/fixtures/retrieval_eval.yaml` — tiny fixture eval set.
- `tests/test_config.py` (extend), `tests/test_cli.py` (extend).

**Generated artifacts (gitignored except the committed eval set):**

- `data/embeddings/<model_key>.npz` — embedding cache (gitignore).
- `data/eval/retrieval_eval.yaml` — committed (produced by the manual generation task).
- `data/eval/benchmark_scorecard.{md,json}` — committed benchmark output.

---

## Conventions for every task

- Use `uv run pytest <path> -v` to run tests; `make check` runs ruff + mypy + the **fast** suite (`-m "not slow"`).
- ruff line-length is 100; sort imports; new CLI options need `# noqa: B008` (matching `cli/main.py`).
- pydantic v2 models; `plain def test_x():` (mypy does not type-check tests).
- Commit after each task with the message shown.

---

## PHASE 2 — KNOWLEDGE LAYER

### Task 2.1: Dependencies, config, and test/type tooling

**Files:**

- Modify: `pyproject.toml`
- Modify: `src/vet_agent/config.py`
- Modify: `.env.example`
- Test: `tests/test_config.py`

- [ ] **Step 1: Add runtime + dev dependencies via uv**

uv resolves the latest compatible versions (honors the project's "prefer latest" rule — do not hand-pin stale floors):

```bash
uv add sentence-transformers qdrant-client numpy pyyaml
uv add --dev anthropic types-PyYAML
```

Expected: `pyproject.toml` `dependencies` gains the four runtime libs; `dependency-groups.dev` gains `anthropic` + `types-PyYAML`; `uv.lock` updates; install succeeds (this pulls in `torch` CPU transitively — may take a few minutes).

- [ ] **Step 2: Add pytest markers + a slow-deselect default, and a mypy override**

Edit `pyproject.toml`. Replace the `[tool.pytest.ini_options]` block with:

```toml
[tool.pytest.ini_options]
addopts = "-q -m 'not slow'"
testpaths = ["tests"]
pythonpath = ["src", "."]
markers = [
    "slow: needs a model download or network; deselected by default (run with -m slow)",
]
```

Append after the `[tool.mypy]` block:

```toml
[[tool.mypy.overrides]]
module = ["sentence_transformers.*"]
ignore_missing_imports = true
```

(`pythonpath` gains `"."` so tests can import the shared `tests.knowledge.fakes` module.)

- [ ] **Step 3: Write the failing config test**

Add to `tests/test_config.py`:

```python
def test_phase2_defaults():
    s = Settings()
    assert s.embedding_model == "medembed-base"
    assert s.qdrant_collection_prefix == "vet_chunks"
    assert s.rerank_enabled is False
    assert s.reranker_model == "bge-reranker-v2-m3"
    assert s.embedding_batch_size == 64
    assert s.retrieval_top_k == 5


def test_phase2_env_override(monkeypatch):
    monkeypatch.setenv("VET_EMBEDDING_MODEL", "bge-base")
    monkeypatch.setenv("VET_RERANK_ENABLED", "true")
    s = Settings()
    assert s.embedding_model == "bge-base"
    assert s.rerank_enabled is True
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `AttributeError`/validation error (fields don't exist yet).

- [ ] **Step 5: Add the settings**

In `src/vet_agent/config.py`, add inside `Settings` (after the `data_dir` field):

```python
    # Knowledge layer (Phase 2)
    embedding_model: str = "medembed-base"
    qdrant_collection_prefix: str = "vet_chunks"
    rerank_enabled: bool = False
    reranker_model: str = "bge-reranker-v2-m3"
    embedding_batch_size: int = 64
    retrieval_top_k: int = 5
```

- [ ] **Step 6: Document the new env vars**

Append to `.env.example`:

```bash
VET_EMBEDDING_MODEL=medembed-base
VET_QDRANT_COLLECTION_PREFIX=vet_chunks
VET_RERANK_ENABLED=false
VET_RERANKER_MODEL=bge-reranker-v2-m3
VET_EMBEDDING_BATCH_SIZE=64
VET_RETRIEVAL_TOP_K=5
```

- [ ] **Step 7: Run tests + full gate**

Run: `uv run pytest tests/test_config.py -v` → PASS.
Run: `make check` → ruff clean, mypy `Success`, all tests pass.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock src/vet_agent/config.py .env.example tests/test_config.py
git commit -m "chore: add Phase 2 deps, knowledge-layer settings, slow-test marker"
```

---

### Task 2.2: Knowledge interfaces + shared test fakes

**Files:**

- Create: `src/vet_agent/knowledge/__init__.py`
- Create: `src/vet_agent/knowledge/interfaces.py`
- Create: `tests/__init__.py`, `tests/knowledge/__init__.py`, `tests/knowledge/fakes.py`
- Test: `tests/knowledge/test_interfaces.py`

- [ ] **Step 1: Write the failing test**

`tests/knowledge/test_interfaces.py`:

```python
from vet_agent.ingestion.models import SectionType
from vet_agent.knowledge.interfaces import Embedder, Passage, PointPayload
from tests.knowledge.fakes import FakeEmbedder


def test_passage_carries_citation_fields():
    p = Passage(
        drug_name="Metronidazole",
        section_type=SectionType.DOSES,
        species=["dog"],
        book_page=873,
        text="DOGS: 25 mg/kg PO q12h",
        logical_key="metronidazole|doses|dog|0",
    )
    assert p.score is None
    assert p.section_type is SectionType.DOSES


def test_point_payload_holds_vector_and_hash():
    pp = PointPayload(
        point_id="abc",
        vector=[0.1, 0.2],
        drug_name="X",
        section_type=SectionType.STORAGE,
        species=["all"],
        book_page=1,
        text="Store cool.",
        logical_key="x|storage|all|0",
        content_hash="deadbeef",
    )
    assert pp.vector == [0.1, 0.2]


def test_fake_embedder_satisfies_protocol_and_is_deterministic():
    emb = FakeEmbedder(dim=8)
    assert isinstance(emb, Embedder)  # runtime_checkable structural check
    assert emb.dim == 8
    assert emb.embed_query("giardia") == emb.embed_query("giardia")
    assert emb.embed_query("a") != emb.embed_query("b")
    vec = emb.embed_query("giardia")
    assert abs(sum(v * v for v in vec) - 1.0) < 1e-9  # unit length
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/knowledge/test_interfaces.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vet_agent.knowledge'`.

- [ ] **Step 3: Create the package + interfaces**

`src/vet_agent/knowledge/__init__.py`:

```python
"""Knowledge layer: pluggable Embedder / Reranker / VectorStore + Qdrant loading."""
```

`src/vet_agent/knowledge/interfaces.py`:

```python
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from vet_agent.ingestion.models import SectionType


class Passage(BaseModel):
    """A retrieved chunk with its citation fields and similarity score."""

    drug_name: str
    section_type: SectionType
    species: list[str]
    book_page: int
    text: str
    logical_key: str
    score: float | None = None


class PointPayload(BaseModel):
    """A fully-formed Qdrant point: id, vector, and stored payload."""

    point_id: str
    vector: list[float]
    drug_name: str
    section_type: SectionType
    species: list[str]
    book_page: int
    text: str
    logical_key: str
    content_hash: str


@runtime_checkable
class Embedder(Protocol):
    """Turns text into L2-normalized vectors. Implementations must normalize."""

    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class Reranker(Protocol):
    """Reorders candidate passages by query relevance and returns the top_k."""

    name: str

    def rerank(self, query: str, passages: list[Passage], top_k: int) -> list[Passage]: ...


class VectorStore(Protocol):
    """Idempotent vector storage with metadata-filtered search."""

    def ensure_collection(self, dim: int) -> None: ...

    def existing_hashes(self) -> dict[str, str]: ...  # point_id -> content_hash

    def upsert(self, points: list[PointPayload]) -> None: ...

    def delete(self, ids: list[str]) -> None: ...

    def search(
        self,
        vector: list[float],
        *,
        drug: str | None = None,
        section: SectionType | None = None,
        species: str | None = None,
        top_k: int = 5,
    ) -> list[Passage]: ...
```

- [ ] **Step 4: Create the test-package markers + FakeEmbedder**

`tests/__init__.py`: (empty file)
`tests/knowledge/__init__.py`: (empty file)

`tests/knowledge/fakes.py`:

```python
"""Deterministic, offline test doubles for the knowledge interfaces."""

import hashlib
import math

from vet_agent.knowledge.interfaces import Passage


class FakeEmbedder:
    """Maps text -> unit vector via SHA-256 bytes.

    Identical text yields an identical vector (so a query equal to a chunk's text
    scores cosine 1.0); distinct text yields a different vector. No semantics — it
    exercises wiring (idempotency, filtering, ranking), not retrieval quality.
    """

    name = "fake"

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [float(digest[i % len(digest)]) for i in range(self.dim)]
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeReranker:
    """Reverses the candidate order (so tests can prove rerank ran), returns top_k."""

    name = "fake-reranker"

    def rerank(self, query: str, passages: list[Passage], top_k: int) -> list[Passage]:
        return list(reversed(passages))[:top_k]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/knowledge/test_interfaces.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add src/vet_agent/knowledge/__init__.py src/vet_agent/knowledge/interfaces.py \
        tests/__init__.py tests/knowledge/
git commit -m "feat(knowledge): add Embedder/Reranker/VectorStore interfaces + test fakes"
```

---

### Task 2.3: Retrieval metrics (pure functions)

**Files:**

- Create: `src/vet_agent/eval/__init__.py`
- Create: `src/vet_agent/eval/metrics.py`
- Create: `tests/eval/__init__.py`
- Test: `tests/eval/test_metrics.py`

Notes (from spec §11): metrics operate on an **already-ranked** list of `logical_key`s vs. a **set** of relevant keys. Recall's denominator is the **total** relevant count (not `min(k, …)`). `rank_by_score` breaks score ties **deterministically by key** so ranks are reproducible.

- [ ] **Step 1: Write the failing test**

`tests/eval/test_metrics.py`:

```python
from vet_agent.eval.metrics import (
    evaluate_query,
    hit_rate_at_k,
    mean_metrics,
    rank_by_score,
    recall_at_k,
    reciprocal_rank,
)


def test_rank_by_score_breaks_ties_by_key():
    # b and a tie on score 0.5; deterministic tie-break is key ascending -> a before b.
    ranked = rank_by_score([("b", 0.5), ("a", 0.5), ("c", 0.9)])
    assert ranked == ["c", "a", "b"]


def test_recall_uses_total_relevant_as_denominator():
    ranked = ["k1", "x", "k2", "y"]
    relevant = {"k1", "k2", "k3"}  # 3 relevant total, 2 retrieved in top-4
    assert recall_at_k(ranked, relevant, k=4) == 2 / 3
    assert recall_at_k(ranked, relevant, k=1) == 1 / 3


def test_hit_rate_is_binary():
    assert hit_rate_at_k(["x", "k1"], {"k1"}, k=2) == 1.0
    assert hit_rate_at_k(["x", "y"], {"k1"}, k=2) == 0.0


def test_reciprocal_rank_is_first_relevant_position():
    assert reciprocal_rank(["x", "k1", "k2"], {"k1", "k2"}) == 0.5
    assert reciprocal_rank(["x", "y"], {"k1"}) == 0.0


def test_evaluate_query_and_mean():
    a = evaluate_query(["k1", "x"], {"k1"}, ks=[1, 3])
    assert a["recall@1"] == 1.0
    assert a["mrr"] == 1.0
    b = evaluate_query(["x", "y"], {"k1"}, ks=[1, 3])
    assert b["recall@1"] == 0.0
    avg = mean_metrics([a, b])
    assert avg["recall@1"] == 0.5
    assert avg["mrr"] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vet_agent.eval'`.

- [ ] **Step 3: Write the implementation**

`src/vet_agent/eval/__init__.py`:

```python
"""Evaluation: retrieval metrics + the embedder benchmark harness."""
```

`tests/eval/__init__.py`: (empty file)

`src/vet_agent/eval/metrics.py`:

```python
from collections.abc import Sequence


def rank_by_score(scored: Sequence[tuple[str, float]]) -> list[str]:
    """Rank keys by score descending, breaking ties by key ascending (deterministic).

    Exact float ties are improbable for cosine over distinct vectors, but the
    secondary key guarantees reproducible ranks regardless (see spec §11).
    """
    return [key for key, _ in sorted(scored, key=lambda kv: (-kv[1], kv[0]))]


def recall_at_k(ranked_keys: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of ALL relevant keys that appear in the top-k."""
    if not relevant:
        return 0.0
    found = len(set(ranked_keys[:k]) & relevant)
    return found / len(relevant)


def hit_rate_at_k(ranked_keys: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant key is in the top-k, else 0.0."""
    return 1.0 if set(ranked_keys[:k]) & relevant else 0.0


def reciprocal_rank(ranked_keys: Sequence[str], relevant: set[str]) -> float:
    """1 / (1-based rank of the first relevant key); 0.0 if none present."""
    for i, key in enumerate(ranked_keys):
        if key in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_query(
    ranked_keys: Sequence[str], relevant: set[str], ks: Sequence[int]
) -> dict[str, float]:
    """All metrics for one query: recall@k + hit_rate@k for each k, plus mrr."""
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(ranked_keys, relevant, k)
        out[f"hit_rate@{k}"] = hit_rate_at_k(ranked_keys, relevant, k)
    out["mrr"] = reciprocal_rank(ranked_keys, relevant)
    return out


def mean_metrics(per_query: Sequence[dict[str, float]]) -> dict[str, float]:
    """Average each metric across queries. Empty input -> empty dict."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: sum(q[k] for q in per_query) / len(per_query) for k in keys}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_metrics.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vet_agent/eval/__init__.py src/vet_agent/eval/metrics.py tests/eval/
git commit -m "feat(eval): add pure retrieval metrics with deterministic tie-handling"
```

---

### Task 2.4: Embedders (registry + sentence-transformers wrapper)

**Files:**

- Create: `src/vet_agent/knowledge/embedders.py`
- Test: `tests/knowledge/test_embedders.py`

Notes: `_postprocess` (truncate to `matryoshka_dim` then L2-normalize) is the unit-tested pure core. Real model loading is a `slow` test (downloads weights). MedEmbed-base and bge-base are 768-d natively; Qwen3-0.6B is truncated to 768 via Matryoshka. Query prefixes apply in `embed_query` only.

- [ ] **Step 1: Write the failing test**

`tests/knowledge/test_embedders.py`:

```python
import math

import pytest

from vet_agent.knowledge.embedders import (
    MODEL_REGISTRY,
    SentenceTransformerEmbedder,
    _postprocess,
    get_embedder,
)


def test_registry_has_three_768d_candidates():
    assert set(MODEL_REGISTRY) == {"medembed-base", "bge-base", "qwen3-0.6b"}
    for spec in MODEL_REGISTRY.values():
        assert spec.dim == 768
    # Qwen3 is natively larger -> truncated via Matryoshka to the common 768.
    assert MODEL_REGISTRY["qwen3-0.6b"].matryoshka_dim == 768
    assert MODEL_REGISTRY["medembed-base"].matryoshka_dim is None


def test_postprocess_truncates_then_normalizes():
    out = _postprocess([3.0, 4.0, 99.0], matryoshka_dim=2)
    assert len(out) == 2
    assert math.isclose(math.sqrt(out[0] ** 2 + out[1] ** 2), 1.0, rel_tol=1e-9)
    assert math.isclose(out[0], 0.6) and math.isclose(out[1], 0.8)


def test_postprocess_normalizes_without_truncation():
    out = _postprocess([0.0, 0.0, 5.0], matryoshka_dim=None)
    assert out == [0.0, 0.0, 1.0]


def test_get_embedder_rejects_unknown_key():
    with pytest.raises(KeyError):
        get_embedder("not-a-model")


@pytest.mark.slow
def test_real_medembed_loads_and_embeds_768d():
    emb = get_embedder("medembed-base")
    assert isinstance(emb, SentenceTransformerEmbedder)
    assert emb.dim == 768
    vecs = emb.embed_documents(["metronidazole treats giardia in dogs"])
    assert len(vecs[0]) == 768
    assert math.isclose(math.sqrt(sum(v * v for v in vecs[0])), 1.0, rel_tol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/knowledge/test_embedders.py -v`
Expected: FAIL with `ModuleNotFoundError` for `vet_agent.knowledge.embedders`.

- [ ] **Step 3: Write the implementation**

`src/vet_agent/knowledge/embedders.py`:

```python
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """How to load and post-process one embedding model."""

    hf_id: str
    dim: int
    query_prefix: str | None = None  # prepended in embed_query only
    matryoshka_dim: int | None = None  # truncate native dim down to this


# All three normalized to 768-d so the benchmark compares like-for-like.
# MedEmbed-base is a direct fine-tune of bge-base-en-v1.5 (isolates the medical gain);
# Qwen3-0.6B is a modern strong-model reference, truncated from its native dim.
# Verify exact hf ids / query-prompt format against the live model cards before first run.
MODEL_REGISTRY: dict[str, "ModelSpec"] = {
    "medembed-base": ModelSpec("abhinand/MedEmbed-base-v0.1", dim=768),
    "bge-base": ModelSpec("BAAI/bge-base-en-v1.5", dim=768),
    "qwen3-0.6b": ModelSpec(
        "Qwen/Qwen3-Embedding-0.6B",
        dim=768,
        query_prefix="Instruct: Given a query, retrieve relevant passages.\nQuery: ",
        matryoshka_dim=768,
    ),
}


def _postprocess(vector: list[float], matryoshka_dim: int | None) -> list[float]:
    """Truncate to matryoshka_dim (if set) then L2-normalize."""
    v = vector[:matryoshka_dim] if matryoshka_dim is not None else vector
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class SentenceTransformerEmbedder:
    """Embedder backed by a `sentence-transformers` SentenceTransformer model."""

    def __init__(self, key: str, spec: ModelSpec) -> None:
        from sentence_transformers import SentenceTransformer  # lazy: avoids import cost

        self.name = key
        self.dim = spec.dim
        self._spec = spec
        self._model = SentenceTransformer(spec.hf_id)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        raw = self._model.encode(texts, normalize_embeddings=False)
        return [_postprocess(list(map(float, row)), self._spec.matryoshka_dim) for row in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        prefixed = (self._spec.query_prefix or "") + text
        return self._encode([prefixed])[0]


def get_embedder(key: str) -> SentenceTransformerEmbedder:
    """Construct the embedder for a registry key (raises KeyError if unknown)."""
    spec = MODEL_REGISTRY[key]
    return SentenceTransformerEmbedder(key, spec)
```

- [ ] **Step 4: Run the fast tests to verify they pass**

Run: `uv run pytest tests/knowledge/test_embedders.py -v`
Expected: PASS for the 4 fast tests; the `slow` test is **deselected** (not run) by the default `-m 'not slow'`.

- [ ] **Step 5: (Optional, online) run the slow test once**

Run: `uv run pytest tests/knowledge/test_embedders.py -m slow -v`
Expected: PASS (downloads MedEmbed-base on first run; needs network). Skip if offline — the manual verification task (2.13) covers real models.

- [ ] **Step 6: Commit**

```bash
git add src/vet_agent/knowledge/embedders.py tests/knowledge/test_embedders.py
git commit -m "feat(knowledge): add model registry + sentence-transformers embedder"
```

---

### Task 2.5: Reranker (cross-encoder)

**Files:**

- Create: `src/vet_agent/knowledge/rerankers.py`
- Test: `tests/knowledge/test_rerankers.py`

Notes: `_apply_scores` (attach scores, sort desc, take top_k) is the pure tested core. The real `bge-reranker-v2-m3` load is a `slow` test.

- [ ] **Step 1: Write the failing test**

`tests/knowledge/test_rerankers.py`:

```python
import pytest

from vet_agent.ingestion.models import SectionType
from vet_agent.knowledge.interfaces import Passage
from vet_agent.knowledge.rerankers import CrossEncoderReranker, _apply_scores


def _p(key: str, text: str) -> Passage:
    return Passage(
        drug_name="X",
        section_type=SectionType.DOSES,
        species=["dog"],
        book_page=1,
        text=text,
        logical_key=key,
    )


def test_apply_scores_sorts_desc_and_truncates():
    passages = [_p("a", "aa"), _p("b", "bb"), _p("c", "cc")]
    out = _apply_scores(passages, [0.1, 0.9, 0.5], top_k=2)
    assert [p.logical_key for p in out] == ["b", "c"]
    assert out[0].score == 0.9


@pytest.mark.slow
def test_real_bge_reranker_reorders_by_relevance():
    rr = CrossEncoderReranker("BAAI/bge-reranker-v2-m3")
    passages = [
        _p("off", "Storage: keep refrigerated."),
        _p("on", "DOGS: metronidazole 25 mg/kg PO q12h for giardia."),
    ]
    out = rr.rerank("metronidazole dose for a dog with giardia", passages, top_k=2)
    assert out[0].logical_key == "on"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/knowledge/test_rerankers.py -v`
Expected: FAIL with `ModuleNotFoundError` for `vet_agent.knowledge.rerankers`.

- [ ] **Step 3: Write the implementation**

`src/vet_agent/knowledge/rerankers.py`:

```python
from vet_agent.knowledge.interfaces import Passage


def _apply_scores(passages: list[Passage], scores: list[float], top_k: int) -> list[Passage]:
    """Attach scores, sort by score descending, return the top_k passages."""
    scored = [p.model_copy(update={"score": float(s)}) for p, s in zip(passages, scores, strict=True)]
    scored.sort(key=lambda p: p.score or 0.0, reverse=True)
    return scored[:top_k]


class CrossEncoderReranker:
    """Reranker backed by a `sentence-transformers` CrossEncoder (e.g. bge-reranker-v2-m3)."""

    def __init__(self, hf_id: str) -> None:
        from sentence_transformers import CrossEncoder  # lazy

        self.name = hf_id
        self._model = CrossEncoder(hf_id)

    def rerank(self, query: str, passages: list[Passage], top_k: int) -> list[Passage]:
        if not passages:
            return []
        scores = self._model.predict([(query, p.text) for p in passages])
        return _apply_scores(passages, [float(s) for s in scores], top_k)
```

- [ ] **Step 4: Run the fast test to verify it passes**

Run: `uv run pytest tests/knowledge/test_rerankers.py -v`
Expected: PASS for `test_apply_scores...`; the `slow` test is deselected.

- [ ] **Step 5: Commit**

```bash
git add src/vet_agent/knowledge/rerankers.py tests/knowledge/test_rerankers.py
git commit -m "feat(knowledge): add cross-encoder reranker (bge-reranker-v2-m3)"
```

---

### Task 2.6: Qdrant vector store (in-memory testable)

**Files:**

- Create: `src/vet_agent/knowledge/vector_store.py`
- Test: `tests/knowledge/test_vector_store.py`

Notes: tests use `QdrantClient(location=":memory:")` — no Docker, fully offline. Payload indexes are created once at collection creation. `search` builds a filterable-HNSW query; `species` is list-valued in the payload so a `MatchValue` matches if any element equals the value (a "dog" filter matches `["cat","dog"]`).

- [ ] **Step 1: Write the failing test**

`tests/knowledge/test_vector_store.py`:

```python
from qdrant_client import QdrantClient

from vet_agent.ingestion.models import SectionType
from vet_agent.knowledge.interfaces import PointPayload
from vet_agent.knowledge.vector_store import QdrantVectorStore, collection_name


def _store() -> QdrantVectorStore:
    return QdrantVectorStore(QdrantClient(location=":memory:"), "vet_test")


def _point(pid: str, *, species, vector, drug="Metronidazole", text="t", ch="h") -> PointPayload:
    return PointPayload(
        point_id=pid,
        vector=vector,
        drug_name=drug,
        section_type=SectionType.DOSES,
        species=species,
        book_page=873,
        text=text,
        logical_key=pid,
        content_hash=ch,
    )


def test_collection_name_is_model_suffixed_and_sanitized():
    assert collection_name("vet_chunks", "qwen3-0.6b") == "vet_chunks__qwen3_0_6b"
    assert collection_name("vet_chunks", "medembed-base") == "vet_chunks__medembed_base"


def test_upsert_then_existing_hashes_roundtrip():
    store = _store()
    store.ensure_collection(dim=2)
    store.upsert([_point("p1", species=["dog"], vector=[1.0, 0.0], ch="h1")])
    assert store.existing_hashes() == {"p1": "h1"}


def test_existing_hashes_empty_when_no_collection():
    assert _store().existing_hashes() == {}


def test_search_respects_species_filter_including_list_membership():
    store = _store()
    store.ensure_collection(dim=2)
    store.upsert(
        [
            _point("dog", species=["dog"], vector=[1.0, 0.0], text="dog dose"),
            _point("cat", species=["cat"], vector=[1.0, 0.0], text="cat dose"),
            _point("both", species=["cat", "dog"], vector=[1.0, 0.0], text="shared dose"),
        ]
    )
    hits = store.search([1.0, 0.0], species="dog", top_k=10)
    keys = {h.logical_key for h in hits}
    assert keys == {"dog", "both"}  # "cat" excluded; list-valued "both" matches
    assert all(h.section_type is SectionType.DOSES for h in hits)


def test_delete_removes_points():
    store = _store()
    store.ensure_collection(dim=2)
    store.upsert([_point("p1", species=["dog"], vector=[1.0, 0.0])])
    store.delete(["p1"])
    assert store.existing_hashes() == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/knowledge/test_vector_store.py -v`
Expected: FAIL with `ModuleNotFoundError` for `vet_agent.knowledge.vector_store`.

- [ ] **Step 3: Write the implementation**

`src/vet_agent/knowledge/vector_store.py`:

```python
import re

from qdrant_client import QdrantClient, models

from vet_agent.ingestion.models import SectionType
from vet_agent.knowledge.interfaces import Passage, PointPayload


def collection_name(prefix: str, model_key: str) -> str:
    """Derive a per-model collection name: '<prefix>__<sanitized model key>'."""
    safe = re.sub(r"[^a-z0-9]+", "_", model_key.lower()).strip("_")
    return f"{prefix}__{safe}"


def _to_passage(point: models.ScoredPoint) -> Passage:
    payload = point.payload or {}
    return Passage(
        drug_name=payload["drug_name"],
        section_type=SectionType(payload["section_type"]),
        species=list(payload["species"]),
        book_page=int(payload["book_page"]),
        text=payload["text"],
        logical_key=payload["logical_key"],
        score=point.score,
    )


class QdrantVectorStore:
    """VectorStore backed by Qdrant (server URL or `:memory:` local mode)."""

    def __init__(self, client: QdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    def ensure_collection(self, dim: int) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            self._collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        for field in ("drug_name", "section_type", "species"):
            self._client.create_payload_index(
                self._collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def existing_hashes(self) -> dict[str, str]:
        if not self._client.collection_exists(self._collection):
            return {}
        result: dict[str, str] = {}
        offset: str | int | None = None
        while True:
            points, offset = self._client.scroll(
                self._collection,
                with_payload=["content_hash"],
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            for p in points:
                ch = (p.payload or {}).get("content_hash")
                if isinstance(ch, str):
                    result[str(p.id)] = ch
            if offset is None:
                break
        return result

    def upsert(self, points: list[PointPayload]) -> None:
        if not points:
            return
        self._client.upsert(
            self._collection,
            points=[
                models.PointStruct(
                    id=p.point_id,
                    vector=p.vector,
                    payload={
                        "drug_name": p.drug_name,
                        "section_type": p.section_type.value,
                        "species": p.species,
                        "book_page": p.book_page,
                        "text": p.text,
                        "logical_key": p.logical_key,
                        "content_hash": p.content_hash,
                    },
                )
                for p in points
            ],
        )

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        self._client.delete(self._collection, points_selector=models.PointIdsList(points=ids))

    def search(
        self,
        vector: list[float],
        *,
        drug: str | None = None,
        section: SectionType | None = None,
        species: str | None = None,
        top_k: int = 5,
    ) -> list[Passage]:
        must: list[models.FieldCondition] = []
        if drug is not None:
            must.append(
                models.FieldCondition(key="drug_name", match=models.MatchValue(value=drug))
            )
        if section is not None:
            must.append(
                models.FieldCondition(
                    key="section_type", match=models.MatchValue(value=section.value)
                )
            )
        if species is not None:
            must.append(
                models.FieldCondition(key="species", match=models.MatchValue(value=species))
            )
        flt = models.Filter(must=must) if must else None
        response = self._client.query_points(
            self._collection,
            query=vector,
            query_filter=flt,
            limit=top_k,
            with_payload=True,
        )
        return [_to_passage(point) for point in response.points]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/knowledge/test_vector_store.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vet_agent/knowledge/vector_store.py tests/knowledge/test_vector_store.py
git commit -m "feat(knowledge): add Qdrant vector store with filtered search"
```

---

### Task 2.7: Idempotent loader

**Files:**

- Create: `src/vet_agent/knowledge/loader.py`
- Test: `tests/knowledge/test_loader.py`

Notes: point id = `uuid5(NAMESPACE_VET, logical_key)`; reuses `chunker.logical_key`/`content_hash`. Idempotency: skip points whose stored `content_hash` is unchanged, embed+upsert changed/new, prune orphans. Qdrant is the source of truth.

- [ ] **Step 1: Write the failing test**

`tests/knowledge/test_loader.py`:

```python
from qdrant_client import QdrantClient

from vet_agent.ingestion.models import Chunk, SectionType
from vet_agent.knowledge.loader import load_chunks, point_id, read_chunks
from vet_agent.knowledge.vector_store import QdrantVectorStore
from tests.knowledge.fakes import FakeEmbedder


def _chunk(text: str, *, drug="Metronidazole", ordinal=0) -> Chunk:
    return Chunk(
        drug_name=drug,
        section_type=SectionType.DOSES,
        species=["dog"],
        book_page=873,
        text=text,
        ordinal=ordinal,
    )


def _store() -> QdrantVectorStore:
    return QdrantVectorStore(QdrantClient(location=":memory:"), "vet_test")


def test_first_load_embeds_all_then_second_load_skips_all():
    chunks = [_chunk("a", ordinal=0), _chunk("b", ordinal=1)]
    store, emb = _store(), FakeEmbedder(dim=8)
    r1 = load_chunks(chunks, emb, store)
    assert (r1.upserted, r1.skipped, r1.pruned) == (2, 0, 0)
    r2 = load_chunks(chunks, emb, store)
    assert (r2.upserted, r2.skipped, r2.pruned) == (0, 2, 0)


def test_changed_chunk_is_reembedded_only():
    chunks = [_chunk("a", ordinal=0), _chunk("b", ordinal=1)]
    store, emb = _store(), FakeEmbedder(dim=8)
    load_chunks(chunks, emb, store)
    chunks[1] = _chunk("b-edited", ordinal=1)  # same logical_key, new content_hash
    r = load_chunks(chunks, emb, store)
    assert (r.upserted, r.skipped, r.pruned) == (1, 1, 0)


def test_orphan_is_pruned():
    chunks = [_chunk("a", ordinal=0), _chunk("b", ordinal=1)]
    store, emb = _store(), FakeEmbedder(dim=8)
    load_chunks(chunks, emb, store)
    r = load_chunks([chunks[0]], emb, store)  # second chunk disappeared
    assert (r.upserted, r.skipped, r.pruned) == (0, 1, 1)
    assert point_id("metronidazole|doses|dog|1") not in store.existing_hashes()


def test_no_prune_keeps_orphans():
    chunks = [_chunk("a", ordinal=0), _chunk("b", ordinal=1)]
    store, emb = _store(), FakeEmbedder(dim=8)
    load_chunks(chunks, emb, store)
    r = load_chunks([chunks[0]], emb, store, prune=False)
    assert r.pruned == 0
    assert len(store.existing_hashes()) == 2


def test_read_chunks_roundtrips(tmp_path):
    import json

    chunks = [_chunk("a", ordinal=0)]
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps([c.model_dump() for c in chunks]), encoding="utf-8")
    loaded = read_chunks(path)
    assert loaded == chunks
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/knowledge/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError` for `vet_agent.knowledge.loader`.

- [ ] **Step 3: Write the implementation**

`src/vet_agent/knowledge/loader.py`:

```python
import uuid
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

from vet_agent.ingestion.chunker import content_hash, logical_key
from vet_agent.ingestion.models import Chunk
from vet_agent.knowledge.interfaces import Embedder, PointPayload, VectorStore

# Fixed namespace so uuid5(point ids) are stable across runs and machines.
NAMESPACE_VET = uuid.UUID("5b3d2c1a-9e8f-4a7b-8c6d-1f2e3a4b5c6d")

_CHUNKS_ADAPTER = TypeAdapter(list[Chunk])


def point_id(logical_key_value: str) -> str:
    """Deterministic Qdrant point id (UUIDv5) derived from a chunk's logical key."""
    return str(uuid.uuid5(NAMESPACE_VET, logical_key_value))


def read_chunks(path: Path) -> list[Chunk]:
    """Load chunks.json (the Phase-1 artifact) into typed Chunk objects."""
    return _CHUNKS_ADAPTER.validate_json(path.read_text(encoding="utf-8"))


class LoadReport(BaseModel):
    upserted: int
    skipped: int
    pruned: int


def load_chunks(
    chunks: list[Chunk],
    embedder: Embedder,
    store: VectorStore,
    *,
    prune: bool = True,
    batch_size: int = 64,
) -> LoadReport:
    """Idempotently load chunks into the store.

    Skips chunks whose stored content_hash is unchanged, embeds+upserts changed/new
    ones, and (unless prune=False) deletes points whose logical key has disappeared.
    """
    store.ensure_collection(embedder.dim)

    desired: dict[str, Chunk] = {}
    hashes: dict[str, str] = {}
    for chunk in chunks:
        lk = logical_key(chunk)
        pid = point_id(lk)
        desired[pid] = chunk
        hashes[pid] = content_hash(chunk)

    existing = store.existing_hashes()
    to_embed = [pid for pid in desired if existing.get(pid) != hashes[pid]]

    points: list[PointPayload] = []
    for start in range(0, len(to_embed), batch_size):
        batch = to_embed[start : start + batch_size]
        vectors = embedder.embed_documents([desired[pid].text for pid in batch])
        for pid, vector in zip(batch, vectors, strict=True):
            chunk = desired[pid]
            points.append(
                PointPayload(
                    point_id=pid,
                    vector=vector,
                    drug_name=chunk.drug_name,
                    section_type=chunk.section_type,
                    species=chunk.species,
                    book_page=chunk.book_page,
                    text=chunk.text,
                    logical_key=logical_key(chunk),
                    content_hash=hashes[pid],
                )
            )
    store.upsert(points)

    orphans = [pid for pid in existing if pid not in desired]
    if prune and orphans:
        store.delete(orphans)

    return LoadReport(
        upserted=len(points),
        skipped=len(desired) - len(points),
        pruned=len(orphans) if prune else 0,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/knowledge/test_loader.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vet_agent/knowledge/loader.py tests/knowledge/test_loader.py
git commit -m "feat(knowledge): add idempotent Qdrant loader (skip/upsert/prune)"
```

---

### Task 2.8: Retriever (filtered semantic search + optional rerank)

**Files:**

- Create: `src/vet_agent/knowledge/retrieval.py`
- Test: `tests/knowledge/test_retrieval.py`

Notes: this is the shape Phase 3's `retrieve_monograph` tool will call. When reranking, over-fetch `RERANK_FETCH_K` candidates before reordering.

- [ ] **Step 1: Write the failing test**

`tests/knowledge/test_retrieval.py`:

```python
from qdrant_client import QdrantClient

from vet_agent.ingestion.models import Chunk, SectionType
from vet_agent.knowledge.loader import load_chunks
from vet_agent.knowledge.retrieval import Retriever
from vet_agent.knowledge.vector_store import QdrantVectorStore
from tests.knowledge.fakes import FakeEmbedder, FakeReranker


def _chunk(text, *, species, ordinal) -> Chunk:
    return Chunk(
        drug_name="Metronidazole",
        section_type=SectionType.DOSES,
        species=species,
        book_page=873,
        text=text,
        ordinal=ordinal,
    )


def _loaded_retriever(reranker=None) -> Retriever:
    store = QdrantVectorStore(QdrantClient(location=":memory:"), "vet_test")
    emb = FakeEmbedder(dim=8)
    chunks = [
        _chunk("dog dose text", species=["dog"], ordinal=0),
        _chunk("cat dose text", species=["cat"], ordinal=1),
    ]
    load_chunks(chunks, emb, store)
    return Retriever(emb, store, reranker=reranker)


def test_semantic_query_matches_identical_text_first():
    r = _loaded_retriever()
    hits = r.retrieve("dog dose text", top_k=1)
    assert hits[0].text == "dog dose text"  # FakeEmbedder: identical text -> cosine 1.0


def test_species_filter_excludes_other_species():
    r = _loaded_retriever()
    hits = r.retrieve("dose", species="dog", top_k=10)
    assert {h.species[0] for h in hits} == {"dog"}


def test_rerank_path_invokes_reranker():
    r = _loaded_retriever(reranker=FakeReranker())
    plain = r.retrieve("dose", top_k=2)
    reranked = r.retrieve("dose", top_k=2, rerank=True)
    assert [h.logical_key for h in reranked] == [h.logical_key for h in reversed(plain)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/knowledge/test_retrieval.py -v`
Expected: FAIL with `ModuleNotFoundError` for `vet_agent.knowledge.retrieval`.

- [ ] **Step 3: Write the implementation**

`src/vet_agent/knowledge/retrieval.py`:

```python
from vet_agent.ingestion.models import SectionType
from vet_agent.knowledge.interfaces import Embedder, Passage, Reranker, VectorStore

RERANK_FETCH_K = 20  # over-fetch this many candidates before reranking


class Retriever:
    """Filtered semantic search over a VectorStore, with optional reranking."""

    def __init__(
        self, embedder: Embedder, store: VectorStore, reranker: Reranker | None = None
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._reranker = reranker

    def retrieve(
        self,
        query: str,
        *,
        drug: str | None = None,
        section: SectionType | None = None,
        species: str | None = None,
        top_k: int = 5,
        rerank: bool = False,
    ) -> list[Passage]:
        vector = self._embedder.embed_query(query)
        fetch_k = max(top_k, RERANK_FETCH_K) if rerank else top_k
        hits = self._store.search(
            vector, drug=drug, section=section, species=species, top_k=fetch_k
        )
        if rerank and self._reranker is not None:
            hits = self._reranker.rerank(query, hits, top_k)
        return hits[:top_k]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/knowledge/test_retrieval.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vet_agent/knowledge/retrieval.py tests/knowledge/test_retrieval.py
git commit -m "feat(knowledge): add filtered-semantic Retriever with optional rerank"
```

---

### Task 2.9: Eval-set model, loader, and label derivation

**Files:**

- Create: `src/vet_agent/eval/eval_set.py`
- Create: `tests/eval/fixtures/retrieval_eval.yaml`
- Test: `tests/eval/test_eval_set.py`

Notes: `EvalCase` is the frozen YAML schema. `derive_relevant_keys` computes ground-truth `logical_key`s from chunk metadata: same drug (case-insensitive) + same section, and (if the case names species) a species overlap — treating a prose chunk tagged `["all"]` as a match.

- [ ] **Step 1: Write the fixture eval set**

`tests/eval/fixtures/retrieval_eval.yaml`:

```yaml
- query: "What is the dose of metronidazole for a dog?"
  flow: dose
  drug: Metronidazole
  section: doses
  species: [dog]
  relevant_logical_keys:
    - "metronidazole|doses|dog|0"
- query: "What does metronidazole treat?"
  flow: indication
  drug: Metronidazole
  section: indications
  species: []
  relevant_logical_keys:
    - "metronidazole|indications|all|0"
```

- [ ] **Step 2: Write the failing test**

`tests/eval/test_eval_set.py`:

```python
from pathlib import Path

from vet_agent.eval.eval_set import EvalCase, derive_relevant_keys, load_eval_set
from vet_agent.ingestion.models import Chunk, SectionType

FIXTURE = Path("tests/eval/fixtures/retrieval_eval.yaml")


def _chunk(section, species, ordinal, drug="Metronidazole") -> Chunk:
    return Chunk(
        drug_name=drug,
        section_type=section,
        species=species,
        book_page=873,
        text="t",
        ordinal=ordinal,
    )


def test_load_eval_set_parses_cases():
    cases = load_eval_set(FIXTURE)
    assert len(cases) == 2
    assert isinstance(cases[0], EvalCase)
    assert cases[0].section is SectionType.DOSES
    assert cases[1].species == []


def test_derive_relevant_keys_matches_drug_section_species():
    chunks = [
        _chunk(SectionType.DOSES, ["dog"], 0),
        _chunk(SectionType.DOSES, ["cat"], 1),
        _chunk(SectionType.DOSES, ["cat", "dog"], 2),
        _chunk(SectionType.INDICATIONS, ["all"], 0),
        _chunk(SectionType.DOSES, ["dog"], 0, drug="Other"),
    ]
    keys = derive_relevant_keys(chunks, drug="metronidazole", section=SectionType.DOSES, species=["dog"])
    assert set(keys) == {"metronidazole|doses|dog|0", "metronidazole|doses|cat+dog|2"}


def test_derive_relevant_keys_empty_species_matches_any():
    chunks = [_chunk(SectionType.INDICATIONS, ["all"], 0)]
    keys = derive_relevant_keys(
        chunks, drug="Metronidazole", section=SectionType.INDICATIONS, species=[]
    )
    assert keys == ["metronidazole|indications|all|0"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_eval_set.py -v`
Expected: FAIL with `ModuleNotFoundError` for `vet_agent.eval.eval_set`.

- [ ] **Step 4: Write the implementation**

`src/vet_agent/eval/eval_set.py`:

```python
from pathlib import Path

import yaml
from pydantic import BaseModel, TypeAdapter

from vet_agent.ingestion.chunker import logical_key
from vet_agent.ingestion.models import Chunk, SectionType


class EvalCase(BaseModel):
    """One frozen retrieval-eval query with its metadata-derived ground truth."""

    query: str
    flow: str  # dose | contraindication | indication | other
    drug: str
    section: SectionType
    species: list[str]
    relevant_logical_keys: list[str]


_ADAPTER = TypeAdapter(list[EvalCase])


def load_eval_set(path: Path) -> list[EvalCase]:
    """Parse + validate a retrieval_eval.yaml file into EvalCase objects."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _ADAPTER.validate_python(data)


def derive_relevant_keys(
    chunks: list[Chunk], *, drug: str, section: SectionType, species: list[str]
) -> list[str]:
    """All logical_keys whose chunk matches (drug, section) and the species constraint.

    Species rule: if `species` is empty, any species matches; otherwise the chunk must
    share at least one species OR be a prose chunk tagged ["all"].
    """
    wanted = set(species)
    keys: list[str] = []
    for chunk in chunks:
        if chunk.drug_name.lower() != drug.lower():
            continue
        if chunk.section_type != section:
            continue
        if wanted and not (set(chunk.species) & wanted) and "all" not in chunk.species:
            continue
        keys.append(logical_key(chunk))
    return keys
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_eval_set.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add src/vet_agent/eval/eval_set.py tests/eval/test_eval_set.py tests/eval/fixtures/
git commit -m "feat(eval): add eval-set schema, loader, and relevance-label derivation"
```

---

### Task 2.10: Benchmark harness + scorecard

**Files:**

- Create: `src/vet_agent/eval/benchmark.py`
- Test: `tests/eval/test_benchmark.py`

Notes: `embed_corpus` embeds all chunk texts into a normalized matrix (optionally cached to `.npz`, keyed by model + a corpus hash so it invalidates when chunks change). `rank_for_query` ranks by cosine (= dot, since normalized) with deterministic tie-breaking. `benchmark_model` averages per-query metrics; `choose_default` picks the winner (recall@5, then mrr).

- [ ] **Step 1: Write the failing test**

`tests/eval/test_benchmark.py`:

```python
from vet_agent.eval.benchmark import benchmark_model, choose_default, render_scorecard
from vet_agent.eval.eval_set import EvalCase
from vet_agent.ingestion.models import Chunk, SectionType
from tests.knowledge.fakes import FakeEmbedder


def _chunk(text, ordinal) -> Chunk:
    return Chunk(
        drug_name="Metronidazole",
        section_type=SectionType.DOSES,
        species=["dog"],
        book_page=873,
        text=text,
        ordinal=ordinal,
    )


def _cases():
    # FakeEmbedder maps identical text -> identical vector, so a query equal to a
    # chunk's text ranks that chunk #1 deterministically.
    return [
        EvalCase(
            query="alpha dose",
            flow="dose",
            drug="Metronidazole",
            section=SectionType.DOSES,
            species=["dog"],
            relevant_logical_keys=["metronidazole|doses|dog|0"],
        )
    ]


def test_benchmark_model_scores_a_perfect_match():
    chunks = [_chunk("alpha dose", 0), _chunk("unrelated text", 1)]
    score = benchmark_model("fake", FakeEmbedder(dim=8), chunks, _cases(), ks=[1, 3])
    assert score.model == "fake"
    assert score.metrics["recall@1"] == 1.0
    assert score.metrics["mrr"] == 1.0


def test_choose_default_picks_best_recall_at_5():
    chunks = [_chunk("alpha dose", 0)]
    good = benchmark_model("good", FakeEmbedder(dim=8), chunks, _cases(), ks=[5])
    # A model that never matches: empty corpus -> recall 0.
    bad = benchmark_model("bad", FakeEmbedder(dim=8), [], _cases(), ks=[5])
    assert choose_default([good, bad]) == "good"


def test_render_scorecard_lists_models_and_winner():
    chunks = [_chunk("alpha dose", 0)]
    score = benchmark_model("fake", FakeEmbedder(dim=8), chunks, _cases(), ks=[1, 5])
    md = render_scorecard([score], ks=[1, 5])
    assert "fake" in md
    assert "Winner" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_benchmark.py -v`
Expected: FAIL with `ModuleNotFoundError` for `vet_agent.eval.benchmark`.

- [ ] **Step 3: Write the implementation**

`src/vet_agent/eval/benchmark.py`:

```python
import hashlib
import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from vet_agent.eval.eval_set import EvalCase
from vet_agent.eval.metrics import evaluate_query, mean_metrics, rank_by_score
from vet_agent.ingestion.chunker import logical_key
from vet_agent.ingestion.models import Chunk
from vet_agent.knowledge.interfaces import Embedder


def _corpus_hash(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(logical_key(chunk).encode("utf-8"))
        h.update(chunk.text.encode("utf-8"))
    return h.hexdigest()[:16]


def embed_corpus(
    embedder: Embedder, chunks: list[Chunk], cache_dir: Path | None = None
) -> tuple[np.ndarray, list[str]]:
    """Embed every chunk's text into a [N, dim] matrix; cache by model + corpus hash."""
    keys = [logical_key(c) for c in chunks]
    cache_path = (
        cache_dir / f"{embedder.name}_{_corpus_hash(chunks)}.npz" if cache_dir else None
    )
    if cache_path is not None and cache_path.exists():
        return np.asarray(np.load(cache_path)["vectors"], dtype=np.float32), keys
    vectors = np.asarray(embedder.embed_documents([c.text for c in chunks]), dtype=np.float32)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, vectors=vectors)
    return vectors, keys


def rank_for_query(query_vector: list[float], corpus: np.ndarray, keys: list[str]) -> list[str]:
    """Rank corpus keys by cosine similarity to the query (vectors are normalized)."""
    if corpus.size == 0:
        return []
    sims = corpus @ np.asarray(query_vector, dtype=np.float32)
    return rank_by_score(list(zip(keys, sims.tolist(), strict=True)))


class ModelScore(BaseModel):
    model: str
    metrics: dict[str, float]


def benchmark_model(
    model_key: str,
    embedder: Embedder,
    chunks: list[Chunk],
    cases: list[EvalCase],
    ks: list[int],
    cache_dir: Path | None = None,
) -> ModelScore:
    """Embed the corpus once, then score every eval case in-memory."""
    corpus, keys = embed_corpus(embedder, chunks, cache_dir)
    per_query: list[dict[str, float]] = []
    for case in cases:
        ranked = rank_for_query(embedder.embed_query(case.query), corpus, keys)
        per_query.append(evaluate_query(ranked, set(case.relevant_logical_keys), ks))
    return ModelScore(model=model_key, metrics=mean_metrics(per_query))


def _primary(score: ModelScore) -> tuple[float, float]:
    return (score.metrics.get("recall@5", 0.0), score.metrics.get("mrr", 0.0))


def choose_default(scores: list[ModelScore]) -> str:
    """Winner = highest recall@5, tie-broken by mrr."""
    return max(scores, key=_primary).model


def render_scorecard(scores: list[ModelScore], ks: list[int]) -> str:
    """Render a markdown table (model x metric) and declare the winner."""
    cols = [f"recall@{k}" for k in ks] + [f"hit_rate@{k}" for k in ks] + ["mrr"]
    lines = ["| model | " + " | ".join(cols) + " |"]
    lines.append("|" + "---|" * (len(cols) + 1))
    for s in scores:
        cells = [f"{s.metrics.get(c, 0.0):.3f}" for c in cols]
        lines.append(f"| {s.model} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"**Winner:** {choose_default(scores)} (by recall@5, tie-broken by mrr)")
    return "\n".join(lines)


def write_scorecard(scores: list[ModelScore], ks: list[int], out_dir: Path) -> None:
    """Write benchmark_scorecard.md and .json to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark_scorecard.md").write_text(
        render_scorecard(scores, ks), encoding="utf-8"
    )
    (out_dir / "benchmark_scorecard.json").write_text(
        json.dumps([s.model_dump() for s in scores], indent=2), encoding="utf-8"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_benchmark.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vet_agent/eval/benchmark.py tests/eval/test_benchmark.py
git commit -m "feat(eval): add in-memory embedder benchmark + scorecard"
```

---

### Task 2.11: Eval-set generation (LLM phrasing)

**Files:**

- Create: `src/vet_agent/eval/eval_set_builder.py`
- Create: `scripts/build_eval_set.py`
- Test: `tests/eval/test_build_eval_set.py`

Notes: `build_eval_set` is pure + deterministic (seeded sampling, injected `QueryPhraser`), so it's fully testable with a `FakeQueryPhraser`. The builder lives in the package (mypy-checked, import-clean); the `scripts/` file is a thin CLI. The real `AnthropicQueryPhraser` is only exercised when the script is run by hand (Task 2.13). Generation samples drugs per flow that actually have the target section, derives labels via `derive_relevant_keys`, and phrases each query. `anthropic` is imported lazily inside `AnthropicQueryPhraser` (it's a dev-only dep; the package must import without it at runtime).

**Human review gate (spec §10):** the model-phrased questions are reviewed before they're frozen, in a batch flow — `generate` writes a draft, the author edits the `query` phrasings in the YAML, then `promote` validates and freezes it to the committed set. `promote_eval_set` re-parses the draft through `load_eval_set` (so a malformed hand-edit fails loudly) and re-serializes canonically; only `query` strings are meant to be hand-edited (labels are deterministic).

- [ ] **Step 1: Write the failing test**

`tests/eval/test_build_eval_set.py`:

```python
from vet_agent.eval.eval_set import load_eval_set
from vet_agent.eval.eval_set_builder import (
    FLOW_SECTIONS,
    build_eval_set,
    promote_eval_set,
    write_eval_set,
)
from vet_agent.ingestion.models import Chunk, SectionType


class FakeQueryPhraser:
    def phrase(self, drug, section, species, sample_text):
        return f"Q: {drug}/{section.value}/{'+'.join(species) or 'any'}"


def _chunks():
    return [
        Chunk(drug_name="Metronidazole", section_type=SectionType.DOSES,
              species=["dog"], book_page=1, text="dog dose", ordinal=0),
        Chunk(drug_name="Metronidazole", section_type=SectionType.INDICATIONS,
              species=["all"], book_page=1, text="treats giardia", ordinal=0),
    ]


def test_flow_sections_cover_three_flows():
    assert FLOW_SECTIONS["dose"] is SectionType.DOSES
    assert FLOW_SECTIONS["indication"] is SectionType.INDICATIONS
    assert FLOW_SECTIONS["contraindication"] is SectionType.CONTRAINDICATIONS


def test_build_eval_set_labels_and_phrases_deterministically():
    cases = build_eval_set(_chunks(), FakeQueryPhraser(), per_flow=5, seed=0)
    flows = {c.flow for c in cases}
    assert "dose" in flows and "indication" in flows
    dose = next(c for c in cases if c.flow == "dose")
    assert dose.relevant_logical_keys == ["metronidazole|doses|dog|0"]
    assert dose.query.startswith("Q: Metronidazole/doses/")
    # Deterministic: same seed -> identical output.
    again = build_eval_set(_chunks(), FakeQueryPhraser(), per_flow=5, seed=0)
    assert [c.model_dump() for c in cases] == [c.model_dump() for c in again]


def test_write_and_reload_roundtrip(tmp_path):
    cases = build_eval_set(_chunks(), FakeQueryPhraser(), per_flow=5, seed=0)
    out = tmp_path / "retrieval_eval.yaml"
    write_eval_set(cases, out)
    assert [c.model_dump() for c in load_eval_set(out)] == [c.model_dump() for c in cases]


def test_promote_validates_and_preserves_hand_edited_phrasing(tmp_path):
    cases = build_eval_set(_chunks(), FakeQueryPhraser(), per_flow=5, seed=0)
    draft = tmp_path / "retrieval_eval.draft.yaml"
    write_eval_set(cases, draft)
    # Simulate a human editing a query phrasing in the draft.
    edited = draft.read_text(encoding="utf-8").replace("Q: Metronidazole/doses/", "Edited dog dose question? ")
    draft.write_text(edited, encoding="utf-8")

    final = tmp_path / "retrieval_eval.yaml"
    count = promote_eval_set(draft, final)
    promoted = load_eval_set(final)
    assert count == len(promoted)
    dose = next(c for c in promoted if c.flow == "dose")
    assert dose.query.startswith("Edited dog dose question?")
    assert dose.relevant_logical_keys == ["metronidazole|doses|dog|0"]  # labels intact
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_build_eval_set.py -v`
Expected: FAIL with `ModuleNotFoundError` for `vet_agent.eval.eval_set_builder`.

- [ ] **Step 3: Write the in-package builder**

`src/vet_agent/eval/eval_set_builder.py`:

```python
import random
from pathlib import Path
from typing import Protocol

import yaml

from vet_agent.eval.eval_set import EvalCase, derive_relevant_keys
from vet_agent.ingestion.models import Chunk, SectionType

FLOW_SECTIONS: dict[str, SectionType] = {
    "dose": SectionType.DOSES,
    "contraindication": SectionType.CONTRAINDICATIONS,
    "indication": SectionType.INDICATIONS,
}


class QueryPhraser(Protocol):
    def phrase(
        self, drug: str, section: SectionType, species: list[str], sample_text: str
    ) -> str: ...


class AnthropicQueryPhraser:
    """Phrases a natural-language vet question for a (drug, section, species) target."""

    def __init__(self, api_key: str, model: str) -> None:
        from anthropic import Anthropic  # lazy: anthropic is a dev-only dependency

        self._client = Anthropic(api_key=api_key)
        self._model = model

    def phrase(
        self, drug: str, section: SectionType, species: list[str], sample_text: str
    ) -> str:
        who = " and ".join(species) if species else "an animal"
        prompt = (
            f"Write ONE natural question a veterinarian would ask about the drug {drug}, "
            f"specifically its {section.value.replace('_', ' ')} for {who}. "
            f"Base it on this source text:\n\n{sample_text[:600]}\n\n"
            "Return only the question, no preamble."
        )
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [getattr(block, "text", "") for block in msg.content if block.type == "text"]
        return "".join(parts).strip()


def build_eval_set(
    chunks: list[Chunk], phraser: QueryPhraser, *, per_flow: int, seed: int
) -> list[EvalCase]:
    """Sample targets per flow, derive labels, and phrase a query for each (deterministic)."""
    rng = random.Random(seed)
    cases: list[EvalCase] = []
    for flow, section in FLOW_SECTIONS.items():
        targets = sorted(
            {(c.drug_name, tuple(c.species)) for c in chunks if c.section_type == section}
        )
        rng.shuffle(targets)
        for drug, species_tuple in targets[:per_flow]:
            species = list(species_tuple)
            sample = next(
                (c.text for c in chunks if c.drug_name == drug and c.section_type == section),
                "",
            )
            keys = derive_relevant_keys(chunks, drug=drug, section=section, species=species)
            if not keys:
                continue
            cases.append(
                EvalCase(
                    query=phraser.phrase(drug, section, species, sample),
                    flow=flow,
                    drug=drug,
                    section=section,
                    species=species,
                    relevant_logical_keys=keys,
                )
            )
    return cases


def write_eval_set(cases: list[EvalCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [c.model_dump(mode="json") for c in cases]
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def promote_eval_set(draft_path: Path, final_path: Path) -> int:
    """Validate a human-reviewed draft and freeze it to the committed eval-set path.

    Re-parses the draft through load_eval_set (a malformed hand-edit fails loudly here,
    not later in the benchmark) and re-serializes canonically. Returns the case count.
    """
    from vet_agent.eval.eval_set import load_eval_set

    cases = load_eval_set(draft_path)
    write_eval_set(cases, final_path)
    return len(cases)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_build_eval_set.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Write the thin CLI script (generate + promote)**

`scripts/build_eval_set.py`:

```python
"""Offline generation of the frozen retrieval eval set, with a human review gate.

Run by hand (NOT in CI):
    # 1) draft (needs VET_ANTHROPIC_API_KEY)
    uv run python scripts/build_eval_set.py generate --chunks data/ingest/chunks.json
    # 2) review/edit the query phrasings in data/eval/retrieval_eval.draft.yaml
    # 3) freeze the reviewed draft
    uv run python scripts/build_eval_set.py promote
"""

from pathlib import Path

import typer

from vet_agent.config import Settings
from vet_agent.eval.eval_set_builder import (
    AnthropicQueryPhraser,
    build_eval_set,
    promote_eval_set,
    write_eval_set,
)
from vet_agent.knowledge.loader import read_chunks

app = typer.Typer(help="Generate + promote the frozen retrieval eval set.")

DRAFT = Path("data/eval/retrieval_eval.draft.yaml")
FINAL = Path("data/eval/retrieval_eval.yaml")


@app.command()
def generate(
    chunks: Path = typer.Option(Path("data/ingest/chunks.json")),  # noqa: B008
    draft: Path = typer.Option(DRAFT),  # noqa: B008
    per_flow: int = typer.Option(25, help="Targets sampled per flow"),  # noqa: B008
    seed: int = typer.Option(0),  # noqa: B008
) -> None:
    """Phrase queries with Claude and write a REVIEWABLE DRAFT (not the frozen set)."""
    settings = Settings()
    if not settings.anthropic_api_key:
        typer.echo("Error: VET_ANTHROPIC_API_KEY is required to phrase queries.")
        raise typer.Exit(code=1)
    phraser = AnthropicQueryPhraser(settings.anthropic_api_key, settings.reasoning_model)
    cases = build_eval_set(read_chunks(chunks), phraser, per_flow=per_flow, seed=seed)
    write_eval_set(cases, draft)
    typer.echo(
        f"Wrote {len(cases)} DRAFT cases -> {draft}\n"
        "Review/edit the 'query' phrasings (labels are derived — leave them), "
        "then run: scripts/build_eval_set.py promote"
    )


@app.command()
def promote(
    draft: Path = typer.Option(DRAFT),  # noqa: B008
    out: Path = typer.Option(FINAL),  # noqa: B008
) -> None:
    """Validate the reviewed draft and freeze it to the committed eval set."""
    if not draft.is_file():
        typer.echo(f"Error: draft not found at {draft} (run 'generate' first).")
        raise typer.Exit(code=1)
    count = promote_eval_set(draft, out)
    typer.echo(f"Promoted {count} reviewed cases -> {out}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 6: Commit**

```bash
git add src/vet_agent/eval/eval_set_builder.py scripts/build_eval_set.py \
        tests/eval/test_build_eval_set.py
git commit -m "feat(eval): add eval-set generation with draft->review->promote gate"
```

---

### Task 2.12: CLI commands — benchmark, load, retrieve

**Files:**

- Modify: `src/vet_agent/cli/main.py`
- Test: `tests/test_cli.py`

Notes: the heavy wiring (real model + Qdrant) is exercised manually in Task 2.13. CLI tests stay offline: they assert the commands are registered and that pre-flight validation (missing files / empty query) exits non-zero before any model loads. Commands resolve config defaults and delegate to the already-tested functions.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:

```python
def test_help_lists_phase2_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("benchmark", "load", "retrieve"):
        assert cmd in result.stdout


def test_load_requires_existing_chunks(tmp_path):
    missing = tmp_path / "nope.json"
    result = runner.invoke(app, ["load", str(missing)])
    assert result.exit_code != 0
    assert "not found" in result.stdout.lower()


def test_retrieve_rejects_blank_query():
    result = runner.invoke(app, ["retrieve", "   "])
    assert result.exit_code != 0
    assert "query" in result.stdout.lower()


def test_benchmark_requires_existing_eval_set(tmp_path):
    chunks = tmp_path / "chunks.json"
    chunks.write_text("[]", encoding="utf-8")
    missing_eval = tmp_path / "eval.yaml"
    result = runner.invoke(
        app, ["benchmark", "--chunks", str(chunks), "--eval-set", str(missing_eval)]
    )
    assert result.exit_code != 0
    assert "not found" in result.stdout.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — the new commands don't exist yet.

- [ ] **Step 3: Write the implementation**

Add to `src/vet_agent/cli/main.py` these imports at the top (alongside the existing ones), then add the three commands before `if __name__`:

```python
from qdrant_client import QdrantClient

from vet_agent.config import Settings
from vet_agent.eval.benchmark import ModelScore, benchmark_model, write_scorecard
from vet_agent.eval.eval_set import load_eval_set
from vet_agent.ingestion.models import SectionType
from vet_agent.knowledge.embedders import MODEL_REGISTRY, get_embedder
from vet_agent.knowledge.interfaces import Reranker
from vet_agent.knowledge.loader import load_chunks, read_chunks
from vet_agent.knowledge.retrieval import Retriever
from vet_agent.knowledge.vector_store import QdrantVectorStore, collection_name
```

```python
@app.command()
def benchmark(
    chunks: Path = typer.Option(Path("data/ingest/chunks.json")),  # noqa: B008
    eval_set: Path = typer.Option(Path("data/eval/retrieval_eval.yaml")),  # noqa: B008
    models: str = typer.Option("medembed-base,bge-base,qwen3-0.6b"),  # noqa: B008
    ks: str = typer.Option("1,3,5,10", help="Comma-separated k cutoffs"),  # noqa: B008
    cache_dir: Path = typer.Option(Path("data/embeddings")),  # noqa: B008
    out_dir: Path = typer.Option(Path("data/eval")),  # noqa: B008
) -> None:
    """Benchmark candidate embedders in-memory and write a scorecard."""
    if not chunks.is_file():
        typer.echo(f"Error: chunks file not found at {chunks}")
        raise typer.Exit(code=1)
    if not eval_set.is_file():
        typer.echo(f"Error: eval set not found at {eval_set}")
        raise typer.Exit(code=1)
    k_values = [int(x) for x in ks.split(",")]
    parsed = read_chunks(chunks)
    cases = load_eval_set(eval_set)
    scores: list[ModelScore] = []
    for key in [m.strip() for m in models.split(",")]:
        if key not in MODEL_REGISTRY:
            typer.echo(f"Error: unknown model '{key}'. Known: {sorted(MODEL_REGISTRY)}")
            raise typer.Exit(code=1)
        typer.echo(f"Benchmarking {key} ...")
        scores.append(
            benchmark_model(key, get_embedder(key), parsed, cases, k_values, cache_dir)
        )
    write_scorecard(scores, k_values, out_dir)
    typer.echo(f"Scorecard -> {out_dir / 'benchmark_scorecard.md'}")


@app.command()
def load(
    chunks: Path = typer.Argument(Path("data/ingest/chunks.json")),  # noqa: B008
    model: str = typer.Option("", help="Override the configured embedding model"),  # noqa: B008
    no_prune: bool = typer.Option(False, "--no-prune"),  # noqa: B008
) -> None:
    """Idempotently embed + load chunks into Qdrant."""
    if not chunks.is_file():
        typer.echo(f"Error: chunks file not found at {chunks}")
        raise typer.Exit(code=1)
    settings = Settings()
    model_key = model or settings.embedding_model
    embedder = get_embedder(model_key)
    client = QdrantClient(url=settings.qdrant_url)
    store = QdrantVectorStore(
        client, collection_name(settings.qdrant_collection_prefix, model_key)
    )
    report = load_chunks(
        read_chunks(chunks),
        embedder,
        store,
        prune=not no_prune,
        batch_size=settings.embedding_batch_size,
    )
    typer.echo(
        f"Loaded into '{collection_name(settings.qdrant_collection_prefix, model_key)}': "
        f"upserted={report.upserted} skipped={report.skipped} pruned={report.pruned}"
    )


@app.command()
def retrieve(
    query: str = typer.Argument(...),  # noqa: B008
    drug: str = typer.Option("", help="Filter by drug name"),  # noqa: B008
    section: str = typer.Option("", help="Filter by section_type, e.g. doses"),  # noqa: B008
    species: str = typer.Option("", help="Filter by species, e.g. dog"),  # noqa: B008
    top_k: int = typer.Option(5),  # noqa: B008
    rerank: bool = typer.Option(False, "--rerank"),  # noqa: B008
) -> None:
    """Filtered semantic search against the loaded Qdrant collection."""
    if not query.strip():
        typer.echo("Error: query must not be blank.")
        raise typer.Exit(code=1)
    settings = Settings()
    model_key = settings.embedding_model
    embedder = get_embedder(model_key)
    client = QdrantClient(url=settings.qdrant_url)
    store = QdrantVectorStore(
        client, collection_name(settings.qdrant_collection_prefix, model_key)
    )
    reranker: Reranker | None = None
    if rerank:
        from vet_agent.knowledge.rerankers import CrossEncoderReranker

        reranker = CrossEncoderReranker(settings.reranker_model)
    retriever = Retriever(embedder, store, reranker=reranker)
    section_type = SectionType(section) if section else None
    hits = retriever.retrieve(
        query,
        drug=drug or None,
        section=section_type,
        species=species or None,
        top_k=top_k,
        rerank=rerank,
    )
    if not hits:
        typer.echo("No results.")
        return
    for h in hits:
        score = f"{h.score:.3f}" if h.score is not None else "n/a"
        typer.echo(f"[{score}] {h.drug_name} / {h.section_type.value} / p.{h.book_page}")
        typer.echo(f"    {h.text[:200]}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (existing ingest tests + 4 new).

- [ ] **Step 5: Run the full gate**

Run: `make check`
Expected: ruff clean, mypy `Success`, all fast tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/vet_agent/cli/main.py tests/test_cli.py
git commit -m "feat(cli): add benchmark, load, and retrieve commands"
```

---

### Task 2.13: Real-model + live-Qdrant verification (manual)

**Files:**

- Create: `data/eval/retrieval_eval.yaml` (committed)
- Create: `data/eval/benchmark_scorecard.{md,json}` (committed)
- Modify: `.gitignore` (ignore `data/embeddings/`)
- Modify: `src/vet_agent/config.py` (set `embedding_model` default to the benchmark winner)

This task is **manual** (needs network for model weights, an Anthropic key, and a running Qdrant). It is the Phase-2 acceptance check, analogous to Phase-1's real-PDF verification.

> **✅ COMPLETED (2026-07) — actual results** (steps below kept as the runbook of record):
> - **Step 1** ✅ gitignore done via a `/data/*` un-ignore-exception so the committed eval set +
>   scorecard stay tracked while the embedding cache + eval-set draft are ignored.
> - **Step 2** ✅ real models exercised directly through `embed`/`benchmark`/`load` (MedEmbed + bge
>   loaded and embedded all 15k chunks); `pytest -m slow` optional.
> - **Steps 3a–3c** ✅ generated a **200-case** draft (8 flows × 25, cat/dog-focused by default),
>   human-reviewed the phrasings, and promoted to `data/eval/retrieval_eval.yaml`.
> - **Step 4** ✅ benchmarked **two** models (Qwen3 dropped) → **bge-base won** (recall@5 0.764 vs
>   0.741; mrr 0.781 vs 0.746). Scorecard committed.
> - **Step 5** ✅ pinned `embedding_model = "bge-base"`.
> - **Step 6** ✅ loaded 15,292 chunks into `vet_chunks__bge_base`; re-run confirmed idempotency
>   (`upserted=0 skipped=15292 pruned=0`).
> - **Step 7** ✅ filtered retrieval returns correctly-cited Metronidazole dog-dose passages with no
>   cat-only leakage; enrofloxacin/meloxicam filtered queries also validated.
> - **Step 8** ✅ `make check` green; eval set, scorecard, and config pin committed.

- [ ] **Step 1: Ignore the embedding cache and the eval-set draft**

Add to `.gitignore`:

```
data/embeddings/
data/eval/retrieval_eval.draft.yaml
```

- [ ] **Step 2: Run the slow unit tests once (real embedder + reranker)**

Run: `uv run pytest -m slow -v`
Expected: PASS — MedEmbed-base loads and embeds 768-d; bge-reranker reorders by relevance. (First run downloads weights.)

- [ ] **Step 3a: Generate the eval-set DRAFT**

Ensure `VET_ANTHROPIC_API_KEY` is set (e.g. in `.env`). Then:

Run: `uv run python scripts/build_eval_set.py generate --chunks data/ingest/chunks.json --per-flow 25 --seed 0`
Expected: writes ~60–75 draft cases (25 per flow, minus any with no derivable labels) to `data/eval/retrieval_eval.draft.yaml`.

- [ ] **Step 3b: Review the draft (human gate — spec §10)**

Open `data/eval/retrieval_eval.draft.yaml` and review the `**query` phrasings** in one pass: fix any that are awkward, leading, or off-target, and delete weak cases. Leave `relevant_logical_keys` untouched (they're derived). Confirm queries read like real vet questions and the labels are non-empty and match the drug/section/species.

- [ ] **Step 3c: Promote the reviewed draft to the frozen set**

Run: `uv run python scripts/build_eval_set.py promote`
Expected: validates the edited draft (fails loudly on a malformed hand-edit) and writes the committed `data/eval/retrieval_eval.yaml`.

- [ ] **Step 4: Run the benchmark**

Run: `uv run vet-agent benchmark`
Expected: prints progress per model, writes `data/eval/benchmark_scorecard.md`. Open it: confirm three models scored across recall@k / hit_rate@k / mrr, a per-model row each, and a declared **Winner**. Note the winner's key.

- [ ] **Step 5: Pin the winning model as the default**

Edit `src/vet_agent/config.py`: set `embedding_model` to the winner from Step 4 (e.g. keep `"medembed-base"` if it won, else `"bge-base"` / `"qwen3-0.6b"`).

- [ ] **Step 6: Bring up Qdrant and load**

Run: `docker compose up -d qdrant`
Run: `uv run vet-agent load data/ingest/chunks.json`
Expected: `upserted=<~15292> skipped=0 pruned=0` on first load. Re-run the same command:
Run: `uv run vet-agent load data/ingest/chunks.json`
Expected: `upserted=0 skipped=<~15292> pruned=0` (idempotency confirmed against a live server).

- [ ] **Step 7: Validate filtered retrieval (the Phase-2 demo)**

Run: `uv run vet-agent retrieve "metronidazole dose for a 12 kg dog with giardia" --section doses --species dog`
Expected: top hits are Metronidazole `doses` chunks tagged `dog` (or `cat+dog`), with sensible text + `p.<page>` citations. Confirm a cat-only dose never appears. Try `--rerank` and confirm it still returns dog-dose passages.

- [ ] **Step 8: Run the full gate + commit artifacts**

Run: `make check`
Expected: green (fast suite; slow deselected).

```bash
git add .gitignore data/eval/retrieval_eval.yaml data/eval/benchmark_scorecard.md \
        data/eval/benchmark_scorecard.json src/vet_agent/config.py
git commit -m "feat(eval): commit frozen eval set + benchmark scorecard; pin default embedder"
```

---

## Definition of Done (Phase 2)

- [x] `make check` is green (ruff + mypy strict + the fast, offline suite).
- [x] All automated tests run **offline**: `FakeEmbedder` + Qdrant `:memory:` mode; no network, no Docker, no model downloads in the default suite.
- [x] `data/eval/retrieval_eval.yaml` is committed and frozen; `vet-agent benchmark` produces `benchmark_scorecard.md` and a default model is chosen **on data**; reranker lift is measurable via `--rerank` in retrieval (lift not yet *quantified* on the frozen set — deferred).
- [x] `vet-agent load` is proven idempotent — unit tests (skip/re-embed/prune) plus the live re-run in Task 2.13 Step 6.
- [x] `vet-agent retrieve "..." --section doses --species dog` returns correctly filtered, cited passages from a loaded Qdrant collection.
- [x] `config.embedding_model` is pinned to the benchmark winner; collections are name-suffixed per model.
- [x] `README.md` documents the full command workflow end-to-end — every `vet-agent` step in order (`ingest` → `embed` → `benchmark` → `load` → `retrieve`) with its purpose and a runnable example invocation.

## Spec Coverage Map

- Interfaces (spec §3, §4) → Task 2.2.
- Embedders + registry (§5) → Task 2.4.
- Reranker (§6) → Task 2.5; lift measured via retrieval `--rerank` (§11) + Task 2.13 Step 7.
- Qdrant schema, name-suffixed collections, payload indexes, uuid5 ids (§7) → Tasks 2.6, 2.7.
- Idempotent loader (§8) → Task 2.7.
- Retriever (§9) → Task 2.8.
- Eval set + generation (§10) → Tasks 2.9, 2.11, 2.13.
- Metrics + benchmark + tie-handling (§11) → Tasks 2.3, 2.10.
- CLI (§12) → Task 2.12.
- Config (§13) → Task 2.1 (+ winner pin in 2.13).
- Testing strategy (§14) → fakes/in-memory across all tasks; slow markers in 2.4/2.5/2.13.
- Dependencies (§15, anthropic dev-only) → Task 2.1.
- Definition of Done (§16) → above.

## Known follow-ups (deferred, not blocking Phase 2)

- The benchmark embeds the full corpus per model in `vet-agent benchmark`; for three models this is a few minutes on CPU (cached after first run). Acceptable for a one-time choice.
- `retrieve`/`load` assume the configured model's collection exists; a friendly "collection missing — run `vet-agent load`" message is a nice-to-have, deferred.
- Exact Qwen3 query-prompt format and current `sentence-transformers` API are confirmed against live model cards at implementation time (spec §5); adjust `MODEL_REGISTRY` if the card differs.

