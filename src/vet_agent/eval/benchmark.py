import hashlib
import json
import logging
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from vet_agent.eval.eval_set import EvalCase
from vet_agent.eval.metrics import evaluate_query, mean_metrics, rank_by_score
from vet_agent.ingestion.chunker import logical_key
from vet_agent.ingestion.models import Chunk
from vet_agent.knowledge.interfaces import Embedder

logger = logging.getLogger(__name__)


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
    cache_path = cache_dir / f"{embedder.name}_{_corpus_hash(chunks)}.npz" if cache_dir else None
    if cache_path is not None and cache_path.exists():
        logger.info("[%s] reusing cached embeddings for %d chunks", embedder.name, len(keys))
        return np.asarray(np.load(cache_path)["vectors"], dtype=np.float32), keys
    logger.info(
        "[%s] embedding %d chunks — this is the one-time slow step (a few minutes); "
        "a progress bar follows ...",
        embedder.name,
        len(chunks),
    )
    vectors = np.asarray(embedder.embed_documents([c.text for c in chunks]), dtype=np.float32)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, vectors=vectors)
        logger.info("[%s] cached embeddings -> %s", embedder.name, cache_path)
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
    logger.info("[%s] scoring %d eval queries ...", model_key, len(cases))
    per_query: list[dict[str, float]] = []
    for case in cases:
        ranked = rank_for_query(embedder.embed_query(case.query), corpus, keys)
        per_query.append(evaluate_query(ranked, set(case.relevant_logical_keys), ks))
    score = ModelScore(model=model_key, metrics=mean_metrics(per_query))
    logger.info(
        "[%s] done: recall@5=%.3f mrr=%.3f",
        model_key,
        score.metrics.get("recall@5", 0.0),
        score.metrics.get("mrr", 0.0),
    )
    return score


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
    (out_dir / "benchmark_scorecard.md").write_text(render_scorecard(scores, ks), encoding="utf-8")
    (out_dir / "benchmark_scorecard.json").write_text(
        json.dumps([s.model_dump() for s in scores], indent=2), encoding="utf-8"
    )
