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
