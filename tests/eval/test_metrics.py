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
