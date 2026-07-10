from vet_agent.knowledge.interfaces import Passage


def _apply_scores(passages: list[Passage], scores: list[float], top_k: int) -> list[Passage]:
    """Attach scores, sort by score descending, return the top_k passages."""
    scored = [
        p.model_copy(update={"score": float(s)}) for p, s in zip(passages, scores, strict=True)
    ]
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
