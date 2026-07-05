import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """How to load and post-process one embedding model."""

    hf_id: str
    dim: int
    query_prefix: str | None = None  # prepended in embed_query only
    matryoshka_dim: int | None = None  # truncate native dim down to this


# Both normalized to 768-d so the benchmark compares like-for-like.
# MedEmbed-base is a direct fine-tune of bge-base-en-v1.5 (isolates the medical gain).
# Verify exact hf ids against the live model cards before first run.
MODEL_REGISTRY: dict[str, "ModelSpec"] = {
    "medembed-base": ModelSpec("abhinand/MedEmbed-base-v0.1", dim=768),
    "bge-base": ModelSpec("BAAI/bge-base-en-v1.5", dim=768),
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

        import torch

        device = "mps" if torch.backends.mps.is_available() else None

        self.name = key
        self.dim = spec.dim
        self._spec = spec
        self._model = SentenceTransformer(spec.hf_id, device=device)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        # Show a live progress bar for large batches (the corpus embed) but not for a
        # single query, so `benchmark`/`embed` don't look stuck during the slow step.
        raw = self._model.encode(
            texts, normalize_embeddings=False, show_progress_bar=len(texts) > 64
        )
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
