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
