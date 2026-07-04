from pathlib import Path

from vet_agent.config import Settings

# NOTE: default-value tests pass _env_file=None so they assert the code defaults and
# never read a developer's real .env (which may hold a live VET_ANTHROPIC_API_KEY).


def test_defaults():
    s = Settings(_env_file=None)
    assert s.reasoning_model == "claude-sonnet-5"
    assert s.qdrant_url == "http://localhost:6333"
    assert s.data_dir == Path("data")
    assert s.anthropic_api_key is None


def test_env_override(monkeypatch):
    monkeypatch.setenv("VET_REASONING_MODEL", "claude-3-5-sonnet-latest")
    monkeypatch.setenv("VET_ANTHROPIC_API_KEY", "sk-test")
    s = Settings(_env_file=None)
    assert s.reasoning_model == "claude-3-5-sonnet-latest"
    assert s.anthropic_api_key is not None
    assert s.anthropic_api_key.get_secret_value() == "sk-test"


def test_phase2_defaults():
    s = Settings(_env_file=None)
    assert s.embedding_model == "medembed-base"
    assert s.qdrant_collection_prefix == "vet_chunks"
    assert s.rerank_enabled is False
    assert s.reranker_model == "bge-reranker-v2-m3"
    assert s.embedding_batch_size == 64
    assert s.retrieval_top_k == 5


def test_phase2_env_override(monkeypatch):
    monkeypatch.setenv("VET_EMBEDDING_MODEL", "bge-base")
    monkeypatch.setenv("VET_RERANK_ENABLED", "true")
    s = Settings(_env_file=None)
    assert s.embedding_model == "bge-base"
    assert s.rerank_enabled is True
