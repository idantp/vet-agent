from pathlib import Path

from vet_agent.config import Settings


def test_defaults():
    s = Settings()
    assert s.reasoning_model == "claude-sonnet-4-6"
    assert s.qdrant_url == "http://localhost:6333"
    assert s.data_dir == Path("data")
    assert s.anthropic_api_key is None


def test_env_override(monkeypatch):
    monkeypatch.setenv("VET_REASONING_MODEL", "claude-3-5-sonnet-latest")
    monkeypatch.setenv("VET_ANTHROPIC_API_KEY", "sk-test")
    s = Settings()
    assert s.reasoning_model == "claude-3-5-sonnet-latest"
    assert s.anthropic_api_key == "sk-test"
