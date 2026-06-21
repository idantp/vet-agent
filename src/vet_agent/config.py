from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, overridable via VET_-prefixed env vars or .env."""

    model_config = SettingsConfigDict(env_prefix="VET_", env_file=".env", extra="ignore")

    # LLM
    anthropic_api_key: str | None = None
    reasoning_model: str = "claude-sonnet-4-6"

    # Vector DB (used from Phase 2 onward)
    qdrant_url: str = "http://localhost:6333"

    # Paths
    data_dir: Path = Path("data")
