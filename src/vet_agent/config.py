from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, overridable via VET_-prefixed env vars or .env."""

    model_config = SettingsConfigDict(env_prefix="VET_", env_file=".env", extra="ignore")

    # LLM. SecretStr so the key renders as '**********' in any repr/log/traceback;
    # read the raw value with .get_secret_value() only where it's actually needed.
    anthropic_api_key: SecretStr | None = None
    reasoning_model: str = "claude-sonnet-5"

    # Vector DB (used from Phase 2 onward)
    qdrant_url: str = "http://localhost:6333"

    # Paths
    data_dir: Path = Path("data")

    # Knowledge layer (Phase 2)
    embedding_model: str = "medembed-base"
    qdrant_collection_prefix: str = "vet_chunks"
    rerank_enabled: bool = False
    reranker_model: str = "bge-reranker-v2-m3"
    embedding_batch_size: int = 64
    retrieval_top_k: int = 5
