"""
Application settings management.

This module defines the `Settings` class which loads configuration
values from environment variables.  It uses Pydantic's `BaseSettings`
to automatically read a `.env` file when present and provides type
checked attributes for all configurable parameters.  Default values
reflect sensible defaults for local development while secret or
sensitive values should always be provided via environment variables
or a `.env` file that is **not** checked into source control.

The configuration supports switching between different large language
models (OpenAI, Bedrock/Claude, or a local Ollama instance), tuning
retrieval parameters, specifying a reranker model and controlling
guardrails such as rate limiting.
"""

from __future__ import annotations

from pydantic import BaseSettings, Field, validator
from typing import Optional
import os


class Settings(BaseSettings):
    """Centralised configuration for the RAG application.

    Values are loaded from environment variables prefixed with the
    corresponding attribute name.  A `.env` file in the project root
    will be automatically read if present.
    """

    # Application
    app_name: str = Field("rag-industrial-eval-guardrails", description="Nom de l'application")

    # Database configuration
    postgres_host: str = Field("localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(5432, env="POSTGRES_PORT")
    postgres_db: str = Field("rag", env="POSTGRES_DB")
    postgres_user: str = Field("rag_user", env="POSTGRES_USER")
    postgres_password: str = Field("rag_password", env="POSTGRES_PASSWORD")

    # OpenAI configuration
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    openai_api_base: str = Field("https://api.openai.com/v1", env="OPENAI_API_BASE")

    # Bedrock/Claude configuration
    bedrock_aws_access_key_id: Optional[str] = Field(None, env="BEDROCK_AWS_ACCESS_KEY_ID")
    bedrock_aws_secret_access_key: Optional[str] = Field(None, env="BEDROCK_AWS_SECRET_ACCESS_KEY")
    bedrock_aws_region: Optional[str] = Field(None, env="BEDROCK_AWS_REGION")
    bedrock_model_id: str = Field("anthropic.claude-3.5-sonnet-v1", env="BEDROCK_MODEL_ID")

    # Local model configuration
    ollama_base_url: str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field("mistral", env="OLLAMA_MODEL")

    # Pipeline parameters
    default_model: str = Field("openai:gpt-4.1-mini", env="DEFAULT_MODEL")
    retrieval_top_k: int = Field(8, env="RETRIEVAL_TOP_K")
    hybrid_weight_bm25: float = Field(0.5, env="HYBRID_WEIGHT_BM25")
    hybrid_weight_vector: float = Field(0.5, env="HYBRID_WEIGHT_VECTOR")
    chunk_size: int = Field(512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(64, env="CHUNK_OVERLAP")

    # Reranker configuration
    reranker_model: str = Field("BAAI/bge-reranker-large", env="RERANKER_MODEL")

    # Guardrails / rate limiting
    rate_limit_requests: int = Field(60, env="RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(60, env="RATE_LIMIT_PERIOD")
    max_tokens: int = Field(4096, env="MAX_TOKENS")

    # Observability
    log_level: str = Field("INFO", env="LOG_LEVEL")
    metrics_port: int = Field(8001, env="METRICS_PORT")

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def database_url(self) -> str:
        """Return the SQLAlchemy/asyncpg connection string."""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def database_url_sync(self) -> str:
        """Return a synchronous Postgres connection URL for scripts."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @validator("hybrid_weight_vector")
    def validate_weights(cls, v: float, values: dict[str, object]) -> float:
        """Ensure the hybrid weights sum to ~1.0."""
        w_bm25 = values.get("hybrid_weight_bm25") or 0.0
        if abs(w_bm25 + v - 1.0) > 0.01:
            raise ValueError("Hybrid weights for BM25 and vector search must sum to 1.0")
        return v
