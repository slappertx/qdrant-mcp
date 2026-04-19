"""Settings management for Qdrant MCP server."""

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for Qdrant MCP server."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Qdrant settings
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="URL of the Qdrant server"
    )
    qdrant_api_key: str | None = Field(
        default=None,
        description="API key for Qdrant (optional)"
    )
    collection_name: str = Field(
        default="mcp_memory",
        description="Name of the Qdrant collection to use"
    )
    
    # Embedding settings
    embedding_provider: Literal["openai", "sentence-transformers"] = Field(
        default="openai",
        description="Embedding provider to use"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name"
    )
    
    # OpenAI settings (optional, used when provider is "openai")
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key (required for OpenAI embeddings)"
    )
    
    # Sentence Transformers settings (optional)
    device: str | None = Field(
        default=None,
        description="Device to run sentence transformers on (cpu, cuda, etc.)"
    )
    
    # Search settings
    default_limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Default number of results to return"
    )
    score_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum score threshold for search results"
    )
    
    # Server settings
    server_name: str = Field(
        default="qdrant-mcp",
        description="Name of the MCP server"
    )
    server_version: str = Field(
        default="0.1.0",
        description="Version of the MCP server"
    )
    
    @field_validator("embedding_model")
    @classmethod
    def validate_embedding_model(cls, v: str, info) -> str:
        """Validate embedding model. Cloud OpenAI names are checked strictly.

        Self-hosted OpenAI-compatible servers (OPENAI_BASE_URL pointing
        somewhere other than api.openai.com) can serve any model name.
        """
        import os
        provider = info.data.get("embedding_provider", "openai")
        base_url = os.environ.get("OPENAI_BASE_URL") or ""
        is_local = "api.openai.com" not in base_url and base_url != ""

        cloud_openai = {"text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"}
        if provider == "openai" and not is_local and v not in cloud_openai:
            raise ValueError(
                f"Invalid OpenAI model: {v}. Valid cloud models: {sorted(cloud_openai)}. "
                f"For self-hosted, set OPENAI_BASE_URL."
            )
        return v

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_api_key(cls, v: str | None, info) -> str | None:
        """API key is required for cloud OpenAI. Self-hosted servers may
        accept any string (or no key), so we allow None there."""
        import os
        provider = info.data.get("embedding_provider", "openai")
        base_url = os.environ.get("OPENAI_BASE_URL") or ""
        is_local = "api.openai.com" not in base_url and base_url != ""
        if provider == "openai" and not v and not is_local:
            raise ValueError(
                "OpenAI API key is required when using OpenAI embedding provider. "
                "Set OPENAI_API_KEY environment variable."
            )
        return v


def get_settings() -> Settings:
    """Get settings instance."""
    return Settings()