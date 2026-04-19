"""OpenAI embeddings provider implementation.

Patched for local-compatible base URLs (e.g. Qwen3 MLX server on port 1235).
Uses the official openai SDK so OPENAI_BASE_URL / OPENAI_API_KEY env vars are
honored automatically, and base64 encoding_format is handled client-side.
"""

import os

from openai import AsyncOpenAI

from .base import EmbeddingProvider


# Known cloud OpenAI model dimensions. Local / self-hosted servers that are
# OpenAI-compatible may serve arbitrary models; dim is auto-detected on first
# call in that case.
KNOWN_OPENAI_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible embeddings provider (cloud or self-hosted)."""

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str | None = None):
        base_url = os.environ.get("OPENAI_BASE_URL")
        is_local = bool(base_url) and ("api.openai.com" not in base_url)

        # If caller used an unknown model name against the real API, fail fast.
        # For local-compatible servers, accept any model name and auto-detect dim.
        if not is_local and model_name not in KNOWN_OPENAI_DIMENSIONS:
            raise ValueError(
                f"Unknown OpenAI model: {model_name}. "
                f"Supported cloud models: {list(KNOWN_OPENAI_DIMENSIONS)}. "
                f"For self-hosted OpenAI-compatible servers, set OPENAI_BASE_URL."
            )

        dim = KNOWN_OPENAI_DIMENSIONS.get(model_name, 0)
        super().__init__(model_name, dim)

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or ("sk-local" if is_local else None)
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable."
            )

        # AsyncOpenAI picks up OPENAI_BASE_URL automatically
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=base_url)
        self._auto_dim = 0  # filled on first embed for local/self-hosted

    async def embed_text(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = await self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        data = sorted(resp.data, key=lambda x: x.index)
        embeddings = [list(item.embedding) for item in data]

        # Auto-detect dim on first successful call for local servers
        if self.dimensions == 0 and embeddings:
            self._auto_dim = len(embeddings[0])
            self.dimensions = self._auto_dim
        return embeddings

    @property
    def provider_name(self) -> str:
        return "openai"

    async def close(self):
        await self.client.close()
