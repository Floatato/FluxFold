"""Benchmark runtime configuration and provider construction."""

from __future__ import annotations

import os
from pathlib import Path

from fluxfold.config import FluxFoldConfig
from fluxfold.errors import ValidationError
from fluxfold.providers import (
    EmbeddingModelInfo,
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleGenerationProvider,
)


def load_config(path: str | None) -> FluxFoldConfig:
    return FluxFoldConfig.from_toml(path) if path else FluxFoldConfig()


def generation_provider(config: FluxFoldConfig) -> OpenAICompatibleGenerationProvider:
    return OpenAICompatibleGenerationProvider(
        model=_required_env("FLUXFOLD_GENERATION_MODEL"),
        api_key=os.environ.get("FLUXFOLD_GENERATION_API_KEY", "EMPTY"),
        base_url=os.environ.get("FLUXFOLD_GENERATION_BASE_URL") or None,
        transport_max_retries=config.transport_max_retries,
        retry_initial_seconds=config.retry_initial_seconds,
        retry_multiplier=config.retry_multiplier,
    )


def embedding_provider(config: FluxFoldConfig) -> OpenAICompatibleEmbeddingProvider:
    dimension_value = _required_env("FLUXFOLD_EMBEDDING_DIMENSION")
    try:
        dimension = int(dimension_value)
    except ValueError as error:
        raise ValidationError(
            "FLUXFOLD_EMBEDDING_DIMENSION must be an integer"
        ) from error
    info = EmbeddingModelInfo(
        provider="openai-compatible",
        model=_required_env("FLUXFOLD_EMBEDDING_MODEL"),
        revision=os.environ.get("FLUXFOLD_EMBEDDING_REVISION", "unspecified"),
        dimension=dimension,
        normalization="l2",
        query_mode=os.environ.get("FLUXFOLD_EMBEDDING_QUERY_MODE", "plain"),
        document_mode=os.environ.get("FLUXFOLD_EMBEDDING_DOCUMENT_MODE", "plain"),
    )
    return OpenAICompatibleEmbeddingProvider(
        model_info=info,
        api_key=os.environ.get("FLUXFOLD_EMBEDDING_API_KEY", "EMPTY"),
        base_url=os.environ.get("FLUXFOLD_EMBEDDING_BASE_URL") or None,
        transport_max_retries=config.embedding_transport_max_retries,
        retry_initial_seconds=config.retry_initial_seconds,
        retry_multiplier=config.retry_multiplier,
    )


def generation_model_id() -> str:
    return _required_env("FLUXFOLD_GENERATION_MODEL")


def dataset_hash(paths: tuple[str, ...]) -> str:
    from hashlib import sha256

    digest = sha256()
    for value in paths:
        path = Path(value)
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise ValidationError(f"required environment variable is missing: {name}")
    return value
