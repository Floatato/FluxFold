"""Benchmark runtime configuration and provider construction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fluxfold.config import FluxFoldConfig
from fluxfold.errors import ValidationError
from fluxfold.providers import (
    EmbeddingModelInfo,
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleGenerationProvider,
)

GenerationStage = Literal["build", "answer", "score"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LONGMEMEVAL_PATH = (
    PROJECT_ROOT / "data" / "LongMemEval" / "data" / "longmemeval_s_cleaned.json"
)
DEFAULT_LOCOMO_CONVERSATIONS_PATH = (
    PROJECT_ROOT / "data" / "LoCoMo_refined" / "data" / "public" / "conversations.jsonl"
)
DEFAULT_LOCOMO_QUESTIONS_PATH = (
    PROJECT_ROOT / "data" / "LoCoMo_refined" / "data" / "public" / "questions.jsonl"
)


def load_project_env() -> None:
    load_env_file(PROJECT_ROOT / ".env")


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if not separator or not key.strip():
            raise ValidationError(f"invalid environment line in {path}: {raw_line}")
        name = key.strip()
        if name in os.environ:
            continue
        os.environ[name] = _decode_env_value(value)


def load_config(path: str | None) -> FluxFoldConfig:
    return FluxFoldConfig.from_toml(path) if path else FluxFoldConfig()


def generation_provider(
    config: FluxFoldConfig, *, stage: GenerationStage
) -> OpenAICompatibleGenerationProvider:
    prefix = f"FLUXFOLD_{stage.upper()}"
    return OpenAICompatibleGenerationProvider(
        model=_required_env(f"{prefix}_MODEL"),
        api_key=os.environ.get(f"{prefix}_API_KEY", "EMPTY"),
        base_url=os.environ.get(f"{prefix}_BASE_URL") or None,
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


def _decode_env_value(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        body = stripped[1:-1]
        if stripped[0] == '"':
            return (
                body.replace(r"\\", "\0")
                .replace(r"\n", "\n")
                .replace(r"\t", "\t")
                .replace(r"\"", '"')
                .replace("\0", "\\")
            )
        return body
    return stripped.replace(r"\n", "\n")


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise ValidationError(f"required environment variable is missing: {name}")
    return value
