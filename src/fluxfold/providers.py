"""Model-provider protocols and an OpenAI-compatible implementation."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Protocol, runtime_checkable

import numpy as np
from openai import (
    NOT_GIVEN,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
    RateLimitError,
)

from fluxfold.errors import ErrorClass, ProviderError


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    stage: str
    system_prompt: str
    user_prompt: str
    temperature: float
    seed: int | None = None
    timeout_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class GenerationResponse:
    text: str
    total_tokens: int
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    vectors: tuple[np.ndarray, ...]
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingModelInfo:
    provider: str
    model: str
    revision: str
    dimension: int
    normalization: str = "l2"
    query_mode: str = "plain"
    document_mode: str = "plain"


@runtime_checkable
class GenerationProvider(Protocol):
    """Provider contract for all write-side and benchmark generation calls."""

    @property
    def model_id(self) -> str: ...

    async def generate(self, request: GenerationRequest) -> GenerationResponse: ...

    async def close(self) -> None: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Provider contract for retrieval embeddings."""

    @property
    def model_info(self) -> EmbeddingModelInfo: ...

    async def embed(
        self,
        texts: Sequence[str],
        *,
        input_type: str,
        timeout_seconds: float,
    ) -> EmbeddingResponse: ...

    async def close(self) -> None: ...


class OpenAICompatibleGenerationProvider:
    """Chat Completions adapter for OpenAI-compatible endpoints."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        transport_max_retries: int = 5,
        retry_initial_seconds: float = 1.0,
        retry_multiplier: float = 2.0,
    ) -> None:
        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
        )
        self._transport_max_retries = transport_max_retries
        self._retry_initial_seconds = retry_initial_seconds
        self._retry_multiplier = retry_multiplier

    @property
    def model_id(self) -> str:
        return self._model

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        for retry_index in range(self._transport_max_retries + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": request.system_prompt},
                        {"role": "user", "content": request.user_prompt},
                    ],
                    temperature=request.temperature,
                    seed=request.seed,
                    timeout=(
                        request.timeout_seconds
                        if request.timeout_seconds is not None
                        else NOT_GIVEN
                    ),
                )
                choice = response.choices[0]
                if choice.finish_reason == "length":
                    raise ProviderError(
                        ErrorClass.INCOMPLETE_OUTPUT,
                        "provider stopped because the output limit was reached",
                        request_id=response.id,
                    )
                if choice.finish_reason == "content_filter":
                    raise ProviderError(
                        ErrorClass.POLICY_REJECTED,
                        "provider content policy rejected the completion",
                        request_id=response.id,
                    )
                content = choice.message.content
                if not content:
                    raise ProviderError(
                        ErrorClass.INCOMPLETE_OUTPUT,
                        "provider returned an empty completion",
                        request_id=response.id,
                    )
                total_tokens = response.usage.total_tokens if response.usage else 0
                return GenerationResponse(
                    text=content,
                    total_tokens=total_tokens,
                    request_id=response.id,
                )
            except ProviderError:
                raise
            except Exception as error:
                normalized = _normalize_openai_error(error)
                if normalized.error_class not in _TRANSIENT_ERRORS:
                    raise normalized from error
                if retry_index >= self._transport_max_retries:
                    raise normalized from error
                await _wait_before_retry(
                    normalized,
                    retry_index,
                    self._retry_initial_seconds,
                    self._retry_multiplier,
                )
        raise AssertionError("unreachable provider retry state")

    async def close(self) -> None:
        await self._client.close()


class OpenAICompatibleEmbeddingProvider:
    """Embeddings adapter for OpenAI-compatible endpoints."""

    def __init__(
        self,
        *,
        model_info: EmbeddingModelInfo,
        api_key: str,
        base_url: str | None = None,
        transport_max_retries: int = 5,
        retry_initial_seconds: float = 1.0,
        retry_multiplier: float = 2.0,
    ) -> None:
        self._model_info = model_info
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
        )
        self._transport_max_retries = transport_max_retries
        self._retry_initial_seconds = retry_initial_seconds
        self._retry_multiplier = retry_multiplier

    @property
    def model_info(self) -> EmbeddingModelInfo:
        return self._model_info

    async def embed(
        self,
        texts: Sequence[str],
        *,
        input_type: str,
        timeout_seconds: float,
    ) -> EmbeddingResponse:
        if input_type not in {"query", "document"}:
            raise ValueError(f"unknown embedding input_type: {input_type}")
        if not texts:
            return EmbeddingResponse(vectors=())
        encoded = [_encode_text(text, input_type, self._model_info) for text in texts]
        for retry_index in range(self._transport_max_retries + 1):
            try:
                response = await self._client.embeddings.create(
                    model=self._model_info.model,
                    input=encoded,
                    encoding_format="float",
                    timeout=timeout_seconds,
                )
                ordered = sorted(response.data, key=lambda item: item.index)
                vectors = tuple(
                    _normalize_vector(item.embedding, self._model_info.dimension)
                    for item in ordered
                )
                if len(vectors) != len(texts):
                    raise ProviderError(
                        ErrorClass.INCOMPLETE_OUTPUT,
                        "embedding response count does not match request count",
                    )
                return EmbeddingResponse(vectors=vectors)
            except ProviderError:
                raise
            except Exception as error:
                normalized = _normalize_openai_error(error)
                if normalized.error_class not in _TRANSIENT_ERRORS:
                    raise normalized from error
                if retry_index >= self._transport_max_retries:
                    raise normalized from error
                await _wait_before_retry(
                    normalized,
                    retry_index,
                    self._retry_initial_seconds,
                    self._retry_multiplier,
                )
        raise AssertionError("unreachable provider retry state")

    async def close(self) -> None:
        await self._client.close()


_TRANSIENT_ERRORS = {
    ErrorClass.TRANSIENT_TRANSPORT,
    ErrorClass.RATE_LIMITED,
    ErrorClass.SERVICE_UNAVAILABLE,
}


def _encode_text(text: str, input_type: str, info: EmbeddingModelInfo) -> str:
    mode = info.query_mode if input_type == "query" else info.document_mode
    if mode == "plain":
        return text
    if "{text}" not in mode:
        raise ValueError("embedding encoding mode must be 'plain' or contain {text}")
    return mode.format(text=text)


def _normalize_vector(raw: Sequence[float], expected_dimension: int) -> np.ndarray:
    vector = np.ascontiguousarray(raw, dtype="<f4")
    if vector.ndim != 1 or vector.shape[0] != expected_dimension:
        raise ProviderError(
            ErrorClass.INVALID_REQUEST,
            f"embedding dimension is {vector.shape}, expected ({expected_dimension},)",
        )
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ProviderError(
            ErrorClass.INVALID_REQUEST, "embedding vector has zero norm"
        )
    return np.ascontiguousarray(vector / norm, dtype="<f4")


async def _wait_before_retry(
    error: ProviderError,
    retry_index: int,
    initial_seconds: float,
    multiplier: float,
) -> None:
    if error.retry_after_seconds is not None:
        delay = error.retry_after_seconds
    else:
        delay = random.uniform(0, initial_seconds * multiplier**retry_index)
    await asyncio.sleep(delay)


def _normalize_openai_error(error: Exception) -> ProviderError:
    if isinstance(error, (APIConnectionError, APITimeoutError, TimeoutError)):
        return ProviderError(ErrorClass.TRANSIENT_TRANSPORT, str(error))
    if isinstance(error, RateLimitError):
        retry_after = _retry_after(error)
        body = getattr(error, "body", None)
        body_text = str(body).lower()
        error_class = (
            ErrorClass.QUOTA_EXHAUSTED
            if any(
                marker in body_text
                for marker in ("insufficient_quota", "billing", "hard limit")
            )
            else ErrorClass.RATE_LIMITED
        )
        return ProviderError(
            error_class,
            str(error),
            retry_after_seconds=retry_after,
            request_id=_request_id(error),
            http_status=getattr(error, "status_code", None),
        )
    if isinstance(error, (AuthenticationError, PermissionDeniedError)):
        return ProviderError(
            ErrorClass.AUTHENTICATION_OR_CONFIGURATION,
            str(error),
            request_id=_request_id(error),
            http_status=getattr(error, "status_code", None),
        )
    if isinstance(error, BadRequestError):
        lower = str(getattr(error, "body", error)).lower()
        if any(marker in lower for marker in ("context", "maximum", "too many tokens")):
            error_class = ErrorClass.CONTEXT_OVERFLOW
        elif any(marker in lower for marker in ("safety", "policy", "content filter")):
            error_class = ErrorClass.POLICY_REJECTED
        else:
            error_class = ErrorClass.INVALID_REQUEST
        return ProviderError(
            error_class,
            str(error),
            request_id=_request_id(error),
            http_status=getattr(error, "status_code", None),
        )
    if isinstance(error, APIStatusError) and error.status_code >= 500:
        return ProviderError(
            ErrorClass.SERVICE_UNAVAILABLE,
            str(error),
            request_id=_request_id(error),
            http_status=error.status_code,
        )
    if isinstance(error, APIStatusError):
        return ProviderError(
            ErrorClass.INVALID_REQUEST,
            str(error),
            request_id=_request_id(error),
            http_status=error.status_code,
        )
    return ProviderError(
        ErrorClass.INVALID_REQUEST, f"unrecognized provider error: {error}"
    )


def _request_id(error: Exception) -> str | None:
    return getattr(error, "request_id", None)


def _retry_after(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max(0.0, (retry_at - datetime.now(tz=UTC)).total_seconds())
