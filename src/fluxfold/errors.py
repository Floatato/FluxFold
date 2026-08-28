"""Public and internal FluxFold errors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FluxFoldError(Exception):
    """Base class for FluxFold failures."""


class ValidationError(FluxFoldError):
    """A library boundary or provider output failed deterministic validation."""


class SourceConflictError(FluxFoldError):
    """A source identity was replayed with a different canonical payload."""


class NotFoundError(FluxFoldError):
    """A requested memory-space object does not exist."""


class ConcurrentUpdateError(FluxFoldError):
    """A state change was based on a stale subject revision."""


class ErrorClass(StrEnum):
    """Normalized model-provider failure categories."""

    TRANSIENT_TRANSPORT = "transient_transport"
    RATE_LIMITED = "rate_limited"
    SERVICE_UNAVAILABLE = "service_unavailable"
    AUTHENTICATION_OR_CONFIGURATION = "authentication_or_configuration"
    QUOTA_EXHAUSTED = "quota_exhausted"
    INVALID_REQUEST = "invalid_request"
    CONTEXT_OVERFLOW = "context_overflow"
    POLICY_REJECTED = "policy_rejected"
    INVALID_STRUCTURED_OUTPUT = "invalid_structured_output"
    INCOMPLETE_OUTPUT = "incomplete_output"


@dataclass(slots=True)
class ProviderError(FluxFoldError):
    """A provider failure normalized for FluxFold retry policy."""

    error_class: ErrorClass
    message: str
    retry_after_seconds: float | None = None
    request_id: str | None = None
    http_status: int | None = None

    def __str__(self) -> str:
        return f"{self.error_class}: {self.message}"


@dataclass(slots=True)
class StageFailure(FluxFoldError):
    """A terminal failure of one pipeline stage."""

    stage: str
    error_class: ErrorClass
    message: str

    def __str__(self) -> str:
        return f"{self.stage}: {self.error_class}: {self.message}"
