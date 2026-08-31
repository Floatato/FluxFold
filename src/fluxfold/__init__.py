"""FluxFold's experimental, benchmark-oriented memory engine."""

from fluxfold.config import FluxFoldConfig
from fluxfold.engine import FluxFold
from fluxfold.errors import (
    ConcurrentUpdateError,
    FluxFoldError,
    NotFoundError,
    ProviderError,
    SourceConflictError,
    StageFailure,
    ValidationError,
)
from fluxfold.models import (
    AddResult,
    EmbeddingRebuildResult,
    EpisodeBlock,
    MemorySpace,
    NormalizedEpisode,
    Role,
    SearchResult,
)
from fluxfold.providers import (
    EmbeddingModelInfo,
    EmbeddingProvider,
    EmbeddingResponse,
    GenerationProvider,
    GenerationRequest,
    GenerationResponse,
    LocalMiniLMEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleGenerationProvider,
)

__version__ = "0.1.0"

__all__ = [
    "AddResult",
    "ConcurrentUpdateError",
    "EmbeddingModelInfo",
    "EmbeddingProvider",
    "EmbeddingRebuildResult",
    "EmbeddingResponse",
    "EpisodeBlock",
    "FluxFold",
    "FluxFoldConfig",
    "FluxFoldError",
    "GenerationProvider",
    "GenerationRequest",
    "GenerationResponse",
    "LocalMiniLMEmbeddingProvider",
    "MemorySpace",
    "NormalizedEpisode",
    "NotFoundError",
    "OpenAICompatibleEmbeddingProvider",
    "OpenAICompatibleGenerationProvider",
    "ProviderError",
    "Role",
    "SearchResult",
    "SourceConflictError",
    "StageFailure",
    "ValidationError",
    "__version__",
]
