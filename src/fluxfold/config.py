"""Central experimental-version policy and runtime configuration."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, fields, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

from fluxfold.errors import ValidationError


@dataclass(frozen=True, slots=True)
class FluxFoldConfig:
    """All configurable experimental Memory Engine policy values."""

    domain_name_max_chars: int = 120
    generated_domain_summary_max_chars: int = 2_000
    memory_content_max_chars: int = 1_000
    review_memory_content_max_chars: int = 800
    episode_message_max: int = 256
    episode_chars_max: int = 96_000
    message_chars_max: int = 32_000
    longmemeval_message_chars_max: int = 80_000
    entity_domain_candidate_top_k: int = 3
    name_resolution_min_similarity: float = 0.8
    name_resolution_bm25_threshold: float = 2.0
    extraction_memory_warning_threshold: int = 20
    linking_candidate_warning_threshold: int = 30
    domain_candidate_min_similarity: float = 0.0
    memory_link_preferred_min: int = 1
    memory_link_preferred_max: int = 4
    memory_active_domain_link_max: int = 5
    domain_review_new_memory_threshold: int = 8
    review_provenance_memory_max: int = 8
    memory_provenance_episode_max: int = 6
    domain_split_memory_count_threshold: int = 24
    domain_split_total_memory_chars_threshold: int = 8_000
    domain_split_result_domain_min: int = 2
    domain_split_result_domain_max: int = 5
    domain_split_result_min_memories: int = 3
    domain_split_result_target_memory_max: int = 20
    domain_split_memory_membership_max: int = 2
    search_domain_top_k: int = 5
    search_domain_min_similarity: float = 0.0
    search_domain_attached_memory_k: int = 1
    search_memory_top_k: int = 15
    search_memory_min_similarity: float = 0.0
    search_memory_attached_domain_k: int = 1
    extraction_temperature: float = 0.1
    linking_temperature: float = 0.0
    review_temperature: float = 0.1
    split_temperature: float = 0.1
    structured_output_max_retries: int = 5
    provider_max_retries: int = 5
    provider_connect_timeout_seconds: float = 30.0
    provider_read_timeout_seconds: float = 600.0
    provider_write_timeout_seconds: float = 600.0
    provider_pool_timeout_seconds: float = 30.0
    embedding_batch_size: int = 100
    embedding_batch_concurrency_per_operation: int = 4
    domain_summary_refresh_llm_link_threshold: int = 2
    domain_summary_refresh_concurrency_per_episode: int = 5
    exact_scan_batch_rows: int = 8_192
    sqlite_busy_timeout_ms: int = 5_000
    sqlite_transaction_max_retries: int = 5
    sqlite_transaction_retry_initial_seconds: float = 0.01
    sqlite_transaction_retry_multiplier: float = 2.0
    sqlite_wal_autocheckpoint_pages: int = 1_000
    benchmark_memory_space_build_concurrency: int = 10
    benchmark_search_concurrency: int = 5
    benchmark_seed: int = 42
    benchmark_checkpoint_interval_items: int = 1

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not type(field.default):
                raise ValidationError(
                    f"configuration value has invalid type: {field.name}"
                )
        positive_names = {
            field.name
            for field in fields(self)
            if field.name.endswith(
                (
                    "_max",
                    "_k",
                    "_rows",
                    "_size",
                    "_ms",
                    "_seconds",
                    "_threshold",
                    "_retries",
                )
            )
        }
        positive_names.update(
            {
                "embedding_batch_concurrency_per_operation",
                "domain_summary_refresh_concurrency_per_episode",
                "benchmark_memory_space_build_concurrency",
            }
        )
        for name in positive_names:
            value = getattr(self, name)
            if isinstance(value, (int, float)) and value <= 0:
                raise ValidationError(f"configuration value must be positive: {name}")
        for name in (
            "name_resolution_min_similarity",
            "domain_candidate_min_similarity",
            "search_domain_min_similarity",
            "search_memory_min_similarity",
        ):
            value = getattr(self, name)
            if not -1.0 <= value <= 1.0:
                raise ValidationError(f"cosine threshold must be in [-1, 1]: {name}")
        if self.memory_link_preferred_max > self.memory_active_domain_link_max:
            raise ValidationError("preferred link maximum exceeds the hard maximum")
        if (
            self.domain_split_result_min_memories
            > self.domain_split_result_target_memory_max
        ):
            raise ValidationError(
                "split result minimum exceeds the target memory maximum"
            )

    def with_overrides(self, **overrides: Any) -> FluxFoldConfig:
        """Return a validated copy containing explicit overrides."""

        unknown = set(overrides) - {field.name for field in fields(self)}
        if unknown:
            raise ValidationError(f"unknown configuration values: {sorted(unknown)}")
        return replace(self, **overrides)

    @property
    def signature(self) -> str:
        """Return the deterministic signature used by benchmark manifests."""

        payload = {field.name: getattr(self, field.name) for field in fields(self)}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()

    @classmethod
    def from_toml(cls, path: str | Path) -> FluxFoldConfig:
        """Load `[fluxfold]` overrides from a TOML file."""

        with Path(path).open("rb") as handle:
            document = tomllib.load(handle)
        section = document.get("fluxfold", {})
        if not isinstance(section, dict):
            raise ValidationError("[fluxfold] must be a TOML table")
        return cls().with_overrides(**section)
