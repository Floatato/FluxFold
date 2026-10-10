"""SQLite persistence and exact retrieval for the experimental engine."""

from __future__ import annotations

import random
import sqlite3
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np

from fluxfold.config import FluxFoldConfig
from fluxfold.errors import (
    ConcurrentUpdateError,
    NotFoundError,
    SourceConflictError,
    ValidationError,
)
from fluxfold.models import (
    DomainSnapshot,
    EpisodeBlock,
    MemoryBankDomain,
    MemoryBankSpace,
    MemorySnapshot,
    MemorySpace,
    NormalizedEpisode,
    SearchDomain,
    SearchLink,
    SearchMemory,
    SearchResult,
    canonical_json,
    text_hash,
    utc_milliseconds,
)
from fluxfold.names import NameEntry, normalized_name
from fluxfold.providers import EmbeddingModelInfo

SCHEMA_VERSION = "7"


@dataclass(frozen=True, slots=True)
class PersistedEpisode:
    episode_id: str
    replayed: bool
    extraction_completed: bool
    completed_operation_id: str | None
    terminal_error_class: str | None = None
    terminal_error_message: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedMemory:
    memory_id: str
    memory_version_id: str
    content: str
    embedding: np.ndarray


@dataclass(frozen=True, slots=True)
class PreparedDomain:
    domain_id: str
    name: str
    summary: str | None
    name_embedding: np.ndarray
    name_summary_embedding: np.ndarray | None


@dataclass(frozen=True, slots=True)
class PreparedLink:
    memory_id: str
    domain_id: str
    basis: Literal["direct", "contextual"]


@dataclass(frozen=True, slots=True)
class ExistingDomainTouch:
    domain_id: str
    expected_revision: int
    new_memory_increment: int


@dataclass(frozen=True, slots=True)
class PreparedMemoryUpdate:
    memory_id: str
    memory_version_id: str
    content: str
    provenance_episode_ids: tuple[str, ...]
    embedding: np.ndarray


@dataclass(frozen=True, slots=True)
class PreparedSplitDomain:
    domain: PreparedDomain
    links: tuple[PreparedLink, ...]
    expected_revision: int | None = None


@dataclass(frozen=True, slots=True)
class RetrievalMemorySource:
    memory_version_id: str
    content: str


@dataclass(frozen=True, slots=True)
class RetrievalDomainSource:
    domain_id: str
    name: str
    summary: str | None


class Store:
    """A small transaction service over the single SQLite source of truth."""

    def __init__(self, path: str | Path, config: FluxFoldConfig) -> None:
        self.path = Path(path)
        self.config = config
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self, *, management: bool = False) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.config.sqlite_busy_timeout_ms / 1000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.create_function(
            "fluxfold_management_mode", 0, lambda: int(management)
        )
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute(
            f"PRAGMA busy_timeout = {self.config.sqlite_busy_timeout_ms}"
        )
        connection.execute(
            f"PRAGMA wal_autocheckpoint = {self.config.sqlite_wal_autocheckpoint_pages}"
        )
        return connection

    def _initialize(self) -> None:
        schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
        with self._connect(management=True) as connection:
            has_metadata = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_metadata'"
            ).fetchone()
            if has_metadata:
                version = connection.execute(
                    "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
                ).fetchone()
                if version is not None and version["value"] != SCHEMA_VERSION:
                    raise ValidationError(
                        f"database schema {version['value']} is incompatible with {SCHEMA_VERSION}"
                    )
            connection.executescript(schema)
            row = connection.execute(
                "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO schema_metadata(key, value) VALUES ('schema_version', ?)",
                    (SCHEMA_VERSION,),
                )

    @contextmanager
    def _transaction(self, *, management: bool = False) -> Iterator[sqlite3.Connection]:
        connection: sqlite3.Connection | None = None
        for retry_index in range(self.config.sqlite_transaction_max_retries + 1):
            candidate = self._connect(management=management)
            try:
                candidate.execute("BEGIN IMMEDIATE")
                connection = candidate
                break
            except sqlite3.OperationalError as error:
                candidate.close()
                if (
                    "locked" not in str(error).lower()
                    or retry_index >= self.config.sqlite_transaction_max_retries
                ):
                    raise
                delay = random.uniform(
                    0,
                    self.config.sqlite_transaction_retry_initial_seconds
                    * self.config.sqlite_transaction_retry_multiplier**retry_index,
                )
                time.sleep(delay)
        if connection is None:
            raise AssertionError("transaction retry loop ended without a connection")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def create_or_open_space(self, space_key: str) -> MemorySpace:
        if not space_key.strip():
            raise ValidationError("space_key must not be blank")
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT memory_space_id, space_key, created_at FROM memory_spaces WHERE space_key = ?",
                (space_key,),
            ).fetchone()
            if row is None:
                row_values = (str(uuid4()), space_key, utc_milliseconds())
                connection.execute(
                    "INSERT INTO memory_spaces(memory_space_id, space_key, created_at) VALUES (?, ?, ?)",
                    row_values,
                )
                return MemorySpace(*row_values)
            return MemorySpace(
                row["memory_space_id"], row["space_key"], row["created_at"]
            )

    def get_space(self, memory_space_id: str) -> MemorySpace:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT memory_space_id, space_key, created_at FROM memory_spaces WHERE memory_space_id = ?",
                (memory_space_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError(f"memory space not found: {memory_space_id}")
        return MemorySpace(row["memory_space_id"], row["space_key"], row["created_at"])

    def delete_space(self, memory_space_id: str) -> None:
        with self._transaction(management=True) as connection:
            exists = connection.execute(
                "SELECT 1 FROM memory_spaces WHERE memory_space_id = ?",
                (memory_space_id,),
            ).fetchone()
            if exists is None:
                raise NotFoundError(f"memory space not found: {memory_space_id}")
            self._delete_space_rows(connection, memory_space_id)

    def clear_spaces(self) -> None:
        with self._transaction(management=True) as connection:
            ids = [
                row[0]
                for row in connection.execute(
                    "SELECT memory_space_id FROM memory_spaces"
                ).fetchall()
            ]
            for memory_space_id in ids:
                self._delete_space_rows(connection, memory_space_id)

    def _delete_space_rows(
        self, connection: sqlite3.Connection, memory_space_id: str
    ) -> None:
        memory_versions = "SELECT memory_version_id FROM memory_versions WHERE memory_id IN (SELECT memory_id FROM memories WHERE memory_space_id = ?)"
        domains = "SELECT domain_id FROM domains WHERE memory_space_id = ?"
        memories = "SELECT memory_id FROM memories WHERE memory_space_id = ?"
        episodes = "SELECT episode_id FROM episodes WHERE memory_space_id = ?"
        operations = (
            "SELECT operation_id FROM memory_operations WHERE memory_space_id = ?"
        )
        connection.execute(
            f"DELETE FROM memory_operation_effects WHERE operation_id IN ({operations})",
            (memory_space_id,),
        )
        connection.execute(
            f"DELETE FROM episode_summary_refresh_targets WHERE add_operation_id IN ({operations})",
            (memory_space_id,),
        )
        connection.execute(
            f"DELETE FROM memory_embeddings WHERE memory_version_id IN ({memory_versions})",
            (memory_space_id,),
        )
        connection.execute(
            f"DELETE FROM domain_embeddings WHERE domain_id IN ({domains})",
            (memory_space_id,),
        )
        connection.execute(
            f"DELETE FROM memory_version_provenance WHERE memory_version_id IN ({memory_versions})",
            (memory_space_id,),
        )
        connection.execute(
            f"DELETE FROM domain_memory_links WHERE domain_id IN ({domains}) OR memory_id IN ({memories})",
            (memory_space_id, memory_space_id),
        )
        connection.execute(
            f"DELETE FROM episode_extractions WHERE episode_id IN ({episodes})",
            (memory_space_id,),
        )
        connection.execute(
            f"DELETE FROM memory_versions WHERE memory_id IN ({memories})",
            (memory_space_id,),
        )
        connection.execute(
            f"DELETE FROM memory_entities WHERE memory_id IN ({memories})",
            (memory_space_id,),
        )
        connection.execute(
            f"DELETE FROM entity_domains WHERE domain_id IN ({domains})",
            (memory_space_id,),
        )
        connection.execute(
            "DELETE FROM entities WHERE memory_space_id=?", (memory_space_id,)
        )
        connection.execute(
            "DELETE FROM domains WHERE memory_space_id = ?", (memory_space_id,)
        )
        connection.execute(
            "DELETE FROM memories WHERE memory_space_id = ?", (memory_space_id,)
        )
        connection.execute(
            f"DELETE FROM episode_blocks WHERE episode_id IN ({episodes})",
            (memory_space_id,),
        )
        connection.execute(
            "DELETE FROM episodes WHERE memory_space_id = ?",
            (memory_space_id,),
        )
        connection.execute(
            "DELETE FROM memory_space_model_signatures WHERE memory_space_id = ?",
            (memory_space_id,),
        )
        connection.execute(
            "DELETE FROM memory_operations WHERE memory_space_id = ?",
            (memory_space_id,),
        )
        connection.execute(
            "DELETE FROM memory_spaces WHERE memory_space_id = ?", (memory_space_id,)
        )

    def ensure_retrieval_signature(
        self, memory_space_id: str, info: EmbeddingModelInfo
    ) -> str:
        signature_value = canonical_json(
            {
                "dimension": info.dimension,
                "document_mode": info.document_mode,
                "dtype": "float32",
                "model": info.model,
                "normalization": info.normalization,
                "provider": info.provider,
                "query_mode": info.query_mode,
                "revision": info.revision,
            }
        )
        signature_hash = text_hash(signature_value)
        with self._transaction() as connection:
            self._require_space(connection, memory_space_id)
            row = connection.execute(
                "SELECT model_signature_id FROM embedding_model_signatures WHERE signature_hash = ?",
                (signature_hash,),
            ).fetchone()
            if row is None:
                signature_id = str(uuid4())
                connection.execute(
                    """
                    INSERT INTO embedding_model_signatures(
                        model_signature_id, provider, model_id, revision, dimension, dtype,
                        normalization, query_mode, document_mode, signature_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, 'float32', ?, ?, ?, ?, ?)
                    """,
                    (
                        signature_id,
                        info.provider,
                        info.model,
                        info.revision,
                        info.dimension,
                        info.normalization,
                        info.query_mode,
                        info.document_mode,
                        signature_hash,
                        utc_milliseconds(),
                    ),
                )
            else:
                signature_id = row["model_signature_id"]
            active = connection.execute(
                "SELECT model_signature_id FROM memory_space_model_signatures WHERE memory_space_id = ? AND purpose = 'retrieval'",
                (memory_space_id,),
            ).fetchone()
            if active is None:
                connection.execute(
                    "INSERT INTO memory_space_model_signatures(memory_space_id, purpose, model_signature_id, activated_at) VALUES (?, 'retrieval', ?, ?)",
                    (memory_space_id, signature_id, utc_milliseconds()),
                )
            elif active["model_signature_id"] != signature_id:
                self._switch_signature(connection, memory_space_id, signature_id)
            return signature_id

    def prepare_retrieval_signature(self, info: EmbeddingModelInfo) -> str:
        """Register an embedding signature without activating it."""

        signature_value = canonical_json(
            {
                "dimension": info.dimension,
                "document_mode": info.document_mode,
                "dtype": "float32",
                "model": info.model,
                "normalization": info.normalization,
                "provider": info.provider,
                "query_mode": info.query_mode,
                "revision": info.revision,
            }
        )
        signature_hash = text_hash(signature_value)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT model_signature_id FROM embedding_model_signatures WHERE signature_hash = ?",
                (signature_hash,),
            ).fetchone()
            if row is not None:
                return str(row["model_signature_id"])
            signature_id = str(uuid4())
            connection.execute(
                """
                INSERT INTO embedding_model_signatures(
                    model_signature_id, provider, model_id, revision, dimension, dtype,
                    normalization, query_mode, document_mode, signature_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, 'float32', ?, ?, ?, ?, ?)
                """,
                (
                    signature_id,
                    info.provider,
                    info.model,
                    info.revision,
                    info.dimension,
                    info.normalization,
                    info.query_mode,
                    info.document_mode,
                    signature_hash,
                    utc_milliseconds(),
                ),
            )
            return signature_id

    def retrieval_embedding_sources(
        self, memory_space_id: str
    ) -> tuple[tuple[RetrievalMemorySource, ...], tuple[RetrievalDomainSource, ...]]:
        with self._connect() as connection:
            self._require_space(connection, memory_space_id)
            memories = tuple(
                RetrievalMemorySource(row["memory_version_id"], row["content"])
                for row in connection.execute(
                    """
                    SELECT v.memory_version_id, v.content FROM memories u
                    JOIN memory_versions v ON v.memory_id = u.memory_id AND v.is_latest = 1
                    WHERE u.memory_space_id = ? AND u.lifecycle_status = 'active'
                    ORDER BY u.memory_id
                    """,
                    (memory_space_id,),
                )
            )
            domains = tuple(
                RetrievalDomainSource(row["domain_id"], row["name"], row["summary"])
                for row in connection.execute(
                    """
                    SELECT domain_id, name, summary FROM domains
                    WHERE memory_space_id = ?
                    ORDER BY domain_id
                    """,
                    (memory_space_id,),
                )
            )
        return memories, domains

    def commit_retrieval_embedding_switch(
        self,
        *,
        memory_space_id: str,
        signature_id: str,
        memories: Sequence[tuple[RetrievalMemorySource, np.ndarray]],
        domains: Sequence[tuple[RetrievalDomainSource, np.ndarray, np.ndarray | None]],
        entities: Sequence[NameEntry],
    ) -> None:
        now = utc_milliseconds()
        with self._transaction() as connection:
            self._require_space(connection, memory_space_id)
            current_memories = {
                row["memory_version_id"]: row["content"]
                for row in connection.execute(
                    """
                    SELECT v.memory_version_id, v.content FROM memories u
                    JOIN memory_versions v ON v.memory_id = u.memory_id AND v.is_latest = 1
                    WHERE u.memory_space_id = ? AND u.lifecycle_status = 'active'
                    """,
                    (memory_space_id,),
                )
            }
            current_domains = {
                row["domain_id"]: (row["name"], row["summary"])
                for row in connection.execute(
                    "SELECT domain_id, name, summary FROM domains WHERE memory_space_id = ?",
                    (memory_space_id,),
                )
            }
            supplied_memories = {
                source.memory_version_id: source.content for source, _ in memories
            }
            supplied_domains = {
                source.domain_id: (source.name, source.summary)
                for source, _, _ in domains
            }
            if (
                supplied_memories != current_memories
                or supplied_domains != current_domains
            ):
                raise ConcurrentUpdateError(
                    "retrieval sources changed while embeddings were being rebuilt"
                )
            current_entities = {
                row["entity_id"]: row["name"]
                for row in connection.execute(
                    "SELECT entity_id, name FROM entities WHERE memory_space_id=?",
                    (memory_space_id,),
                )
            }
            if current_entities != {
                entity.object_id: entity.name for entity in entities
            }:
                raise ConcurrentUpdateError(
                    "entities changed while embeddings were being rebuilt"
                )
            for entity in entities:
                connection.execute(
                    "UPDATE entities SET vector=?, model_signature_id=? WHERE entity_id=?",
                    (_vector_blob(entity.vector), signature_id, entity.object_id),
                )
            for memory_source, vector in memories:
                connection.execute(
                    """
                    INSERT INTO memory_embeddings(
                        memory_version_id, model_signature_id, dimension, source_hash,
                        vector, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(memory_version_id, model_signature_id) DO UPDATE SET
                        dimension = excluded.dimension, source_hash = excluded.source_hash,
                        vector = excluded.vector, created_at = excluded.created_at
                    """,
                    (
                        memory_source.memory_version_id,
                        signature_id,
                        vector.shape[0],
                        text_hash(memory_source.content),
                        _vector_blob(vector),
                        now,
                    ),
                )
            for domain_source, name_vector, summary_vector in domains:
                self._upsert_domain_embedding(
                    connection,
                    domain_source.domain_id,
                    "name",
                    domain_source.name,
                    name_vector,
                    signature_id,
                    now,
                )
                if domain_source.summary is not None:
                    if summary_vector is None:
                        raise ValidationError("summary embedding is missing")
                    self._upsert_domain_embedding(
                        connection,
                        domain_source.domain_id,
                        "name_summary",
                        _domain_embedding_text(
                            domain_source.name, domain_source.summary
                        ),
                        summary_vector,
                        signature_id,
                        now,
                    )
            active = connection.execute(
                "SELECT 1 FROM memory_space_model_signatures WHERE memory_space_id = ? AND purpose = 'retrieval'",
                (memory_space_id,),
            ).fetchone()
            if active is None:
                connection.execute(
                    "INSERT INTO memory_space_model_signatures(memory_space_id, purpose, model_signature_id, activated_at) VALUES (?, 'retrieval', ?, ?)",
                    (memory_space_id, signature_id, now),
                )
            else:
                connection.execute(
                    "UPDATE memory_space_model_signatures SET model_signature_id = ?, activated_at = ? WHERE memory_space_id = ? AND purpose = 'retrieval'",
                    (signature_id, now, memory_space_id),
                )

    def _switch_signature(
        self, connection: sqlite3.Connection, memory_space_id: str, signature_id: str
    ) -> None:
        active_memories = connection.execute(
            """
            SELECT count(*) AS n FROM memories u
            JOIN memory_versions v ON v.memory_id = u.memory_id AND v.is_latest = 1
            LEFT JOIN memory_embeddings e ON e.memory_version_id = v.memory_version_id AND e.model_signature_id = ?
            WHERE u.memory_space_id = ? AND u.lifecycle_status = 'active' AND e.memory_version_id IS NULL
            """,
            (signature_id, memory_space_id),
        ).fetchone()["n"]
        active_domain_embeddings = connection.execute(
            """
            SELECT count(*) AS n FROM domains s
            CROSS JOIN (SELECT 'name' AS kind UNION ALL SELECT 'name_summary') kinds
            LEFT JOIN domain_embeddings e ON e.domain_id = s.domain_id
                AND e.embedding_kind = kinds.kind AND e.model_signature_id = ?
            WHERE s.memory_space_id = ?
                AND (kinds.kind = 'name' OR s.summary IS NOT NULL)
                AND e.domain_id IS NULL
            """,
            (signature_id, memory_space_id),
        ).fetchone()["n"]
        if active_memories or active_domain_embeddings:
            raise ValidationError(
                "cannot switch retrieval signature before all embeddings exist"
            )
        connection.execute(
            "UPDATE memory_space_model_signatures SET model_signature_id = ?, activated_at = ? WHERE memory_space_id = ? AND purpose = 'retrieval'",
            (signature_id, utc_milliseconds(), memory_space_id),
        )

    def persist_episode(
        self, memory_space_id: str, episode: NormalizedEpisode
    ) -> PersistedEpisode:
        with self._transaction() as connection:
            self._require_space(connection, memory_space_id)
            existing = connection.execute(
                """
                SELECT episode_id, content_hash FROM episodes
                WHERE memory_space_id = ? AND source_type = ? AND source_key = ?
                """,
                (memory_space_id, episode.source_type, episode.source_key),
            ).fetchone()
            if existing is not None:
                if existing["content_hash"] != episode.content_hash:
                    raise SourceConflictError(
                        f"source identity changed payload: {episode.source_type}/{episode.source_key}"
                    )
                extraction = connection.execute(
                    "SELECT status, completed_operation_id, error_class, error_message FROM episode_extractions WHERE episode_id = ?",
                    (existing["episode_id"],),
                ).fetchone()
                return PersistedEpisode(
                    existing["episode_id"],
                    True,
                    extraction is not None and extraction["status"] == "completed",
                    extraction["completed_operation_id"]
                    if extraction is not None
                    else None,
                    extraction["error_class"] if extraction is not None else None,
                    extraction["error_message"] if extraction is not None else None,
                )
            episode_id = str(uuid4())
            created_at = utc_milliseconds()
            connection.execute(
                """
                INSERT INTO episodes(
                    episode_id, memory_space_id, source_type, source_key, source_sequence,
                    payload_version, source_started_at, source_ended_at, source_timezone,
                    content_hash, source_chars, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    memory_space_id,
                    episode.source_type,
                    episode.source_key,
                    episode.source_sequence,
                    episode.payload_version,
                    episode.source_started_at,
                    episode.source_ended_at,
                    episode.source_timezone,
                    episode.content_hash,
                    episode.source_chars,
                    created_at,
                ),
            )
            for block in episode.blocks:
                self._insert_block(connection, episode_id, block)
            return PersistedEpisode(episode_id, False, False, None)

    def record_episode_terminal_failure(
        self,
        *,
        memory_space_id: str,
        episode_id: str,
        input_hash: str,
        config_signature: str,
        error_class: str,
        error_message: str,
    ) -> None:
        with self._transaction() as connection:
            episode = connection.execute(
                "SELECT content_hash FROM episodes WHERE episode_id = ? AND memory_space_id = ?",
                (episode_id, memory_space_id),
            ).fetchone()
            if episode is None:
                raise NotFoundError(f"episode not found: {episode_id}")
            if episode["content_hash"] != input_hash:
                raise SourceConflictError(f"episode input hash changed: {episode_id}")
            existing = connection.execute(
                "SELECT status FROM episode_extractions WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()
            if existing is not None:
                if existing["status"] == "terminal_failure":
                    return
                raise ValidationError("cannot replace a completed episode extraction")
            connection.execute(
                """
                INSERT INTO episode_extractions(
                    episode_id, input_hash, extractor_config_signature, status,
                    completed_operation_id, completed_at, error_class, error_message
                ) VALUES (?, ?, ?, 'terminal_failure', NULL, ?, ?, ?)
                """,
                (
                    episode_id,
                    input_hash,
                    config_signature,
                    utc_milliseconds(),
                    error_class,
                    error_message,
                ),
            )

    def _insert_block(
        self, connection: sqlite3.Connection, episode_id: str, block: EpisodeBlock
    ) -> None:
        connection.execute(
            """
            INSERT INTO episode_blocks(
                block_id, episode_id, sequence_no, role, message_phase, speaker_id,
                speaker_name, content, observed_at, metadata_json, preprocessor_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                block.block_id,
                episode_id,
                block.sequence_no,
                block.role.value,
                block.message_phase,
                block.speaker_id,
                block.speaker_name,
                block.content,
                block.observed_at,
                canonical_json(block.metadata) if block.metadata is not None else None,
                block.preprocessor_version,
            ),
        )

    def commit_add(
        self,
        *,
        memory_space_id: str,
        episode_id: str,
        input_hash: str,
        memories: Sequence[PreparedMemory],
        domains: Sequence[PreparedDomain],
        links: Sequence[PreparedLink],
        domain_touches: Sequence[ExistingDomainTouch],
        entities: Sequence[NameEntry],
        entity_domains: Sequence[tuple[str, str]],
        memory_entities: Sequence[tuple[str, str]],
        signature_id: str,
        actor: str,
        config_signature: str,
    ) -> str:
        operation_id = str(uuid4())
        now = utc_milliseconds()
        with self._transaction() as connection:
            self._require_space(connection, memory_space_id)
            episode = connection.execute(
                "SELECT source_started_at, source_ended_at, content_hash FROM episodes WHERE episode_id = ? AND memory_space_id = ?",
                (episode_id, memory_space_id),
            ).fetchone()
            if episode is None:
                raise NotFoundError(f"episode not found: {episode_id}")
            existing_completion = connection.execute(
                "SELECT completed_operation_id FROM episode_extractions WHERE episode_id = ? AND status = 'completed'",
                (episode_id,),
            ).fetchone()
            if existing_completion is not None:
                return str(existing_completion["completed_operation_id"])
            self._insert_operation(
                connection,
                operation_id,
                memory_space_id,
                "add_episode_memories",
                actor,
                config_signature,
                "Add memories extracted from one dataset episode.",
                now,
            )
            source_time = episode["source_ended_at"] or episode["source_started_at"]
            for memory in memories:
                self._insert_memory(
                    connection,
                    memory_space_id,
                    memory,
                    (episode_id,),
                    source_time,
                    operation_id,
                    signature_id,
                    now,
                )
            for domain in domains:
                self._insert_domain(
                    connection,
                    memory_space_id,
                    domain,
                    operation_id,
                    signature_id,
                    now,
                )
            for entity in entities:
                connection.execute(
                    "INSERT INTO entities(entity_id, memory_space_id, name, normalized_name, model_signature_id, vector, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        entity.object_id,
                        memory_space_id,
                        entity.name,
                        normalized_name(entity.name),
                        signature_id,
                        _vector_blob(entity.vector),
                        now,
                    ),
                )
            connection.executemany(
                "INSERT OR IGNORE INTO entity_domains(entity_id, domain_id) VALUES (?, ?)",
                entity_domains,
            )
            connection.executemany(
                "INSERT INTO memory_entities(memory_id, entity_id) VALUES (?, ?)",
                memory_entities,
            )
            for touch in domain_touches:
                self._reactivate_domain(connection, touch.domain_id, operation_id, now)
                cursor = connection.execute(
                    """
                    UPDATE domains SET new_memory_count = new_memory_count + ?,
                        updated_at = ?
                    WHERE domain_id = ? AND memory_space_id = ? AND lifecycle_status = 'active'
                        AND summary_revision = ?
                    """,
                    (
                        touch.new_memory_increment,
                        now,
                        touch.domain_id,
                        memory_space_id,
                        touch.expected_revision,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ConcurrentUpdateError(f"stale domain: {touch.domain_id}")
            for link in links:
                self._insert_link(connection, link, operation_id, now)
            self._insert_summary_refresh_targets(
                connection,
                operation_id,
                {link.domain_id for link in links}
                | {domain.domain_id for domain in domains},
            )
            self._validate_active_memory_links(connection, memory_space_id)
            connection.execute(
                """
                INSERT INTO episode_extractions(
                    episode_id, input_hash, extractor_config_signature, status,
                    completed_operation_id, completed_at
                ) VALUES (?, ?, ?, 'completed', ?, ?)
                """,
                (episode_id, input_hash, config_signature, operation_id, now),
            )
            self._effect(connection, operation_id, "episode", episode_id, "completed")
        return operation_id

    def _insert_memory(
        self,
        connection: sqlite3.Connection,
        memory_space_id: str,
        memory: PreparedMemory,
        provenance: Sequence[str],
        latest_source_at: int | None,
        operation_id: str,
        signature_id: str,
        now: int,
    ) -> None:
        connection.execute(
            "INSERT INTO memories(memory_id, memory_space_id, lifecycle_status, latest_source_at, created_at, updated_at) VALUES (?, ?, 'active', ?, ?, ?)",
            (memory.memory_id, memory_space_id, latest_source_at, now, now),
        )
        connection.execute(
            """
            INSERT INTO memory_versions(
                memory_version_id, memory_id, version_no, is_latest, content,
                content_hash, char_count, created_by_operation_id, created_at
            ) VALUES (?, ?, 1, 1, ?, ?, ?, ?, ?)
            """,
            (
                memory.memory_version_id,
                memory.memory_id,
                memory.content,
                text_hash(memory.content),
                len(memory.content),
                operation_id,
                now,
            ),
        )
        for episode_id in provenance:
            connection.execute(
                "INSERT INTO memory_version_provenance(memory_version_id, episode_id) VALUES (?, ?)",
                (memory.memory_version_id, episode_id),
            )
        self._insert_memory_embedding(connection, memory, signature_id, now)
        self._effect(connection, operation_id, "memory", memory.memory_id, "created")
        self._effect(
            connection,
            operation_id,
            "memory_version",
            memory.memory_version_id,
            "created",
        )

    def _reactivate_domain(
        self,
        connection: sqlite3.Connection,
        domain_id: str,
        operation_id: str,
        now: int,
    ) -> None:
        cursor = connection.execute(
            """UPDATE domains SET lifecycle_status='active', retired_at=NULL,
            retired_by_operation_id=NULL, updated_at=?
            WHERE domain_id=? AND lifecycle_status='retired'""",
            (now, domain_id),
        )
        if cursor.rowcount:
            self._effect(connection, operation_id, "domain", domain_id, "reactivated")

    def _insert_domain(
        self,
        connection: sqlite3.Connection,
        memory_space_id: str,
        domain: PreparedDomain,
        operation_id: str,
        signature_id: str,
        now: int,
    ) -> None:
        connection.execute(
            """
            INSERT INTO domains(
                domain_id, memory_space_id, name, normalized_name, summary, lifecycle_status,
                new_memory_count, summary_revision, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 'active', 0, 0, ?, ?)
            """,
            (
                domain.domain_id,
                memory_space_id,
                domain.name,
                normalized_name(domain.name),
                domain.summary,
                now,
                now,
            ),
        )
        self._upsert_domain_embedding(
            connection,
            domain.domain_id,
            "name",
            domain.name,
            domain.name_embedding,
            signature_id,
            now,
        )
        if domain.summary is not None:
            if domain.name_summary_embedding is None:
                raise ValidationError("summary embedding is missing")
            self._upsert_domain_embedding(
                connection,
                domain.domain_id,
                "name_summary",
                _domain_embedding_text(domain.name, domain.summary),
                domain.name_summary_embedding,
                signature_id,
                now,
            )
        self._effect(connection, operation_id, "domain", domain.domain_id, "created")

    def _insert_link(
        self,
        connection: sqlite3.Connection,
        link: PreparedLink,
        operation_id: str,
        now: int,
    ) -> str:
        link_id = str(uuid4())
        connection.execute(
            """
            INSERT INTO domain_memory_links(
                link_id, domain_id, memory_id, link_basis, linked_at, opened_by_operation_id
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (link_id, link.domain_id, link.memory_id, link.basis, now, operation_id),
        )
        self._effect(connection, operation_id, "link", link_id, "opened")
        return link_id

    def _insert_memory_embedding(
        self,
        connection: sqlite3.Connection,
        memory: PreparedMemory | PreparedMemoryUpdate,
        signature_id: str,
        now: int,
    ) -> None:
        connection.execute(
            """
            INSERT INTO memory_embeddings(
                memory_version_id, model_signature_id, dimension, source_hash, vector, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                memory.memory_version_id,
                signature_id,
                memory.embedding.shape[0],
                text_hash(memory.content),
                _vector_blob(memory.embedding),
                now,
            ),
        )

    def _upsert_domain_embedding(
        self,
        connection: sqlite3.Connection,
        domain_id: str,
        kind: str,
        source_text: str,
        vector: np.ndarray,
        signature_id: str,
        now: int,
    ) -> None:
        connection.execute(
            """
            INSERT INTO domain_embeddings(
                domain_id, embedding_kind, model_signature_id, dimension,
                source_hash, vector, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(domain_id, embedding_kind, model_signature_id) DO UPDATE SET
                dimension = excluded.dimension, source_hash = excluded.source_hash,
                vector = excluded.vector, created_at = excluded.created_at
            """,
            (
                domain_id,
                kind,
                signature_id,
                vector.shape[0],
                text_hash(source_text),
                _vector_blob(vector),
                now,
            ),
        )

    def entity_names(self, memory_space_id: str) -> tuple[NameEntry, ...]:
        with self._connect() as connection:
            return tuple(
                NameEntry(
                    row["entity_id"],
                    row["name"],
                    np.frombuffer(row["vector"], dtype="<f4"),
                )
                for row in connection.execute(
                    "SELECT entity_id, name, vector FROM entities WHERE memory_space_id = ? ORDER BY normalized_name",
                    (memory_space_id,),
                )
            )

    def domain_names(self, memory_space_id: str) -> tuple[NameEntry, ...]:
        with self._connect() as connection:
            return tuple(
                NameEntry(
                    row["domain_id"],
                    row["name"],
                    np.frombuffer(row["vector"], dtype="<f4"),
                )
                for row in connection.execute(
                    """SELECT s.domain_id, s.name, e.vector FROM domains s
                JOIN memory_space_model_signatures a ON a.memory_space_id=s.memory_space_id AND a.purpose='retrieval'
                JOIN domain_embeddings e ON e.domain_id=s.domain_id AND e.embedding_kind='name' AND e.model_signature_id=a.model_signature_id
                WHERE s.memory_space_id=? ORDER BY s.normalized_name""",
                    (memory_space_id,),
                )
            )

    def domain_entity_names(self, memory_space_id: str) -> dict[str, set[str]]:
        with self._connect() as connection:
            result: dict[str, set[str]] = {}
            for row in connection.execute(
                """SELECT r.domain_id, a.name FROM entity_domains r
                JOIN entities a USING(entity_id) JOIN domains s USING(domain_id)
                WHERE a.memory_space_id=?""",
                (memory_space_id,),
            ):
                result.setdefault(row["domain_id"], set()).add(row["name"])
            return result

    def active_link_bases(self, memory_space_id: str) -> dict[tuple[str, str], str]:
        with self._connect() as connection:
            return {
                (row["memory_id"], row["domain_id"]): row["link_basis"]
                for row in connection.execute(
                    """SELECT l.memory_id, l.domain_id, l.link_basis FROM domain_memory_links l
                JOIN domains s USING(domain_id) WHERE s.memory_space_id=? AND l.unlinked_at IS NULL""",
                    (memory_space_id,),
                )
            }

    def _scan_domains(
        self,
        memory_space_id: str,
        query_vector: np.ndarray,
        top_k: int,
        min_similarity: float,
    ) -> list[tuple[str, float]]:
        sql = """
            SELECT s.domain_id AS object_id, e.vector FROM domains s
            JOIN memory_space_model_signatures a ON a.memory_space_id = s.memory_space_id AND a.purpose = 'retrieval'
            JOIN domain_embeddings e ON e.domain_id = s.domain_id
                AND e.embedding_kind = 'name' AND e.model_signature_id = a.model_signature_id
            WHERE s.memory_space_id = ?
            AND s.lifecycle_status = 'active'
            AND EXISTS (
                SELECT 1 FROM domain_memory_links l
                JOIN memories u ON u.memory_id=l.memory_id AND u.lifecycle_status='active'
                WHERE l.domain_id=s.domain_id AND l.unlinked_at IS NULL
            )
            ORDER BY s.domain_id
        """
        return self._exact_scan(
            sql, memory_space_id, query_vector, top_k, min_similarity
        )

    def _scan_memories(
        self,
        memory_space_id: str,
        query_vector: np.ndarray,
        top_k: int,
        min_similarity: float,
    ) -> list[tuple[str, float]]:
        sql = """
            SELECT u.memory_id AS object_id, e.vector FROM memories u
            JOIN memory_versions v ON v.memory_id = u.memory_id AND v.is_latest = 1
            JOIN memory_space_model_signatures a ON a.memory_space_id = u.memory_space_id AND a.purpose = 'retrieval'
            JOIN memory_embeddings e ON e.memory_version_id = v.memory_version_id AND e.model_signature_id = a.model_signature_id
            WHERE u.memory_space_id = ? AND u.lifecycle_status = 'active'
            ORDER BY u.memory_id
        """
        return self._exact_scan(
            sql, memory_space_id, query_vector, top_k, min_similarity
        )

    def _exact_scan(
        self,
        sql: str,
        memory_space_id: str,
        query_vector: np.ndarray,
        top_k: int,
        min_similarity: float,
    ) -> list[tuple[str, float]]:
        query = _normalized_query(query_vector)
        candidates: list[tuple[str, float]] = []
        with self._connect() as connection:
            cursor = connection.execute(sql, (memory_space_id,))
            while rows := cursor.fetchmany(self.config.exact_scan_batch_rows):
                matrix = np.vstack(
                    [np.frombuffer(row["vector"], dtype="<f4") for row in rows]
                )
                if matrix.shape[1] != query.shape[0]:
                    raise ValidationError(
                        "stored embedding dimension does not match query"
                    )
                scores = matrix @ query
                candidates.extend(
                    (row["object_id"], float(score))
                    for row, score in zip(rows, scores, strict=True)
                    if float(score) >= min_similarity
                )
        candidates.sort(key=lambda item: (-item[1], item[0]))
        return candidates[:top_k]

    def domain_snapshot(self, memory_space_id: str, domain_id: str) -> DomainSnapshot:
        with self._connect() as connection:
            domain = connection.execute(
                """
                SELECT name, summary, new_memory_count, summary_revision FROM domains
                WHERE domain_id = ? AND memory_space_id = ?
                """,
                (domain_id, memory_space_id),
            ).fetchone()
            if domain is None:
                raise NotFoundError(f"domain not found: {domain_id}")
            rows = connection.execute(
                """
                SELECT u.memory_id, u.latest_source_at, v.content, l.link_basis
                FROM domain_memory_links l
                JOIN memories u ON u.memory_id = l.memory_id AND u.lifecycle_status = 'active'
                JOIN memory_versions v ON v.memory_id = u.memory_id AND v.is_latest = 1
                WHERE l.domain_id = ? AND l.unlinked_at IS NULL
                ORDER BY u.memory_id
                """,
                (domain_id,),
            ).fetchall()
            memories = tuple(
                MemorySnapshot(
                    row["memory_id"],
                    row["content"],
                    row["latest_source_at"],
                    self._provenance(connection, row["memory_id"]),
                    row["link_basis"],
                )
                for row in rows
            )
            return DomainSnapshot(
                domain_id,
                domain["name"],
                domain["summary"],
                domain["new_memory_count"],
                domain["summary_revision"],
                memories,
            )

    def provenance_episodes(
        self, memory_space_id: str, memory_ids: Sequence[str]
    ) -> dict[str, tuple[dict[str, object], ...]]:
        output: dict[str, tuple[dict[str, object], ...]] = {}
        with self._connect() as connection:
            for memory_id in memory_ids:
                owner = connection.execute(
                    "SELECT 1 FROM memories WHERE memory_id = ? AND memory_space_id = ?",
                    (memory_id, memory_space_id),
                ).fetchone()
                if owner is None:
                    raise NotFoundError(f"memory not found: {memory_id}")
                episodes = connection.execute(
                    """
                    SELECT ep.* FROM memory_versions v
                    JOIN memory_version_provenance p ON p.memory_version_id = v.memory_version_id
                    JOIN episodes ep ON ep.episode_id = p.episode_id
                    WHERE v.memory_id = ? AND v.is_latest = 1
                    ORDER BY ep.source_sequence
                    """,
                    (memory_id,),
                ).fetchall()
                rendered: list[dict[str, object]] = []
                for episode in episodes:
                    blocks = connection.execute(
                        "SELECT * FROM episode_blocks WHERE episode_id = ? ORDER BY sequence_no",
                        (episode["episode_id"],),
                    ).fetchall()
                    rendered.append(
                        {
                            "episode_id": episode["episode_id"],
                            "source_started_at": episode["source_started_at"],
                            "source_ended_at": episode["source_ended_at"],
                            "source_timezone": episode["source_timezone"],
                            "blocks": [
                                {
                                    "speaker_id": block["speaker_id"] or block["role"],
                                    "content": block["content"],
                                    "observed_at": block["observed_at"],
                                }
                                for block in blocks
                            ],
                        }
                    )
                output[memory_id] = tuple(rendered)
        return output

    def existing_episode_ids(
        self, memory_space_id: str, episode_ids: Sequence[str]
    ) -> frozenset[str]:
        unique = tuple(dict.fromkeys(episode_ids))
        if not unique:
            return frozenset()
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT episode_id FROM episodes WHERE memory_space_id = ? AND episode_id IN ({_placeholders(unique)})",
                (memory_space_id, *unique),
            ).fetchall()
        return frozenset(row["episode_id"] for row in rows)

    def commit_review(
        self,
        *,
        add_operation_id: str,
        memory_space_id: str,
        domain_id: str,
        expected_revision: int,
        updates: Sequence[PreparedMemoryUpdate],
        retirements: Sequence[str],
        signature_id: str,
        actor: str,
        config_signature: str,
    ) -> str:
        operation_id = str(uuid4())
        now = utc_milliseconds()
        with self._transaction() as connection:
            domain = connection.execute(
                "SELECT summary_revision FROM domains WHERE domain_id = ? AND memory_space_id = ? AND lifecycle_status = 'active'",
                (domain_id, memory_space_id),
            ).fetchone()
            if domain is None:
                raise NotFoundError(f"active domain not found: {domain_id}")
            if domain["summary_revision"] != expected_revision:
                raise ConcurrentUpdateError(f"stale domain: {domain_id}")
            current_ids = self._domain_memory_ids(connection, domain_id)
            changed_ids = {update.memory_id for update in updates}
            retired_ids = set(retirements)
            if not changed_ids | retired_ids <= current_ids:
                raise ValidationError("review references a memory outside the domain")
            if changed_ids & retired_ids:
                raise ValidationError("review cannot update and retire the same memory")
            self._insert_operation(
                connection,
                operation_id,
                memory_space_id,
                "review_domain",
                actor,
                config_signature,
                "Review all active memories in one domain.",
                now,
            )
            for update in updates:
                self._replace_memory_version(
                    connection, memory_space_id, update, operation_id, signature_id, now
                )
            for memory_id in retirements:
                self._retire_memory(
                    connection, memory_space_id, memory_id, operation_id, now
                )
            cursor = connection.execute(
                """
                UPDATE domains SET new_memory_count = 0, updated_at = ?
                WHERE domain_id = ? AND summary_revision = ?
                """,
                (now, domain_id, expected_revision),
            )
            if cursor.rowcount != 1:
                raise ConcurrentUpdateError(f"stale domain: {domain_id}")
            self._effect(connection, operation_id, "domain", domain_id, "reviewed")
            self._insert_summary_refresh_targets(
                connection, add_operation_id, {domain_id}
            )
            self._validate_active_memory_links(connection, memory_space_id)
        return operation_id

    def commit_summary_refresh(
        self,
        *,
        add_operation_id: str,
        memory_space_id: str,
        snapshot: DomainSnapshot,
        summary: str,
        summary_embedding: np.ndarray,
        signature_id: str,
        actor: str,
        config_signature: str,
    ) -> str:
        operation_id = str(uuid4())
        now = utc_milliseconds()
        with self._transaction() as connection:
            domain = connection.execute(
                """
                SELECT name, summary_revision FROM domains
                WHERE domain_id = ? AND memory_space_id = ?
                    AND lifecycle_status = 'active'
                """,
                (snapshot.domain_id, memory_space_id),
            ).fetchone()
            if domain is None:
                raise NotFoundError(f"active domain not found: {snapshot.domain_id}")
            if domain["summary_revision"] != snapshot.summary_revision:
                raise ConcurrentUpdateError(f"stale domain: {snapshot.domain_id}")
            rows = connection.execute(
                """
                SELECT u.memory_id, u.latest_source_at, v.content, l.link_basis
                FROM domain_memory_links l
                JOIN memories u ON u.memory_id = l.memory_id
                    AND u.lifecycle_status = 'active'
                JOIN memory_versions v ON v.memory_id = u.memory_id AND v.is_latest = 1
                WHERE l.domain_id = ? AND l.unlinked_at IS NULL
                ORDER BY u.memory_id
                """,
                (snapshot.domain_id,),
            ).fetchall()
            current_memories = tuple(
                (
                    row["memory_id"],
                    row["content"],
                    row["latest_source_at"],
                    row["link_basis"],
                )
                for row in rows
            )
            supplied_memories = tuple(
                (
                    memory.memory_id,
                    memory.content,
                    memory.latest_source_at,
                    memory.link_basis,
                )
                for memory in snapshot.memories
            )
            if domain["name"] != snapshot.name or current_memories != supplied_memories:
                raise ConcurrentUpdateError(
                    f"domain evidence changed: {snapshot.domain_id}"
                )
            self._insert_operation(
                connection,
                operation_id,
                memory_space_id,
                "refresh_domain_summary",
                actor,
                config_signature,
                "Rewrite a linked domain summary from all active memories.",
                now,
            )
            connection.execute(
                """
                UPDATE domains SET summary = ?, summary_revision = summary_revision + 1,
                    updated_at = ? WHERE domain_id = ?
                """,
                (summary, now, snapshot.domain_id),
            )
            self._upsert_domain_embedding(
                connection,
                snapshot.domain_id,
                "name_summary",
                _domain_embedding_text(snapshot.name, summary),
                summary_embedding,
                signature_id,
                now,
            )
            self._effect(
                connection,
                operation_id,
                "domain",
                snapshot.domain_id,
                "summary_refreshed",
            )
            cursor = connection.execute(
                """
                UPDATE episode_summary_refresh_targets
                SET completed_by_operation_id = ?
                WHERE add_operation_id = ? AND domain_id = ?
                    AND completed_by_operation_id IS NULL
                """,
                (operation_id, add_operation_id, snapshot.domain_id),
            )
            if cursor.rowcount != 1:
                raise ConcurrentUpdateError(
                    f"summary refresh target is no longer pending: {snapshot.domain_id}"
                )
        return operation_id

    def _replace_memory_version(
        self,
        connection: sqlite3.Connection,
        memory_space_id: str,
        update: PreparedMemoryUpdate,
        operation_id: str,
        signature_id: str,
        now: int,
    ) -> None:
        current = connection.execute(
            """
            SELECT v.memory_version_id, v.version_no, v.content FROM memories u
            JOIN memory_versions v ON v.memory_id = u.memory_id AND v.is_latest = 1
            WHERE u.memory_id = ? AND u.memory_space_id = ? AND u.lifecycle_status = 'active'
            """,
            (update.memory_id, memory_space_id),
        ).fetchone()
        if current is None:
            raise NotFoundError(f"active memory not found: {update.memory_id}")
        if len(set(update.provenance_episode_ids)) != len(
            update.provenance_episode_ids
        ):
            raise ValidationError("memory provenance contains duplicate episodes")
        episode_rows = connection.execute(
            f"SELECT episode_id, source_started_at, source_ended_at FROM episodes WHERE memory_space_id = ? AND episode_id IN ({_placeholders(update.provenance_episode_ids)})",
            (memory_space_id, *update.provenance_episode_ids),
        ).fetchall()
        if len(episode_rows) != len(update.provenance_episode_ids):
            raise ValidationError("memory provenance contains an invalid episode")
        connection.execute(
            "UPDATE memory_versions SET is_latest = 0 WHERE memory_version_id = ?",
            (current["memory_version_id"],),
        )
        connection.execute(
            """
            INSERT INTO memory_versions(
                memory_version_id, memory_id, version_no, is_latest, content,
                content_hash, char_count, created_by_operation_id, created_at
            ) VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)
            """,
            (
                update.memory_version_id,
                update.memory_id,
                current["version_no"] + 1,
                update.content,
                text_hash(update.content),
                len(update.content),
                operation_id,
                now,
            ),
        )
        for episode_id in update.provenance_episode_ids:
            connection.execute(
                "INSERT INTO memory_version_provenance(memory_version_id, episode_id) VALUES (?, ?)",
                (update.memory_version_id, episode_id),
            )
        latest_source_at = _latest_source_time(episode_rows)
        connection.execute(
            "UPDATE memories SET latest_source_at = ?, updated_at = ? WHERE memory_id = ?",
            (latest_source_at, now, update.memory_id),
        )
        self._insert_memory_embedding(connection, update, signature_id, now)
        self._effect(
            connection,
            operation_id,
            "memory_version",
            update.memory_version_id,
            "created",
        )
        self._effect(connection, operation_id, "memory", update.memory_id, "updated")

    def _retire_memory(
        self,
        connection: sqlite3.Connection,
        memory_space_id: str,
        memory_id: str,
        operation_id: str,
        now: int,
    ) -> None:
        cursor = connection.execute(
            """
            UPDATE memories SET lifecycle_status = 'retired', retired_at = ?,
                retired_by_operation_id = ?, updated_at = ?
            WHERE memory_id = ? AND memory_space_id = ? AND lifecycle_status = 'active'
            """,
            (now, operation_id, now, memory_id, memory_space_id),
        )
        if cursor.rowcount != 1:
            raise NotFoundError(f"active memory not found: {memory_id}")
        connection.execute(
            """
            UPDATE domain_memory_links SET unlinked_at = ?, closed_by_operation_id = ?
            WHERE memory_id = ? AND unlinked_at IS NULL
            """,
            (now, operation_id, memory_id),
        )
        connection.execute(
            "DELETE FROM memory_embeddings WHERE memory_version_id IN (SELECT memory_version_id FROM memory_versions WHERE memory_id = ? AND is_latest = 1)",
            (memory_id,),
        )
        self._effect(connection, operation_id, "memory", memory_id, "retired")

    def commit_split(
        self,
        *,
        add_operation_id: str,
        memory_space_id: str,
        original: DomainSnapshot,
        result: Literal["full_split", "partial_split"],
        new_domains: Sequence[PreparedSplitDomain],
        signature_id: str,
        actor: str,
        config_signature: str,
    ) -> str:
        operation_id = str(uuid4())
        now = utc_milliseconds()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT name, summary_revision FROM domains WHERE domain_id = ? AND memory_space_id = ? AND lifecycle_status = 'active'",
                (original.domain_id, memory_space_id),
            ).fetchone()
            if row is None:
                raise NotFoundError(f"active domain not found: {original.domain_id}")
            if row["summary_revision"] != original.summary_revision:
                raise ConcurrentUpdateError(f"stale domain: {original.domain_id}")
            current_ids = self._domain_memory_ids(connection, original.domain_id)
            input_ids = {memory.memory_id for memory in original.memories}
            if current_ids != input_ids:
                raise ConcurrentUpdateError(
                    f"domain membership changed: {original.domain_id}"
                )
            self._insert_operation(
                connection,
                operation_id,
                memory_space_id,
                "split_domain",
                actor,
                config_signature,
                f"Apply {result} to an over-capacity domain.",
                now,
            )
            moved_ids = {
                link.memory_id for prepared in new_domains for link in prepared.links
            }
            if result == "full_split":
                close_ids = current_ids
                connection.execute(
                    """
                    DELETE FROM episode_summary_refresh_targets
                    WHERE add_operation_id = ? AND domain_id = ?
                    """,
                    (add_operation_id, original.domain_id),
                )
                connection.execute(
                    """
                    UPDATE domains SET lifecycle_status = 'retired', new_memory_count = 0, retired_at = ?,
                        retired_by_operation_id = ?, updated_at = ?
                    WHERE domain_id = ?
                    """,
                    (now, operation_id, now, original.domain_id),
                )
                self._effect(
                    connection, operation_id, "domain", original.domain_id, "retired"
                )
            else:
                close_ids = moved_ids
                connection.execute(
                    """
                    UPDATE domains SET new_memory_count = 0, updated_at = ?
                    WHERE domain_id = ?
                    """,
                    (now, original.domain_id),
                )
                self._effect(
                    connection,
                    operation_id,
                    "domain",
                    original.domain_id,
                    "split_retained",
                )
            if close_ids:
                connection.execute(
                    f"""
                    UPDATE domain_memory_links SET unlinked_at = ?, closed_by_operation_id = ?
                    WHERE domain_id = ? AND unlinked_at IS NULL
                        AND memory_id IN ({_placeholders(tuple(close_ids))})
                    """,
                    (now, operation_id, original.domain_id, *close_ids),
                )
            for prepared in new_domains:
                target_id = prepared.domain.domain_id
                if prepared.expected_revision is None:
                    self._insert_domain(
                        connection,
                        memory_space_id,
                        prepared.domain,
                        operation_id,
                        signature_id,
                        now,
                    )
                else:
                    target = connection.execute(
                        "SELECT summary_revision FROM domains WHERE domain_id=? AND memory_space_id=?",
                        (target_id, memory_space_id),
                    ).fetchone()
                    if (
                        target is None
                        or target["summary_revision"] != prepared.expected_revision
                    ):
                        raise ConcurrentUpdateError(f"stale split target: {target_id}")
                    self._reactivate_domain(connection, target_id, operation_id, now)
                connection.execute(
                    """INSERT OR IGNORE INTO entity_domains(entity_id, domain_id)
                    SELECT entity_id, ? FROM entity_domains WHERE domain_id=?""",
                    (target_id, original.domain_id),
                )
                added = 0
                for link in prepared.links:
                    existing = connection.execute(
                        "SELECT link_id, link_basis FROM domain_memory_links WHERE domain_id=? AND memory_id=? AND unlinked_at IS NULL",
                        (target_id, link.memory_id),
                    ).fetchone()
                    if existing is None:
                        self._insert_link(connection, link, operation_id, now)
                        added += 1
                    elif (
                        existing["link_basis"] == "contextual"
                        and link.basis == "direct"
                    ):
                        connection.execute(
                            "UPDATE domain_memory_links SET unlinked_at=?, closed_by_operation_id=? WHERE link_id=?",
                            (now, operation_id, existing["link_id"]),
                        )
                        self._insert_link(connection, link, operation_id, now)
                if prepared.expected_revision is not None:
                    connection.execute(
                        "UPDATE domains SET new_memory_count=new_memory_count+?, updated_at=? WHERE domain_id=?",
                        (added, now, target_id),
                    )
                    self._effect(
                        connection, operation_id, "domain", target_id, "split_reused"
                    )
            self._insert_summary_refresh_targets(
                connection,
                add_operation_id,
                {prepared.domain.domain_id for prepared in new_domains},
            )
            self._validate_active_memory_links(connection, memory_space_id)
        return operation_id

    def record_deferred_split(
        self,
        *,
        memory_space_id: str,
        domain_id: str,
        expected_revision: int,
        reason: str,
        actor: str,
        config_signature: str,
    ) -> str:
        operation_id = str(uuid4())
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT summary_revision FROM domains WHERE domain_id = ? AND memory_space_id = ? AND lifecycle_status = 'active'",
                (domain_id, memory_space_id),
            ).fetchone()
            if row is None:
                raise NotFoundError(f"active domain not found: {domain_id}")
            if row["summary_revision"] != expected_revision:
                raise ConcurrentUpdateError(f"stale domain: {domain_id}")
            self._insert_operation(
                connection,
                operation_id,
                memory_space_id,
                "split_domain",
                actor,
                config_signature,
                reason,
                utc_milliseconds(),
            )
            self._effect(
                connection, operation_id, "domain", domain_id, "split_deferred"
            )
        return operation_id

    def public_search(
        self, memory_space_id: str, query: str, query_vector: np.ndarray
    ) -> SearchResult:
        domain_hits = self._scan_domains(
            memory_space_id,
            query_vector,
            self.config.search_domain_top_k,
            self.config.search_domain_min_similarity,
        )
        memory_hits = self._scan_memories(
            memory_space_id,
            query_vector,
            self.config.search_memory_top_k,
            self.config.search_memory_min_similarity,
        )
        with self._connect() as connection:
            signature_id = self._active_signature(connection, memory_space_id)
            domain_refs: list[tuple[str, float, tuple[str, ...]]] = []
            memory_refs: list[tuple[str, float, tuple[str, ...]]] = []
            domain_similarities = dict(domain_hits)
            memory_similarities = dict(memory_hits)
            direct_domain_ids = tuple(item[0] for item in domain_hits)
            domain_order = list(direct_domain_ids)
            memory_order = [item[0] for item in memory_hits]
            for domain_id, similarity in domain_hits:
                rows = connection.execute(
                    """
                    SELECT u.memory_id, e.vector FROM domain_memory_links l
                    JOIN memories u ON u.memory_id = l.memory_id AND u.lifecycle_status = 'active'
                    JOIN memory_versions v ON v.memory_id = u.memory_id AND v.is_latest = 1
                    JOIN memory_embeddings e ON e.memory_version_id = v.memory_version_id AND e.model_signature_id = ?
                    WHERE l.domain_id = ? AND l.unlinked_at IS NULL
                    """,
                    (signature_id, domain_id),
                ).fetchall()
                attached_hits = _top_embedded_items(
                    rows,
                    query_vector,
                    "memory_id",
                    self.config.search_domain_attached_memory_k,
                )
                attached = tuple(item[0] for item in attached_hits)
                memory_similarities.update(attached_hits)
                for memory_id in attached:
                    if memory_id not in memory_order:
                        memory_order.append(memory_id)
                domain_refs.append((domain_id, similarity, attached))
            for memory_id, similarity in memory_hits:
                rows = connection.execute(
                    """
                    SELECT s.domain_id, e.vector FROM domain_memory_links l
                    JOIN domains s ON s.domain_id = l.domain_id AND s.lifecycle_status = 'active'
                    JOIN domain_embeddings e ON e.domain_id = s.domain_id
                        AND e.embedding_kind = 'name' AND e.model_signature_id = ?
                    WHERE l.memory_id = ? AND l.unlinked_at IS NULL
                    """,
                    (signature_id, memory_id),
                ).fetchall()
                attached_hits = _top_embedded_items(
                    rows,
                    query_vector,
                    "domain_id",
                    self.config.search_memory_attached_domain_k,
                )
                attached = tuple(item[0] for item in attached_hits)
                domain_similarities.update(attached_hits)
                for domain_id in attached:
                    if domain_id not in domain_order:
                        domain_order.append(domain_id)
                memory_refs.append((memory_id, similarity, attached))
            domain_ids = set(domain_order)
            memory_ids = set(memory_order)
            domains = self._load_search_domains(
                connection,
                domain_order,
                set(direct_domain_ids),
                domain_similarities,
            )
            memories = self._load_search_memories(
                connection, memory_order, memory_similarities
            )
            links = self._load_search_links(connection, domain_ids, memory_ids)
        from fluxfold.models import RankedDomainRef, RankedMemoryRef

        return SearchResult(
            query,
            domains,
            memories,
            links,
            tuple(RankedDomainRef(*item) for item in domain_refs),
            tuple(RankedMemoryRef(*item) for item in memory_refs),
        )

    def active_domain_ids(self, memory_space_id: str) -> tuple[str, ...]:
        with self._connect() as connection:
            return tuple(
                row[0]
                for row in connection.execute(
                    "SELECT domain_id FROM domains WHERE memory_space_id = ? AND lifecycle_status = 'active' ORDER BY domain_id",
                    (memory_space_id,),
                )
            )

    def operation_active_domain_ids(
        self, memory_space_id: str, operation_id: str | None
    ) -> tuple[str, ...]:
        if operation_id is None:
            return ()
        with self._connect() as connection:
            return tuple(
                row[0]
                for row in connection.execute(
                    """
                    SELECT DISTINCT l.domain_id FROM domain_memory_links l
                    JOIN domains s ON s.domain_id = l.domain_id
                        AND s.lifecycle_status = 'active'
                    WHERE s.memory_space_id = ? AND l.opened_by_operation_id = ?
                        AND l.unlinked_at IS NULL
                    ORDER BY l.domain_id
                    """,
                    (memory_space_id, operation_id),
                )
            )

    def pending_summary_refresh_domain_ids(
        self, memory_space_id: str, add_operation_id: str | None
    ) -> tuple[str, ...]:
        if add_operation_id is None:
            return ()
        with self._connect() as connection:
            return tuple(
                row[0]
                for row in connection.execute(
                    """
                    SELECT t.domain_id FROM episode_summary_refresh_targets t
                    JOIN domains s ON s.domain_id = t.domain_id
                        AND s.lifecycle_status = 'active'
                    WHERE t.add_operation_id = ? AND s.memory_space_id = ?
                        AND t.completed_by_operation_id IS NULL
                    ORDER BY t.domain_id
                    """,
                    (add_operation_id, memory_space_id),
                )
            )

    def memory_bank(self) -> tuple[MemoryBankSpace, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    ms.space_key AS space_key,
                    s.domain_id AS domain_id,
                    s.name AS name,
                    s.summary AS summary,
                    u.memory_id AS memory_id,
                    v.content AS content
                FROM memory_spaces ms
                LEFT JOIN domains s
                    ON s.memory_space_id = ms.memory_space_id
                    AND s.lifecycle_status = 'active'
                LEFT JOIN domain_memory_links l
                    ON l.domain_id = s.domain_id AND l.unlinked_at IS NULL
                LEFT JOIN memories u
                    ON u.memory_id = l.memory_id AND u.lifecycle_status = 'active'
                LEFT JOIN memory_versions v
                    ON v.memory_id = u.memory_id AND v.is_latest = 1
                ORDER BY ms.space_key, s.name, s.domain_id, u.memory_id
                """
            ).fetchall()
        spaces: dict[str, dict[str, MemoryBankDomain]] = {}
        domain_order: dict[str, list[str]] = {}
        for row in rows:
            space_key = str(row["space_key"])
            spaces.setdefault(space_key, {})
            domain_order.setdefault(space_key, [])
            domain_id = row["domain_id"]
            if domain_id is None:
                continue
            current = spaces[space_key].get(domain_id)
            if current is None:
                current = MemoryBankDomain(row["name"], row["summary"], ())
                spaces[space_key][domain_id] = current
                domain_order[space_key].append(domain_id)
            if row["memory_id"] is not None:
                spaces[space_key][domain_id] = MemoryBankDomain(
                    current.name,
                    current.summary,
                    (*current.memory_contents, row["content"]),
                )
        return tuple(
            MemoryBankSpace(
                space_key,
                tuple(spaces[space_key][domain_id] for domain_id in domain_ids),
            )
            for space_key, domain_ids in domain_order.items()
        )

    def space_statistics(self, memory_space_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            source_chars = connection.execute(
                "SELECT coalesce(sum(source_chars), 0) FROM episodes WHERE memory_space_id = ?",
                (memory_space_id,),
            ).fetchone()[0]
            episode_count = connection.execute(
                "SELECT count(*) FROM episodes WHERE memory_space_id = ?",
                (memory_space_id,),
            ).fetchone()[0]
            memory_content_chars = connection.execute(
                """
                SELECT coalesce(sum(v.char_count), 0) FROM memories u
                JOIN memory_versions v ON v.memory_id = u.memory_id AND v.is_latest = 1
                WHERE u.memory_space_id = ? AND u.lifecycle_status = 'active'
                """,
                (memory_space_id,),
            ).fetchone()[0]
            summary_chars = connection.execute(
                "SELECT coalesce(sum(length(summary)), 0) FROM domains WHERE memory_space_id = ? AND lifecycle_status = 'active'",
                (memory_space_id,),
            ).fetchone()[0]
            active_memories = connection.execute(
                "SELECT count(*) FROM memories WHERE memory_space_id = ? AND lifecycle_status = 'active'",
                (memory_space_id,),
            ).fetchone()[0]
            retired_memories = connection.execute(
                "SELECT count(*) FROM memories WHERE memory_space_id = ? AND lifecycle_status = 'retired'",
                (memory_space_id,),
            ).fetchone()[0]
            active_domains = connection.execute(
                "SELECT count(*) FROM domains WHERE memory_space_id = ? AND lifecycle_status = 'active'",
                (memory_space_id,),
            ).fetchone()[0]
            retired_domains = connection.execute(
                "SELECT count(*) FROM domains WHERE memory_space_id = ? AND lifecycle_status = 'retired'",
                (memory_space_id,),
            ).fetchone()[0]
            review_count = connection.execute(
                "SELECT count(*) FROM memory_operations WHERE memory_space_id = ? AND operation_type = 'review_domain'",
                (memory_space_id,),
            ).fetchone()[0]
            split_count = connection.execute(
                "SELECT count(*) FROM memory_operations WHERE memory_space_id = ? AND operation_type = 'split_domain'",
                (memory_space_id,),
            ).fetchone()[0]
            refresh_count = connection.execute(
                "SELECT count(*) FROM memory_operations WHERE memory_space_id = ? AND operation_type = 'refresh_domain_summary'",
                (memory_space_id,),
            ).fetchone()[0]
            rewritten_memories = connection.execute(
                """
                SELECT count(*) FROM memories u
                JOIN memory_versions v ON v.memory_id = u.memory_id AND v.is_latest = 1
                WHERE u.memory_space_id = ? AND u.lifecycle_status = 'active'
                    AND v.version_no > 1
                """,
                (memory_space_id,),
            ).fetchone()[0]
            memory_link_rows = connection.execute(
                """
                SELECT
                    coalesce(sum(l.link_basis = 'direct'), 0) AS direct_links,
                    coalesce(sum(l.link_basis = 'contextual'), 0) AS contextual_links
                FROM memories u
                LEFT JOIN domain_memory_links l
                    ON l.memory_id = u.memory_id AND l.unlinked_at IS NULL
                LEFT JOIN domains s
                    ON s.domain_id = l.domain_id AND s.lifecycle_status = 'active'
                WHERE u.memory_space_id = ? AND u.lifecycle_status = 'active'
                GROUP BY u.memory_id
                """,
                (memory_space_id,),
            ).fetchall()
            max_provenance_per_memory = connection.execute(
                """
                SELECT coalesce(max(n), 0) FROM (
                    SELECT count(p.episode_id) AS n
                    FROM memories u
                    JOIN memory_versions v
                        ON v.memory_id = u.memory_id AND v.is_latest = 1
                    LEFT JOIN memory_version_provenance p
                        ON p.memory_version_id = v.memory_version_id
                    WHERE u.memory_space_id = ? AND u.lifecycle_status = 'active'
                    GROUP BY u.memory_id
                )
                """,
                (memory_space_id,),
            ).fetchone()[0]
            domain_rows = connection.execute(
                """
                SELECT
                    s.name AS name,
                    coalesce(sum(l.link_basis = 'direct'), 0) AS direct_links,
                    coalesce(sum(l.link_basis = 'contextual'), 0) AS contextual_links
                FROM domains s
                LEFT JOIN domain_memory_links l
                    ON l.domain_id = s.domain_id AND l.unlinked_at IS NULL
                LEFT JOIN memories u
                    ON u.memory_id = l.memory_id AND u.lifecycle_status = 'active'
                WHERE s.memory_space_id = ? AND s.lifecycle_status = 'active'
                GROUP BY s.domain_id, s.name
                ORDER BY s.name, s.domain_id
                """,
                (memory_space_id,),
            ).fetchall()
        memory_chars = memory_content_chars + summary_chars
        rate = 0.0 if source_chars == 0 else 1 - memory_chars / source_chars
        link_counts = [
            row["direct_links"] + row["contextual_links"] for row in memory_link_rows
        ]
        memories_per_domain = [
            row["direct_links"] + row["contextual_links"] for row in domain_rows
        ]
        return {
            "source_chars": source_chars,
            "memory_chars": memory_chars,
            "memory_compression_rate": rate,
            "active_memories": active_memories,
            "active_domains": active_domains,
            "episode_count": episode_count,
            "direct_links": sum(row["direct_links"] for row in memory_link_rows),
            "contextual_links": sum(
                row["contextual_links"] for row in memory_link_rows
            ),
            "max_links_per_memory": max(link_counts, default=0),
            "max_direct_links_per_memory": max(
                (row["direct_links"] for row in memory_link_rows), default=0
            ),
            "max_contextual_links_per_memory": max(
                (row["contextual_links"] for row in memory_link_rows), default=0
            ),
            "mean_links_per_memory": (
                0.0 if not link_counts else sum(link_counts) / len(link_counts)
            ),
            "max_memories_per_domain": max(memories_per_domain, default=0),
            "mean_memories_per_domain": (
                0.0
                if not memories_per_domain
                else sum(memories_per_domain) / len(memories_per_domain)
            ),
            "max_provenance_per_memory": max_provenance_per_memory,
            "retired_memories": retired_memories,
            "retired_domains": retired_domains,
            "rewritten_memories": rewritten_memories,
            "domain_review_count": review_count,
            "domain_split_count": split_count,
            "domain_summary_refresh_count": refresh_count,
            "domains": [
                {
                    "name": row["name"],
                    "direct_links": row["direct_links"],
                    "contextual_links": row["contextual_links"],
                }
                for row in domain_rows
            ],
        }

    def _load_search_domains(
        self,
        connection: sqlite3.Connection,
        ids: Sequence[str],
        summary_ids: set[str],
        similarities: dict[str, float],
    ) -> tuple[SearchDomain, ...]:
        if not ids:
            return ()
        stable_ids = tuple(ids)
        rows = connection.execute(
            f"SELECT domain_id, name, summary FROM domains WHERE domain_id IN ({_placeholders(stable_ids)})",
            stable_ids,
        ).fetchall()
        by_id = {row["domain_id"]: row for row in rows}
        return tuple(
            SearchDomain(
                domain_id,
                by_id[domain_id]["name"],
                by_id[domain_id]["summary"] if domain_id in summary_ids else None,
                similarities[domain_id],
            )
            for domain_id in stable_ids
        )

    def _load_search_memories(
        self,
        connection: sqlite3.Connection,
        ids: Sequence[str],
        similarities: dict[str, float],
    ) -> tuple[SearchMemory, ...]:
        if not ids:
            return ()
        stable_ids = tuple(ids)
        rows = connection.execute(
            f"""
            SELECT u.memory_id, u.latest_source_at, v.content FROM memories u
            JOIN memory_versions v ON v.memory_id = u.memory_id AND v.is_latest = 1
            WHERE u.memory_id IN ({_placeholders(stable_ids)})
            """,
            stable_ids,
        ).fetchall()
        by_id = {row["memory_id"]: row for row in rows}
        return tuple(
            SearchMemory(
                memory_id,
                by_id[memory_id]["content"],
                by_id[memory_id]["latest_source_at"],
                similarities[memory_id],
            )
            for memory_id in stable_ids
        )

    def _load_search_links(
        self,
        connection: sqlite3.Connection,
        domain_ids: set[str],
        memory_ids: set[str],
    ) -> tuple[SearchLink, ...]:
        if not domain_ids or not memory_ids:
            return ()
        stable_domain_ids = tuple(sorted(domain_ids))
        stable_memory_ids = tuple(sorted(memory_ids))
        rows = connection.execute(
            f"""
            SELECT domain_id, memory_id, link_basis FROM domain_memory_links
            WHERE unlinked_at IS NULL
                AND domain_id IN ({_placeholders(stable_domain_ids)})
                AND memory_id IN ({_placeholders(stable_memory_ids)})
            ORDER BY domain_id, memory_id
            """,
            (*stable_domain_ids, *stable_memory_ids),
        ).fetchall()
        return tuple(
            SearchLink(row["domain_id"], row["memory_id"], row["link_basis"])
            for row in rows
        )

    def _active_signature(
        self, connection: sqlite3.Connection, memory_space_id: str
    ) -> str:
        row = connection.execute(
            "SELECT model_signature_id FROM memory_space_model_signatures WHERE memory_space_id = ? AND purpose = 'retrieval'",
            (memory_space_id,),
        ).fetchone()
        if row is None:
            raise ValidationError(
                f"memory space has no retrieval signature: {memory_space_id}"
            )
        return str(row["model_signature_id"])

    def _provenance(
        self, connection: sqlite3.Connection, memory_id: str
    ) -> tuple[str, ...]:
        return tuple(
            row[0]
            for row in connection.execute(
                """
                SELECT p.episode_id FROM memory_versions v
                JOIN memory_version_provenance p ON p.memory_version_id = v.memory_version_id
                WHERE v.memory_id = ? AND v.is_latest = 1 ORDER BY p.episode_id
                """,
                (memory_id,),
            )
        )

    def _domain_memory_ids(
        self, connection: sqlite3.Connection, domain_id: str
    ) -> set[str]:
        return {
            row[0]
            for row in connection.execute(
                """
                SELECT l.memory_id FROM domain_memory_links l
                JOIN memories u ON u.memory_id = l.memory_id AND u.lifecycle_status = 'active'
                WHERE l.domain_id = ? AND l.unlinked_at IS NULL
                """,
                (domain_id,),
            )
        }

    def memory_link_limit(self, memory_id: str) -> int:
        with self._connect() as connection:
            entity_count = connection.execute(
                "SELECT count(*) FROM memory_entities WHERE memory_id=?",
                (memory_id,),
            ).fetchone()[0]
        return max(self.config.memory_active_domain_link_max, int(entity_count))

    def _validate_active_memory_links(
        self, connection: sqlite3.Connection, memory_space_id: str
    ) -> None:
        invalid = connection.execute(
            """
            SELECT u.memory_id,
                sum(CASE WHEN l.unlinked_at IS NULL THEN 1 ELSE 0 END) AS active_links,
                sum(CASE WHEN l.unlinked_at IS NULL AND l.link_basis = 'direct' THEN 1 ELSE 0 END) AS direct_links
            FROM memories u
            LEFT JOIN domain_memory_links l ON l.memory_id = u.memory_id
            WHERE u.memory_space_id = ? AND u.lifecycle_status = 'active'
            GROUP BY u.memory_id
            HAVING active_links < 1 OR direct_links < 1 OR active_links > max(?,
                (SELECT count(*) FROM memory_entities ma WHERE ma.memory_id=u.memory_id))
            LIMIT 1
            """,
            (memory_space_id, self.config.memory_active_domain_link_max),
        ).fetchone()
        if invalid is not None:
            raise ValidationError(
                f"active memory link invariant failed for {invalid['memory_id']}"
            )

    def _require_space(
        self, connection: sqlite3.Connection, memory_space_id: str
    ) -> None:
        if (
            connection.execute(
                "SELECT 1 FROM memory_spaces WHERE memory_space_id = ?",
                (memory_space_id,),
            ).fetchone()
            is None
        ):
            raise NotFoundError(f"memory space not found: {memory_space_id}")

    def _insert_operation(
        self,
        connection: sqlite3.Connection,
        operation_id: str,
        memory_space_id: str,
        operation_type: str,
        actor: str,
        config_signature: str,
        reason: str,
        now: int,
    ) -> None:
        connection.execute(
            """
            INSERT INTO memory_operations(
                operation_id, memory_space_id, operation_type, actor,
                config_signature, reason, committed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                operation_id,
                memory_space_id,
                operation_type,
                actor,
                config_signature,
                reason,
                now,
            ),
        )

    def _effect(
        self,
        connection: sqlite3.Connection,
        operation_id: str,
        object_type: str,
        object_id: str,
        effect_type: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO memory_operation_effects(
                effect_id, operation_id, object_type, object_id, effect_type, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid4()),
                operation_id,
                object_type,
                object_id,
                effect_type,
                canonical_json(metadata) if metadata is not None else None,
            ),
        )

    def _insert_summary_refresh_targets(
        self,
        connection: sqlite3.Connection,
        add_operation_id: str,
        domain_ids: set[str],
    ) -> None:
        connection.executemany(
            """
            INSERT OR IGNORE INTO episode_summary_refresh_targets(
                add_operation_id, domain_id
            ) VALUES (?, ?)
            """,
            [(add_operation_id, domain_id) for domain_id in sorted(domain_ids)],
        )


def _vector_blob(vector: np.ndarray) -> bytes:
    contiguous = np.ascontiguousarray(vector, dtype="<f4")
    if contiguous.ndim != 1:
        raise ValidationError("embedding must be one-dimensional")
    return contiguous.tobytes(order="C")


def _normalized_query(vector: np.ndarray) -> np.ndarray:
    query = np.ascontiguousarray(vector, dtype="<f4")
    if query.ndim != 1:
        raise ValidationError("query embedding must be one-dimensional")
    norm = float(np.linalg.norm(query))
    if norm == 0:
        raise ValidationError("query embedding has zero norm")
    return np.ascontiguousarray(query / norm, dtype="<f4")


def _top_embedded_items(
    rows: Sequence[sqlite3.Row],
    query_vector: np.ndarray,
    id_field: str,
    top_k: int,
) -> tuple[tuple[str, float], ...]:
    query = _normalized_query(query_vector)
    ranked = sorted(
        (
            (row[id_field], float(np.frombuffer(row["vector"], dtype="<f4") @ query))
            for row in rows
        ),
        key=lambda item: (-item[1], item[0]),
    )
    return tuple(ranked[:top_k])


def _domain_embedding_text(name: str, summary: str) -> str:
    return f"{name}\n\n{summary}"


def _placeholders(values: Sequence[object]) -> str:
    if not values:
        raise ValidationError("SQL value list must not be empty")
    return ",".join("?" for _ in values)


def _latest_source_time(rows: Iterable[sqlite3.Row]) -> int | None:
    timestamps = [row["source_ended_at"] or row["source_started_at"] for row in rows]
    known = [timestamp for timestamp in timestamps if timestamp is not None]
    return max(known) if known else None
