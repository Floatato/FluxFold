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
    CandidateMemory,
    CandidateSubject,
    EpisodeBlock,
    MemoryBankSpace,
    MemoryBankSubject,
    MemorySnapshot,
    MemorySpace,
    NormalizedEpisode,
    SearchLink,
    SearchMemory,
    SearchResult,
    SearchSubject,
    SubjectSnapshot,
    canonical_json,
    text_hash,
    utc_milliseconds,
)
from fluxfold.providers import EmbeddingModelInfo

SCHEMA_VERSION = "4"


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
class PreparedSubject:
    subject_id: str
    name: str
    summary: str | None
    name_embedding: np.ndarray
    name_summary_embedding: np.ndarray | None


@dataclass(frozen=True, slots=True)
class PreparedLink:
    memory_id: str
    subject_id: str
    basis: Literal["direct", "contextual"]


@dataclass(frozen=True, slots=True)
class ExistingSubjectTouch:
    subject_id: str
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
class PreparedSplitSubject:
    subject: PreparedSubject
    links: tuple[PreparedLink, ...]


@dataclass(frozen=True, slots=True)
class RetrievalMemorySource:
    memory_version_id: str
    content: str


@dataclass(frozen=True, slots=True)
class RetrievalSubjectSource:
    subject_id: str
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
            connection.executescript(schema)
            row = connection.execute(
                "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO schema_metadata(key, value) VALUES ('schema_version', ?)",
                    (SCHEMA_VERSION,),
                )
            elif row["value"] != SCHEMA_VERSION:
                raise ValidationError(
                    f"database schema {row['value']} is incompatible with {SCHEMA_VERSION}"
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
        memory_versions = "SELECT memory_version_id FROM memory_versions WHERE memory_id IN (SELECT memory_id FROM memory_units WHERE memory_space_id = ?)"
        subjects = "SELECT subject_id FROM subjects WHERE memory_space_id = ?"
        memories = "SELECT memory_id FROM memory_units WHERE memory_space_id = ?"
        episodes = (
            "SELECT episode_id FROM situational_episodes WHERE memory_space_id = ?"
        )
        operations = (
            "SELECT operation_id FROM domain_operations WHERE memory_space_id = ?"
        )
        connection.execute(
            f"DELETE FROM domain_operation_effects WHERE operation_id IN ({operations})",
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
            f"DELETE FROM subject_embeddings WHERE subject_id IN ({subjects})",
            (memory_space_id,),
        )
        connection.execute(
            f"DELETE FROM memory_version_provenance WHERE memory_version_id IN ({memory_versions})",
            (memory_space_id,),
        )
        connection.execute(
            f"DELETE FROM subject_memory_links WHERE subject_id IN ({subjects}) OR memory_id IN ({memories})",
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
            "DELETE FROM subjects WHERE memory_space_id = ?", (memory_space_id,)
        )
        connection.execute(
            "DELETE FROM memory_units WHERE memory_space_id = ?", (memory_space_id,)
        )
        connection.execute(
            f"DELETE FROM episode_blocks WHERE episode_id IN ({episodes})",
            (memory_space_id,),
        )
        connection.execute(
            "DELETE FROM situational_episodes WHERE memory_space_id = ?",
            (memory_space_id,),
        )
        connection.execute(
            "DELETE FROM memory_space_model_signatures WHERE memory_space_id = ?",
            (memory_space_id,),
        )
        connection.execute(
            "DELETE FROM domain_operations WHERE memory_space_id = ?",
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
    ) -> tuple[tuple[RetrievalMemorySource, ...], tuple[RetrievalSubjectSource, ...]]:
        with self._connect() as connection:
            self._require_space(connection, memory_space_id)
            memories = tuple(
                RetrievalMemorySource(row["memory_version_id"], row["content"])
                for row in connection.execute(
                    """
                    SELECT v.memory_version_id, v.content FROM memory_units u
                    JOIN memory_versions v ON v.memory_id = u.memory_id AND v.is_latest = 1
                    WHERE u.memory_space_id = ? AND u.lifecycle_status = 'active'
                    ORDER BY u.memory_id
                    """,
                    (memory_space_id,),
                )
            )
            subjects = tuple(
                RetrievalSubjectSource(row["subject_id"], row["name"], row["summary"])
                for row in connection.execute(
                    """
                    SELECT subject_id, name, summary FROM subjects
                    WHERE memory_space_id = ? AND lifecycle_status = 'active'
                    ORDER BY subject_id
                    """,
                    (memory_space_id,),
                )
            )
        return memories, subjects

    def commit_retrieval_embedding_switch(
        self,
        *,
        memory_space_id: str,
        signature_id: str,
        memories: Sequence[tuple[RetrievalMemorySource, np.ndarray]],
        subjects: Sequence[
            tuple[RetrievalSubjectSource, np.ndarray, np.ndarray | None]
        ],
    ) -> None:
        now = utc_milliseconds()
        with self._transaction() as connection:
            self._require_space(connection, memory_space_id)
            current_memories = {
                row["memory_version_id"]: row["content"]
                for row in connection.execute(
                    """
                    SELECT v.memory_version_id, v.content FROM memory_units u
                    JOIN memory_versions v ON v.memory_id = u.memory_id AND v.is_latest = 1
                    WHERE u.memory_space_id = ? AND u.lifecycle_status = 'active'
                    """,
                    (memory_space_id,),
                )
            }
            current_subjects = {
                row["subject_id"]: (row["name"], row["summary"])
                for row in connection.execute(
                    "SELECT subject_id, name, summary FROM subjects WHERE memory_space_id = ? AND lifecycle_status = 'active'",
                    (memory_space_id,),
                )
            }
            supplied_memories = {
                source.memory_version_id: source.content for source, _ in memories
            }
            supplied_subjects = {
                source.subject_id: (source.name, source.summary)
                for source, _, _ in subjects
            }
            if (
                supplied_memories != current_memories
                or supplied_subjects != current_subjects
            ):
                raise ConcurrentUpdateError(
                    "retrieval sources changed while embeddings were being rebuilt"
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
            for subject_source, name_vector, summary_vector in subjects:
                self._upsert_subject_embedding(
                    connection,
                    subject_source.subject_id,
                    "name",
                    subject_source.name,
                    name_vector,
                    signature_id,
                    now,
                )
                if subject_source.summary is not None:
                    if summary_vector is None:
                        raise ValidationError("summary embedding is missing")
                    self._upsert_subject_embedding(
                        connection,
                        subject_source.subject_id,
                        "name_summary",
                        _subject_embedding_text(
                            subject_source.name, subject_source.summary
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
            SELECT count(*) AS n FROM memory_units u
            JOIN memory_versions v ON v.memory_id = u.memory_id AND v.is_latest = 1
            LEFT JOIN memory_embeddings e ON e.memory_version_id = v.memory_version_id AND e.model_signature_id = ?
            WHERE u.memory_space_id = ? AND u.lifecycle_status = 'active' AND e.memory_version_id IS NULL
            """,
            (signature_id, memory_space_id),
        ).fetchone()["n"]
        active_subject_embeddings = connection.execute(
            """
            SELECT count(*) AS n FROM subjects s
            CROSS JOIN (SELECT 'name' AS kind UNION ALL SELECT 'name_summary') kinds
            LEFT JOIN subject_embeddings e ON e.subject_id = s.subject_id
                AND e.embedding_kind = kinds.kind AND e.model_signature_id = ?
            WHERE s.memory_space_id = ? AND s.lifecycle_status = 'active'
                AND (kinds.kind = 'name' OR s.summary IS NOT NULL)
                AND e.subject_id IS NULL
            """,
            (signature_id, memory_space_id),
        ).fetchone()["n"]
        if active_memories or active_subject_embeddings:
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
                SELECT episode_id, content_hash FROM situational_episodes
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
                INSERT INTO situational_episodes(
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
                "SELECT content_hash FROM situational_episodes WHERE episode_id = ? AND memory_space_id = ?",
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
        subjects: Sequence[PreparedSubject],
        links: Sequence[PreparedLink],
        subject_touches: Sequence[ExistingSubjectTouch],
        signature_id: str,
        actor: str,
        config_signature: str,
    ) -> str:
        operation_id = str(uuid4())
        now = utc_milliseconds()
        with self._transaction() as connection:
            self._require_space(connection, memory_space_id)
            episode = connection.execute(
                "SELECT source_started_at, source_ended_at, content_hash FROM situational_episodes WHERE episode_id = ? AND memory_space_id = ?",
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
            source_anchor = episode["source_ended_at"] or episode["source_started_at"]
            for memory in memories:
                self._insert_memory(
                    connection,
                    memory_space_id,
                    memory,
                    (episode_id,),
                    source_anchor,
                    operation_id,
                    signature_id,
                    now,
                )
            for subject in subjects:
                self._insert_subject(
                    connection,
                    memory_space_id,
                    subject,
                    operation_id,
                    signature_id,
                    now,
                )
            for touch in subject_touches:
                cursor = connection.execute(
                    """
                    UPDATE subjects SET new_memory_count = new_memory_count + ?,
                        updated_at = ?
                    WHERE subject_id = ? AND memory_space_id = ? AND lifecycle_status = 'active'
                        AND summary_revision = ?
                    """,
                    (
                        touch.new_memory_increment,
                        now,
                        touch.subject_id,
                        memory_space_id,
                        touch.expected_revision,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ConcurrentUpdateError(f"stale subject: {touch.subject_id}")
            for link in links:
                self._insert_link(connection, link, operation_id, now)
            self._insert_summary_refresh_targets(
                connection, operation_id, {link.subject_id for link in links}
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
            "INSERT INTO memory_units(memory_id, memory_space_id, lifecycle_status, latest_source_at, created_at, updated_at) VALUES (?, ?, 'active', ?, ?, ?)",
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

    def _insert_subject(
        self,
        connection: sqlite3.Connection,
        memory_space_id: str,
        subject: PreparedSubject,
        operation_id: str,
        signature_id: str,
        now: int,
    ) -> None:
        connection.execute(
            """
            INSERT INTO subjects(
                subject_id, memory_space_id, name, summary, lifecycle_status,
                new_memory_count, summary_revision, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'active', 0, 0, ?, ?)
            """,
            (
                subject.subject_id,
                memory_space_id,
                subject.name,
                subject.summary,
                now,
                now,
            ),
        )
        self._upsert_subject_embedding(
            connection,
            subject.subject_id,
            "name",
            subject.name,
            subject.name_embedding,
            signature_id,
            now,
        )
        if subject.summary is not None:
            if subject.name_summary_embedding is None:
                raise ValidationError("summary embedding is missing")
            self._upsert_subject_embedding(
                connection,
                subject.subject_id,
                "name_summary",
                _subject_embedding_text(subject.name, subject.summary),
                subject.name_summary_embedding,
                signature_id,
                now,
            )
        self._effect(connection, operation_id, "subject", subject.subject_id, "created")

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
            INSERT INTO subject_memory_links(
                link_id, subject_id, memory_id, link_basis, linked_at, opened_by_operation_id
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (link_id, link.subject_id, link.memory_id, link.basis, now, operation_id),
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

    def _upsert_subject_embedding(
        self,
        connection: sqlite3.Connection,
        subject_id: str,
        kind: str,
        source_text: str,
        vector: np.ndarray,
        signature_id: str,
        now: int,
    ) -> None:
        connection.execute(
            """
            INSERT INTO subject_embeddings(
                subject_id, embedding_kind, model_signature_id, dimension,
                source_hash, vector, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(subject_id, embedding_kind, model_signature_id) DO UPDATE SET
                dimension = excluded.dimension, source_hash = excluded.source_hash,
                vector = excluded.vector, created_at = excluded.created_at
            """,
            (
                subject_id,
                kind,
                signature_id,
                vector.shape[0],
                text_hash(source_text),
                _vector_blob(vector),
                now,
            ),
        )

    def candidate_subjects(
        self,
        memory_space_id: str,
        query_vector: np.ndarray,
        *,
        top_k: int,
        min_similarity: float,
    ) -> tuple[CandidateSubject, ...]:
        hits = self._scan_subjects(memory_space_id, query_vector, top_k, min_similarity)
        output: list[CandidateSubject] = []
        with self._connect() as connection:
            signature_id = self._active_signature(connection, memory_space_id)
            for subject_id, similarity in hits:
                row = connection.execute(
                    "SELECT name FROM subjects WHERE subject_id = ?",
                    (subject_id,),
                ).fetchone()
                attached = connection.execute(
                    """
                    SELECT u.memory_id, v.content, e.vector FROM subject_memory_links l
                    JOIN memory_units u ON u.memory_id = l.memory_id AND u.lifecycle_status = 'active'
                    JOIN memory_versions v ON v.memory_id = u.memory_id AND v.is_latest = 1
                    JOIN memory_embeddings e ON e.memory_version_id = v.memory_version_id AND e.model_signature_id = ?
                    WHERE l.subject_id = ? AND l.unlinked_at IS NULL
                    """,
                    (signature_id, subject_id),
                ).fetchall()
                best = _best_embedded_row(attached, query_vector, "memory_id")
                output.append(
                    CandidateSubject(
                        subject_id,
                        row["name"],
                        similarity,
                        best["memory_id"] if best is not None else None,
                        best["content"] if best is not None else None,
                    )
                )
        return tuple(output)

    def candidate_subject_names(
        self,
        memory_space_id: str,
        query_vector: np.ndarray,
        *,
        top_k: int,
        min_similarity: float,
    ) -> tuple[CandidateSubject, ...]:
        """Return name-only subject hits for the initial linking pool."""

        hits = self._scan_subjects(memory_space_id, query_vector, top_k, min_similarity)
        with self._connect() as connection:
            return tuple(
                CandidateSubject(
                    subject_id,
                    connection.execute(
                        "SELECT name FROM subjects WHERE subject_id = ?", (subject_id,)
                    ).fetchone()["name"],
                    similarity,
                )
                for subject_id, similarity in hits
            )

    def candidate_memories(
        self,
        memory_space_id: str,
        query_vector: np.ndarray,
        *,
        top_k: int,
        min_similarity: float,
    ) -> tuple[CandidateMemory, ...]:
        hits = self._scan_memories(memory_space_id, query_vector, top_k, min_similarity)
        output: list[CandidateMemory] = []
        with self._connect() as connection:
            signature_id = self._active_signature(connection, memory_space_id)
            for memory_id, similarity in hits:
                row = connection.execute(
                    """
                    SELECT u.latest_source_at, v.content FROM memory_units u
                    JOIN memory_versions v ON v.memory_id = u.memory_id AND v.is_latest = 1
                    WHERE u.memory_id = ?
                    """,
                    (memory_id,),
                ).fetchone()
                attached = connection.execute(
                    """
                    SELECT s.subject_id, s.name, e.vector FROM subject_memory_links l
                    JOIN subjects s ON s.subject_id = l.subject_id AND s.lifecycle_status = 'active'
                    JOIN subject_embeddings e ON e.subject_id = s.subject_id
                        AND e.embedding_kind = 'name' AND e.model_signature_id = ?
                    WHERE l.memory_id = ? AND l.unlinked_at IS NULL
                    """,
                    (signature_id, memory_id),
                ).fetchall()
                best = _best_embedded_row(attached, query_vector, "subject_id")
                output.append(
                    CandidateMemory(
                        memory_id,
                        row["content"],
                        row["latest_source_at"],
                        similarity,
                        best["subject_id"] if best is not None else None,
                        best["name"] if best is not None else None,
                    )
                )
        return tuple(output)

    def _scan_subjects(
        self,
        memory_space_id: str,
        query_vector: np.ndarray,
        top_k: int,
        min_similarity: float,
    ) -> list[tuple[str, float]]:
        sql = """
            SELECT s.subject_id AS object_id, e.vector FROM subjects s
            JOIN memory_space_model_signatures a ON a.memory_space_id = s.memory_space_id AND a.purpose = 'retrieval'
            JOIN subject_embeddings e ON e.subject_id = s.subject_id
                AND e.embedding_kind = 'name' AND e.model_signature_id = a.model_signature_id
            WHERE s.memory_space_id = ? AND s.lifecycle_status = 'active'
            ORDER BY s.subject_id
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
            SELECT u.memory_id AS object_id, e.vector FROM memory_units u
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

    def subject_snapshot(
        self, memory_space_id: str, subject_id: str
    ) -> SubjectSnapshot:
        with self._connect() as connection:
            subject = connection.execute(
                """
                SELECT name, summary, new_memory_count, summary_revision FROM subjects
                WHERE subject_id = ? AND memory_space_id = ? AND lifecycle_status = 'active'
                """,
                (subject_id, memory_space_id),
            ).fetchone()
            if subject is None:
                raise NotFoundError(f"active subject not found: {subject_id}")
            rows = connection.execute(
                """
                SELECT u.memory_id, u.latest_source_at, v.content, l.link_basis
                FROM subject_memory_links l
                JOIN memory_units u ON u.memory_id = l.memory_id AND u.lifecycle_status = 'active'
                JOIN memory_versions v ON v.memory_id = u.memory_id AND v.is_latest = 1
                WHERE l.subject_id = ? AND l.unlinked_at IS NULL
                ORDER BY u.memory_id
                """,
                (subject_id,),
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
            return SubjectSnapshot(
                subject_id,
                subject["name"],
                subject["summary"],
                subject["new_memory_count"],
                subject["summary_revision"],
                memories,
            )

    def provenance_episodes(
        self, memory_space_id: str, memory_ids: Sequence[str]
    ) -> dict[str, tuple[dict[str, object], ...]]:
        output: dict[str, tuple[dict[str, object], ...]] = {}
        with self._connect() as connection:
            for memory_id in memory_ids:
                owner = connection.execute(
                    "SELECT 1 FROM memory_units WHERE memory_id = ? AND memory_space_id = ?",
                    (memory_id, memory_space_id),
                ).fetchone()
                if owner is None:
                    raise NotFoundError(f"memory not found: {memory_id}")
                episodes = connection.execute(
                    """
                    SELECT ep.* FROM memory_versions v
                    JOIN memory_version_provenance p ON p.memory_version_id = v.memory_version_id
                    JOIN situational_episodes ep ON ep.episode_id = p.episode_id
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

    def commit_review(
        self,
        *,
        add_operation_id: str,
        memory_space_id: str,
        subject_id: str,
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
            subject = connection.execute(
                "SELECT summary_revision FROM subjects WHERE subject_id = ? AND memory_space_id = ? AND lifecycle_status = 'active'",
                (subject_id, memory_space_id),
            ).fetchone()
            if subject is None:
                raise NotFoundError(f"active subject not found: {subject_id}")
            if subject["summary_revision"] != expected_revision:
                raise ConcurrentUpdateError(f"stale subject: {subject_id}")
            current_ids = self._subject_memory_ids(connection, subject_id)
            changed_ids = {update.memory_id for update in updates}
            retired_ids = set(retirements)
            if not changed_ids | retired_ids <= current_ids:
                raise ValidationError("review references a memory outside the subject")
            if changed_ids & retired_ids:
                raise ValidationError("review cannot update and retire the same memory")
            self._insert_operation(
                connection,
                operation_id,
                memory_space_id,
                "review_subject",
                actor,
                config_signature,
                "Review all active memories in one subject.",
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
                UPDATE subjects SET new_memory_count = 0, updated_at = ?
                WHERE subject_id = ? AND summary_revision = ?
                """,
                (now, subject_id, expected_revision),
            )
            if cursor.rowcount != 1:
                raise ConcurrentUpdateError(f"stale subject: {subject_id}")
            self._effect(connection, operation_id, "subject", subject_id, "reviewed")
            self._insert_summary_refresh_targets(
                connection, add_operation_id, {subject_id}
            )
            self._validate_active_memory_links(connection, memory_space_id)
        return operation_id

    def commit_summary_refresh(
        self,
        *,
        add_operation_id: str,
        memory_space_id: str,
        snapshot: SubjectSnapshot,
        summary: str,
        summary_embedding: np.ndarray,
        signature_id: str,
        actor: str,
        config_signature: str,
    ) -> str:
        operation_id = str(uuid4())
        now = utc_milliseconds()
        with self._transaction() as connection:
            subject = connection.execute(
                """
                SELECT name, summary_revision FROM subjects
                WHERE subject_id = ? AND memory_space_id = ?
                    AND lifecycle_status = 'active'
                """,
                (snapshot.subject_id, memory_space_id),
            ).fetchone()
            if subject is None:
                raise NotFoundError(f"active subject not found: {snapshot.subject_id}")
            if subject["summary_revision"] != snapshot.summary_revision:
                raise ConcurrentUpdateError(f"stale subject: {snapshot.subject_id}")
            rows = connection.execute(
                """
                SELECT u.memory_id, u.latest_source_at, v.content, l.link_basis
                FROM subject_memory_links l
                JOIN memory_units u ON u.memory_id = l.memory_id
                    AND u.lifecycle_status = 'active'
                JOIN memory_versions v ON v.memory_id = u.memory_id AND v.is_latest = 1
                WHERE l.subject_id = ? AND l.unlinked_at IS NULL
                ORDER BY u.memory_id
                """,
                (snapshot.subject_id,),
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
            if (
                subject["name"] != snapshot.name
                or current_memories != supplied_memories
            ):
                raise ConcurrentUpdateError(
                    f"subject evidence changed: {snapshot.subject_id}"
                )
            self._insert_operation(
                connection,
                operation_id,
                memory_space_id,
                "refresh_subject_summary",
                actor,
                config_signature,
                "Rewrite a linked subject summary from all active memories.",
                now,
            )
            connection.execute(
                """
                UPDATE subjects SET summary = ?, summary_revision = summary_revision + 1,
                    updated_at = ? WHERE subject_id = ?
                """,
                (summary, now, snapshot.subject_id),
            )
            self._upsert_subject_embedding(
                connection,
                snapshot.subject_id,
                "name_summary",
                _subject_embedding_text(snapshot.name, summary),
                summary_embedding,
                signature_id,
                now,
            )
            self._effect(
                connection,
                operation_id,
                "subject",
                snapshot.subject_id,
                "summary_refreshed",
            )
            cursor = connection.execute(
                """
                UPDATE episode_summary_refresh_targets
                SET completed_by_operation_id = ?
                WHERE add_operation_id = ? AND subject_id = ?
                    AND completed_by_operation_id IS NULL
                """,
                (operation_id, add_operation_id, snapshot.subject_id),
            )
            if cursor.rowcount != 1:
                raise ConcurrentUpdateError(
                    f"summary refresh target is no longer pending: {snapshot.subject_id}"
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
            SELECT v.memory_version_id, v.version_no, v.content FROM memory_units u
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
            f"SELECT episode_id, source_started_at, source_ended_at FROM situational_episodes WHERE memory_space_id = ? AND episode_id IN ({_placeholders(update.provenance_episode_ids)})",
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
        latest_source_at = _latest_source_anchor(episode_rows)
        connection.execute(
            "UPDATE memory_units SET latest_source_at = ?, updated_at = ? WHERE memory_id = ?",
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
            UPDATE memory_units SET lifecycle_status = 'retired', retired_at = ?,
                retired_by_operation_id = ?, updated_at = ?
            WHERE memory_id = ? AND memory_space_id = ? AND lifecycle_status = 'active'
            """,
            (now, operation_id, now, memory_id, memory_space_id),
        )
        if cursor.rowcount != 1:
            raise NotFoundError(f"active memory not found: {memory_id}")
        connection.execute(
            """
            UPDATE subject_memory_links SET unlinked_at = ?, closed_by_operation_id = ?
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
        original: SubjectSnapshot,
        result: Literal["full_split", "partial_split"],
        new_subjects: Sequence[PreparedSplitSubject],
        signature_id: str,
        actor: str,
        config_signature: str,
    ) -> str:
        operation_id = str(uuid4())
        now = utc_milliseconds()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT name, summary_revision FROM subjects WHERE subject_id = ? AND memory_space_id = ? AND lifecycle_status = 'active'",
                (original.subject_id, memory_space_id),
            ).fetchone()
            if row is None:
                raise NotFoundError(f"active subject not found: {original.subject_id}")
            if row["summary_revision"] != original.summary_revision:
                raise ConcurrentUpdateError(f"stale subject: {original.subject_id}")
            current_ids = self._subject_memory_ids(connection, original.subject_id)
            input_ids = {memory.memory_id for memory in original.memories}
            if current_ids != input_ids:
                raise ConcurrentUpdateError(
                    f"subject membership changed: {original.subject_id}"
                )
            self._insert_operation(
                connection,
                operation_id,
                memory_space_id,
                "split_subject",
                actor,
                config_signature,
                f"Apply {result} to an over-capacity subject.",
                now,
            )
            moved_ids = {
                link.memory_id for prepared in new_subjects for link in prepared.links
            }
            if result == "full_split":
                close_ids = current_ids
                connection.execute(
                    """
                    DELETE FROM episode_summary_refresh_targets
                    WHERE add_operation_id = ? AND subject_id = ?
                    """,
                    (add_operation_id, original.subject_id),
                )
                connection.execute(
                    """
                    UPDATE subjects SET lifecycle_status = 'retired', retired_at = ?,
                        retired_by_operation_id = ?, updated_at = ?
                    WHERE subject_id = ?
                    """,
                    (now, operation_id, now, original.subject_id),
                )
                connection.execute(
                    "DELETE FROM subject_embeddings WHERE subject_id = ?",
                    (original.subject_id,),
                )
                self._effect(
                    connection, operation_id, "subject", original.subject_id, "retired"
                )
            else:
                close_ids = moved_ids
                connection.execute(
                    """
                    UPDATE subjects SET new_memory_count = 0, updated_at = ?
                    WHERE subject_id = ?
                    """,
                    (now, original.subject_id),
                )
                self._effect(
                    connection,
                    operation_id,
                    "subject",
                    original.subject_id,
                    "split_retained",
                )
            if close_ids:
                connection.execute(
                    f"""
                    UPDATE subject_memory_links SET unlinked_at = ?, closed_by_operation_id = ?
                    WHERE subject_id = ? AND unlinked_at IS NULL
                        AND memory_id IN ({_placeholders(tuple(close_ids))})
                    """,
                    (now, operation_id, original.subject_id, *close_ids),
                )
            for prepared in new_subjects:
                self._insert_subject(
                    connection,
                    memory_space_id,
                    prepared.subject,
                    operation_id,
                    signature_id,
                    now,
                )
                for link in prepared.links:
                    self._insert_link(connection, link, operation_id, now)
            self._insert_summary_refresh_targets(
                connection,
                add_operation_id,
                {prepared.subject.subject_id for prepared in new_subjects},
            )
            self._validate_active_memory_links(connection, memory_space_id)
        return operation_id

    def record_deferred_split(
        self,
        *,
        memory_space_id: str,
        subject_id: str,
        expected_revision: int,
        reason: str,
        actor: str,
        config_signature: str,
    ) -> str:
        operation_id = str(uuid4())
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT summary_revision FROM subjects WHERE subject_id = ? AND memory_space_id = ? AND lifecycle_status = 'active'",
                (subject_id, memory_space_id),
            ).fetchone()
            if row is None:
                raise NotFoundError(f"active subject not found: {subject_id}")
            if row["summary_revision"] != expected_revision:
                raise ConcurrentUpdateError(f"stale subject: {subject_id}")
            self._insert_operation(
                connection,
                operation_id,
                memory_space_id,
                "split_subject",
                actor,
                config_signature,
                reason,
                utc_milliseconds(),
            )
            self._effect(
                connection, operation_id, "subject", subject_id, "split_deferred"
            )
        return operation_id

    def public_search(
        self, memory_space_id: str, query: str, query_vector: np.ndarray
    ) -> SearchResult:
        subject_hits = self._scan_subjects(
            memory_space_id,
            query_vector,
            self.config.search_subject_top_k,
            self.config.search_subject_min_similarity,
        )
        memory_hits = self._scan_memories(
            memory_space_id,
            query_vector,
            self.config.search_memory_top_k,
            self.config.search_memory_min_similarity,
        )
        with self._connect() as connection:
            signature_id = self._active_signature(connection, memory_space_id)
            subject_refs: list[tuple[str, float, tuple[str, ...]]] = []
            memory_refs: list[tuple[str, float, tuple[str, ...]]] = []
            direct_subject_ids = tuple(item[0] for item in subject_hits)
            subject_order = list(direct_subject_ids)
            memory_order = [item[0] for item in memory_hits]
            for subject_id, similarity in subject_hits:
                rows = connection.execute(
                    """
                    SELECT u.memory_id, e.vector FROM subject_memory_links l
                    JOIN memory_units u ON u.memory_id = l.memory_id AND u.lifecycle_status = 'active'
                    JOIN memory_versions v ON v.memory_id = u.memory_id AND v.is_latest = 1
                    JOIN memory_embeddings e ON e.memory_version_id = v.memory_version_id AND e.model_signature_id = ?
                    WHERE l.subject_id = ? AND l.unlinked_at IS NULL
                    """,
                    (signature_id, subject_id),
                ).fetchall()
                attached = _top_embedded_ids(
                    rows,
                    query_vector,
                    "memory_id",
                    self.config.search_subject_attached_memory_k,
                )
                for memory_id in attached:
                    if memory_id not in memory_order:
                        memory_order.append(memory_id)
                subject_refs.append((subject_id, similarity, attached))
            for memory_id, similarity in memory_hits:
                rows = connection.execute(
                    """
                    SELECT s.subject_id, e.vector FROM subject_memory_links l
                    JOIN subjects s ON s.subject_id = l.subject_id AND s.lifecycle_status = 'active'
                    JOIN subject_embeddings e ON e.subject_id = s.subject_id
                        AND e.embedding_kind = 'name' AND e.model_signature_id = ?
                    WHERE l.memory_id = ? AND l.unlinked_at IS NULL
                    """,
                    (signature_id, memory_id),
                ).fetchall()
                attached = _top_embedded_ids(
                    rows,
                    query_vector,
                    "subject_id",
                    self.config.search_memory_attached_subject_k,
                )
                for subject_id in attached:
                    if subject_id not in subject_order:
                        subject_order.append(subject_id)
                memory_refs.append((memory_id, similarity, attached))
            subject_ids = set(subject_order)
            memory_ids = set(memory_order)
            subjects = self._load_search_subjects(
                connection, subject_order, set(direct_subject_ids)
            )
            memories = self._load_search_memories(connection, memory_order)
            links = self._load_search_links(connection, subject_ids, memory_ids)
        from fluxfold.models import RankedMemoryRef, RankedSubjectRef

        return SearchResult(
            query,
            subjects,
            memories,
            links,
            tuple(RankedSubjectRef(*item) for item in subject_refs),
            tuple(RankedMemoryRef(*item) for item in memory_refs),
        )

    def active_subject_ids(self, memory_space_id: str) -> tuple[str, ...]:
        with self._connect() as connection:
            return tuple(
                row[0]
                for row in connection.execute(
                    "SELECT subject_id FROM subjects WHERE memory_space_id = ? AND lifecycle_status = 'active' ORDER BY subject_id",
                    (memory_space_id,),
                )
            )

    def operation_active_subject_ids(
        self, memory_space_id: str, operation_id: str | None
    ) -> tuple[str, ...]:
        if operation_id is None:
            return ()
        with self._connect() as connection:
            return tuple(
                row[0]
                for row in connection.execute(
                    """
                    SELECT DISTINCT l.subject_id FROM subject_memory_links l
                    JOIN subjects s ON s.subject_id = l.subject_id
                        AND s.lifecycle_status = 'active'
                    WHERE s.memory_space_id = ? AND l.opened_by_operation_id = ?
                        AND l.unlinked_at IS NULL
                    ORDER BY l.subject_id
                    """,
                    (memory_space_id, operation_id),
                )
            )

    def pending_summary_refresh_subject_ids(
        self, memory_space_id: str, add_operation_id: str | None
    ) -> tuple[str, ...]:
        if add_operation_id is None:
            return ()
        with self._connect() as connection:
            return tuple(
                row[0]
                for row in connection.execute(
                    """
                    SELECT t.subject_id FROM episode_summary_refresh_targets t
                    JOIN subjects s ON s.subject_id = t.subject_id
                        AND s.lifecycle_status = 'active'
                    WHERE t.add_operation_id = ? AND s.memory_space_id = ?
                        AND t.completed_by_operation_id IS NULL
                    ORDER BY t.subject_id
                    """,
                    (add_operation_id, memory_space_id),
                )
            )

    def memory_active_link_counts(
        self, memory_space_id: str, memory_ids: Sequence[str]
    ) -> dict[str, int]:
        if not memory_ids:
            return {}
        ids = tuple(memory_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT u.memory_id, count(l.link_id) AS n FROM memory_units u
                LEFT JOIN subject_memory_links l ON l.memory_id = u.memory_id AND l.unlinked_at IS NULL
                WHERE u.memory_space_id = ? AND u.memory_id IN ({_placeholders(ids)})
                GROUP BY u.memory_id
                """,
                (memory_space_id, *ids),
            ).fetchall()
        if len(rows) != len(ids):
            raise ValidationError("link count requested for an invalid memory")
        return {row["memory_id"]: row["n"] for row in rows}

    def memory_active_direct_link_counts(
        self,
        memory_space_id: str,
        memory_ids: Sequence[str],
        *,
        excluding_subject_id: str,
    ) -> dict[str, int]:
        if not memory_ids:
            return {}
        ids = tuple(memory_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT u.memory_id,
                    count(CASE WHEN l.link_basis = 'direct' AND l.subject_id != ? THEN 1 END) AS n
                FROM memory_units u
                LEFT JOIN subject_memory_links l
                    ON l.memory_id = u.memory_id AND l.unlinked_at IS NULL
                WHERE u.memory_space_id = ? AND u.memory_id IN ({_placeholders(ids)})
                GROUP BY u.memory_id
                """,
                (excluding_subject_id, memory_space_id, *ids),
            ).fetchall()
        if len(rows) != len(ids):
            raise ValidationError("link count requested for an invalid memory")
        return {row["memory_id"]: row["n"] for row in rows}

    def memory_bank(self) -> tuple[MemoryBankSpace, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    ms.space_key AS space_key,
                    s.subject_id AS subject_id,
                    s.name AS name,
                    s.summary AS summary,
                    u.memory_id AS memory_id,
                    v.content AS content
                FROM memory_spaces ms
                LEFT JOIN subjects s
                    ON s.memory_space_id = ms.memory_space_id
                    AND s.lifecycle_status = 'active'
                LEFT JOIN subject_memory_links l
                    ON l.subject_id = s.subject_id AND l.unlinked_at IS NULL
                LEFT JOIN memory_units u
                    ON u.memory_id = l.memory_id AND u.lifecycle_status = 'active'
                LEFT JOIN memory_versions v
                    ON v.memory_id = u.memory_id AND v.is_latest = 1
                ORDER BY ms.space_key, s.name, s.subject_id, u.memory_id
                """
            ).fetchall()
        spaces: dict[str, dict[str, MemoryBankSubject]] = {}
        subject_order: dict[str, list[str]] = {}
        for row in rows:
            space_key = str(row["space_key"])
            spaces.setdefault(space_key, {})
            subject_order.setdefault(space_key, [])
            subject_id = row["subject_id"]
            if subject_id is None:
                continue
            current = spaces[space_key].get(subject_id)
            if current is None:
                current = MemoryBankSubject(row["name"], row["summary"], ())
                spaces[space_key][subject_id] = current
                subject_order[space_key].append(subject_id)
            if row["memory_id"] is not None:
                spaces[space_key][subject_id] = MemoryBankSubject(
                    current.name,
                    current.summary,
                    (*current.memory_contents, row["content"]),
                )
        return tuple(
            MemoryBankSpace(
                space_key,
                tuple(spaces[space_key][subject_id] for subject_id in subject_ids),
            )
            for space_key, subject_ids in subject_order.items()
        )

    def space_statistics(self, memory_space_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            source_chars = connection.execute(
                "SELECT coalesce(sum(source_chars), 0) FROM situational_episodes WHERE memory_space_id = ?",
                (memory_space_id,),
            ).fetchone()[0]
            episode_count = connection.execute(
                "SELECT count(*) FROM situational_episodes WHERE memory_space_id = ?",
                (memory_space_id,),
            ).fetchone()[0]
            memory_content_chars = connection.execute(
                """
                SELECT coalesce(sum(v.char_count), 0) FROM memory_units u
                JOIN memory_versions v ON v.memory_id = u.memory_id AND v.is_latest = 1
                WHERE u.memory_space_id = ? AND u.lifecycle_status = 'active'
                """,
                (memory_space_id,),
            ).fetchone()[0]
            summary_chars = connection.execute(
                "SELECT coalesce(sum(length(summary)), 0) FROM subjects WHERE memory_space_id = ? AND lifecycle_status = 'active'",
                (memory_space_id,),
            ).fetchone()[0]
            active_memories = connection.execute(
                "SELECT count(*) FROM memory_units WHERE memory_space_id = ? AND lifecycle_status = 'active'",
                (memory_space_id,),
            ).fetchone()[0]
            retired_memories = connection.execute(
                "SELECT count(*) FROM memory_units WHERE memory_space_id = ? AND lifecycle_status = 'retired'",
                (memory_space_id,),
            ).fetchone()[0]
            active_subjects = connection.execute(
                "SELECT count(*) FROM subjects WHERE memory_space_id = ? AND lifecycle_status = 'active'",
                (memory_space_id,),
            ).fetchone()[0]
            retired_subjects = connection.execute(
                "SELECT count(*) FROM subjects WHERE memory_space_id = ? AND lifecycle_status = 'retired'",
                (memory_space_id,),
            ).fetchone()[0]
            review_count = connection.execute(
                "SELECT count(*) FROM domain_operations WHERE memory_space_id = ? AND operation_type = 'review_subject'",
                (memory_space_id,),
            ).fetchone()[0]
            split_count = connection.execute(
                "SELECT count(*) FROM domain_operations WHERE memory_space_id = ? AND operation_type = 'split_subject'",
                (memory_space_id,),
            ).fetchone()[0]
            refresh_count = connection.execute(
                "SELECT count(*) FROM domain_operations WHERE memory_space_id = ? AND operation_type = 'refresh_subject_summary'",
                (memory_space_id,),
            ).fetchone()[0]
            rewritten_memories = connection.execute(
                """
                SELECT count(*) FROM memory_units u
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
                FROM memory_units u
                LEFT JOIN subject_memory_links l
                    ON l.memory_id = u.memory_id AND l.unlinked_at IS NULL
                LEFT JOIN subjects s
                    ON s.subject_id = l.subject_id AND s.lifecycle_status = 'active'
                WHERE u.memory_space_id = ? AND u.lifecycle_status = 'active'
                GROUP BY u.memory_id
                """,
                (memory_space_id,),
            ).fetchall()
            max_provenance_per_memory = connection.execute(
                """
                SELECT coalesce(max(n), 0) FROM (
                    SELECT count(p.episode_id) AS n
                    FROM memory_units u
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
            subject_rows = connection.execute(
                """
                SELECT
                    s.name AS name,
                    coalesce(sum(l.link_basis = 'direct'), 0) AS direct_links,
                    coalesce(sum(l.link_basis = 'contextual'), 0) AS contextual_links
                FROM subjects s
                LEFT JOIN subject_memory_links l
                    ON l.subject_id = s.subject_id AND l.unlinked_at IS NULL
                LEFT JOIN memory_units u
                    ON u.memory_id = l.memory_id AND u.lifecycle_status = 'active'
                WHERE s.memory_space_id = ? AND s.lifecycle_status = 'active'
                GROUP BY s.subject_id, s.name
                ORDER BY s.name, s.subject_id
                """,
                (memory_space_id,),
            ).fetchall()
        memory_chars = memory_content_chars + summary_chars
        rate = 0.0 if source_chars == 0 else 1 - memory_chars / source_chars
        link_counts = [
            row["direct_links"] + row["contextual_links"] for row in memory_link_rows
        ]
        memories_per_subject = [
            row["direct_links"] + row["contextual_links"] for row in subject_rows
        ]
        return {
            "source_chars": source_chars,
            "memory_chars": memory_chars,
            "memory_compression_rate": rate,
            "active_memories": active_memories,
            "active_subjects": active_subjects,
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
            "max_memories_per_subject": max(memories_per_subject, default=0),
            "mean_memories_per_subject": (
                0.0
                if not memories_per_subject
                else sum(memories_per_subject) / len(memories_per_subject)
            ),
            "max_provenance_per_memory": max_provenance_per_memory,
            "retired_memories": retired_memories,
            "retired_subjects": retired_subjects,
            "rewritten_memories": rewritten_memories,
            "subject_review_count": review_count,
            "subject_split_count": split_count,
            "subject_summary_refresh_count": refresh_count,
            "subjects": [
                {
                    "name": row["name"],
                    "direct_links": row["direct_links"],
                    "contextual_links": row["contextual_links"],
                }
                for row in subject_rows
            ],
        }

    def _load_search_subjects(
        self,
        connection: sqlite3.Connection,
        ids: Sequence[str],
        summary_ids: set[str],
    ) -> tuple[SearchSubject, ...]:
        if not ids:
            return ()
        stable_ids = tuple(ids)
        rows = connection.execute(
            f"SELECT subject_id, name, summary FROM subjects WHERE subject_id IN ({_placeholders(stable_ids)})",
            stable_ids,
        ).fetchall()
        by_id = {row["subject_id"]: row for row in rows}
        return tuple(
            SearchSubject(
                subject_id,
                by_id[subject_id]["name"],
                by_id[subject_id]["summary"] if subject_id in summary_ids else None,
            )
            for subject_id in stable_ids
        )

    def _load_search_memories(
        self, connection: sqlite3.Connection, ids: Sequence[str]
    ) -> tuple[SearchMemory, ...]:
        if not ids:
            return ()
        stable_ids = tuple(ids)
        rows = connection.execute(
            f"""
            SELECT u.memory_id, u.latest_source_at, v.content FROM memory_units u
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
            )
            for memory_id in stable_ids
        )

    def _load_search_links(
        self,
        connection: sqlite3.Connection,
        subject_ids: set[str],
        memory_ids: set[str],
    ) -> tuple[SearchLink, ...]:
        if not subject_ids or not memory_ids:
            return ()
        stable_subject_ids = tuple(sorted(subject_ids))
        stable_memory_ids = tuple(sorted(memory_ids))
        rows = connection.execute(
            f"""
            SELECT subject_id, memory_id, link_basis FROM subject_memory_links
            WHERE unlinked_at IS NULL
                AND subject_id IN ({_placeholders(stable_subject_ids)})
                AND memory_id IN ({_placeholders(stable_memory_ids)})
            ORDER BY subject_id, memory_id
            """,
            (*stable_subject_ids, *stable_memory_ids),
        ).fetchall()
        return tuple(
            SearchLink(row["subject_id"], row["memory_id"], row["link_basis"])
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

    def _subject_memory_ids(
        self, connection: sqlite3.Connection, subject_id: str
    ) -> set[str]:
        return {
            row[0]
            for row in connection.execute(
                """
                SELECT l.memory_id FROM subject_memory_links l
                JOIN memory_units u ON u.memory_id = l.memory_id AND u.lifecycle_status = 'active'
                WHERE l.subject_id = ? AND l.unlinked_at IS NULL
                """,
                (subject_id,),
            )
        }

    def _validate_active_memory_links(
        self, connection: sqlite3.Connection, memory_space_id: str
    ) -> None:
        invalid = connection.execute(
            """
            SELECT u.memory_id,
                sum(CASE WHEN l.unlinked_at IS NULL THEN 1 ELSE 0 END) AS active_links,
                sum(CASE WHEN l.unlinked_at IS NULL AND l.link_basis = 'direct' THEN 1 ELSE 0 END) AS direct_links
            FROM memory_units u
            LEFT JOIN subject_memory_links l ON l.memory_id = u.memory_id
            WHERE u.memory_space_id = ? AND u.lifecycle_status = 'active'
            GROUP BY u.memory_id
            HAVING active_links < 1 OR direct_links < 1 OR active_links > ?
            LIMIT 1
            """,
            (memory_space_id, self.config.memory_active_subject_link_max),
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
            INSERT INTO domain_operations(
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
            INSERT INTO domain_operation_effects(
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
        subject_ids: set[str],
    ) -> None:
        connection.executemany(
            """
            INSERT OR IGNORE INTO episode_summary_refresh_targets(
                add_operation_id, subject_id
            ) VALUES (?, ?)
            """,
            [(add_operation_id, subject_id) for subject_id in sorted(subject_ids)],
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


def _best_embedded_row(
    rows: Sequence[sqlite3.Row], query_vector: np.ndarray, id_field: str
) -> sqlite3.Row | None:
    if not rows:
        return None
    query = _normalized_query(query_vector)
    return min(
        rows,
        key=lambda row: (
            -float(np.frombuffer(row["vector"], dtype="<f4") @ query),
            row[id_field],
        ),
    )


def _top_embedded_ids(
    rows: Sequence[sqlite3.Row],
    query_vector: np.ndarray,
    id_field: str,
    top_k: int,
) -> tuple[str, ...]:
    query = _normalized_query(query_vector)
    ranked = sorted(
        (
            (row[id_field], float(np.frombuffer(row["vector"], dtype="<f4") @ query))
            for row in rows
        ),
        key=lambda item: (-item[1], item[0]),
    )
    return tuple(item[0] for item in ranked[:top_k])


def _subject_embedding_text(name: str, summary: str) -> str:
    return f"{name}\n\n{summary}"


def _placeholders(values: Sequence[object]) -> str:
    if not values:
        raise ValidationError("SQL value list must not be empty")
    return ",".join("?" for _ in values)


def _latest_source_anchor(rows: Iterable[sqlite3.Row]) -> int | None:
    anchors = [row["source_ended_at"] or row["source_started_at"] for row in rows]
    known = [anchor for anchor in anchors if anchor is not None]
    return max(known) if known else None
