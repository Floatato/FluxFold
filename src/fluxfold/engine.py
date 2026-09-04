"""Experimental FluxFold Memory Engine orchestration."""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast
from uuid import uuid4

import numpy as np
from pydantic import TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from fluxfold.config import FluxFoldConfig
from fluxfold.errors import (
    ConcurrentUpdateError,
    ErrorClass,
    ProviderError,
    StageFailure,
    ValidationError,
)
from fluxfold.models import (
    AddResult,
    AssociationSearchOutput,
    DeferSplitOutput,
    EmbeddingRebuildResult,
    ExtractionOutput,
    FullSplitOutput,
    LinkingOutput,
    LinkingStageOutput,
    MaintenanceResult,
    MemoriesExtraction,
    MemoryBankSpace,
    MemorySpace,
    NormalizedEpisode,
    ProvenanceRequestOutput,
    ReplaceContent,
    ReplaceProvenance,
    ReviewOutput,
    ReviewStageOutput,
    SearchResult,
    SplitStageOutput,
    SubjectSnapshot,
    SummaryRefreshOutput,
)
from fluxfold.prompts import (
    EXTRACTION_SYSTEM,
    LINKING_SYSTEM,
    REVIEW_SYSTEM,
    SPLIT_SYSTEM,
    SUMMARY_REFRESH_SYSTEM,
    extraction_input,
    linking_input,
    prompt_timestamp,
    repair_input,
    review_input,
    split_input,
    summary_refresh_input,
    validation_feedback,
)
from fluxfold.providers import (
    EmbeddingProvider,
    GenerationProvider,
    GenerationRequest,
    GenerationResponse,
    LocalMiniLMEmbeddingProvider,
)
from fluxfold.storage import (
    ExistingSubjectTouch,
    PersistedEpisode,
    PreparedLink,
    PreparedMemory,
    PreparedMemoryUpdate,
    PreparedSplitSubject,
    PreparedSubject,
    Store,
)

EventSink = Callable[[dict[str, object]], None]

LLM_IO_SAMPLE_QUOTAS: dict[str, int] = {
    "extract": 2,
    "link": 2,
    "review": 2,
    "split": 2,
    "summary": 2,
    "link_association_search": 1,
    "review_provenance": 1,
}

EPISODE_TERMINAL_ERROR_CLASSES = frozenset(
    {
        ErrorClass.CONTEXT_OVERFLOW,
        ErrorClass.POLICY_REJECTED,
        ErrorClass.INVALID_STRUCTURED_OUTPUT,
        ErrorClass.INCOMPLETE_OUTPUT,
    }
)


@dataclass(frozen=True, slots=True)
class _LlmTurn:
    first_attempt: bool
    stage: str
    system_prompt: str
    user_prompt: str
    output: str
    request_id: str | None


@dataclass(frozen=True, slots=True)
class _PreparedAdd:
    memory_space_id: str
    episode: NormalizedEpisode
    actor: str
    signature_id: str
    persisted: PersistedEpisode
    extraction: ExtractionOutput | None
    contents: tuple[str, ...]
    memory_refs: tuple[str, ...]
    memory_ids: tuple[str, ...]
    memory_vectors: tuple[Any, ...]
    prepared_memories: tuple[PreparedMemory, ...]


@dataclass(frozen=True, slots=True)
class _LinkingMetrics:
    candidate_memory_count: int
    association_search_called: bool
    association_additional_candidate_memory_count: int


class FluxFold:
    """Public experimental-version library façade."""

    def __init__(
        self,
        *,
        store: Store,
        generation_provider: GenerationProvider,
        embedding_provider: EmbeddingProvider,
        config: FluxFoldConfig,
        event_sink: EventSink | None,
        benchmark_seed: int | None,
    ) -> None:
        self._store = store
        self._generation = generation_provider
        self._embedding = embedding_provider
        self.config = config
        self._event_sink = event_sink
        self._benchmark_seed = benchmark_seed
        self._space_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._space_pipeline_failures: dict[str, StageFailure] = {}
        self._llm_io_sample_counts: Counter[str] = Counter()
        self._llm_io_sample_lock = threading.Lock()

    @classmethod
    async def open(
        cls,
        *,
        db_path: str,
        generation_provider: GenerationProvider,
        embedding_provider: EmbeddingProvider | None = None,
        config: FluxFoldConfig | None = None,
        event_sink: EventSink | None = None,
        benchmark_seed: int | None = None,
    ) -> FluxFold:
        """Open or initialize one SQLite-backed engine."""

        resolved = config or FluxFoldConfig()
        resolved_embedding = embedding_provider or LocalMiniLMEmbeddingProvider()
        return cls(
            store=Store(db_path, resolved),
            generation_provider=generation_provider,
            embedding_provider=resolved_embedding,
            config=resolved,
            event_sink=event_sink,
            benchmark_seed=benchmark_seed,
        )

    async def __aenter__(self) -> FluxFold:
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close both configured provider clients."""

        await asyncio.gather(self._generation.close(), self._embedding.close())

    async def create_or_open_space(self, space_key: str) -> MemorySpace:
        space = self._store.create_or_open_space(space_key)
        self._store.ensure_retrieval_signature(
            space.memory_space_id, self._embedding.model_info
        )
        return space

    async def delete_space(self, memory_space_id: str) -> None:
        async with self._space_locks[memory_space_id]:
            self._store.delete_space(memory_space_id)

    async def clear_spaces(self) -> None:
        self._store.clear_spaces()

    async def add_episode(
        self,
        memory_space_id: str,
        episode: NormalizedEpisode,
        *,
        actor: str = "dataset",
    ) -> AddResult:
        """Extract and add one complete normalized episode to a memory space."""

        try:
            async with self._space_locks[memory_space_id]:
                self._raise_if_pipeline_failed(memory_space_id)
                prepared = await self._prepare_add(
                    memory_space_id, episode, actor=actor
                )
                return await self._commit_prepared_add(prepared)
        except StageFailure as failure:
            self._handle_episode_failure(memory_space_id, episode, failure)
            raise
        except ProviderError as error:
            normalized = StageFailure("embedding", error.error_class, error.message)
            self._handle_episode_failure(memory_space_id, episode, normalized)
            raise normalized from error

    async def _prepare_add(
        self,
        memory_space_id: str,
        episode: NormalizedEpisode,
        *,
        actor: str = "dataset",
    ) -> _PreparedAdd:
        self._validate_episode(episode)
        signature_id = self._store.ensure_retrieval_signature(
            memory_space_id, self._embedding.model_info
        )
        persisted = self._store.persist_episode(memory_space_id, episode)
        if persisted.terminal_error_class is not None:
            raise StageFailure(
                "memory_extraction",
                ErrorClass(persisted.terminal_error_class),
                str(persisted.terminal_error_message),
            )
        if persisted.extraction_completed:
            return _PreparedAdd(
                memory_space_id,
                episode,
                actor,
                signature_id,
                persisted,
                None,
                (),
                (),
                (),
                (),
                (),
            )
        extraction, extraction_turn = await self._structured_output(
            stage="memory_extraction",
            system_prompt=EXTRACTION_SYSTEM,
            user_prompt=extraction_input(episode),
            adapter=TypeAdapter(ExtractionOutput),
            temperature=self.config.extraction_temperature,
            validator=self._validate_extraction,
        )
        extraction = cast(ExtractionOutput, extraction)
        self._offer_llm_sample("extract", extraction_turn)
        self._event(
            "extraction_completed",
            memory_space_id=memory_space_id,
            episode_id=persisted.episode_id,
            result=extraction.result,
            memories_extracted=(
                len(extraction.memories)
                if isinstance(extraction, MemoriesExtraction)
                else 0
            ),
        )
        contents = (
            tuple(memory.content for memory in extraction.memories)
            if isinstance(extraction, MemoriesExtraction)
            else ()
        )
        memory_refs = tuple(f"memory_{index + 1}" for index in range(len(contents)))
        memory_ids = tuple(str(uuid4()) for _ in contents)
        memory_vectors = await self._embed_documents(contents)
        prepared_memories = tuple(
            PreparedMemory(memory_id, str(uuid4()), content, vector)
            for memory_id, content, vector in zip(
                memory_ids, contents, memory_vectors, strict=True
            )
        )
        return _PreparedAdd(
            memory_space_id,
            episode,
            actor,
            signature_id,
            persisted,
            extraction,
            contents,
            memory_refs,
            memory_ids,
            memory_vectors,
            prepared_memories,
        )

    async def _commit_prepared_add(self, prepared: _PreparedAdd) -> AddResult:
        memory_space_id = prepared.memory_space_id
        episode = prepared.episode
        current = self._store.persist_episode(memory_space_id, episode)
        if current.extraction_completed:
            replay_subject_ids = self._store.operation_active_subject_ids(
                memory_space_id, current.completed_operation_id
            )
            maintenance = await self._complete_maintenance(
                memory_space_id,
                cast(str, current.completed_operation_id),
                replay_subject_ids,
                prepared.signature_id,
                prepared.actor,
            )
            return AddResult(
                current.episode_id,
                current.completed_operation_id,
                True,
                0,
                0,
                0,
                tuple(maintenance),
            )
        if prepared.extraction is None:
            raise ConcurrentUpdateError(
                "episode completion disappeared after preparation"
            )
        extraction = prepared.extraction
        contents = prepared.contents
        memory_refs = prepared.memory_refs
        memory_ids = prepared.memory_ids
        memory_vectors = prepared.memory_vectors
        prepared_memories = prepared.prepared_memories
        if contents:
            linking, linking_metrics = await self._link_memories(
                memory_space_id,
                memory_refs,
                contents,
                memory_vectors,
            )
        else:
            linking = LinkingOutput(new_subjects=[], links=[])
            linking_metrics = _LinkingMetrics(0, False, 0)
        self._event(
            "subject_linking_completed",
            memory_space_id=memory_space_id,
            episode_id=current.episode_id,
            new_subjects=len(linking.new_subjects),
            links=len(linking.links),
            candidate_memory_count=linking_metrics.candidate_memory_count,
            association_search_called=linking_metrics.association_search_called,
            association_additional_candidate_memory_count=(
                linking_metrics.association_additional_candidate_memory_count
            ),
        )
        (
            prepared_subjects,
            prepared_links,
            subject_touches,
            affected_subject_ids,
        ) = await self._prepare_link_commit(
            memory_space_id,
            memory_refs,
            memory_ids,
            linking,
        )
        operation_id = self._store.commit_add(
            memory_space_id=memory_space_id,
            episode_id=current.episode_id,
            input_hash=episode.content_hash,
            memories=prepared_memories,
            subjects=prepared_subjects,
            links=prepared_links,
            subject_touches=subject_touches,
            signature_id=prepared.signature_id,
            actor=prepared.actor,
            config_signature=self.config.signature,
        )
        self._event(
            "episode_committed",
            memory_space_id=memory_space_id,
            episode_id=current.episode_id,
            operation_id=operation_id,
            memories_created=len(prepared_memories),
            subjects_created=len(prepared_subjects),
            links_created=len(prepared_links),
        )
        self._event(
            "audit_episode_decision",
            memory_space_id=memory_space_id,
            episode_id=current.episode_id,
            operation_id=operation_id,
            extraction=self._audit_extraction(
                memory_space_id, extraction, memory_refs, contents, linking
            ),
            new_subjects=[
                {
                    **subject.model_dump(mode="json"),
                    "subject_id": prepared_subject.subject_id,
                }
                for subject, prepared_subject in zip(
                    linking.new_subjects, prepared_subjects, strict=True
                )
            ],
        )
        maintenance = await self._complete_maintenance(
            memory_space_id,
            operation_id,
            sorted(affected_subject_ids),
            prepared.signature_id,
            prepared.actor,
        )
        return AddResult(
            current.episode_id,
            operation_id,
            current.replayed,
            len(prepared_memories),
            len(prepared_subjects),
            len(prepared_links),
            tuple(maintenance),
        )

    def _handle_episode_failure(
        self,
        memory_space_id: str,
        episode: NormalizedEpisode,
        failure: StageFailure,
    ) -> None:
        persisted = self._store.persist_episode(memory_space_id, episode)
        if persisted.extraction_completed:
            self._space_pipeline_failures[memory_space_id] = failure
            return
        if failure.error_class not in EPISODE_TERMINAL_ERROR_CLASSES:
            self._space_pipeline_failures[memory_space_id] = failure
            return
        self._store.record_episode_terminal_failure(
            memory_space_id=memory_space_id,
            episode_id=persisted.episode_id,
            input_hash=episode.content_hash,
            config_signature=self.config.signature,
            error_class=failure.error_class.value,
            error_message=failure.message,
        )
        if persisted.terminal_error_class is None:
            self._event(
                "episode_terminal_failure",
                severity="error",
                memory_space_id=memory_space_id,
                source_sequence=episode.source_sequence,
                error_class=failure.error_class.value,
                reason=failure.message,
            )

    def _raise_if_pipeline_failed(self, memory_space_id: str) -> None:
        failure = self._space_pipeline_failures.get(memory_space_id)
        if failure is not None:
            raise failure

    async def search(self, memory_space_id: str, query: str) -> SearchResult:
        """Search active subjects and memories using only exact vector retrieval."""

        if not isinstance(query, str):
            raise ValidationError("query must be a string")
        self._store.get_space(memory_space_id)
        vector = (await self._embed_queries([query]))[0]
        return self._store.public_search(memory_space_id, query, vector)

    async def rebuild_retrieval_embeddings(
        self, memory_space_id: str
    ) -> EmbeddingRebuildResult:
        """Rebuild all active retrieval vectors and atomically switch signatures."""

        async with self._space_locks[memory_space_id]:
            return await self._rebuild_retrieval_embeddings_locked(memory_space_id)

    async def _rebuild_retrieval_embeddings_locked(
        self, memory_space_id: str
    ) -> EmbeddingRebuildResult:
        self._store.get_space(memory_space_id)
        signature_id = self._store.prepare_retrieval_signature(
            self._embedding.model_info
        )
        memory_sources, subject_sources = self._store.retrieval_embedding_sources(
            memory_space_id
        )
        vectors = await self._embed_documents(
            [source.content for source in memory_sources]
            + [source.name for source in subject_sources]
            + [
                _name_summary(source.name, source.summary)
                for source in subject_sources
                if source.summary is not None
            ]
        )
        memory_vectors = vectors[: len(memory_sources)]
        subject_vectors = iter(vectors[len(memory_sources) :])
        name_vectors = [next(subject_vectors) for _ in subject_sources]
        summary_vectors = iter(subject_vectors)
        prepared_subjects = [
            (
                source,
                name_vector,
                next(summary_vectors) if source.summary is not None else None,
            )
            for source, name_vector in zip(subject_sources, name_vectors, strict=True)
        ]
        self._store.commit_retrieval_embedding_switch(
            memory_space_id=memory_space_id,
            signature_id=signature_id,
            memories=tuple(zip(memory_sources, memory_vectors, strict=True)),
            subjects=prepared_subjects,
        )
        return EmbeddingRebuildResult(
            memory_space_id,
            signature_id,
            len(memory_sources),
            len(subject_sources),
        )

    def space_statistics(self, memory_space_id: str) -> dict[str, Any]:
        """Return the experiment metrics derived from final database state."""

        return self._store.space_statistics(memory_space_id)

    def memory_bank(self) -> tuple[MemoryBankSpace, ...]:
        """Return active subjects and memories of every space for inspect logs."""

        return self._store.memory_bank()

    async def _link_memories(
        self,
        memory_space_id: str,
        memory_refs: Sequence[str],
        contents: Sequence[str],
        vectors: Sequence[Any],
    ) -> tuple[LinkingOutput, _LinkingMetrics]:
        candidates, legal_subject_ids = self._recall_initial_subject_candidates(
            memory_space_id, vectors
        )
        legal_by_memory = {
            memory_ref: set(legal_subject_ids) for memory_ref in memory_refs
        }
        new_memories = [
            {"memory_ref": memory_ref, "content": content}
            for memory_ref, content in zip(memory_refs, contents, strict=True)
        ]
        association_results: list[dict[str, Any]] = []
        association_memory_ids: set[str] = set()
        turns: list[_LlmTurn] = []
        max_searches = (
            self.config.association_search_max_calls
            if self.config.association_search_enabled
            else 0
        )

        while True:
            searches_remaining = max_searches - len(association_results)

            def validate(value: LinkingStageOutput) -> None:
                if isinstance(value, AssociationSearchOutput):
                    if searches_remaining == 0:
                        raise ValidationError(
                            "association_search call limit has been reached"
                        )
                    if not value.query.strip():
                        raise ValidationError(
                            "association_search query must not be blank"
                        )
                    return
                self._validate_linking(value, memory_refs, legal_by_memory)

            output, turn = await self._structured_output(
                stage="subject_linking",
                system_prompt=LINKING_SYSTEM,
                user_prompt=linking_input(
                    new_memories,
                    candidates,
                    association_results,
                    searches_remaining,
                ),
                adapter=TypeAdapter(LinkingStageOutput),
                temperature=self.config.linking_temperature,
                validator=validate,
            )
            turns.append(turn)
            output = cast(LinkingStageOutput, output)
            if isinstance(output, LinkingOutput):
                sample_kind = (
                    "link_association_search" if association_results else "link"
                )
                self._offer_llm_sample(sample_kind, *turns)
                return output, _LinkingMetrics(
                    len(association_memory_ids),
                    bool(association_results),
                    len(association_memory_ids),
                )

            association_vector = (await self._embed_queries([output.query]))[0]
            association_candidates, association_legal = (
                self._recall_association_candidates(
                    memory_space_id,
                    memory_refs,
                    [association_vector] * len(memory_refs),
                )
            )
            association_memory_ids.update(_candidate_memory_ids(association_candidates))
            for memory_ref, subject_ids in association_legal.items():
                legal_by_memory[memory_ref].update(subject_ids)
            association_results.append(
                {
                    "query": output.query,
                    "candidates_by_memory": association_candidates,
                }
            )

    def _recall_initial_subject_candidates(
        self,
        memory_space_id: str,
        vectors: Sequence[Any],
    ) -> tuple[list[dict[str, str]], set[str]]:
        scores: dict[str, float] = {}
        names: dict[str, str] = {}
        direct_order: list[str] = []
        for vector in vectors:
            subjects = self._store.candidate_subject_names(
                memory_space_id,
                vector,
                top_k=self.config.subject_candidate_top_k,
                min_similarity=self.config.subject_candidate_min_similarity,
            )
            for index, subject in enumerate(subjects):
                names[subject.subject_id] = subject.name
                scores[subject.subject_id] = max(
                    scores.get(subject.subject_id, -1.0), subject.similarity
                )
                if (
                    index < self.config.subject_candidate_direct_top_k
                    and subject.subject_id not in direct_order
                ):
                    direct_order.append(subject.subject_id)

        pooled = sorted(
            scores, key=lambda subject_id: (-scores[subject_id], subject_id)
        )[: self.config.subject_candidate_pool_top_k]
        final_ids = list(dict.fromkeys([*direct_order, *pooled]))
        return (
            [
                {"subject_id": subject_id, "name": names[subject_id]}
                for subject_id in final_ids
            ],
            set(final_ids),
        )

    def _recall_association_candidates(
        self,
        memory_space_id: str,
        memory_refs: Sequence[str],
        vectors: Sequence[Any],
    ) -> tuple[dict[str, dict[str, object]], dict[str, set[str]]]:
        candidates: dict[str, dict[str, object]] = {}
        legal: dict[str, set[str]] = {}
        for memory_ref, vector in zip(memory_refs, vectors, strict=True):
            subjects = self._store.candidate_subjects(
                memory_space_id,
                vector,
                top_k=self.config.association_subject_candidate_top_k,
                min_similarity=self.config.subject_candidate_min_similarity,
            )
            memories = self._store.candidate_memories(
                memory_space_id,
                vector,
                top_k=self.config.memory_candidate_top_k,
                min_similarity=self.config.memory_candidate_min_similarity,
            )
            groups: dict[str, dict[str, object]] = {}
            for subject in subjects:
                group: dict[str, object] = {
                    "subject_id": subject.subject_id,
                    "name": subject.name,
                    "subject_similarity": subject.similarity,
                    "memory_hits": [],
                }
                if subject.attached_memory_id is not None:
                    cast(list[dict[str, object]], group["memory_hits"]).append(
                        {
                            "memory_id": subject.attached_memory_id,
                            "content": subject.attached_memory_content,
                            "source": "subject_channel_attachment",
                        }
                    )
                groups[subject.subject_id] = group
            unattached_memories: list[dict[str, object]] = []
            for memory in memories:
                memory_value: dict[str, object] = {
                    "memory_id": memory.memory_id,
                    "content": memory.content,
                    "last_mentioned_at": prompt_timestamp(memory.latest_source_at),
                    "similarity": memory.similarity,
                    "source": "memory_channel",
                }
                subject_id = memory.attached_subject_id
                if subject_id is None:
                    unattached_memories.append(memory_value)
                    continue
                if subject_id not in groups:
                    groups[subject_id] = {
                        "subject_id": subject_id,
                        "name": memory.attached_subject_name,
                        "subject_similarity": None,
                        "memory_hits": [],
                    }
                hits = cast(list[dict[str, object]], groups[subject_id]["memory_hits"])
                if not any(hit["memory_id"] == memory.memory_id for hit in hits):
                    hits.append(memory_value)
            legal_ids = set(groups)
            candidates[memory_ref] = {
                "subjects": list(groups.values()),
                "unattached_memories": unattached_memories,
            }
            legal[memory_ref] = legal_ids
        return candidates, legal

    def _validate_linking(
        self,
        output: LinkingOutput,
        memory_refs: Sequence[str],
        legal_by_memory: dict[str, set[str]],
    ) -> None:
        expected = set(memory_refs)
        subject_refs = [subject.subject_ref for subject in output.new_subjects]
        if len(subject_refs) != len(set(subject_refs)):
            raise ValidationError("new subject_ref values must be unique")
        new_refs = set(subject_refs)
        referenced_new: set[str] = set()
        seen_pairs: set[tuple[str, str, str]] = set()
        by_memory: defaultdict[str, list[Any]] = defaultdict(list)
        for link in output.links:
            if link.memory_ref not in expected:
                raise ValidationError(f"unknown memory_ref: {link.memory_ref}")
            if link.subject.kind == "existing":
                target = link.subject.subject_id
                if target not in legal_by_memory[link.memory_ref]:
                    raise ValidationError(
                        f"subject {target} was not a candidate for {link.memory_ref}"
                    )
                target_kind = "existing"
            else:
                target = link.subject.subject_ref
                if target not in new_refs:
                    raise ValidationError(f"unknown new subject_ref: {target}")
                referenced_new.add(target)
                target_kind = "new"
            pair = (link.memory_ref, target_kind, target)
            if pair in seen_pairs:
                raise ValidationError("duplicate memory-subject link")
            seen_pairs.add(pair)
            by_memory[link.memory_ref].append(link)
        if referenced_new != new_refs:
            raise ValidationError("every new subject must be linked to a batch memory")
        for memory_ref in expected:
            memory_links = by_memory[memory_ref]
            if not memory_links or not any(
                link.basis == "direct" for link in memory_links
            ):
                raise ValidationError(f"{memory_ref} needs at least one direct link")
            if len(memory_links) > self.config.memory_active_subject_link_max:
                raise ValidationError(f"{memory_ref} exceeds the active link maximum")
        for subject in output.new_subjects:
            self._validate_subject_name(subject.name)

    async def _prepare_link_commit(
        self,
        memory_space_id: str,
        memory_refs: Sequence[str],
        memory_ids: Sequence[str],
        linking: LinkingOutput,
    ) -> tuple[
        list[PreparedSubject],
        list[PreparedLink],
        list[ExistingSubjectTouch],
        set[str],
    ]:
        ref_to_memory_id = dict(zip(memory_refs, memory_ids, strict=True))
        new_ref_to_id = {
            subject.subject_ref: str(uuid4()) for subject in linking.new_subjects
        }
        subject_vectors = iter(
            await self._embed_documents(
                [subject.name for subject in linking.new_subjects]
            )
        )
        prepared_subjects = [
            PreparedSubject(
                new_ref_to_id[subject.subject_ref],
                subject.name,
                None,
                next(subject_vectors),
                None,
            )
            for subject in linking.new_subjects
        ]
        prepared_links: list[PreparedLink] = []
        existing_link_counts: Counter[str] = Counter()
        affected: set[str] = set()
        for link in linking.links:
            if link.subject.kind == "existing":
                subject_id = link.subject.subject_id
                existing_link_counts[subject_id] += 1
            else:
                subject_id = new_ref_to_id[link.subject.subject_ref]
            prepared_links.append(
                PreparedLink(ref_to_memory_id[link.memory_ref], subject_id, link.basis)
            )
            affected.add(subject_id)
        touches: list[ExistingSubjectTouch] = []
        for subject_id, increment in sorted(existing_link_counts.items()):
            snapshot = self._store.subject_snapshot(memory_space_id, subject_id)
            touches.append(
                ExistingSubjectTouch(subject_id, snapshot.summary_revision, increment)
            )
        return (
            prepared_subjects,
            prepared_links,
            touches,
            affected,
        )

    async def _complete_maintenance(
        self,
        memory_space_id: str,
        add_operation_id: str,
        affected_subject_ids: Sequence[str],
        signature_id: str,
        actor: str,
    ) -> list[MaintenanceResult]:
        outcomes: list[MaintenanceResult] = []
        for subject_id in affected_subject_ids:
            try:
                subject_outcomes = await self._maintain_subject(
                    memory_space_id,
                    add_operation_id,
                    subject_id,
                    signature_id,
                    actor,
                )
                outcomes.extend(subject_outcomes)
            except StageFailure as error:
                self._handle_maintenance_failure(memory_space_id, subject_id, error)
            except ProviderError as error:
                self._handle_maintenance_failure(
                    memory_space_id,
                    subject_id,
                    StageFailure(
                        "maintenance_embedding", error.error_class, error.message
                    ),
                )

        refresh_subject_ids = self._store.pending_summary_refresh_subject_ids(
            memory_space_id, add_operation_id
        )
        semaphore = asyncio.Semaphore(
            self.config.subject_summary_refresh_concurrency_per_episode
        )

        async def refresh_one(
            subject_id: str,
        ) -> MaintenanceResult | StageFailure:
            async with semaphore:
                try:
                    return await self._refresh_subject_summary(
                        memory_space_id,
                        add_operation_id,
                        subject_id,
                        signature_id,
                        actor,
                    )
                except StageFailure as error:
                    self._record_summary_refresh_failure(
                        memory_space_id, subject_id, error
                    )
                    return error
                except ProviderError as error:
                    failure = StageFailure(
                        "maintenance_embedding", error.error_class, error.message
                    )
                    self._record_summary_refresh_failure(
                        memory_space_id, subject_id, failure
                    )
                    return failure

        refresh_results = await asyncio.gather(
            *(refresh_one(subject_id) for subject_id in refresh_subject_ids)
        )
        failures: list[StageFailure] = []
        for result in refresh_results:
            if isinstance(result, StageFailure):
                failures.append(result)
            else:
                outcomes.append(result)
        if failures:
            raise failures[0]
        return outcomes

    def _record_summary_refresh_failure(
        self, memory_space_id: str, subject_id: str, error: StageFailure
    ) -> None:
        self._event(
            "subject_summary_refresh_failed",
            severity="error",
            memory_space_id=memory_space_id,
            subject_id=subject_id,
            error_class=error.error_class.value,
            reason=error.message,
        )

    def _handle_maintenance_failure(
        self, memory_space_id: str, subject_id: str, error: StageFailure
    ) -> None:
        if error.error_class not in {
            ErrorClass.CONTEXT_OVERFLOW,
            ErrorClass.POLICY_REJECTED,
            ErrorClass.INVALID_STRUCTURED_OUTPUT,
            ErrorClass.INCOMPLETE_OUTPUT,
        }:
            raise error
        self._event(
            "maintenance_terminal_failure",
            severity="error",
            memory_space_id=memory_space_id,
            subject_id=subject_id,
            error_class=error.error_class.value,
            reason=error.message,
        )

    async def _refresh_subject_summary(
        self,
        memory_space_id: str,
        add_operation_id: str,
        subject_id: str,
        signature_id: str,
        actor: str,
    ) -> MaintenanceResult:
        snapshot = self._store.subject_snapshot(memory_space_id, subject_id)
        self._event(
            "subject_summary_refresh_started",
            memory_space_id=memory_space_id,
            subject_id=subject_id,
            memory_count=len(snapshot.memories),
        )
        output, summary_turn = await self._structured_output(
            stage="subject_summary_refresh",
            system_prompt=SUMMARY_REFRESH_SYSTEM,
            user_prompt=summary_refresh_input(snapshot),
            adapter=TypeAdapter(SummaryRefreshOutput),
            temperature=self.config.review_temperature,
            validator=lambda value: self._validate_summary(value.summary),
        )
        output = cast(SummaryRefreshOutput, output)
        self._offer_llm_sample("summary", summary_turn)
        vector = (
            await self._embed_documents([_name_summary(snapshot.name, output.summary)])
        )[0]
        operation_id = self._store.commit_summary_refresh(
            add_operation_id=add_operation_id,
            memory_space_id=memory_space_id,
            snapshot=snapshot,
            summary=output.summary,
            summary_embedding=vector,
            signature_id=signature_id,
            actor=actor,
            config_signature=self.config.signature,
        )
        self._event(
            "subject_summary_refresh_completed",
            memory_space_id=memory_space_id,
            subject_id=subject_id,
            operation_id=operation_id,
        )
        self._event(
            "audit_subject_summary_refresh",
            memory_space_id=memory_space_id,
            subject_id=subject_id,
            operation_id=operation_id,
            evidence={
                "name": snapshot.name,
                "memories": [asdict(memory) for memory in snapshot.memories],
            },
            decision=output.model_dump(mode="json"),
        )
        return MaintenanceResult(subject_id, "summary_refresh", operation_id)

    async def _maintain_subject(
        self,
        memory_space_id: str,
        add_operation_id: str,
        subject_id: str,
        signature_id: str,
        actor: str,
    ) -> list[MaintenanceResult]:
        snapshot = self._store.subject_snapshot(memory_space_id, subject_id)
        memory_chars = sum(len(memory.content) for memory in snapshot.memories)
        split_due = (
            len(snapshot.memories) >= self.config.subject_split_memory_count_threshold
            or memory_chars >= self.config.subject_split_total_memory_chars_threshold
        )
        outcomes: list[MaintenanceResult] = []
        if split_due:
            self._event(
                "subject_split_started",
                memory_space_id=memory_space_id,
                subject_id=subject_id,
                memory_count=len(snapshot.memories),
                memory_chars=memory_chars,
            )
            split = await self._split_subject(
                memory_space_id,
                add_operation_id,
                snapshot,
                signature_id,
                actor,
            )
            outcomes.append(split)
            self._event(
                "subject_split_completed",
                memory_space_id=memory_space_id,
                subject_id=subject_id,
                operation_id=split.operation_id,
                result=split.operation,
                reason=split.reason,
            )
            if split.operation != "defer_split":
                return outcomes
            snapshot = self._store.subject_snapshot(memory_space_id, subject_id)
        if snapshot.new_memory_count >= self.config.subject_review_new_memory_threshold:
            self._event(
                "subject_review_started",
                memory_space_id=memory_space_id,
                subject_id=subject_id,
                memory_count=len(snapshot.memories),
            )
            operation_id, provenance_viewed = await self._review_subject(
                memory_space_id,
                add_operation_id,
                snapshot,
                signature_id,
                actor,
            )
            outcomes.append(MaintenanceResult(subject_id, "review", operation_id))
            self._event(
                "subject_review_completed",
                memory_space_id=memory_space_id,
                subject_id=subject_id,
                operation_id=operation_id,
                provenance_viewed=provenance_viewed,
            )
        return outcomes

    async def _review_subject(
        self,
        memory_space_id: str,
        add_operation_id: str,
        snapshot: SubjectSnapshot,
        signature_id: str,
        actor: str,
    ) -> tuple[str, bool]:
        current_ids = {memory.memory_id for memory in snapshot.memories}

        def validate_first(value: ReviewStageOutput) -> None:
            if isinstance(value, ProvenanceRequestOutput):
                requested = value.memory_ids
                if (
                    not requested
                    or len(requested) > self.config.review_provenance_memory_max
                ):
                    raise ValidationError("provenance request size is invalid")
                if (
                    len(set(requested)) != len(requested)
                    or not set(requested) <= current_ids
                ):
                    raise ValidationError(
                        "provenance request references invalid memories"
                    )
                return
            self._validate_review(value, snapshot)

        first, first_turn = await self._structured_output(
            stage="subject_review",
            system_prompt=REVIEW_SYSTEM,
            user_prompt=review_input(snapshot),
            adapter=TypeAdapter(ReviewStageOutput),
            temperature=self.config.review_temperature,
            validator=validate_first,
        )
        first = cast(ReviewStageOutput, first)
        if isinstance(first, ProvenanceRequestOutput):
            provenance_viewed = True
            provenance = self._store.provenance_episodes(
                memory_space_id, first.memory_ids
            )

            def validate_final(value: ReviewStageOutput) -> None:
                if isinstance(value, ProvenanceRequestOutput):
                    raise ValidationError("provenance may only be requested once")
                self._validate_review(value, snapshot)

            final, final_turn = await self._structured_output(
                stage="subject_review",
                system_prompt=REVIEW_SYSTEM,
                user_prompt=review_input(snapshot, provenance),
                adapter=TypeAdapter(ReviewStageOutput),
                temperature=self.config.review_temperature,
                validator=validate_final,
            )
            self._offer_llm_sample("review_provenance", first_turn, final_turn)
            review = cast(ReviewOutput, final)
        else:
            provenance_viewed = False
            self._offer_llm_sample("review", first_turn)
            review = first
        snapshots = {memory.memory_id: memory for memory in snapshot.memories}
        prepared_values: list[tuple[str, str, tuple[str, ...]]] = []
        for update in review.updates:
            old = snapshots[update.memory_id]
            content = (
                update.content_change.content
                if isinstance(update.content_change, ReplaceContent)
                else old.content
            )
            provenance_ids = (
                tuple(update.provenance_change.episode_ids)
                if isinstance(update.provenance_change, ReplaceProvenance)
                else old.provenance_episode_ids
            )
            prepared_values.append((update.memory_id, content, provenance_ids))
        vectors = await self._embed_documents(
            [content for _, content, _ in prepared_values]
        )
        updates = [
            PreparedMemoryUpdate(memory_id, str(uuid4()), content, provenance, vector)
            for (memory_id, content, provenance), vector in zip(
                prepared_values, vectors, strict=True
            )
        ]
        operation_id = self._store.commit_review(
            add_operation_id=add_operation_id,
            memory_space_id=memory_space_id,
            subject_id=snapshot.subject_id,
            expected_revision=snapshot.summary_revision,
            updates=updates,
            retirements=review.retirements,
            signature_id=signature_id,
            actor=actor,
            config_signature=self.config.signature,
        )
        self._event(
            "audit_subject_review",
            memory_space_id=memory_space_id,
            subject_id=snapshot.subject_id,
            operation_id=operation_id,
            before=asdict(snapshot),
            decision=review.model_dump(mode="json"),
        )
        return operation_id, provenance_viewed

    def _validate_review(self, review: ReviewOutput, snapshot: SubjectSnapshot) -> None:
        current = {memory.memory_id: memory for memory in snapshot.memories}
        update_ids = [update.memory_id for update in review.updates]
        if len(update_ids) != len(set(update_ids)):
            raise ValidationError("review contains duplicate memory updates")
        if len(review.retirements) != len(set(review.retirements)):
            raise ValidationError("review contains duplicate retirements")
        if not set(update_ids) | set(review.retirements) <= set(current):
            raise ValidationError("review references a memory outside the subject")
        if set(update_ids) & set(review.retirements):
            raise ValidationError("review cannot update and retire the same memory")
        for update in review.updates:
            old = current[update.memory_id]
            content_changed = isinstance(update.content_change, ReplaceContent)
            provenance_changed = isinstance(update.provenance_change, ReplaceProvenance)
            if not content_changed and not provenance_changed:
                raise ValidationError(
                    "a review update must change content or provenance"
                )
            if content_changed:
                replacement_content = cast(
                    ReplaceContent, update.content_change
                ).content
                self._validate_memory_content(replacement_content)
                if replacement_content == old.content and not provenance_changed:
                    raise ValidationError("replacement content is unchanged")
            if provenance_changed:
                episodes = cast(ReplaceProvenance, update.provenance_change).episode_ids
                if not 1 <= len(episodes) <= self.config.memory_provenance_episode_max:
                    raise ValidationError("replacement provenance size is invalid")
                if len(episodes) != len(set(episodes)):
                    raise ValidationError("replacement provenance contains duplicates")
                if (
                    tuple(sorted(episodes)) == tuple(sorted(old.provenance_episode_ids))
                    and not content_changed
                ):
                    raise ValidationError("replacement provenance is unchanged")

    async def _split_subject(
        self,
        memory_space_id: str,
        add_operation_id: str,
        snapshot: SubjectSnapshot,
        signature_id: str,
        actor: str,
    ) -> MaintenanceResult:
        def validate(value: SplitStageOutput) -> None:
            self._validate_split(memory_space_id, value, snapshot)

        output, split_turn = await self._structured_output(
            stage="subject_split",
            system_prompt=SPLIT_SYSTEM,
            user_prompt=split_input(snapshot),
            adapter=TypeAdapter(SplitStageOutput),
            temperature=self.config.split_temperature,
            validator=validate,
        )
        output = cast(SplitStageOutput, output)
        self._offer_llm_sample("split", split_turn)
        if isinstance(output, DeferSplitOutput):
            operation_id = self._store.record_deferred_split(
                memory_space_id=memory_space_id,
                subject_id=snapshot.subject_id,
                expected_revision=snapshot.summary_revision,
                reason=output.reason,
                actor=actor,
                config_signature=self.config.signature,
            )
            self._event(
                "audit_subject_split",
                memory_space_id=memory_space_id,
                subject_id=snapshot.subject_id,
                operation_id=operation_id,
                before=asdict(snapshot),
                decision=output.model_dump(mode="json"),
            )
            return MaintenanceResult(
                snapshot.subject_id, "defer_split", operation_id, output.reason
            )
        subjects = (
            output.subjects
            if isinstance(output, FullSplitOutput)
            else output.new_subjects
        )
        embedding_texts = [subject.name for subject in subjects]
        vectors = iter(await self._embed_documents(embedding_texts))
        prepared: list[PreparedSplitSubject] = []
        for subject in subjects:
            subject_id = str(uuid4())
            prepared.append(
                PreparedSplitSubject(
                    PreparedSubject(
                        subject_id,
                        subject.name,
                        None,
                        next(vectors),
                        None,
                    ),
                    tuple(
                        PreparedLink(link.memory_id, subject_id, link.basis)
                        for link in subject.links
                    ),
                )
            )
        result: Literal["full_split", "partial_split"] = (
            "full_split" if isinstance(output, FullSplitOutput) else "partial_split"
        )
        operation_id = self._store.commit_split(
            add_operation_id=add_operation_id,
            memory_space_id=memory_space_id,
            original=snapshot,
            result=result,
            new_subjects=prepared,
            signature_id=signature_id,
            actor=actor,
            config_signature=self.config.signature,
        )
        self._event(
            "audit_subject_split",
            memory_space_id=memory_space_id,
            subject_id=snapshot.subject_id,
            operation_id=operation_id,
            before=asdict(snapshot),
            decision=output.model_dump(mode="json"),
        )
        return MaintenanceResult(snapshot.subject_id, result, operation_id)

    def _validate_split(
        self,
        memory_space_id: str,
        output: SplitStageOutput,
        snapshot: SubjectSnapshot,
    ) -> None:
        if isinstance(output, DeferSplitOutput):
            if not output.reason.strip():
                raise ValidationError("defer_split requires a reason")
            return
        subjects = (
            output.subjects
            if isinstance(output, FullSplitOutput)
            else output.new_subjects
        )
        if isinstance(output, FullSplitOutput):
            if (
                not self.config.subject_split_result_subject_min
                <= len(subjects)
                <= self.config.subject_split_result_subject_max
            ):
                raise ValidationError("full split result subject count is invalid")
        elif not 1 <= len(subjects) <= self.config.subject_split_result_subject_max - 1:
            raise ValidationError("partial split new subject count is invalid")
        refs = [subject.subject_ref for subject in subjects]
        if len(refs) != len(set(refs)):
            raise ValidationError("split subject_ref values must be unique")
        input_ids = {memory.memory_id for memory in snapshot.memories}
        membership: Counter[str] = Counter()
        for subject in subjects:
            self._validate_subject_name(subject.name)
            if subject.name.strip().lower() in {
                "other",
                "misc",
                "general",
                "others",
                "miscellaneous",
            }:
                raise ValidationError("split subject name has no semantic boundary")
            ids = [link.memory_id for link in subject.links]
            if len(ids) != len(set(ids)):
                raise ValidationError("a split subject contains duplicate memories")
            if len(ids) < self.config.subject_split_result_min_memories:
                raise ValidationError("a split subject has too few memories")
            if len(ids) > self.config.subject_split_result_target_memory_max:
                raise ValidationError("a split subject has too many memories")
            if not set(ids) <= input_ids:
                raise ValidationError(
                    "split references a memory outside the input subject"
                )
            if not any(link.basis == "direct" for link in subject.links):
                raise ValidationError("every split subject requires a direct link")
            membership.update(ids)
        if any(
            count > self.config.subject_split_memory_membership_max
            for count in membership.values()
        ):
            raise ValidationError("a memory exceeds split membership maximum")
        moved = set(membership)
        if isinstance(output, FullSplitOutput):
            if moved != input_ids or any(membership[item] < 1 for item in input_ids):
                raise ValidationError("full split must cover every input memory")
        else:
            if not moved or moved == input_ids:
                raise ValidationError(
                    "partial split must move a non-empty proper subset"
                )
        current_counts = self._store.memory_active_link_counts(
            memory_space_id, tuple(moved)
        )
        remaining_direct = self._store.memory_active_direct_link_counts(
            memory_space_id,
            tuple(moved),
            excluding_subject_id=snapshot.subject_id,
        )
        new_direct: Counter[str] = Counter()
        for subject in subjects:
            for link in subject.links:
                if link.basis == "direct":
                    new_direct[link.memory_id] += 1
        for memory_id, new_count in membership.items():
            final_count = current_counts[memory_id] - 1 + new_count
            if final_count > self.config.memory_active_subject_link_max:
                raise ValidationError(
                    "split would exceed a memory's active link maximum"
                )
            if remaining_direct[memory_id] + new_direct[memory_id] < 1:
                raise ValidationError(
                    f"split would leave memory {memory_id} without a direct link"
                )

    async def _structured_output(
        self,
        *,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        adapter: TypeAdapter[Any],
        temperature: float,
        validator: Callable[[Any], None],
    ) -> tuple[Any, _LlmTurn]:
        started = time.monotonic()
        current_input = user_prompt
        for attempt in range(self.config.structured_output_max_retries + 1):
            failed_output = ""
            response: GenerationResponse | None = None
            try:
                response = await self._generation.generate(
                    GenerationRequest(
                        stage=stage,
                        system_prompt=system_prompt,
                        user_prompt=current_input,
                        temperature=temperature,
                        seed=self._benchmark_seed,
                    )
                )
                failed_output = response.text
                raw = _parse_json_object(response.text)
                value = adapter.validate_python(raw)
                validator(value)
                self._event(
                    "llm_call",
                    stage=stage,
                    attempt=attempt + 1,
                    result="success",
                    elapsed_seconds=time.monotonic() - started,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    total_tokens=response.total_tokens,
                    request_id=response.request_id,
                )
                return value, _LlmTurn(
                    first_attempt=attempt == 0,
                    stage=stage,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output=response.text,
                    request_id=response.request_id,
                )
            except ProviderError as error:
                if error.error_class not in {
                    ErrorClass.INVALID_STRUCTURED_OUTPUT,
                    ErrorClass.INCOMPLETE_OUTPUT,
                }:
                    self._event(
                        "llm_call",
                        severity="error",
                        stage=stage,
                        attempt=attempt + 1,
                        result="failed",
                        elapsed_seconds=time.monotonic() - started,
                        input_tokens=0,
                        output_tokens=0,
                        total_tokens=0,
                        request_id=None,
                        error_class=error.error_class.value,
                        reason=error.message,
                    )
                    raise StageFailure(
                        stage, error.error_class, error.message
                    ) from error
                output_error: Exception = error
                error_class = error.error_class
            except (
                json.JSONDecodeError,
                PydanticValidationError,
                ValidationError,
            ) as error:
                output_error = error
                error_class = ErrorClass.INVALID_STRUCTURED_OUTPUT
            feedback = validation_feedback(output_error)
            self._event(
                "llm_call",
                severity="warning",
                stage=stage,
                attempt=attempt + 1,
                result="failed",
                elapsed_seconds=time.monotonic() - started,
                input_tokens=response.input_tokens if response is not None else 0,
                output_tokens=response.output_tokens if response is not None else 0,
                total_tokens=response.total_tokens if response is not None else 0,
                request_id=response.request_id if response is not None else None,
                error_class=error_class.value,
                reason=feedback,
            )
            self._event(
                "structured_output_retry",
                severity="warning",
                stage=stage,
                attempt=attempt + 1,
                reason=feedback,
            )
            if attempt >= self.config.structured_output_max_retries:
                raise StageFailure(
                    stage, ErrorClass.INVALID_STRUCTURED_OUTPUT, feedback
                )
            current_input = repair_input(user_prompt, failed_output, feedback)
        raise AssertionError("unreachable structured-output retry state")

    async def _embed_documents(self, texts: Sequence[str]) -> tuple[Any, ...]:
        return await self._embed(texts, "document")

    async def _embed_queries(self, texts: Sequence[str]) -> tuple[Any, ...]:
        return await self._embed(texts, "query")

    async def _embed(self, texts: Sequence[str], input_type: str) -> tuple[Any, ...]:
        if not texts:
            return ()
        batches = [
            texts[index : index + self.config.embedding_batch_size]
            for index in range(0, len(texts), self.config.embedding_batch_size)
        ]
        semaphore = asyncio.Semaphore(
            self.config.embedding_batch_concurrency_per_operation
        )

        async def one(batch: Sequence[str]) -> tuple[Any, ...]:
            async with semaphore:
                response = await self._embedding.embed(
                    batch,
                    input_type=input_type,
                    timeout_seconds=self.config.embedding_request_timeout_seconds,
                )
                if len(response.vectors) != len(batch):
                    raise ProviderError(
                        ErrorClass.INCOMPLETE_OUTPUT,
                        "embedding response count does not match request count",
                    )
                vectors: list[np.ndarray] = []
                for vector in response.vectors:
                    checked = np.ascontiguousarray(vector, dtype="<f4")
                    if checked.shape != (self._embedding.model_info.dimension,):
                        raise ProviderError(
                            ErrorClass.INVALID_REQUEST,
                            "embedding response dimension does not match model signature",
                        )
                    norm = float(np.linalg.norm(checked))
                    if not np.isfinite(checked).all() or abs(norm - 1.0) > 1e-4:
                        raise ProviderError(
                            ErrorClass.INVALID_REQUEST,
                            "embedding response must contain finite L2-normalized vectors",
                        )
                    vectors.append(checked)
                return tuple(vectors)

        results = await asyncio.gather(*(one(batch) for batch in batches))
        return tuple(vector for batch in results for vector in batch)

    def _validate_episode(self, episode: NormalizedEpisode) -> None:
        if not episode.source_type.strip() or not episode.source_key.strip():
            raise ValidationError("episode source identity must not be blank")
        if episode.source_sequence < 0:
            raise ValidationError("source_sequence must be non-negative")
        if not episode.blocks:
            raise ValidationError("episode must contain at least one message")
        if len(episode.blocks) > self.config.episode_message_max:
            raise ValidationError("episode exceeds message count limit")
        if episode.source_chars > self.config.episode_chars_max:
            raise ValidationError("episode exceeds total character limit")
        expected_sequence = list(range(len(episode.blocks)))
        if [block.sequence_no for block in episode.blocks] != expected_sequence:
            raise ValidationError(
                "episode block sequence numbers must be contiguous from zero"
            )
        block_ids = [block.block_id for block in episode.blocks]
        if len(block_ids) != len(set(block_ids)):
            raise ValidationError("episode block IDs must be unique")
        message_limit = (
            self.config.longmemeval_message_chars_max
            if episode.source_type == "longmemeval"
            else self.config.message_chars_max
        )
        for block in episode.blocks:
            if len(block.content) > message_limit:
                raise ValidationError("episode message exceeds character limit")
            if block.role.value == "user" and block.message_phase is not None:
                raise ValidationError("user messages cannot have an assistant phase")

    def _validate_extraction(self, extraction: ExtractionOutput) -> None:
        if not isinstance(extraction, MemoriesExtraction):
            if not extraction.reason.strip():
                raise ValidationError("no_valuable_memory requires a reason")
            return
        for memory in extraction.memories:
            self._validate_memory_content(memory.content)

    def _validate_memory_content(self, content: str) -> None:
        if not content.strip():
            raise ValidationError("memory content must not be blank")
        if len(content) > self.config.memory_content_max_chars:
            raise ValidationError("memory content exceeds character limit")
        if _SECRET_RE.search(content):
            raise ValidationError(
                "memory content appears to contain an authentication secret"
            )

    def _validate_subject_name(self, name: str) -> None:
        if not name.strip() or len(name) > self.config.subject_name_max_chars:
            raise ValidationError("subject name is blank or exceeds character limit")

    def _validate_summary(self, summary: str) -> None:
        if (
            not summary.strip()
            or len(summary) > self.config.generated_subject_summary_max_chars
        ):
            raise ValidationError(
                "generated subject summary is blank or exceeds character limit"
            )

    def _audit_extraction(
        self,
        memory_space_id: str,
        extraction: ExtractionOutput,
        memory_refs: Sequence[str],
        contents: Sequence[str],
        linking: LinkingOutput,
    ) -> dict[str, object]:
        if extraction.result != "memories":
            return extraction.model_dump(mode="json")
        new_names = {
            subject.subject_ref: subject.name for subject in linking.new_subjects
        }
        existing_ids = {
            link.subject.subject_id
            for link in linking.links
            if link.subject.kind == "existing"
        }
        existing_names = {
            subject_id: self._store.subject_snapshot(memory_space_id, subject_id).name
            for subject_id in existing_ids
        }
        subjects_by_ref: dict[str, list[str]] = {
            memory_ref: [] for memory_ref in memory_refs
        }
        for link in linking.links:
            name = (
                existing_names[link.subject.subject_id]
                if link.subject.kind == "existing"
                else new_names[link.subject.subject_ref]
            )
            subjects_by_ref[link.memory_ref].append(name)
        return {
            "result": "memories",
            "memories": [
                {
                    "content": content,
                    "subjects": subjects_by_ref[memory_ref],
                }
                for memory_ref, content in zip(memory_refs, contents, strict=True)
            ],
        }

    def _offer_llm_sample(self, kind: str, *turns: _LlmTurn) -> None:
        if self._event_sink is None or any(not turn.first_attempt for turn in turns):
            return
        with self._llm_io_sample_lock:
            if self._llm_io_sample_counts[kind] >= LLM_IO_SAMPLE_QUOTAS[kind]:
                return
            self._llm_io_sample_counts[kind] += 1
        self._event(
            "llm_io_sample",
            kind=kind,
            rounds=[
                {
                    "stage": turn.stage,
                    "request_id": turn.request_id,
                    "system_prompt": turn.system_prompt,
                    "user_prompt": turn.user_prompt,
                    "output": turn.output,
                }
                for turn in turns
            ],
        )

    def _event(self, event_type: str, **fields: object) -> None:
        if self._event_sink is not None:
            self._event_sink(
                {
                    "event_type": event_type,
                    "timestamp_ms": int(time.time() * 1000),
                    "severity": fields.pop("severity", "info"),
                    **fields,
                }
            )


_SECRET_RE = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"\b(?:api[_ -]?key|session[_ -]?token|verification[_ -]?code|password)\s*[:=]\s*\S+",
    re.IGNORECASE,
)


def _candidate_memory_ids(
    candidates: dict[str, dict[str, object]],
) -> set[str]:
    memory_ids: set[str] = set()
    for value in candidates.values():
        for group in cast(list[dict[str, object]], value["subjects"]):
            for memory in cast(list[dict[str, object]], group["memory_hits"]):
                memory_ids.add(cast(str, memory["memory_id"]))
        for memory in cast(list[dict[str, object]], value["unattached_memories"]):
            memory_ids.add(cast(str, memory["memory_id"]))
    return memory_ids


def _parse_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValidationError("structured output must be one JSON object")
    return value


def _name_summary(name: str, summary: str) -> str:
    return f"{name}\n\n{summary}"
