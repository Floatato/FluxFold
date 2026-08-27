"""Experimental FluxFold Memory Engine orchestration."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeVar, cast
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
    MemorySpace,
    NormalizedEpisode,
    PartialSplitOutput,
    ProvenanceRequestOutput,
    ReplaceContent,
    ReplaceProvenance,
    ReviewOutput,
    ReviewStageOutput,
    SearchResult,
    SplitStageOutput,
    SubjectSnapshot,
)
from fluxfold.prompts import (
    EXTRACTION_SYSTEM,
    LINKING_SYSTEM,
    REVIEW_SYSTEM,
    SPLIT_SYSTEM,
    extraction_input,
    linking_input,
    repair_input,
    review_input,
    split_input,
)
from fluxfold.providers import (
    EmbeddingProvider,
    GenerationProvider,
    GenerationRequest,
)
from fluxfold.storage import (
    ExistingSubjectAppend,
    PersistedEpisode,
    PreparedLink,
    PreparedMemory,
    PreparedMemoryUpdate,
    PreparedSplitSubject,
    PreparedSubject,
    Store,
)

EventSink = Callable[[dict[str, object]], None]
T = TypeVar("T")


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

    @classmethod
    async def open(
        cls,
        *,
        db_path: str,
        generation_provider: GenerationProvider,
        embedding_provider: EmbeddingProvider,
        config: FluxFoldConfig | None = None,
        event_sink: EventSink | None = None,
        benchmark_seed: int | None = None,
    ) -> FluxFold:
        """Open or initialize one SQLite-backed engine."""

        resolved = config or FluxFoldConfig()
        return cls(
            store=Store(db_path, resolved),
            generation_provider=generation_provider,
            embedding_provider=embedding_provider,
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

    async def add(
        self,
        memory_space_id: str,
        episode: NormalizedEpisode,
        *,
        actor: str = "dataset",
    ) -> AddResult:
        """Add one complete normalized dataset episode."""

        prepared = await self._prepare_add(memory_space_id, episode, actor=actor)
        return await self._commit_prepared_add(prepared)

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
        extraction = cast(
            ExtractionOutput,
            await self._structured_output(
                stage="memory_extraction",
                system_prompt=EXTRACTION_SYSTEM,
                user_prompt=extraction_input(episode),
                adapter=TypeAdapter(ExtractionOutput),
                temperature=self.config.extraction_temperature,
                request_timeout=self.config.extraction_request_timeout_seconds,
                stage_deadline=self.config.extraction_stage_deadline_seconds,
                validator=self._validate_extraction,
            ),
        )
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
        async with self._space_locks[memory_space_id]:
            current = self._store.persist_episode(memory_space_id, episode)
            if current.extraction_completed:
                return AddResult(
                    current.episode_id,
                    current.completed_operation_id,
                    True,
                    0,
                    0,
                    0,
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
            linking = (
                await self._link_batch(
                    memory_space_id,
                    memory_refs,
                    contents,
                    memory_vectors,
                )
                if contents
                else LinkingOutput(new_subjects=[], links=[])
            )
            self._event(
                "subject_linking_completed",
                memory_space_id=memory_space_id,
                episode_id=current.episode_id,
                new_subjects=len(linking.new_subjects),
                links=len(linking.links),
            )
            (
                prepared_subjects,
                prepared_links,
                subject_appends,
                affected_subject_ids,
            ) = await self._prepare_link_commit(
                memory_space_id,
                memory_refs,
                memory_ids,
                contents,
                linking,
            )
            operation_id = self._store.commit_add(
                memory_space_id=memory_space_id,
                episode_id=current.episode_id,
                input_hash=episode.content_hash,
                memories=prepared_memories,
                subjects=prepared_subjects,
                links=prepared_links,
                subject_appends=subject_appends,
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
                extraction=extraction.model_dump(mode="json"),
                memories=[
                    {
                        "memory_ref": memory_ref,
                        "memory_id": memory_id,
                        "content": content,
                    }
                    for memory_ref, memory_id, content in zip(
                        memory_refs, memory_ids, contents, strict=True
                    )
                ],
                new_subjects=[
                    {
                        **subject.model_dump(mode="json"),
                        "subject_id": prepared_subject.subject_id,
                    }
                    for subject, prepared_subject in zip(
                        linking.new_subjects, prepared_subjects, strict=True
                    )
                ],
                links=[link.model_dump(mode="json") for link in linking.links],
            )
            maintenance: list[MaintenanceResult] = []
            for subject_id in sorted(affected_subject_ids):
                try:
                    maintenance.extend(
                        await self._maintain_subject(
                            memory_space_id,
                            subject_id,
                            prepared.signature_id,
                            prepared.actor,
                        )
                    )
                except StageFailure as error:
                    if error.error_class not in {
                        ErrorClass.CONTEXT_OVERFLOW,
                        ErrorClass.POLICY_REJECTED,
                        ErrorClass.INVALID_STRUCTURED_OUTPUT,
                        ErrorClass.INCOMPLETE_OUTPUT,
                        ErrorClass.STAGE_DEADLINE_EXCEEDED,
                    }:
                        raise
                    self._event(
                        "maintenance_terminal_failure",
                        severity="error",
                        memory_space_id=memory_space_id,
                        subject_id=subject_id,
                        error_class=error.error_class.value,
                        reason=error.message,
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

    def record_episode_terminal_failure(
        self,
        memory_space_id: str,
        episode: NormalizedEpisode,
        failure: StageFailure,
    ) -> None:
        """Persist a benchmark episode's deterministic terminal failure."""

        persisted = self._store.persist_episode(memory_space_id, episode)
        self._store.record_episode_terminal_failure(
            memory_space_id=memory_space_id,
            episode_id=persisted.episode_id,
            input_hash=episode.content_hash,
            config_signature=self.config.signature,
            error_class=failure.error_class.value,
            error_message=failure.message,
        )

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
            self._store.get_space(memory_space_id)
            signature_id = self._store.prepare_retrieval_signature(
                self._embedding.model_info
            )
            memory_sources, subject_sources = self._store.retrieval_embedding_sources(
                memory_space_id
            )
            vectors = await self._embed_documents(
                [source.content for source in memory_sources]
                + [
                    text
                    for source in subject_sources
                    for text in (
                        source.name,
                        _name_summary(source.name, source.summary),
                    )
                ]
            )
            memory_vectors = vectors[: len(memory_sources)]
            subject_vectors = iter(vectors[len(memory_sources) :])
            prepared_subjects = [
                (source, next(subject_vectors), next(subject_vectors))
                for source in subject_sources
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

    def space_statistics(self, memory_space_id: str) -> dict[str, int | float]:
        """Return the experiment metrics derived from final database state."""

        return self._store.space_statistics(memory_space_id)

    async def _link_batch(
        self,
        memory_space_id: str,
        memory_refs: Sequence[str],
        contents: Sequence[str],
        vectors: Sequence[Any],
    ) -> LinkingOutput:
        candidates, legal_by_memory = self._recall_candidates(
            memory_space_id, memory_refs, vectors
        )
        new_memories = [
            {"memory_ref": memory_ref, "content": content}
            for memory_ref, content in zip(memory_refs, contents, strict=True)
        ]

        def validate_initial(value: LinkingStageOutput) -> None:
            if isinstance(value, AssociationSearchOutput):
                if not self.config.association_search_enabled:
                    raise ValidationError("association_search is disabled")
                if not value.query.strip():
                    raise ValidationError("association_search query must not be blank")
                return
            self._validate_linking(value, memory_refs, legal_by_memory)

        first = cast(
            LinkingStageOutput,
            await self._structured_output(
                stage="subject_linking",
                system_prompt=LINKING_SYSTEM,
                user_prompt=linking_input(new_memories, candidates),
                adapter=TypeAdapter(LinkingStageOutput),
                temperature=self.config.linking_temperature,
                request_timeout=self.config.linking_request_timeout_seconds,
                stage_deadline=self.config.linking_stage_deadline_seconds,
                validator=validate_initial,
            ),
        )
        if isinstance(first, LinkingOutput):
            return first
        association_vector = (await self._embed_queries([first.query]))[0]
        association_candidates, association_legal = self._recall_candidates(
            memory_space_id, memory_refs, [association_vector] * len(memory_refs)
        )
        for memory_ref, ids in association_legal.items():
            legal_by_memory[memory_ref].update(ids)

        def validate_final(value: LinkingStageOutput) -> None:
            if isinstance(value, AssociationSearchOutput):
                raise ValidationError("association_search may only be requested once")
            self._validate_linking(value, memory_refs, legal_by_memory)

        final = cast(
            LinkingStageOutput,
            await self._structured_output(
                stage="subject_linking",
                system_prompt=LINKING_SYSTEM,
                user_prompt=linking_input(
                    new_memories,
                    candidates,
                    {
                        "query": first.query,
                        "candidates_by_memory": association_candidates,
                    },
                ),
                adapter=TypeAdapter(LinkingStageOutput),
                temperature=self.config.linking_temperature,
                request_timeout=self.config.linking_request_timeout_seconds,
                stage_deadline=self.config.linking_stage_deadline_seconds,
                validator=validate_final,
            ),
        )
        return cast(LinkingOutput, final)

    def _recall_candidates(
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
                top_k=self.config.subject_candidate_top_k,
                min_similarity=self.config.subject_candidate_min_similarity,
            )
            memories = self._store.candidate_memories(
                memory_space_id,
                vector,
                top_k=self.config.memory_candidate_top_k,
                min_similarity=self.config.memory_candidate_min_similarity,
            )
            legal_ids = {subject.subject_id for subject in subjects}
            legal_ids.update(
                memory.attached_subject_id
                for memory in memories
                if memory.attached_subject_id is not None
            )
            candidates[memory_ref] = {
                "subjects": [asdict(subject) for subject in subjects],
                "memories": [asdict(memory) for memory in memories],
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
            links = by_memory[memory_ref]
            if not links or not any(link.basis == "direct" for link in links):
                raise ValidationError(f"{memory_ref} needs at least one direct link")
            if len(links) > self.config.memory_active_subject_link_max:
                raise ValidationError(f"{memory_ref} exceeds the active link maximum")
        for subject in output.new_subjects:
            self._validate_subject_text(subject.name, subject.summary)

    async def _prepare_link_commit(
        self,
        memory_space_id: str,
        memory_refs: Sequence[str],
        memory_ids: Sequence[str],
        contents: Sequence[str],
        linking: LinkingOutput,
    ) -> tuple[
        list[PreparedSubject],
        list[PreparedLink],
        list[ExistingSubjectAppend],
        set[str],
    ]:
        ref_to_memory_id = dict(zip(memory_refs, memory_ids, strict=True))
        new_ref_to_id = {
            subject.subject_ref: str(uuid4()) for subject in linking.new_subjects
        }
        subject_texts: list[str] = []
        for subject in linking.new_subjects:
            subject_texts.extend(
                [subject.name, _name_summary(subject.name, subject.summary)]
            )
        subject_vectors = iter(await self._embed_documents(subject_texts))
        prepared_subjects = [
            PreparedSubject(
                new_ref_to_id[subject.subject_ref],
                subject.name,
                subject.summary,
                next(subject_vectors),
                next(subject_vectors),
            )
            for subject in linking.new_subjects
        ]
        prepared_links: list[PreparedLink] = []
        existing_linked_contents: defaultdict[str, list[str]] = defaultdict(list)
        affected: set[str] = set()
        content_by_ref = dict(zip(memory_refs, contents, strict=True))
        for link in linking.links:
            if link.subject.kind == "existing":
                subject_id = link.subject.subject_id
                existing_linked_contents[subject_id].append(
                    content_by_ref[link.memory_ref]
                )
            else:
                subject_id = new_ref_to_id[link.subject.subject_ref]
            prepared_links.append(
                PreparedLink(ref_to_memory_id[link.memory_ref], subject_id, link.basis)
            )
            affected.add(subject_id)
        appends_data: list[tuple[str, int, str, int]] = []
        for subject_id, appended in sorted(existing_linked_contents.items()):
            snapshot = self._store.subject_snapshot(memory_space_id, subject_id)
            summary = snapshot.summary + "".join(f"\n{content}" for content in appended)
            appends_data.append(
                (subject_id, snapshot.summary_revision, summary, len(appended))
            )
        append_vectors = await self._embed_documents(
            [
                _name_summary(
                    self._store.subject_snapshot(memory_space_id, subject_id).name,
                    summary,
                )
                for subject_id, _, summary, _ in appends_data
            ]
        )
        appends = [
            ExistingSubjectAppend(subject_id, revision, summary, increment, vector)
            for (subject_id, revision, summary, increment), vector in zip(
                appends_data, append_vectors, strict=True
            )
        ]
        return prepared_subjects, prepared_links, appends, affected

    async def _maintain_subject(
        self,
        memory_space_id: str,
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
                memory_space_id, snapshot, signature_id, actor
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
            operation_id = await self._review_subject(
                memory_space_id, snapshot, signature_id, actor
            )
            outcomes.append(MaintenanceResult(subject_id, "review", operation_id))
            self._event(
                "subject_review_completed",
                memory_space_id=memory_space_id,
                subject_id=subject_id,
                operation_id=operation_id,
            )
        return outcomes

    async def _review_subject(
        self,
        memory_space_id: str,
        snapshot: SubjectSnapshot,
        signature_id: str,
        actor: str,
    ) -> str:
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

        first = cast(
            ReviewStageOutput,
            await self._structured_output(
                stage="subject_review",
                system_prompt=REVIEW_SYSTEM,
                user_prompt=review_input(snapshot),
                adapter=TypeAdapter(ReviewStageOutput),
                temperature=self.config.review_temperature,
                request_timeout=self.config.review_request_timeout_seconds,
                stage_deadline=self.config.review_stage_deadline_seconds,
                validator=validate_first,
            ),
        )
        if isinstance(first, ProvenanceRequestOutput):
            provenance = self._store.provenance_episodes(
                memory_space_id, first.memory_ids
            )

            def validate_final(value: ReviewStageOutput) -> None:
                if isinstance(value, ProvenanceRequestOutput):
                    raise ValidationError("provenance may only be requested once")
                self._validate_review(value, snapshot)

            final = cast(
                ReviewStageOutput,
                await self._structured_output(
                    stage="subject_review",
                    system_prompt=REVIEW_SYSTEM,
                    user_prompt=review_input(snapshot, provenance),
                    adapter=TypeAdapter(ReviewStageOutput),
                    temperature=self.config.review_temperature,
                    request_timeout=self.config.review_request_timeout_seconds,
                    stage_deadline=self.config.review_stage_deadline_seconds,
                    validator=validate_final,
                ),
            )
            review = cast(ReviewOutput, final)
        else:
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
            + [_name_summary(snapshot.name, review.summary)]
        )
        updates = [
            PreparedMemoryUpdate(memory_id, str(uuid4()), content, provenance, vector)
            for (memory_id, content, provenance), vector in zip(
                prepared_values, vectors[: len(prepared_values)], strict=True
            )
        ]
        summary_vector = vectors[-1]
        operation_id = self._store.commit_review(
            memory_space_id=memory_space_id,
            subject_id=snapshot.subject_id,
            expected_revision=snapshot.summary_revision,
            updates=updates,
            retirements=review.retirements,
            summary=review.summary,
            summary_embedding=summary_vector,
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
        return operation_id

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
        self._validate_summary(review.summary)

    async def _split_subject(
        self,
        memory_space_id: str,
        snapshot: SubjectSnapshot,
        signature_id: str,
        actor: str,
    ) -> MaintenanceResult:
        def validate(value: SplitStageOutput) -> None:
            self._validate_split(memory_space_id, value, snapshot)

        output = cast(
            SplitStageOutput,
            await self._structured_output(
                stage="subject_split",
                system_prompt=SPLIT_SYSTEM,
                user_prompt=split_input(snapshot),
                adapter=TypeAdapter(SplitStageOutput),
                temperature=self.config.split_temperature,
                request_timeout=self.config.split_request_timeout_seconds,
                stage_deadline=self.config.split_stage_deadline_seconds,
                validator=validate,
            ),
        )
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
        embedding_texts = [
            text
            for subject in subjects
            for text in (subject.name, _name_summary(subject.name, subject.summary))
        ]
        if isinstance(output, PartialSplitOutput):
            embedding_texts.append(
                _name_summary(snapshot.name, output.remaining_summary)
            )
        vectors = iter(await self._embed_documents(embedding_texts))
        prepared: list[PreparedSplitSubject] = []
        for subject in subjects:
            subject_id = str(uuid4())
            prepared.append(
                PreparedSplitSubject(
                    PreparedSubject(
                        subject_id,
                        subject.name,
                        subject.summary,
                        next(vectors),
                        next(vectors),
                    ),
                    tuple(
                        PreparedLink(link.memory_id, subject_id, link.basis)
                        for link in subject.links
                    ),
                )
            )
        remaining_vector = (
            next(vectors) if isinstance(output, PartialSplitOutput) else None
        )
        result: Literal["full_split", "partial_split"] = (
            "full_split" if isinstance(output, FullSplitOutput) else "partial_split"
        )
        operation_id = self._store.commit_split(
            memory_space_id=memory_space_id,
            original=snapshot,
            result=result,
            new_subjects=prepared,
            remaining_summary=(
                output.remaining_summary
                if isinstance(output, PartialSplitOutput)
                else None
            ),
            remaining_summary_embedding=remaining_vector,
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
            self._validate_subject_text(subject.name, subject.summary)
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
            self._validate_summary(output.remaining_summary)
        current_counts = self._store.memory_active_link_counts(
            memory_space_id, tuple(moved)
        )
        for memory_id, new_count in membership.items():
            final_count = current_counts[memory_id] - 1 + new_count
            if final_count > self.config.memory_active_subject_link_max:
                raise ValidationError(
                    "split would exceed a memory's active link maximum"
                )

    async def _structured_output(
        self,
        *,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        adapter: TypeAdapter[Any],
        temperature: float,
        request_timeout: float,
        stage_deadline: float,
        validator: Callable[[Any], None],
    ) -> Any:
        started = time.monotonic()
        failed_output = ""
        current_input = user_prompt
        for attempt in range(self.config.structured_output_max_retries + 1):
            remaining = stage_deadline - (time.monotonic() - started)
            if remaining <= 0:
                raise StageFailure(
                    stage, ErrorClass.STAGE_DEADLINE_EXCEEDED, "stage deadline exceeded"
                )
            try:
                response = await asyncio.wait_for(
                    self._generation.generate(
                        GenerationRequest(
                            stage,
                            system_prompt,
                            current_input,
                            temperature,
                            min(request_timeout, remaining),
                            self._benchmark_seed,
                        )
                    ),
                    timeout=min(request_timeout, remaining),
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
                    total_tokens=response.total_tokens,
                    request_id=response.request_id,
                )
                return value
            except TimeoutError as error:
                provider_error = ProviderError(
                    ErrorClass.TRANSIENT_TRANSPORT, "request timeout"
                )
                raise StageFailure(
                    stage, provider_error.error_class, str(provider_error)
                ) from error
            except ProviderError as error:
                if error.error_class not in {
                    ErrorClass.INVALID_STRUCTURED_OUTPUT,
                    ErrorClass.INCOMPLETE_OUTPUT,
                }:
                    raise StageFailure(
                        stage, error.error_class, error.message
                    ) from error
                validation_message = error.message
            except (
                json.JSONDecodeError,
                PydanticValidationError,
                ValidationError,
            ) as error:
                validation_message = str(error)
            self._event(
                "structured_output_retry",
                severity="warning",
                stage=stage,
                attempt=attempt + 1,
                reason=validation_message,
            )
            if attempt >= self.config.structured_output_max_retries:
                raise StageFailure(
                    stage, ErrorClass.INVALID_STRUCTURED_OUTPUT, validation_message
                )
            current_input = repair_input(user_prompt, failed_output, validation_message)
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
        semaphore = asyncio.Semaphore(self.config.embedding_max_concurrency)

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
            if not block.content.strip():
                raise ValidationError("blank episode messages are invalid")
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

    def _validate_subject_text(self, name: str, summary: str) -> None:
        if not name.strip() or len(name) > self.config.subject_name_max_chars:
            raise ValidationError("subject name is blank or exceeds character limit")
        self._validate_summary(summary)

    def _validate_summary(self, summary: str) -> None:
        if (
            not summary.strip()
            or len(summary) > self.config.generated_subject_summary_max_chars
        ):
            raise ValidationError(
                "generated subject summary is blank or exceeds character limit"
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
