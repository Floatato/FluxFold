"""Experimental FluxFold Memory Engine orchestration."""

from __future__ import annotations

import asyncio
import json
import random
import re
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
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
    DeferSplitOutput,
    DomainSnapshot,
    EmbeddingRebuildResult,
    ExtractionOutput,
    FullSplitOutput,
    LinkingOutput,
    MaintenanceResult,
    MemoriesExtraction,
    MemoryBankSpace,
    MemorySpace,
    NameResolutionOutput,
    NormalizedEpisode,
    ProvenanceRequestOutput,
    ReplaceContent,
    ReplaceProvenance,
    ReviewOutput,
    ReviewStageOutput,
    SearchResult,
    SplitStageOutput,
    SummaryRefreshOutput,
)
from fluxfold.names import NameEntry, bm25_scores, normalized_name
from fluxfold.prompts import (
    EXTRACTION_SYSTEM,
    LINKING_SYSTEM,
    REVIEW_SYSTEM,
    SPLIT_SYSTEM,
    SUMMARY_REFRESH_SYSTEM,
    extraction_input,
    linking_input,
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
    ExistingDomainTouch,
    PersistedEpisode,
    PreparedDomain,
    PreparedLink,
    PreparedMemory,
    PreparedMemoryUpdate,
    PreparedSplitDomain,
    Store,
)

EventSink = Callable[[dict[str, object]], None]

LLM_IO_SAMPLE_QUOTAS: dict[str, int] = {
    "extract": 10,
    "link": 10,
    "review": 10,
    "split": 10,
    "summary": 10,
}
LLM_IO_SAMPLE_PROBABILITY = 0.1

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
    user_prompt: str
    output: str


@dataclass(frozen=True, slots=True)
class _PreparedAdd:
    memory_space_id: str
    episode: NormalizedEpisode
    actor: str
    signature_id: str
    persisted: PersistedEpisode
    extraction: ExtractionOutput | None
    contents: tuple[str, ...]
    memory_ids: tuple[str, ...]
    memory_vectors: tuple[Any, ...]
    prepared_memories: tuple[PreparedMemory, ...]
    entities: tuple[NameEntry, ...] = ()


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
            )
        extraction, _ = await self._structured_output(
            stage="memory_extraction",
            system_prompt=EXTRACTION_SYSTEM,
            user_prompt=extraction_input(episode),
            adapter=TypeAdapter(ExtractionOutput),
            temperature=self.config.extraction_temperature,
            validator=self._validate_extraction,
            sample_kind="extract",
        )
        extraction = cast(ExtractionOutput, extraction)
        entities: tuple[NameEntry, ...] = ()
        if isinstance(extraction, MemoriesExtraction):
            proposed = list(
                dict.fromkeys(
                    name for memory in extraction.memories for name in memory.entities
                )
            )
            mapping, entries = await self._resolve_names(
                memory_space_id,
                proposed,
                self._store.entity_names(memory_space_id),
                "memory_extraction",
                EXTRACTION_SYSTEM,
                extraction_input(episode),
                extraction.model_dump(),
            )
            for memory in extraction.memories:
                memory.entities = list(
                    dict.fromkeys(mapping[name] for name in memory.entities)
                )
            entities = tuple(
                entries[normalized_name(name)]
                for name in dict.fromkeys(mapping.values())
            )
            if (
                len(extraction.memories)
                > self.config.extraction_memory_warning_threshold
            ):
                self._event(
                    "extraction_memory_count_warning",
                    severity="warning",
                    memory_space_id=memory_space_id,
                    episode_id=persisted.episode_id,
                    memory_count=len(extraction.memories),
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
            memory_ids,
            memory_vectors,
            prepared_memories,
            entities,
        )

    async def _commit_prepared_add(self, prepared: _PreparedAdd) -> AddResult:
        memory_space_id = prepared.memory_space_id
        episode = prepared.episode
        current = self._store.persist_episode(memory_space_id, episode)
        if current.extraction_completed:
            replay_domain_ids = self._store.operation_active_domain_ids(
                memory_space_id, current.completed_operation_id
            )
            maintenance = await self._complete_maintenance(
                memory_space_id,
                cast(str, current.completed_operation_id),
                replay_domain_ids,
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
        contents = prepared.contents
        memory_ids = prepared.memory_ids
        memory_vectors = prepared.memory_vectors
        prepared_memories = prepared.prepared_memories
        prepared_domains, entity_domains = self._prepare_entity_domains(
            memory_space_id, prepared.entities
        )
        if contents:
            linking = await self._link_memories(
                memory_space_id,
                memory_ids,
                contents,
                memory_vectors,
                [
                    memory.entities
                    for memory in cast(MemoriesExtraction, prepared.extraction).memories
                ],
            )
        else:
            linking = LinkingOutput(memories=[])
        self._event(
            "domain_linking_completed",
            memory_space_id=memory_space_id,
            episode_id=current.episode_id,
            new_domains=len(prepared_domains),
            links=sum(
                len(
                    {item.domain for item in memory.direct_assignments}
                    | set(memory.contextual_domains)
                )
                for memory in linking.memories
            ),
        )
        prepared_links, domain_touches, affected_domain_ids = self._prepare_link_commit(
            memory_space_id, linking, prepared_domains
        )
        known_entity_ids = {
            entry.object_id for entry in self._store.entity_names(memory_space_id)
        }
        entity_by_name = {entry.name: entry.object_id for entry in prepared.entities}
        operation_id = self._store.commit_add(
            memory_space_id=memory_space_id,
            episode_id=current.episode_id,
            input_hash=episode.content_hash,
            memories=prepared_memories,
            domains=prepared_domains,
            links=prepared_links,
            domain_touches=domain_touches,
            entities=[
                entry
                for entry in prepared.entities
                if entry.object_id not in known_entity_ids
            ],
            entity_domains=entity_domains,
            memory_entities=[
                (memory_id, entity_by_name[name])
                for memory_id, memory in zip(
                    memory_ids,
                    cast(MemoriesExtraction, prepared.extraction).memories,
                    strict=True,
                )
                for name in memory.entities
            ]
            if contents
            else [],
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
            domains_created=len(prepared_domains),
            links_created=len(prepared_links),
        )
        maintenance = await self._complete_maintenance(
            memory_space_id,
            operation_id,
            sorted(affected_domain_ids),
            prepared.signature_id,
            prepared.actor,
        )
        return AddResult(
            current.episode_id,
            operation_id,
            current.replayed,
            len(prepared_memories),
            len(prepared_domains),
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
        """Search active domains and memories using only exact vector retrieval."""

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
        memory_sources, domain_sources = self._store.retrieval_embedding_sources(
            memory_space_id
        )
        entity_sources = self._store.entity_names(memory_space_id)
        entity_vectors = await self._embed_documents(
            [entry.name for entry in entity_sources]
        )
        vectors = await self._embed_documents(
            [source.content for source in memory_sources]
            + [source.name for source in domain_sources]
            + [
                _name_summary(source.name, source.summary)
                for source in domain_sources
                if source.summary is not None
            ]
        )
        memory_vectors = vectors[: len(memory_sources)]
        domain_vectors = iter(vectors[len(memory_sources) :])
        name_vectors = [next(domain_vectors) for _ in domain_sources]
        summary_vectors = iter(domain_vectors)
        prepared_domains = [
            (
                source,
                name_vector,
                next(summary_vectors) if source.summary is not None else None,
            )
            for source, name_vector in zip(domain_sources, name_vectors, strict=True)
        ]
        self._store.commit_retrieval_embedding_switch(
            memory_space_id=memory_space_id,
            signature_id=signature_id,
            memories=tuple(zip(memory_sources, memory_vectors, strict=True)),
            domains=prepared_domains,
            entities=[
                replace(entry, vector=vector)
                for entry, vector in zip(entity_sources, entity_vectors, strict=True)
            ],
        )
        return EmbeddingRebuildResult(
            memory_space_id,
            signature_id,
            len(memory_sources),
            len(domain_sources),
        )

    def space_statistics(self, memory_space_id: str) -> dict[str, Any]:
        """Return the experiment metrics derived from final database state."""

        return self._store.space_statistics(memory_space_id)

    def memory_bank(self) -> tuple[MemoryBankSpace, ...]:
        """Return active domains and memories of every space for inspect logs."""

        return self._store.memory_bank()

    async def _resolve_names(
        self,
        memory_space_id: str,
        proposed: Sequence[str],
        existing: Sequence[NameEntry],
        stage: str,
        system_prompt: str,
        original_input: str,
        original_output: dict[str, Any],
    ) -> tuple[dict[str, str], dict[str, NameEntry]]:
        entries = {normalized_name(entry.name): entry for entry in existing}
        mapping: dict[str, str] = {}
        unique_proposals: dict[str, str] = {}
        for name in proposed:
            if normalized_name(name) not in entries:
                unique_proposals.setdefault(normalized_name(name), name)
        fresh = list(unique_proposals.values())
        vectors = await self._embed_documents(fresh)
        pending: dict[str, list[str]] = {}
        feedback = []
        for name, vector in zip(fresh, vectors, strict=True):
            catalog = list(entries.values())
            lexical = bm25_scores(name, [entry.name for entry in catalog])
            candidates = [
                entry
                for entry, score in zip(catalog, lexical, strict=True)
                if float(entry.vector @ vector)
                >= self.config.name_resolution_min_similarity
                or score >= self.config.name_resolution_bm25_threshold
            ]
            if candidates:
                pending[name] = [entry.name for entry in candidates]
                feedback.append(
                    {
                        "proposed_name": name,
                        "candidates": [{"name": entry.name} for entry in candidates],
                    }
                )
            entries[normalized_name(name)] = NameEntry(str(uuid4()), name, vector)
        if feedback:

            def validate(value: NameResolutionOutput) -> None:
                if len(value.resolutions) != len(pending) or {
                    item.proposed_name for item in value.resolutions
                } != set(pending):
                    raise ValidationError("resolve each proposed name once")
                for item in value.resolutions:
                    if item.canonical_name not in [
                        item.proposed_name,
                        *pending[item.proposed_name],
                    ]:
                        raise ValidationError(
                            "canonical name must be the proposed name or a supplied candidate"
                        )

            resolved, _ = await self._structured_output(
                stage=stage,
                system_prompt=system_prompt
                + "\n\nFor name_resolution feedback, judge whether the names describe the same "
                + "entity."
                + " Reuse its candidate name when they do; otherwise keep the proposed name. Return "
                + '{"resolutions":[{"proposed_name":"...","canonical_name":"..."}]}.',
                user_prompt=json.dumps(
                    {
                        "original_input": json.loads(original_input),
                        "previous_output": original_output,
                        "name_resolution": {"kind": "entity", "items": feedback},
                    },
                    ensure_ascii=False,
                ),
                adapter=TypeAdapter(NameResolutionOutput),
                temperature=0.0,
                validator=validate,
                sample_kind="extract",
            )
            decisions = {
                item.proposed_name: item.canonical_name
                for item in cast(NameResolutionOutput, resolved).resolutions
            }
        else:
            decisions = {}
        canonical: dict[str, str] = {}
        for name in fresh:
            target = decisions.get(name, name)
            # Candidates only reference persisted names or earlier proposals, so this is acyclic.
            canonical[name] = canonical.get(target, target)
        for name in proposed:
            entry = entries[normalized_name(name)]
            target = canonical.get(entry.name, entry.name)
            mapping[name] = target
            if normalized_name(name) != normalized_name(target):
                self._event(
                    "name_normalized",
                    severity="warning",
                    memory_space_id=memory_space_id,
                    stage=stage,
                    object_kind="entity",
                    proposed_name=name,
                    canonical_name=target,
                )
        return mapping, entries

    async def _link_memories(
        self,
        memory_space_id: str,
        memory_ids: Sequence[str],
        contents: Sequence[str],
        vectors: Sequence[Any],
        memory_entities: Sequence[list[str]],
    ) -> LinkingOutput:
        candidates, legal_domain_ids = self._recall_initial_domain_candidates(
            memory_space_id, vectors, memory_entities
        )
        legal_by_memory = {memory_id: set(legal_domain_ids) for memory_id in memory_ids}
        new_memories = [
            {
                "memory_id": memory_id,
                "content": content,
                "entities": entities,
                "link_limit": max(
                    self.config.memory_active_domain_link_max, len(entities)
                ),
            }
            for memory_id, content, entities in zip(
                memory_ids, contents, memory_entities, strict=True
            )
        ]

        def validate(value: LinkingOutput) -> None:
            self._validate_linking(
                value, memory_ids, legal_by_memory, memory_entities, memory_space_id
            )

        output, _ = await self._structured_output(
            stage="domain_linking",
            system_prompt=LINKING_SYSTEM,
            user_prompt=linking_input(new_memories, candidates),
            adapter=TypeAdapter(LinkingOutput),
            temperature=self.config.linking_temperature,
            validator=validate,
            sample_kind="link",
        )
        return cast(LinkingOutput, output)

    def _recall_initial_domain_candidates(
        self,
        memory_space_id: str,
        vectors: Sequence[Any],
        memory_entities: Sequence[list[str]],
    ) -> tuple[list[dict[str, Any]], set[str]]:
        catalog = self._store.domain_names(memory_space_id)
        memberships = self._store.domain_entity_names(memory_space_id)
        pool: dict[str, dict[str, Any]] = {}
        for vector, entities in zip(vectors, memory_entities, strict=True):
            for entity in entities:
                ranked = sorted(
                    (
                        (entry, float(entry.vector @ vector))
                        for entry in catalog
                        if entity in memberships.get(entry.object_id, set())
                        and normalized_name(entry.name) != normalized_name(entity)
                    ),
                    key=lambda item: (-item[1], normalized_name(item[0].name)),
                )
                for entry, score in ranked[: self.config.entity_domain_candidate_top_k]:
                    if score >= self.config.domain_candidate_min_similarity:
                        pool[entry.name] = {
                            "name": entry.name,
                            "entities": sorted(memberships[entry.object_id]),
                        }
        by_name = {normalized_name(entry.name): entry for entry in catalog}
        for entity in dict.fromkeys(
            name for entities in memory_entities for name in entities
        ):
            root = by_name.get(normalized_name(entity))
            name = root.name if root is not None else entity
            linked_entities = memberships.get(root.object_id, set()) if root else set()
            pool[name] = {"name": name, "entities": sorted(linked_entities | {entity})}
        if len(pool) > self.config.linking_candidate_warning_threshold:
            self._event(
                "linking_candidate_count_warning",
                severity="warning",
                memory_space_id=memory_space_id,
                candidate_count=len(pool),
            )
        return list(pool.values()), set(pool)

    def _validate_linking(
        self,
        output: LinkingOutput,
        memory_ids: Sequence[str],
        legal_by_memory: dict[str, set[str]],
        memory_entities: Sequence[list[str]],
        memory_space_id: str,
    ) -> None:
        expected = dict(zip(memory_ids, memory_entities, strict=True))
        if len(output.memories) != len(expected) or {
            item.memory_id for item in output.memories
        } != set(expected):
            raise ValidationError("return each supplied memory_id exactly once")
        memberships_by_id = self._store.domain_entity_names(memory_space_id)
        memberships = {
            entry.name: memberships_by_id.get(entry.object_id, set())
            for entry in self._store.domain_names(memory_space_id)
        }
        known_names = {normalized_name(name): name for name in memberships}
        for entity in {name for entities in memory_entities for name in entities}:
            name = known_names.get(normalized_name(entity), entity)
            memberships.setdefault(name, set()).add(entity)
        for memory in output.memories:
            entities = expected[memory.memory_id]
            assigned = [item.entity for item in memory.direct_assignments]
            if len(assigned) != len(entities) or set(assigned) != set(entities):
                raise ValidationError(
                    f"{memory.memory_id} needs exactly one direct assignment per supplied entity"
                )
            names = {item.domain for item in memory.direct_assignments} | set(
                memory.contextual_domains
            )
            if not names <= legal_by_memory[memory.memory_id]:
                raise ValidationError(
                    f"domain was not a candidate for {memory.memory_id}"
                )
            for item in memory.direct_assignments:
                if item.entity not in memberships[item.domain]:
                    raise ValidationError(
                        f"domain {item.domain} does not belong to entity {item.entity}"
                    )
            if len(names) > max(
                self.config.memory_active_domain_link_max, len(entities)
            ):
                raise ValidationError(f"{memory.memory_id} exceeds its link_limit")

    def _prepare_entity_domains(
        self,
        memory_space_id: str,
        entities: Sequence[NameEntry],
    ) -> tuple[list[PreparedDomain], list[tuple[str, str]]]:
        entries = {
            normalized_name(entry.name): entry
            for entry in self._store.domain_names(memory_space_id)
        }
        domains = []
        relations = []
        for entity in entities:
            key = normalized_name(entity.name)
            if key not in entries:
                entry = NameEntry(str(uuid4()), entity.name, entity.vector)
                entries[key] = entry
                domains.append(
                    PreparedDomain(
                        entry.object_id, entry.name, None, entry.vector, None
                    )
                )
            relations.append((entity.object_id, entries[key].object_id))
        return domains, sorted(relations)

    def _prepare_link_commit(
        self,
        memory_space_id: str,
        linking: LinkingOutput,
        domains: Sequence[PreparedDomain],
    ) -> tuple[list[PreparedLink], list[ExistingDomainTouch], set[str]]:
        known = {
            normalized_name(entry.name): entry
            for entry in self._store.domain_names(memory_space_id)
        }
        entries = dict(known)
        entries.update(
            {
                normalized_name(domain.name): NameEntry(
                    domain.domain_id, domain.name, domain.name_embedding
                )
                for domain in domains
            }
        )
        links = []
        for memory in linking.memories:
            direct = {item.domain for item in memory.direct_assignments}
            for name in sorted(direct | set(memory.contextual_domains)):
                links.append(
                    PreparedLink(
                        memory.memory_id,
                        entries[normalized_name(name)].object_id,
                        "direct" if name in direct else "contextual",
                    )
                )
        known_ids = {entry.object_id for entry in known.values()}
        counts = Counter(
            link.domain_id for link in links if link.domain_id in known_ids
        )
        touches = [
            ExistingDomainTouch(
                domain_id,
                self._store.domain_snapshot(
                    memory_space_id, domain_id
                ).summary_revision,
                count,
            )
            for domain_id, count in counts.items()
        ]
        affected = {link.domain_id for link in links} | {
            domain.domain_id for domain in domains
        }
        return links, touches, affected

    async def _complete_maintenance(
        self,
        memory_space_id: str,
        add_operation_id: str,
        affected_domain_ids: Sequence[str],
        signature_id: str,
        actor: str,
    ) -> list[MaintenanceResult]:
        outcomes: list[MaintenanceResult] = []
        for domain_id in affected_domain_ids:
            try:
                domain_outcomes = await self._maintain_domain(
                    memory_space_id,
                    add_operation_id,
                    domain_id,
                    signature_id,
                    actor,
                )
                outcomes.extend(domain_outcomes)
            except StageFailure as error:
                self._handle_maintenance_failure(memory_space_id, domain_id, error)
            except ProviderError as error:
                self._handle_maintenance_failure(
                    memory_space_id,
                    domain_id,
                    StageFailure(
                        "maintenance_embedding", error.error_class, error.message
                    ),
                )

        refresh_domain_ids = self._store.pending_summary_refresh_domain_ids(
            memory_space_id, add_operation_id
        )
        semaphore = asyncio.Semaphore(
            self.config.domain_summary_refresh_concurrency_per_episode
        )

        async def refresh_one(
            domain_id: str,
        ) -> MaintenanceResult | StageFailure:
            async with semaphore:
                try:
                    return await self._refresh_domain_summary(
                        memory_space_id,
                        add_operation_id,
                        domain_id,
                        signature_id,
                        actor,
                    )
                except StageFailure as error:
                    self._record_summary_refresh_failure(
                        memory_space_id, domain_id, error
                    )
                    return error
                except ProviderError as error:
                    failure = StageFailure(
                        "maintenance_embedding", error.error_class, error.message
                    )
                    self._record_summary_refresh_failure(
                        memory_space_id, domain_id, failure
                    )
                    return failure

        refresh_results = await asyncio.gather(
            *(refresh_one(domain_id) for domain_id in refresh_domain_ids)
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
        self, memory_space_id: str, domain_id: str, error: StageFailure
    ) -> None:
        self._event(
            "domain_summary_refresh_failed",
            severity="error",
            memory_space_id=memory_space_id,
            domain_id=domain_id,
            error_class=error.error_class.value,
            reason=error.message,
        )

    def _handle_maintenance_failure(
        self, memory_space_id: str, domain_id: str, error: StageFailure
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
            domain_id=domain_id,
            error_class=error.error_class.value,
            reason=error.message,
        )

    async def _refresh_domain_summary(
        self,
        memory_space_id: str,
        add_operation_id: str,
        domain_id: str,
        signature_id: str,
        actor: str,
    ) -> MaintenanceResult:
        snapshot = self._store.domain_snapshot(memory_space_id, domain_id)
        self._event(
            "domain_summary_refresh_started",
            memory_space_id=memory_space_id,
            domain_id=domain_id,
            memory_count=len(snapshot.memories),
        )
        if (
            len(snapshot.memories)
            <= self.config.domain_summary_refresh_llm_link_threshold
        ):
            summary = " "
        else:
            output, _ = await self._structured_output(
                stage="domain_summary_refresh",
                system_prompt=SUMMARY_REFRESH_SYSTEM,
                user_prompt=summary_refresh_input(snapshot),
                adapter=TypeAdapter(SummaryRefreshOutput),
                temperature=self.config.review_temperature,
                validator=lambda value: self._validate_summary(value.summary),
                sample_kind="summary",
            )
            output = cast(SummaryRefreshOutput, output)
            summary = output.summary
        vector = (await self._embed_documents([_name_summary(snapshot.name, summary)]))[
            0
        ]
        operation_id = self._store.commit_summary_refresh(
            add_operation_id=add_operation_id,
            memory_space_id=memory_space_id,
            snapshot=snapshot,
            summary=summary,
            summary_embedding=vector,
            signature_id=signature_id,
            actor=actor,
            config_signature=self.config.signature,
        )
        self._event(
            "domain_summary_refresh_completed",
            memory_space_id=memory_space_id,
            domain_id=domain_id,
            operation_id=operation_id,
        )
        return MaintenanceResult(domain_id, "summary_refresh", operation_id)

    async def _maintain_domain(
        self,
        memory_space_id: str,
        add_operation_id: str,
        domain_id: str,
        signature_id: str,
        actor: str,
    ) -> list[MaintenanceResult]:
        snapshot = self._store.domain_snapshot(memory_space_id, domain_id)
        memory_chars = sum(len(memory.content) for memory in snapshot.memories)
        split_due = (
            len(snapshot.memories) >= self.config.domain_split_memory_count_threshold
            or memory_chars >= self.config.domain_split_total_memory_chars_threshold
        )
        outcomes: list[MaintenanceResult] = []
        if split_due:
            self._event(
                "domain_split_started",
                memory_space_id=memory_space_id,
                domain_id=domain_id,
                memory_count=len(snapshot.memories),
                memory_chars=memory_chars,
            )
            split = await self._split_domain(
                memory_space_id,
                add_operation_id,
                snapshot,
                signature_id,
                actor,
            )
            outcomes.append(split)
            self._event(
                "domain_split_completed",
                memory_space_id=memory_space_id,
                domain_id=domain_id,
                operation_id=split.operation_id,
                result=split.operation,
                reason=split.reason,
            )
            if split.operation != "defer_split":
                return outcomes
            snapshot = self._store.domain_snapshot(memory_space_id, domain_id)
        if snapshot.new_memory_count >= self.config.domain_review_new_memory_threshold:
            self._event(
                "domain_review_started",
                memory_space_id=memory_space_id,
                domain_id=domain_id,
                memory_count=len(snapshot.memories),
            )
            operation_id, provenance_viewed = await self._review_domain(
                memory_space_id,
                add_operation_id,
                snapshot,
                signature_id,
                actor,
            )
            outcomes.append(MaintenanceResult(domain_id, "review", operation_id))
            self._event(
                "domain_review_completed",
                memory_space_id=memory_space_id,
                domain_id=domain_id,
                operation_id=operation_id,
                provenance_viewed=provenance_viewed,
            )
        return outcomes

    async def _review_domain(
        self,
        memory_space_id: str,
        add_operation_id: str,
        snapshot: DomainSnapshot,
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
            self._validate_review(memory_space_id, value, snapshot)

        first, _ = await self._structured_output(
            stage="domain_review",
            system_prompt=REVIEW_SYSTEM,
            user_prompt=review_input(snapshot),
            adapter=TypeAdapter(ReviewStageOutput),
            temperature=self.config.review_temperature,
            validator=validate_first,
            sample_kind="review",
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
                self._validate_review(memory_space_id, value, snapshot)

            final, _ = await self._structured_output(
                stage="domain_review",
                system_prompt=REVIEW_SYSTEM,
                user_prompt=review_input(snapshot, provenance),
                adapter=TypeAdapter(ReviewStageOutput),
                temperature=self.config.review_temperature,
                validator=validate_final,
                sample_kind="review",
            )
            review = cast(ReviewOutput, final)
        else:
            provenance_viewed = False
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
            domain_id=snapshot.domain_id,
            expected_revision=snapshot.summary_revision,
            updates=updates,
            retirements=review.retirements,
            signature_id=signature_id,
            actor=actor,
            config_signature=self.config.signature,
        )
        return operation_id, provenance_viewed

    def _validate_review(
        self,
        memory_space_id: str,
        review: ReviewOutput,
        snapshot: DomainSnapshot,
    ) -> None:
        current = {memory.memory_id: memory for memory in snapshot.memories}
        update_ids = [update.memory_id for update in review.updates]
        if len(update_ids) != len(set(update_ids)):
            raise ValidationError("review contains duplicate memory updates")
        if len(review.retirements) != len(set(review.retirements)):
            raise ValidationError("review contains duplicate retirements")
        if not set(update_ids) | set(review.retirements) <= set(current):
            raise ValidationError("review references a memory outside the domain")
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
                self._validate_memory_content(
                    replacement_content,
                    max_chars=self.config.review_memory_content_max_chars,
                )
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
                existing = self._store.existing_episode_ids(memory_space_id, episodes)
                if not set(episodes) <= existing:
                    raise ValidationError(
                        "replacement provenance contains an invalid episode"
                    )

    async def _split_domain(
        self,
        memory_space_id: str,
        add_operation_id: str,
        snapshot: DomainSnapshot,
        signature_id: str,
        actor: str,
    ) -> MaintenanceResult:
        def validate(value: SplitStageOutput) -> None:
            self._validate_split(memory_space_id, value, snapshot)

        output, _ = await self._structured_output(
            stage="domain_split",
            system_prompt=SPLIT_SYSTEM,
            user_prompt=split_input(snapshot),
            adapter=TypeAdapter(SplitStageOutput),
            temperature=self.config.split_temperature,
            validator=validate,
            sample_kind="split",
        )
        output = cast(SplitStageOutput, output)
        if isinstance(output, DeferSplitOutput):
            operation_id = self._store.record_deferred_split(
                memory_space_id=memory_space_id,
                domain_id=snapshot.domain_id,
                expected_revision=snapshot.summary_revision,
                reason=output.reason,
                actor=actor,
                config_signature=self.config.signature,
            )
            return MaintenanceResult(
                snapshot.domain_id, "defer_split", operation_id, output.reason
            )
        domains = (
            output.domains
            if isinstance(output, FullSplitOutput)
            else output.new_domains
        )
        catalog = {
            normalized_name(entry.name): entry
            for entry in self._store.domain_names(memory_space_id)
        }
        fresh_names = [
            domain.name
            for domain in domains
            if normalized_name(domain.name) not in catalog
        ]
        vectors = dict(
            zip(fresh_names, await self._embed_documents(fresh_names), strict=True)
        )
        prepared: list[PreparedSplitDomain] = []
        for domain in domains:
            existing = catalog.get(normalized_name(domain.name))
            domain_id = existing.object_id if existing is not None else str(uuid4())
            revision = (
                self._store.domain_snapshot(memory_space_id, domain_id).summary_revision
                if existing is not None
                else None
            )
            prepared.append(
                PreparedSplitDomain(
                    PreparedDomain(
                        domain_id,
                        existing.name if existing is not None else domain.name,
                        None,
                        existing.vector
                        if existing is not None
                        else vectors[domain.name],
                        None,
                    ),
                    tuple(
                        PreparedLink(link.memory_id, domain_id, link.basis)
                        for link in domain.links
                    ),
                    revision,
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
            new_domains=prepared,
            signature_id=signature_id,
            actor=actor,
            config_signature=self.config.signature,
        )
        return MaintenanceResult(snapshot.domain_id, result, operation_id)

    def _validate_split(
        self,
        memory_space_id: str,
        output: SplitStageOutput,
        snapshot: DomainSnapshot,
    ) -> None:
        if isinstance(output, DeferSplitOutput):
            if not output.reason.strip():
                raise ValidationError("defer_split requires a reason")
            return
        domains = (
            output.domains
            if isinstance(output, FullSplitOutput)
            else output.new_domains
        )
        if isinstance(output, FullSplitOutput):
            if (
                not self.config.domain_split_result_domain_min
                <= len(domains)
                <= self.config.domain_split_result_domain_max
            ):
                raise ValidationError("full split result domain count is invalid")
        elif not 1 <= len(domains) <= self.config.domain_split_result_domain_max - 1:
            raise ValidationError("partial split new domain count is invalid")
        normalized = [normalized_name(domain.name) for domain in domains]
        if len(normalized) != len(set(normalized)):
            raise ValidationError("split domain names must be unique within the batch")
        if normalized_name(snapshot.name) in normalized:
            raise ValidationError("split targets must differ from the original domain")
        input_ids = {memory.memory_id for memory in snapshot.memories}
        membership: Counter[str] = Counter()
        for domain in domains:
            self._validate_domain_name(domain.name)
            if domain.name.strip().lower() in {
                "other",
                "misc",
                "general",
                "others",
                "miscellaneous",
            }:
                raise ValidationError("split domain name has no semantic boundary")
            ids = [link.memory_id for link in domain.links]
            if len(ids) != len(set(ids)):
                raise ValidationError("a split domain contains duplicate memories")
            if len(ids) < self.config.domain_split_result_min_memories:
                raise ValidationError("a split domain has too few memories")
            if len(ids) > self.config.domain_split_result_target_memory_max:
                raise ValidationError("a split domain has too many memories")
            if not set(ids) <= input_ids:
                raise ValidationError(
                    "split references a memory outside the input domain"
                )
            if not any(link.basis == "direct" for link in domain.links):
                raise ValidationError("every split domain requires a direct link")
            membership.update(ids)
        if any(
            count > self.config.domain_split_memory_membership_max
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
        catalog = {
            normalized_name(entry.name): entry.object_id
            for entry in self._store.domain_names(memory_space_id)
        }
        existing_links = self._store.active_link_bases(memory_space_id)
        final_links = {
            pair: basis
            for pair, basis in existing_links.items()
            if pair[1] != snapshot.domain_id
        }
        for domain in domains:
            target = catalog.get(
                normalized_name(domain.name), "new:" + normalized_name(domain.name)
            )
            for link in domain.links:
                pair = (link.memory_id, target)
                if pair not in final_links or link.basis == "direct":
                    final_links[pair] = link.basis
        for memory_id in moved:
            bases = [
                basis
                for (linked_memory, _), basis in final_links.items()
                if linked_memory == memory_id
            ]
            if len(bases) > self._store.memory_link_limit(memory_id):
                raise ValidationError(
                    "split would exceed a memory's active link maximum"
                )
            if "direct" not in bases:
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
        sample_kind: str | None = None,
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
                turn = _LlmTurn(current_input, response.text)
                if sample_kind is not None:
                    self._offer_llm_sample(sample_kind, turn)
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
                return value, turn
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
                    timeout_seconds=self.config.provider_read_timeout_seconds,
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
            for name in memory.entities:
                self._validate_domain_name(name)

    def _validate_memory_content(
        self, content: str, *, max_chars: int | None = None
    ) -> None:
        if not content.strip():
            raise ValidationError("memory content must not be blank")
        limit = self.config.memory_content_max_chars if max_chars is None else max_chars
        if len(content) > limit:
            raise ValidationError("memory content exceeds character limit")
        if _SECRET_RE.search(content):
            raise ValidationError(
                "memory content appears to contain an authentication secret"
            )

    def _validate_domain_name(self, name: str) -> None:
        if not name.strip() or len(name) > self.config.domain_name_max_chars:
            raise ValidationError("domain name is blank or exceeds character limit")

    def _validate_summary(self, summary: str) -> None:
        if (
            not summary.strip()
            or len(summary) > self.config.generated_domain_summary_max_chars
        ):
            raise ValidationError(
                "generated domain summary is blank or exceeds character limit"
            )

    def _offer_llm_sample(self, kind: str, turn: _LlmTurn) -> None:
        if self._event_sink is None:
            return
        with self._llm_io_sample_lock:
            if self._llm_io_sample_counts[kind] >= LLM_IO_SAMPLE_QUOTAS[kind]:
                return
            if random.random() >= LLM_IO_SAMPLE_PROBABILITY:
                return
            self._llm_io_sample_counts[kind] += 1
        self._event(
            "llm_io_sample",
            kind=kind,
            user_prompt=turn.user_prompt,
            output=turn.output,
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
