from __future__ import annotations

import asyncio
import json

import pytest

from fluxfold import EpisodeBlock, FluxFold, FluxFoldConfig, NormalizedEpisode, Role
from fluxfold.errors import ErrorClass, ProviderError, SourceConflictError, StageFailure
from fluxfold.models import CandidateSubject
from fluxfold.providers import GenerationRequest, GenerationResponse
from tests.fakes import FakeEmbeddingProvider, FakeGenerationProvider


def _episode(key: str, content: str, sequence: int = 0) -> NormalizedEpisode:
    return NormalizedEpisode(
        source_type="test",
        source_key=key,
        source_sequence=sequence,
        blocks=(EpisodeBlock(f"{key}:0", 0, Role.USER, content),),
    )


def test_add_search_replay_and_source_conflict(tmp_path) -> None:
    async def scenario() -> None:
        generation = FakeGenerationProvider()
        events: list[dict[str, object]] = []
        engine = await FluxFold.open(
            db_path=str(tmp_path / "engine.sqlite3"),
            generation_provider=generation,
            embedding_provider=FakeEmbeddingProvider(),
            config=FluxFoldConfig().with_overrides(
                subject_candidate_min_similarity=-1.0,
                memory_candidate_min_similarity=-1.0,
                search_subject_min_similarity=-1.0,
                search_memory_min_similarity=-1.0,
            ),
            event_sink=events.append,
        )
        space = await engine.create_or_open_space("test:alice")
        first = await engine.add_episode(
            space.memory_space_id, _episode("session-1", "Alice likes hiking.")
        )
        assert first.memories_created == 1
        assert first.subjects_created == 1
        assert first.links_created == 1
        bank = engine.memory_bank()
        assert len(bank) == 1
        assert bank[0].space_key == "test:alice"
        assert [(subject.name, subject.summary) for subject in bank[0].subjects] == [
            ("Alice's hiking", "Alice likes hiking.")
        ]
        assert bank[0].subjects[0].memory_contents == ("Alice likes hiking.",)
        decision = next(
            event for event in events if event["event_type"] == "audit_episode_decision"
        )
        assert "links" not in decision
        assert "memories" not in decision
        assert decision["extraction"] == {
            "result": "memories",
            "memories": [
                {
                    "content": "Alice likes hiking.",
                    "subjects": ["Alice's hiking"],
                }
            ],
        }

        replay = await engine.add_episode(
            space.memory_space_id, _episode("session-1", "Alice likes hiking.")
        )
        assert replay.replayed is True
        assert replay.memories_created == 0
        assert (
            len(
                [
                    request
                    for request in generation.requests
                    if request.stage == "memory_extraction"
                ]
            )
            == 1
        )

        result = await engine.search(space.memory_space_id, "What does Alice enjoy?")
        assert [memory.content for memory in result.memories] == ["Alice likes hiking."]
        assert result.subjects[0].name == "Alice's hiking"
        assert "Alice likes hiking." in result.render()

        with pytest.raises(SourceConflictError):
            await engine.add_episode(
                space.memory_space_id, _episode("session-1", "Alice dislikes hiking.")
            )
        await engine.close()

    asyncio.run(scenario())


def test_add_episode_pipelines_extraction_ahead_of_ordered_linking(tmp_path) -> None:
    class ObservedPipelineProvider(FakeGenerationProvider):
        def __init__(self) -> None:
            super().__init__(extraction_delay_seconds=0.01)
            self.events: list[tuple[str, str]] = []
            self.second_extraction_started = asyncio.Event()

        async def generate(self, request: GenerationRequest):
            payload = json.loads(request.user_prompt)
            if request.stage == "memory_extraction":
                content = str(payload["episode"]["messages"][-1]["content"])
                self.events.append(("extract_started", content))
                if content == "Alice bought boots.":
                    self.second_extraction_started.set()
                response = await super().generate(request)
                self.events.append(("extract_completed", content))
                return response
            if request.stage == "subject_linking":
                content = str(payload["new_memories"][0]["content"])
                self.events.append(("link_started", content))
                if content == "Alice likes hiking.":
                    await asyncio.wait_for(
                        self.second_extraction_started.wait(), timeout=1
                    )
                response = await super().generate(request)
                self.events.append(("link_completed", content))
                return response
            return await super().generate(request)

    async def scenario() -> None:
        generation = ObservedPipelineProvider()
        engine = await FluxFold.open(
            db_path=str(tmp_path / "pipeline.sqlite3"),
            generation_provider=generation,
            embedding_provider=FakeEmbeddingProvider(),
            config=FluxFoldConfig().with_overrides(
                subject_candidate_min_similarity=-1.0,
                memory_candidate_min_similarity=-1.0,
            ),
        )
        space = await engine.create_or_open_space("test:pipeline")
        episodes = (
            _episode("one", "Alice likes hiking.", 0),
            _episode("two", "Alice bought boots.", 1),
        )
        results = await asyncio.gather(
            *(
                asyncio.create_task(engine.add_episode(space.memory_space_id, episode))
                for episode in episodes
            )
        )

        assert [result.memories_created for result in results] == [1, 1]
        assert generation.max_active_extractions == 1
        assert generation.events.index(
            ("link_started", "Alice likes hiking.")
        ) < generation.events.index(("extract_started", "Alice bought boots."))
        assert generation.events.index(
            ("extract_started", "Alice bought boots.")
        ) < generation.events.index(("link_completed", "Alice likes hiking."))
        assert [
            content for event, content in generation.events if event == "link_completed"
        ] == ["Alice likes hiking.", "Alice bought boots."]
        await engine.close()

    asyncio.run(scenario())


def test_add_episode_extraction_lanes_are_independent_between_spaces(tmp_path) -> None:
    async def scenario() -> None:
        generation = FakeGenerationProvider(extraction_delay_seconds=0.02)
        engine = await FluxFold.open(
            db_path=str(tmp_path / "independent-spaces.sqlite3"),
            generation_provider=generation,
            embedding_provider=FakeEmbeddingProvider(),
        )
        first = await engine.create_or_open_space("test:first-space")
        second = await engine.create_or_open_space("test:second-space")

        await asyncio.gather(
            engine.add_episode(
                first.memory_space_id, _episode("first", "Alice likes hiking.")
            ),
            engine.add_episode(
                second.memory_space_id, _episode("second", "Bob likes cycling.")
            ),
        )

        assert generation.max_active_extractions == 2
        await engine.close()

    asyncio.run(scenario())


def test_episode_links_all_new_memories_in_one_batch(tmp_path) -> None:
    class TwoMemoryExtraction(FakeGenerationProvider):
        async def generate(self, request: GenerationRequest):
            if request.stage == "memory_extraction":
                self.requests.append(request)
                return GenerationResponse(
                    text=json.dumps(
                        {
                            "result": "memories",
                            "memories": [
                                {"content": "Alice likes hiking."},
                                {"content": "Alice bought hiking boots."},
                            ],
                        }
                    ),
                    input_tokens=6,
                    output_tokens=4,
                    total_tokens=10,
                    request_id="two-memory-request",
                )
            return await super().generate(request)

    async def scenario() -> None:
        generation = TwoMemoryExtraction()
        engine = await FluxFold.open(
            db_path=str(tmp_path / "batch-linking.sqlite3"),
            generation_provider=generation,
            embedding_provider=FakeEmbeddingProvider(),
        )
        space = await engine.create_or_open_space("test:batch-linking")

        result = await engine.add_episode(
            space.memory_space_id, _episode("one", "Alice hiking facts.")
        )

        linking_requests = [
            request
            for request in generation.requests
            if request.stage == "subject_linking"
        ]
        assert len(linking_requests) == 1
        payload = json.loads(linking_requests[0].user_prompt)
        assert [item["memory_ref"] for item in payload["new_memories"]] == [
            "memory_1",
            "memory_2",
        ]
        assert payload["candidates"] == []
        assert payload["association_search_results"] == []
        assert payload["association_searches_remaining"] == 5
        assert result.memories_created == 2
        assert result.subjects_created == 1
        assert result.links_created == 2
        await engine.close()

    asyncio.run(scenario())


def test_initial_subject_candidates_union_per_memory_direct_and_global_pool(
    tmp_path, monkeypatch
) -> None:
    async def scenario() -> None:
        engine = await FluxFold.open(
            db_path=str(tmp_path / "candidate-pool.sqlite3"),
            generation_provider=FakeGenerationProvider(),
            embedding_provider=FakeEmbeddingProvider(),
        )
        space = await engine.create_or_open_space("test:candidate-pool")
        batches = iter(
            tuple(
                CandidateSubject(
                    f"subject-{batch}-{rank}",
                    f"Subject {batch}-{rank}",
                    1.0 - batch * 0.1 - rank * 0.01,
                )
                for rank in range(5)
            )
            for batch in range(6)
        )

        def candidate_subject_names(*args, **kwargs):
            del args
            assert kwargs["top_k"] == 5
            assert kwargs["min_similarity"] == 0.25
            return next(batches)

        monkeypatch.setattr(
            engine._store, "candidate_subject_names", candidate_subject_names
        )
        candidates, legal = engine._recall_initial_subject_candidates(
            space.memory_space_id, [object() for _ in range(6)]
        )

        direct_ids = [
            f"subject-{batch}-{rank}" for batch in range(6) for rank in range(2)
        ]
        assert [candidate["subject_id"] for candidate in candidates[:12]] == direct_ids
        assert set(direct_ids) <= legal
        assert len(candidates) == 18
        assert all(set(candidate) == {"subject_id", "name"} for candidate in candidates)
        await engine.close()

    asyncio.run(scenario())


def test_link_agent_allows_five_association_search_calls(tmp_path) -> None:
    class FiveAssociationSearches(FakeGenerationProvider):
        async def generate(self, request: GenerationRequest):
            if request.stage == "subject_linking":
                payload = json.loads(request.user_prompt)
                if payload["association_searches_remaining"] > 0:
                    self.requests.append(request)
                    call_number = len(payload["association_search_results"]) + 1
                    return GenerationResponse(
                        text=json.dumps(
                            {
                                "result": "association_search",
                                "query": f"Alice related topic {call_number}",
                            }
                        ),
                        input_tokens=6,
                        output_tokens=4,
                        total_tokens=10,
                        request_id=f"association-{call_number}",
                    )
            return await super().generate(request)

    async def scenario() -> None:
        generation = FiveAssociationSearches()
        engine = await FluxFold.open(
            db_path=str(tmp_path / "five-association-searches.sqlite3"),
            generation_provider=generation,
            embedding_provider=FakeEmbeddingProvider(),
        )
        space = await engine.create_or_open_space("test:five-association-searches")

        result = await engine.add_episode(
            space.memory_space_id, _episode("one", "Alice likes hiking.")
        )

        linking_requests = [
            request
            for request in generation.requests
            if request.stage == "subject_linking"
        ]
        assert len(linking_requests) == 6
        payloads = [json.loads(request.user_prompt) for request in linking_requests]
        assert [payload["association_searches_remaining"] for payload in payloads] == [
            5,
            4,
            3,
            2,
            1,
            0,
        ]
        assert [len(payload["association_search_results"]) for payload in payloads] == [
            0,
            1,
            2,
            3,
            4,
            5,
        ]
        assert result.memories_created == 1
        assert result.links_created == 1
        await engine.close()

    asyncio.run(scenario())


def test_episode_summary_refresh_concurrency_is_capped_at_five(tmp_path) -> None:
    class ConcurrentSummaries(FakeGenerationProvider):
        def __init__(self) -> None:
            super().__init__(always_new_subject=True)
            self.active_summaries = 0
            self.max_active_summaries = 0

        async def generate(self, request: GenerationRequest):
            if request.stage == "memory_extraction":
                self.requests.append(request)
                return GenerationResponse(
                    text=json.dumps(
                        {
                            "result": "memories",
                            "memories": [
                                {"content": f"Alice fact {index}."}
                                for index in range(7)
                            ],
                        }
                    ),
                    input_tokens=6,
                    output_tokens=4,
                    total_tokens=10,
                    request_id="seven-memory-request",
                )
            if request.stage == "subject_summary_refresh":
                self.active_summaries += 1
                self.max_active_summaries = max(
                    self.max_active_summaries, self.active_summaries
                )
                try:
                    await asyncio.sleep(0.01)
                    return await super().generate(request)
                finally:
                    self.active_summaries -= 1
            return await super().generate(request)

    async def scenario() -> None:
        generation = ConcurrentSummaries()
        engine = await FluxFold.open(
            db_path=str(tmp_path / "summary-concurrency.sqlite3"),
            generation_provider=generation,
            embedding_provider=FakeEmbeddingProvider(),
        )
        space = await engine.create_or_open_space("test:summary-concurrency")

        result = await engine.add_episode(
            space.memory_space_id, _episode("one", "Seven Alice facts.")
        )

        assert result.subjects_created == 7
        assert [item.operation for item in result.maintenance] == [
            "summary_refresh"
        ] * 7
        assert generation.max_active_summaries == 5
        await engine.close()

    asyncio.run(scenario())


def test_add_episode_persists_terminal_failure_and_replays_it(tmp_path) -> None:
    class RejectOneExtraction(FakeGenerationProvider):
        def __init__(self) -> None:
            super().__init__()
            self.rejection_count = 0

        async def generate(self, request: GenerationRequest):
            if request.stage == "memory_extraction":
                payload = json.loads(request.user_prompt)
                content = str(payload["episode"]["messages"][-1]["content"])
                if content == "REJECT":
                    self.rejection_count += 1
                    raise ProviderError(ErrorClass.POLICY_REJECTED, "rejected")
            return await super().generate(request)

    async def scenario() -> None:
        generation = RejectOneExtraction()
        events: list[dict[str, object]] = []
        engine = await FluxFold.open(
            db_path=str(tmp_path / "terminal.sqlite3"),
            generation_provider=generation,
            embedding_provider=FakeEmbeddingProvider(),
            event_sink=events.append,
        )
        space = await engine.create_or_open_space("test:terminal")
        rejected = _episode("rejected", "REJECT", 0)

        for _ in range(2):
            with pytest.raises(StageFailure) as raised:
                await engine.add_episode(space.memory_space_id, rejected)
            assert raised.value.error_class == ErrorClass.POLICY_REJECTED
        accepted = await engine.add_episode(
            space.memory_space_id, _episode("accepted", "Alice likes hiking.", 1)
        )

        assert generation.rejection_count == 1
        assert accepted.memories_created == 1
        assert (
            len(
                [
                    event
                    for event in events
                    if event["event_type"] == "episode_terminal_failure"
                ]
            )
            == 1
        )
        await engine.close()

    asyncio.run(scenario())


def test_existing_subject_link_triggers_review(tmp_path) -> None:
    async def scenario() -> None:
        generation = FakeGenerationProvider()
        events: list[dict[str, object]] = []
        engine = await FluxFold.open(
            db_path=str(tmp_path / "review.sqlite3"),
            generation_provider=generation,
            embedding_provider=FakeEmbeddingProvider(),
            config=FluxFoldConfig().with_overrides(
                subject_candidate_min_similarity=-1.0,
                memory_candidate_min_similarity=-1.0,
                subject_review_new_memory_threshold=1,
            ),
            event_sink=events.append,
        )
        space = await engine.create_or_open_space("test:review")
        await engine.add_episode(
            space.memory_space_id, _episode("one", "Alice likes hiking.", 0)
        )
        second = await engine.add_episode(
            space.memory_space_id, _episode("two", "Alice bought boots.", 1)
        )
        assert second.subjects_created == 0
        assert second.maintenance[0].operation == "review"
        assert (
            engine.space_statistics(space.memory_space_id)["subject_review_count"] == 1
        )
        decisions = [
            event for event in events if event["event_type"] == "audit_episode_decision"
        ]
        assert decisions[1]["extraction"]["memories"] == [
            {
                "content": "Alice bought boots.",
                "subjects": ["Alice's hiking"],
            }
        ]
        assert decisions[1]["new_subjects"] == []
        linking_events = [
            event
            for event in events
            if event["event_type"] == "subject_linking_completed"
        ]
        assert linking_events[0]["candidate_memory_count"] == 0
        assert linking_events[1]["candidate_memory_count"] == 0
        assert linking_events[1]["association_search_called"] is False
        assert linking_events[1]["association_additional_candidate_memory_count"] == 0
        await engine.close()

    asyncio.run(scenario())


def test_linked_existing_subject_summary_is_rewritten_without_old_summary(
    tmp_path,
) -> None:
    async def scenario() -> None:
        generation = FakeGenerationProvider()
        engine = await FluxFold.open(
            db_path=str(tmp_path / "refresh.sqlite3"),
            generation_provider=generation,
            embedding_provider=FakeEmbeddingProvider(),
            config=FluxFoldConfig().with_overrides(
                subject_candidate_min_similarity=-1.0,
                memory_candidate_min_similarity=-1.0,
            ),
        )
        space = await engine.create_or_open_space("test:refresh")
        await engine.add_episode(
            space.memory_space_id, _episode("one", "Alice likes hiking.", 0)
        )
        second = await engine.add_episode(
            space.memory_space_id, _episode("two", "Alice bought boots.", 1)
        )

        assert [item.operation for item in second.maintenance] == ["summary_refresh"]
        request = [
            item
            for item in generation.requests
            if item.stage == "subject_summary_refresh"
        ][-1]
        payload = json.loads(request.user_prompt)
        assert "summary" not in payload["subject"]
        assert {item["content"] for item in payload["subject"]["memories"]} == {
            "Alice likes hiking.",
            "Alice bought boots.",
        }
        snapshot = engine._store.subject_snapshot(
            space.memory_space_id,
            engine._store.active_subject_ids(space.memory_space_id)[0],
        )
        assert "Alice likes hiking." in snapshot.summary
        assert "Alice bought boots." in snapshot.summary
        assert snapshot.new_memory_count == 1
        await engine.close()

    asyncio.run(scenario())


def test_search_groups_subjects_and_only_exposes_five_summaries(tmp_path) -> None:
    async def scenario() -> None:
        generation = FakeGenerationProvider(always_new_subject=True)
        engine = await FluxFold.open(
            db_path=str(tmp_path / "search-groups.sqlite3"),
            generation_provider=generation,
            embedding_provider=FakeEmbeddingProvider(),
            config=FluxFoldConfig().with_overrides(
                subject_candidate_min_similarity=-1.0,
                memory_candidate_min_similarity=-1.0,
                search_subject_min_similarity=-1.0,
                search_memory_min_similarity=-1.0,
            ),
        )
        space = await engine.create_or_open_space("test:search-groups")
        for index in range(9):
            await engine.add_episode(
                space.memory_space_id,
                _episode(
                    str(index), f"Unique fact number {index} token {index * 17}.", index
                ),
            )

        result = await engine.search(space.memory_space_id, "all unique facts")
        assert len(result.subjects) == 9
        assert sum(subject.summary is not None for subject in result.subjects) == 5
        rendered = result.render()
        assert rendered.count("- Subject:") == 9
        assert rendered.count("  Summary:") == 5
        linking_payload = json.loads(
            [
                request.user_prompt
                for request in generation.requests
                if request.stage == "subject_linking"
            ][-1]
        )
        candidates = linking_payload["candidates"]
        assert len(candidates) == 5
        assert len({candidate["subject_id"] for candidate in candidates}) == 5
        assert all(set(candidate) == {"subject_id", "name"} for candidate in candidates)
        await engine.close()

    asyncio.run(scenario())


def test_failed_summary_refresh_is_recovered_by_episode_replay(tmp_path) -> None:
    class FailFirstRefresh(FakeGenerationProvider):
        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        async def generate(self, request: GenerationRequest):
            if request.stage == "subject_summary_refresh" and not self.failed:
                self.failed = True
                raise ProviderError(
                    ErrorClass.TRANSIENT_TRANSPORT, "temporary refresh failure"
                )
            return await super().generate(request)

    async def scenario() -> None:
        generation = FailFirstRefresh()
        events: list[dict[str, object]] = []
        database = tmp_path / "resume-refresh.sqlite3"
        engine = await FluxFold.open(
            db_path=str(database),
            generation_provider=generation,
            embedding_provider=FakeEmbeddingProvider(),
            config=FluxFoldConfig().with_overrides(
                subject_candidate_min_similarity=-1.0,
                memory_candidate_min_similarity=-1.0,
            ),
            event_sink=events.append,
        )
        space = await engine.create_or_open_space("test:resume-refresh")
        episode = _episode("one", "Alice likes hiking.", 0)
        with pytest.raises(StageFailure) as raised:
            await engine.add_episode(space.memory_space_id, episode)
        assert raised.value.error_class == ErrorClass.TRANSIENT_TRANSPORT
        assert any(
            event["event_type"] == "subject_summary_refresh_failed"
            and event["error_class"] == ErrorClass.TRANSIENT_TRANSPORT.value
            for event in events
        )
        assert any(
            event["event_type"] == "llm_call"
            and event["stage"] == "subject_summary_refresh"
            and event["result"] == "failed"
            for event in events
        )

        subject_id = engine._store.active_subject_ids(space.memory_space_id)[0]
        assert (
            engine._store.subject_snapshot(space.memory_space_id, subject_id).summary
            is None
        )
        await engine.close()

        resumed_generation = FakeGenerationProvider()
        resumed = await FluxFold.open(
            db_path=str(database),
            generation_provider=resumed_generation,
            embedding_provider=FakeEmbeddingProvider(),
        )
        replay = await resumed.add_episode(space.memory_space_id, episode)
        assert replay.replayed is True
        assert [item.operation for item in replay.maintenance] == ["summary_refresh"]
        assert {request.stage for request in resumed_generation.requests} == {
            "subject_summary_refresh"
        }
        refreshed = resumed._store.subject_snapshot(space.memory_space_id, subject_id)
        assert "Alice likes hiking." in refreshed.summary
        await resumed.close()

    asyncio.run(scenario())


def test_replay_only_retries_unfinished_summary_refresh_targets(tmp_path) -> None:
    class ThreeMemoriesWithOneRefreshFailure(FakeGenerationProvider):
        def __init__(self) -> None:
            super().__init__(always_new_subject=True)

        async def generate(self, request: GenerationRequest):
            if request.stage == "memory_extraction":
                self.requests.append(request)
                return GenerationResponse(
                    text=json.dumps(
                        {
                            "result": "memories",
                            "memories": [
                                {"content": f"Alice fact {index}."}
                                for index in range(3)
                            ],
                        }
                    ),
                    input_tokens=6,
                    output_tokens=4,
                    total_tokens=10,
                    request_id="three-memory-request",
                )
            if request.stage == "subject_summary_refresh":
                payload = json.loads(request.user_prompt)
                content = payload["subject"]["memories"][0]["content"]
                if content == "Alice fact 1.":
                    self.requests.append(request)
                    raise ProviderError(
                        ErrorClass.TRANSIENT_TRANSPORT,
                        "temporary refresh failure",
                    )
            return await super().generate(request)

    async def scenario() -> None:
        database = tmp_path / "partial-refresh-replay.sqlite3"
        generation = ThreeMemoriesWithOneRefreshFailure()
        engine = await FluxFold.open(
            db_path=str(database),
            generation_provider=generation,
            embedding_provider=FakeEmbeddingProvider(),
        )
        space = await engine.create_or_open_space("test:partial-refresh-replay")
        episode = _episode("one", "Three Alice facts.")

        with pytest.raises(StageFailure):
            await engine.add_episode(space.memory_space_id, episode)

        subject_ids = engine._store.active_subject_ids(space.memory_space_id)
        summaries = [
            engine._store.subject_snapshot(space.memory_space_id, subject_id).summary
            for subject_id in subject_ids
        ]
        assert sum(summary is None for summary in summaries) == 1
        assert sum(summary is not None for summary in summaries) == 2
        await engine.close()

        resumed_generation = FakeGenerationProvider()
        resumed = await FluxFold.open(
            db_path=str(database),
            generation_provider=resumed_generation,
            embedding_provider=FakeEmbeddingProvider(),
        )
        replay = await resumed.add_episode(space.memory_space_id, episode)

        assert replay.replayed is True
        assert [item.operation for item in replay.maintenance] == ["summary_refresh"]
        assert [request.stage for request in resumed_generation.requests] == [
            "subject_summary_refresh"
        ]
        assert all(
            resumed._store.subject_snapshot(space.memory_space_id, subject_id).summary
            is not None
            for subject_id in subject_ids
        )
        await resumed.close()

    asyncio.run(scenario())


def test_rebuild_retrieval_embeddings_switches_model_atomically(tmp_path) -> None:
    async def scenario() -> None:
        database = tmp_path / "rebuild.sqlite3"
        first = await FluxFold.open(
            db_path=str(database),
            generation_provider=FakeGenerationProvider(),
            embedding_provider=FakeEmbeddingProvider(revision="1"),
        )
        space = await first.create_or_open_space("test:rebuild")
        await first.add_episode(
            space.memory_space_id, _episode("one", "Alice likes hiking.")
        )
        await first.close()

        second = await FluxFold.open(
            db_path=str(database),
            generation_provider=FakeGenerationProvider(),
            embedding_provider=FakeEmbeddingProvider(revision="2"),
        )
        rebuilt = await second.rebuild_retrieval_embeddings(space.memory_space_id)
        assert rebuilt.memories_embedded == 1
        assert rebuilt.subjects_embedded == 1
        result = await second.search(space.memory_space_id, "Alice hiking")
        assert result.memories[0].content == "Alice likes hiking."
        await second.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("split_result", "episode_count", "expected_active_subjects", "expected_operation"),
    [
        ("full_split", 2, 2, "full_split"),
        ("partial_split", 3, 2, "partial_split"),
        ("defer_split", 2, 1, "defer_split"),
    ],
)
def test_subject_split_outcomes(
    tmp_path,
    split_result: str,
    episode_count: int,
    expected_active_subjects: int,
    expected_operation: str,
) -> None:
    async def scenario() -> None:
        generation = FakeGenerationProvider(split_result=split_result)
        engine = await FluxFold.open(
            db_path=str(tmp_path / f"{split_result}.sqlite3"),
            generation_provider=generation,
            embedding_provider=FakeEmbeddingProvider(),
            config=FluxFoldConfig().with_overrides(
                subject_candidate_min_similarity=-1.0,
                memory_candidate_min_similarity=-1.0,
                subject_split_memory_count_threshold=episode_count,
                subject_review_new_memory_threshold=10,
            ),
        )
        space = await engine.create_or_open_space(f"test:{split_result}")
        result = None
        for index in range(episode_count):
            result = await engine.add_episode(
                space.memory_space_id,
                _episode(str(index), f"Alice hiking fact {index}.", index),
            )
        assert result is not None
        assert result.maintenance[0].operation == expected_operation
        split_request = next(
            request
            for request in generation.requests
            if request.stage == "subject_split"
        )
        assert "summary" not in json.loads(split_request.user_prompt)["subject"]
        statistics = engine.space_statistics(space.memory_space_id)
        assert statistics["active_subjects"] == expected_active_subjects
        assert statistics["subject_split_count"] == 1
        await engine.close()

    asyncio.run(scenario())


def test_split_retries_when_a_moved_memory_would_lose_its_direct_link(tmp_path) -> None:
    async def scenario() -> None:
        generation = FakeGenerationProvider(
            split_result="full_split", split_missing_direct_first=True
        )
        events: list[dict[str, object]] = []
        engine = await FluxFold.open(
            db_path=str(tmp_path / "split-direct.sqlite3"),
            generation_provider=generation,
            embedding_provider=FakeEmbeddingProvider(),
            config=FluxFoldConfig().with_overrides(
                subject_candidate_min_similarity=-1.0,
                memory_candidate_min_similarity=-1.0,
                subject_split_memory_count_threshold=2,
                subject_review_new_memory_threshold=10,
            ),
            event_sink=events.append,
        )
        space = await engine.create_or_open_space("test:split-direct")
        result = None
        for index in range(2):
            result = await engine.add_episode(
                space.memory_space_id,
                _episode(str(index), f"Alice hiking fact {index}.", index),
            )
        assert result is not None
        assert result.maintenance[0].operation == "full_split"
        split_calls = [
            event
            for event in events
            if event["event_type"] == "llm_call" and event["stage"] == "subject_split"
        ]
        assert [event["result"] for event in split_calls] == ["failed", "success"]
        assert engine.space_statistics(space.memory_space_id)["active_subjects"] == 2
        await engine.close()

    asyncio.run(scenario())


def test_association_search_results_keep_the_detailed_candidate_view(tmp_path) -> None:
    async def scenario() -> None:
        generation = FakeGenerationProvider()
        events: list[dict[str, object]] = []
        engine = await FluxFold.open(
            db_path=str(tmp_path / "association.sqlite3"),
            generation_provider=generation,
            embedding_provider=FakeEmbeddingProvider(),
            config=FluxFoldConfig().with_overrides(
                subject_candidate_min_similarity=-1.0,
                memory_candidate_min_similarity=-1.0,
            ),
            event_sink=events.append,
        )
        space = await engine.create_or_open_space("test:association")
        await engine.add_episode(
            space.memory_space_id, _episode("one", "Alice likes hiking.", 0)
        )
        generation.association_search_once = True
        result = await engine.add_episode(
            space.memory_space_id, _episode("two", "Alice bought boots.", 1)
        )
        assert result.subjects_created == 0
        linking_calls = [
            request
            for request in generation.requests
            if request.stage == "subject_linking"
        ]
        assert len(linking_calls) == 3
        linking_events = [
            event
            for event in events
            if event["event_type"] == "subject_linking_completed"
        ]
        assert linking_events[1]["candidate_memory_count"] == 1
        assert linking_events[1]["association_search_called"] is True
        assert linking_events[1]["association_additional_candidate_memory_count"] == 1
        await engine.close()

    asyncio.run(scenario())


def test_no_valuable_memory_is_a_completed_replay(tmp_path) -> None:
    async def scenario() -> None:
        generation = FakeGenerationProvider()
        engine = await FluxFold.open(
            db_path=str(tmp_path / "empty.sqlite3"),
            generation_provider=generation,
            embedding_provider=FakeEmbeddingProvider(),
        )
        space = await engine.create_or_open_space("test:empty")
        first = await engine.add_episode(
            space.memory_space_id, _episode("one", "NO_MEMORY")
        )
        replay = await engine.add_episode(
            space.memory_space_id, _episode("one", "NO_MEMORY")
        )
        assert first.memories_created == 0
        assert first.operation_id is not None
        assert replay.replayed is True
        assert engine.space_statistics(space.memory_space_id)["active_memories"] == 0
        await engine.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("review_mode", ["provenance_request", "update_retire"])
def test_review_provenance_and_memory_changes(tmp_path, review_mode: str) -> None:
    async def scenario() -> None:
        generation = FakeGenerationProvider(review_mode=review_mode)
        events: list[dict[str, object]] = []
        engine = await FluxFold.open(
            db_path=str(tmp_path / f"review-{review_mode}.sqlite3"),
            generation_provider=generation,
            embedding_provider=FakeEmbeddingProvider(),
            config=FluxFoldConfig().with_overrides(
                subject_candidate_min_similarity=-1.0,
                memory_candidate_min_similarity=-1.0,
                search_subject_min_similarity=-1.0,
                search_memory_min_similarity=-1.0,
                subject_review_new_memory_threshold=1,
            ),
            event_sink=events.append,
        )
        space = await engine.create_or_open_space(f"test:{review_mode}")
        await engine.add_episode(
            space.memory_space_id, _episode("one", "Alice likes hiking.", 0)
        )
        result = await engine.add_episode(
            space.memory_space_id, _episode("two", "Alice bought boots.", 1)
        )
        assert result.maintenance[0].operation == "review"
        review_calls = [
            request
            for request in generation.requests
            if request.stage == "subject_review"
        ]
        assert len(review_calls) == (2 if review_mode == "provenance_request" else 1)
        assert all(
            "summary" not in json.loads(request.user_prompt)["subject"]
            for request in review_calls
        )
        review_completed = next(
            event
            for event in events
            if event["event_type"] == "subject_review_completed"
        )
        assert review_completed["provenance_viewed"] is (
            review_mode == "provenance_request"
        )
        if review_mode == "update_retire":
            search = await engine.search(space.memory_space_id, "Alice hiking")
            assert [memory.content for memory in search.memories] == [
                "Alice enjoys hiking."
            ]
            assert (
                engine.space_statistics(space.memory_space_id)["active_memories"] == 1
            )
        await engine.close()

    asyncio.run(scenario())
