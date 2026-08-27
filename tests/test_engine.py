from __future__ import annotations

import asyncio
import json

import pytest

from fluxfold import EpisodeBlock, FluxFold, FluxFoldConfig, NormalizedEpisode, Role
from fluxfold.errors import ErrorClass, ProviderError, SourceConflictError
from fluxfold.providers import GenerationRequest
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
        first = await engine.add(
            space.memory_space_id, _episode("session-1", "Alice likes hiking.")
        )
        assert first.memories_created == 1
        assert first.subjects_created == 1
        assert first.links_created == 1
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

        replay = await engine.add(
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
            await engine.add(
                space.memory_space_id, _episode("session-1", "Alice dislikes hiking.")
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
        await engine.add(
            space.memory_space_id, _episode("one", "Alice likes hiking.", 0)
        )
        second = await engine.add(
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
        assert linking_events[1]["candidate_memory_count"] == 1
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
        await engine.add(
            space.memory_space_id, _episode("one", "Alice likes hiking.", 0)
        )
        second = await engine.add(
            space.memory_space_id, _episode("two", "Alice bought boots.", 1)
        )

        assert [item.operation for item in second.maintenance] == ["summary_refresh"]
        request = next(
            item
            for item in generation.requests
            if item.stage == "subject_summary_refresh"
        )
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
            await engine.add(
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
        candidate_groups = linking_payload["candidates_by_memory"]["memory_1"][
            "subjects"
        ]
        assert len(candidate_groups) == 8
        assert len({group["subject_id"] for group in candidate_groups}) == 8
        assert sum("summary" in group for group in candidate_groups) == 5
        await engine.close()

    asyncio.run(scenario())


def test_failed_summary_refresh_is_local_and_a_later_link_rewrites_it(tmp_path) -> None:
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
        engine = await FluxFold.open(
            db_path=str(tmp_path / "resume-refresh.sqlite3"),
            generation_provider=generation,
            embedding_provider=FakeEmbeddingProvider(),
            config=FluxFoldConfig().with_overrides(
                subject_candidate_min_similarity=-1.0,
                memory_candidate_min_similarity=-1.0,
            ),
            event_sink=events.append,
        )
        space = await engine.create_or_open_space("test:resume-refresh")
        await engine.add(
            space.memory_space_id, _episode("one", "Alice likes hiking.", 0)
        )
        second_episode = _episode("two", "Alice bought boots.", 1)
        second = await engine.add(space.memory_space_id, second_episode)
        assert second.maintenance == ()
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
        stale = engine._store.subject_snapshot(space.memory_space_id, subject_id)
        assert stale.summary == "Alice likes hiking."

        third = await engine.add(
            space.memory_space_id, _episode("three", "Alice planned a trail.", 2)
        )
        assert [item.operation for item in third.maintenance] == ["summary_refresh"]
        refreshed = engine._store.subject_snapshot(space.memory_space_id, subject_id)
        assert "Alice likes hiking." in refreshed.summary
        assert "Alice bought boots." in refreshed.summary
        assert "Alice planned a trail." in refreshed.summary
        await engine.close()

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
        await first.add(space.memory_space_id, _episode("one", "Alice likes hiking."))
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
            result = await engine.add(
                space.memory_space_id,
                _episode(str(index), f"Alice hiking fact {index}.", index),
            )
        assert result is not None
        assert result.maintenance[0].operation == expected_operation
        statistics = engine.space_statistics(space.memory_space_id)
        assert statistics["active_subjects"] == expected_active_subjects
        assert statistics["subject_split_count"] == 1
        await engine.close()

    asyncio.run(scenario())


def test_association_search_is_called_at_most_once(tmp_path) -> None:
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
        await engine.add(
            space.memory_space_id, _episode("one", "Alice likes hiking.", 0)
        )
        generation.association_search_once = True
        result = await engine.add(
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
        assert linking_events[1]["association_additional_candidate_memory_count"] == 0
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
        first = await engine.add(space.memory_space_id, _episode("one", "NO_MEMORY"))
        replay = await engine.add(space.memory_space_id, _episode("one", "NO_MEMORY"))
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
        await engine.add(
            space.memory_space_id, _episode("one", "Alice likes hiking.", 0)
        )
        result = await engine.add(
            space.memory_space_id, _episode("two", "Alice bought boots.", 1)
        )
        assert result.maintenance[0].operation == "review"
        review_calls = [
            request
            for request in generation.requests
            if request.stage == "subject_review"
        ]
        assert len(review_calls) == (2 if review_mode == "provenance_request" else 1)
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
