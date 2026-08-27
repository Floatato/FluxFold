from __future__ import annotations

import asyncio

import pytest

from fluxfold import EpisodeBlock, FluxFold, FluxFoldConfig, NormalizedEpisode, Role
from fluxfold.errors import SourceConflictError
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
        )
        space = await engine.create_or_open_space("test:alice")
        first = await engine.add(
            space.memory_space_id, _episode("session-1", "Alice likes hiking.")
        )
        assert first.memories_created == 1
        assert first.subjects_created == 1
        assert first.links_created == 1

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


def test_existing_subject_append_triggers_review(tmp_path) -> None:
    async def scenario() -> None:
        generation = FakeGenerationProvider()
        engine = await FluxFold.open(
            db_path=str(tmp_path / "review.sqlite3"),
            generation_provider=generation,
            embedding_provider=FakeEmbeddingProvider(),
            config=FluxFoldConfig().with_overrides(
                subject_candidate_min_similarity=-1.0,
                memory_candidate_min_similarity=-1.0,
                subject_review_new_memory_threshold=1,
            ),
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
        generation = FakeGenerationProvider(association_search_once=True)
        engine = await FluxFold.open(
            db_path=str(tmp_path / "association.sqlite3"),
            generation_provider=generation,
            embedding_provider=FakeEmbeddingProvider(),
        )
        space = await engine.create_or_open_space("test:association")
        result = await engine.add(
            space.memory_space_id, _episode("one", "Alice likes hiking.")
        )
        assert result.subjects_created == 1
        linking_calls = [
            request
            for request in generation.requests
            if request.stage == "subject_linking"
        ]
        assert len(linking_calls) == 2
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
