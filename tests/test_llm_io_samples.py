from __future__ import annotations

import asyncio
import json

from benchmarks.artifacts import ArtifactWriter, RunPaths

from fluxfold import EpisodeBlock, FluxFold, FluxFoldConfig, NormalizedEpisode, Role
from fluxfold.models import MemoryBankSpace, MemoryBankSubject
from tests.fakes import FakeEmbeddingProvider, FakeGenerationProvider


def _episode(key: str, content: str, sequence: int = 0) -> NormalizedEpisode:
    return NormalizedEpisode(
        source_type="test",
        source_key=key,
        source_sequence=sequence,
        blocks=(EpisodeBlock(f"{key}:0", 0, Role.USER, content),),
    )


def _samples(
    events: list[dict[str, object]], kind: str | None = None
) -> list[dict[str, object]]:
    items = [event for event in events if event["event_type"] == "llm_io_sample"]
    if kind is not None:
        items = [event for event in items if event["kind"] == kind]
    return items


async def _open(
    tmp_path,
    *,
    name: str,
    generation: FakeGenerationProvider,
    events: list[dict[str, object]],
    **overrides: object,
) -> FluxFold:
    return await FluxFold.open(
        db_path=str(tmp_path / f"{name}.sqlite3"),
        generation_provider=generation,
        embedding_provider=FakeEmbeddingProvider(),
        config=FluxFoldConfig().with_overrides(
            subject_candidate_min_similarity=-1.0,
            memory_candidate_min_similarity=-1.0,
            **overrides,
        ),
        event_sink=events.append,
    )


def test_extract_link_and_summary_calls_are_sampled_at_random(
    tmp_path, monkeypatch
) -> None:
    async def scenario() -> None:
        draws = iter((0.09, 0.09, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
        monkeypatch.setattr("fluxfold.engine.random.random", lambda: next(draws))
        generation = FakeGenerationProvider()
        events: list[dict[str, object]] = []
        engine = await _open(
            tmp_path, name="quota", generation=generation, events=events
        )
        space = await engine.create_or_open_space("test:quota")
        for index, content in enumerate(
            (
                "Alice likes hiking.",
                "Alice bought boots.",
                "Alice hikes on weekends.",
                "Alice plans a mountain hike.",
            )
        ):
            await engine.add_episode(
                space.memory_space_id, _episode(str(index), content, index)
            )
        assert len(_samples(events, "extract")) == 1
        assert len(_samples(events, "link")) == 1
        assert _samples(events, "summary") == []
        first_extract = _samples(events, "extract")[0]
        assert "Alice likes hiking." in str(first_extract["user_prompt"])
        assert json.loads(str(first_extract["output"]))["result"] == "memories"
        first_link = _samples(events, "link")[0]
        assert json.loads(str(first_link["output"]))["result"] == "links"
        await engine.close()

    asyncio.run(scenario())


def test_sampling_stops_drawing_after_each_category_reaches_its_quota(
    tmp_path, monkeypatch
) -> None:
    async def scenario() -> None:
        draw_count = 0

        def draw() -> float:
            nonlocal draw_count
            draw_count += 1
            return 0.0

        monkeypatch.setattr("fluxfold.engine.random.random", draw)
        generation = FakeGenerationProvider(always_new_subject=True)
        events: list[dict[str, object]] = []
        engine = await _open(
            tmp_path, name="quota", generation=generation, events=events
        )
        space = await engine.create_or_open_space("test:quota")
        for index in range(12):
            await engine.add_episode(
                space.memory_space_id,
                _episode(str(index), f"Alice hiking fact {index}.", index),
            )
        assert len(_samples(events, "extract")) == 10
        assert len(_samples(events, "link")) == 10
        assert draw_count == 20
        await engine.close()

    asyncio.run(scenario())


def test_review_calls_share_one_sample_category(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        monkeypatch.setattr("fluxfold.engine.random.random", lambda: 0.0)
        generation = FakeGenerationProvider()
        events: list[dict[str, object]] = []
        engine = await _open(
            tmp_path,
            name="review",
            generation=generation,
            events=events,
            subject_review_new_memory_threshold=1,
        )
        space = await engine.create_or_open_space("test:review")
        for index, content in enumerate(
            (
                "Alice likes hiking.",
                "Alice bought boots.",
                "Alice hikes on weekends.",
            )
        ):
            await engine.add_episode(
                space.memory_space_id, _episode(str(index), content, index)
            )
        reviews = _samples(events, "review")
        assert len(reviews) == 2
        first = reviews[0]
        assert json.loads(str(first["user_prompt"]))["requested_provenance"] is None
        assert json.loads(str(first["output"]))["result"] == "review"
        await engine.close()

    asyncio.run(scenario())


def test_split_is_sampled_up_to_its_quota(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        monkeypatch.setattr("fluxfold.engine.random.random", lambda: 0.0)
        generation = FakeGenerationProvider(split_result="full_split")
        events: list[dict[str, object]] = []
        engine = await _open(
            tmp_path,
            name="split",
            generation=generation,
            events=events,
            subject_split_memory_count_threshold=3,
            subject_review_new_memory_threshold=10,
        )
        for space_name in ("one", "two"):
            space = await engine.create_or_open_space(f"test:split-{space_name}")
            for index in range(3):
                await engine.add_episode(
                    space.memory_space_id,
                    _episode(
                        f"{space_name}-{index}",
                        f"Alice hiking fact {index}.",
                        index,
                    ),
                )
        splits = _samples(events, "split")
        assert len(splits) == 2
        first = splits[0]
        assert json.loads(str(first["output"]))["result"] == "full_split"
        await engine.close()

    asyncio.run(scenario())


def test_association_search_samples_only_the_final_decision(
    tmp_path, monkeypatch
) -> None:
    async def scenario() -> None:
        monkeypatch.setattr("fluxfold.engine.random.random", lambda: 0.0)
        generation = FakeGenerationProvider()
        events: list[dict[str, object]] = []
        engine = await _open(
            tmp_path, name="association", generation=generation, events=events
        )
        space = await engine.create_or_open_space("test:association")
        await engine.add_episode(
            space.memory_space_id, _episode("one", "Alice likes hiking.", 0)
        )
        generation.association_search_once = True
        await engine.add_episode(
            space.memory_space_id, _episode("two", "Alice bought boots.", 1)
        )
        assert len(_samples(events, "link")) == 1
        samples = _samples(events, "link_association_search")
        assert len(samples) == 1
        sample = samples[0]
        prompt = json.loads(str(sample["user_prompt"]))
        assert prompt["association_search_results"][0]["query"] == ("Alice activities")
        assert json.loads(str(sample["output"]))["result"] == "links"
        assert all(
            json.loads(str(item["output"]))["result"] != "association_search"
            for item in _samples(events)
        )
        await engine.close()

    asyncio.run(scenario())


def test_provenance_review_samples_each_llm_call(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        monkeypatch.setattr("fluxfold.engine.random.random", lambda: 0.0)
        generation = FakeGenerationProvider(review_mode="provenance_request")
        events: list[dict[str, object]] = []
        engine = await _open(
            tmp_path,
            name="provenance",
            generation=generation,
            events=events,
            subject_review_new_memory_threshold=1,
        )
        space = await engine.create_or_open_space("test:provenance")
        await engine.add_episode(
            space.memory_space_id, _episode("one", "Alice likes hiking.", 0)
        )
        await engine.add_episode(
            space.memory_space_id, _episode("two", "Alice bought boots.", 1)
        )
        samples = _samples(events, "review")
        assert len(samples) == 2
        first, second = samples
        assert json.loads(str(first["output"]))["result"] == "provenance_request"
        assert json.loads(str(first["user_prompt"]))["requested_provenance"] is None
        second_prompt = json.loads(str(second["user_prompt"]))
        assert second_prompt["requested_provenance"] is not None
        assert json.loads(str(second["output"]))["result"] == "review"
        await engine.close()

    asyncio.run(scenario())


def test_repaired_success_samples_the_actual_repair_prompt(
    tmp_path, monkeypatch
) -> None:
    async def scenario() -> None:
        monkeypatch.setattr("fluxfold.engine.random.random", lambda: 0.0)
        generation = FakeGenerationProvider(
            split_result="full_split", split_missing_direct_first=True
        )
        events: list[dict[str, object]] = []
        engine = await _open(
            tmp_path,
            name="repair",
            generation=generation,
            events=events,
            subject_split_memory_count_threshold=3,
            subject_review_new_memory_threshold=10,
        )
        space = await engine.create_or_open_space("test:repair")
        for index in range(3):
            await engine.add_episode(
                space.memory_space_id,
                _episode(str(index), f"Alice hiking fact {index}.", index),
            )
        samples = _samples(events, "split")
        assert len(samples) == 2
        assert "Your immediately previous response failed validation" not in str(
            samples[0]["user_prompt"]
        )
        assert "Your immediately previous response failed validation" in str(
            samples[1]["user_prompt"]
        )
        await engine.close()

    asyncio.run(scenario())


def test_artifact_writer_renders_readable_samples_and_enforces_quota(tmp_path) -> None:
    writer = ArtifactWriter(RunPaths(tmp_path / "run"))
    writer.run_id = "run-1"
    extract = {
        "event_type": "llm_io_sample",
        "kind": "extract",
        "timestamp_ms": 1,
        "user_prompt": '{"episode":{"messages":[{"content":"hi"}]}}',
        "output": '{"result":"memories","memories":[{"content":"hi"}]}',
    }
    for _ in range(11):
        writer.event(extract)
    text = writer.paths.llm_io_samples.read_text(encoding="utf-8")
    assert text.count("## extract · sample") == 10
    assert "## extract · sample 11" not in text
    assert "### System prompt" not in text
    assert "run_id" not in text
    assert "request_id" not in text
    assert "timestamp_ms" not in text
    assert "### User prompt" in text
    assert "### Model output" in text
    assert '"content": "hi"' in text
    assert not writer.paths.events.exists()
    resumed = ArtifactWriter(RunPaths(tmp_path / "run"))
    resumed.event(extract)
    resumed_text = resumed.paths.llm_io_samples.read_text(encoding="utf-8")
    assert resumed_text.count("## extract · sample") == 10


def test_artifact_writer_enforces_association_search_quota(tmp_path) -> None:
    writer = ArtifactWriter(RunPaths(tmp_path / "run"))
    event = {
        "event_type": "llm_io_sample",
        "kind": "link_association_search",
        "timestamp_ms": 1,
        "user_prompt": '{"association_search_results":[{"query":"q1"}]}',
        "output": '{"result":"links","memories":[]}',
    }
    for _ in range(6):
        writer.event(event)
    text = writer.paths.llm_io_samples.read_text(encoding="utf-8")
    assert text.count("## link_association_search · sample") == 5
    assert '"query": "q1"' in text
    assert '"result": "association_search"' not in text


def test_artifact_writer_overwrites_memory_bank_snapshot(tmp_path) -> None:
    writer = ArtifactWriter(RunPaths(tmp_path / "run"))
    first = (
        MemoryBankSpace(
            "space-a",
            (
                MemoryBankSubject(
                    "Hiking",
                    "Alice hikes.",
                    ("Alice likes hiking.",),
                ),
            ),
        ),
    )
    second = (
        MemoryBankSpace(
            "space-a",
            (
                MemoryBankSubject(
                    "Hiking",
                    "Alice hikes and owns boots.",
                    ("Alice likes hiking.", "Alice owns hiking boots."),
                ),
            ),
        ),
        MemoryBankSpace("space-b", ()),
    )
    writer.write_memory_bank(first, updated_after="episode `s-1` in `space-a`")
    writer.write_memory_bank(second, updated_after="build completed")
    text = writer.paths.memory_bank.read_text(encoding="utf-8")
    assert text.count("# Memory bank") == 1
    assert "after build completed" in text
    assert "episode `s-1`" not in text
    assert "### Hiking" in text
    assert "Alice hikes and owns boots." in text
    assert "- Alice likes hiking." in text
    assert "- Alice owns hiking boots." in text
    assert "## `space-b`" in text
    assert "_(no active subjects)_" in text
    assert "subject_id" not in text
    assert "memory_id" not in text
