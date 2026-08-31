from __future__ import annotations

import asyncio
import json

from benchmarks.artifacts import ArtifactWriter, RunPaths

from fluxfold import EpisodeBlock, FluxFold, FluxFoldConfig, NormalizedEpisode, Role
from fluxfold.models import MemoryBankSpace, MemoryBankSubject
from fluxfold.prompts import (
    EXTRACTION_SYSTEM,
    LINKING_SYSTEM,
    REVIEW_SYSTEM,
    SPLIT_SYSTEM,
    SUMMARY_REFRESH_SYSTEM,
)
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


def _round(event: dict[str, object], index: int = 0) -> dict[str, object]:
    rounds = event["rounds"]
    assert isinstance(rounds, list)
    payload = rounds[index]
    assert isinstance(payload, dict)
    return payload


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


def test_extract_link_and_summary_are_sampled_twice(tmp_path) -> None:
    async def scenario() -> None:
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
            )
        ):
            await engine.add_episode(
                space.memory_space_id, _episode(str(index), content, index)
            )
        assert [event["kind"] for event in _samples(events, "extract")] == [
            "extract",
            "extract",
        ]
        assert len(_samples(events, "link")) == 2
        assert len(_samples(events, "summary")) == 2
        first_extract = _round(_samples(events, "extract")[0])
        assert first_extract["system_prompt"] == EXTRACTION_SYSTEM
        assert "Alice likes hiking." in str(first_extract["user_prompt"])
        assert json.loads(str(first_extract["output"]))["result"] == "memories"
        first_link = _round(_samples(events, "link")[0])
        assert first_link["system_prompt"] == LINKING_SYSTEM
        assert json.loads(str(first_link["output"]))["result"] == "links"
        first_summary = _round(_samples(events, "summary")[0])
        assert first_summary["system_prompt"] == SUMMARY_REFRESH_SYSTEM
        await engine.close()

    asyncio.run(scenario())


def test_review_without_provenance_is_sampled_twice(tmp_path) -> None:
    async def scenario() -> None:
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
        assert _samples(events, "review_provenance") == []
        first = _round(reviews[0])
        assert first["system_prompt"] == REVIEW_SYSTEM
        assert json.loads(str(first["user_prompt"]))["requested_provenance"] is None
        assert json.loads(str(first["output"]))["result"] == "review"
        await engine.close()

    asyncio.run(scenario())


def test_split_is_sampled_twice(tmp_path) -> None:
    async def scenario() -> None:
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
        first = _round(splits[0])
        assert first["system_prompt"] == SPLIT_SYSTEM
        assert json.loads(str(first["output"]))["result"] == "full_split"
        await engine.close()

    asyncio.run(scenario())


def test_association_search_records_both_rounds(tmp_path) -> None:
    async def scenario() -> None:
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
        first = _round(samples[0], 0)
        second = _round(samples[0], 1)
        assert json.loads(str(first["output"])) == {
            "result": "association_search",
            "query": "Alice activities",
        }
        second_prompt = json.loads(str(second["user_prompt"]))
        assert second_prompt["association_search_results"][0]["query"] == (
            "Alice activities"
        )
        assert json.loads(str(second["output"]))["result"] == "links"
        await engine.close()

    asyncio.run(scenario())


def test_provenance_viewed_records_both_rounds(tmp_path) -> None:
    async def scenario() -> None:
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
        assert _samples(events, "review") == []
        samples = _samples(events, "review_provenance")
        assert len(samples) == 1
        first = _round(samples[0], 0)
        second = _round(samples[0], 1)
        assert json.loads(str(first["output"]))["result"] == "provenance_request"
        assert json.loads(str(first["user_prompt"]))["requested_provenance"] is None
        second_prompt = json.loads(str(second["user_prompt"]))
        assert second_prompt["requested_provenance"] is not None
        assert json.loads(str(second["output"]))["result"] == "review"
        await engine.close()

    asyncio.run(scenario())


def test_repaired_success_is_not_sampled(tmp_path) -> None:
    async def scenario() -> None:
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
        assert _samples(events, "split") == []
        await engine.close()

    asyncio.run(scenario())


def test_artifact_writer_renders_readable_samples_and_enforces_quota(tmp_path) -> None:
    writer = ArtifactWriter(RunPaths(tmp_path / "run"))
    writer.run_id = "run-1"
    extract = {
        "event_type": "llm_io_sample",
        "kind": "extract",
        "timestamp_ms": 1,
        "rounds": [
            {
                "stage": "memory_extraction",
                "request_id": "req-1",
                "system_prompt": "You are a memory extractor.",
                "user_prompt": '{"episode":{"messages":[{"content":"hi"}]}}',
                "output": '{"result":"memories","memories":[{"content":"hi"}]}',
            }
        ],
    }
    writer.event(extract)
    writer.event(extract)
    writer.event(extract)
    writer.event(
        {
            "event_type": "llm_io_sample",
            "kind": "review_provenance",
            "timestamp_ms": 2,
            "rounds": [
                {
                    "stage": "subject_review",
                    "request_id": "req-2",
                    "system_prompt": "Review the subject.",
                    "user_prompt": '{"requested_provenance":null}',
                    "output": '{"result":"provenance_request","memory_ids":["m1"]}',
                },
                {
                    "stage": "subject_review",
                    "request_id": "req-3",
                    "system_prompt": "Review the subject.",
                    "user_prompt": '{"requested_provenance":{"m1":[]}}',
                    "output": '{"result":"review","updates":[],"retirements":[]}',
                },
            ],
        }
    )
    text = writer.paths.llm_io_samples.read_text(encoding="utf-8")
    assert text.count("## extract · sample") == 2
    assert "## extract · sample 3" not in text
    assert "You are a memory extractor." in text
    assert '"content": "hi"' in text
    assert "Round 1 — provenance request" in text
    assert "Round 2 — final review decision" in text
    assert not writer.paths.events.exists()
    resumed = ArtifactWriter(RunPaths(tmp_path / "run"))
    resumed.event(extract)
    resumed_text = resumed.paths.llm_io_samples.read_text(encoding="utf-8")
    assert resumed_text.count("## extract · sample") == 2


def test_artifact_writer_renders_multi_round_association_search(tmp_path) -> None:
    writer = ArtifactWriter(RunPaths(tmp_path / "run"))
    outputs = (
        '{"result":"association_search","query":"q1"}',
        '{"result":"association_search","query":"q2"}',
        '{"result":"links","new_subjects":[],"links":[]}',
    )
    writer.event(
        {
            "event_type": "llm_io_sample",
            "kind": "link_association_search",
            "timestamp_ms": 1,
            "rounds": [
                {
                    "stage": "subject_linking",
                    "request_id": f"req-{index}",
                    "system_prompt": "Link memories.",
                    "user_prompt": "{}",
                    "output": output,
                }
                for index, output in enumerate(outputs, start=1)
            ],
        }
    )
    text = writer.paths.llm_io_samples.read_text(encoding="utf-8")
    assert "### Round 1 — association_search request" in text
    assert "### Round 2 — association_search request" in text
    assert "### Round 3 — final linking decision" in text
    assert '"query": "q1"' in text
    assert '"query": "q2"' in text


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
