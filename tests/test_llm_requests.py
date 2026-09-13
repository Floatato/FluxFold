from __future__ import annotations

import asyncio
import json

from pydantic import TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from fluxfold import EpisodeBlock, FluxFold, NormalizedEpisode, Role
from fluxfold.models import LinkingStageOutput, MemorySnapshot, SubjectSnapshot
from fluxfold.prompts import (
    extraction_input,
    repair_input,
    review_input,
    split_input,
    summary_refresh_input,
    validation_feedback,
)
from fluxfold.providers import GenerationRequest, GenerationResponse
from tests.fakes import FakeEmbeddingProvider, FakeGenerationProvider


class RepairingGenerationProvider(FakeGenerationProvider):
    def __init__(self) -> None:
        super().__init__()
        self._invalid_linking_outputs = iter(
            (
                json.dumps({"result": "links", "first_marker": True}),
                json.dumps({"result": "links", "second_marker": True}),
            )
        )

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        if request.stage == "subject_linking":
            try:
                text = next(self._invalid_linking_outputs)
            except StopIteration:
                pass
            else:
                self.requests.append(request)
                return GenerationResponse(
                    text=text,
                    input_tokens=6,
                    output_tokens=4,
                    total_tokens=10,
                    request_id="invalid-request",
                )
        return await super().generate(request)


def test_llm_inputs_project_only_task_relevant_fields() -> None:
    episode = NormalizedEpisode(
        source_type="test",
        source_key="session-1",
        source_sequence=7,
        payload_version="2",
        source_started_at=1_694_044_800_000,
        source_ended_at=1_694_048_400_000,
        source_timezone="Europe/Paris",
        blocks=(
            EpisodeBlock(
                "block-1",
                3,
                Role.USER,
                "Alice likes hiking.",
                message_phase="history",
                speaker_id="alice",
                speaker_name="Alice Example",
                observed_at=1_694_045_100_000,
                metadata={"unused": True},
                preprocessor_version="9",
            ),
        ),
    )
    assert json.loads(extraction_input(episode)) == {
        "episode": {
            "source_started_at": "2023-09-07T00:00:00Z (Thursday)",
            "source_ended_at": "2023-09-07T01:00:00Z (Thursday)",
            "source_timezone": "Europe/Paris",
            "messages": [
                {
                    "speaker_id": "alice",
                    "content": "Alice likes hiking.",
                    "observed_at": "2023-09-07T00:05:00Z (Thursday)",
                }
            ],
        }
    }

    episode_without_times = NormalizedEpisode(
        source_type="test",
        source_key="session-2",
        source_sequence=8,
        blocks=(EpisodeBlock("block-2", 0, Role.USER, "No time metadata."),),
    )
    assert json.loads(extraction_input(episode_without_times)) == {
        "episode": {
            "messages": [{"speaker_id": "user", "content": "No time metadata."}]
        }
    }

    snapshot = SubjectSnapshot(
        "subject-id",
        "Alice",
        "stale summary",
        4,
        2,
        (
            MemorySnapshot(
                "memory-id",
                "Alice likes hiking.",
                1_694_044_800_000,
                ("episode-id",),
                "direct",
            ),
        ),
    )
    review = json.loads(review_input(snapshot))
    assert set(review["subject"]) == {"name", "memories"}
    assert set(review["subject"]["memories"][0]) == {
        "memory_id",
        "content",
        "link_basis",
        "provenance_episode_ids",
    }
    split = json.loads(split_input(snapshot))
    assert set(split["subject"]) == {"name", "memories"}
    assert set(split["subject"]["memories"][0]) == {
        "memory_id",
        "content",
        "link_basis",
    }
    assert json.loads(summary_refresh_input(snapshot)) == {
        "subject": {
            "name": "Alice",
            "memories": [{"content": "Alice likes hiking."}],
        }
    }


def test_review_provenance_input_uses_only_source_identity_time_and_content() -> None:
    snapshot = SubjectSnapshot(
        "subject-id",
        "Alice",
        None,
        0,
        0,
        (MemorySnapshot("memory-id", "Alice moved.", None, ("episode-id",), "direct"),),
    )
    provenance = {
        "memory-id": (
            {
                "episode_id": "episode-id",
                "source_started_at": 1_694_044_800_000,
                "source_ended_at": 1_694_048_400_000,
                "source_timezone": "Europe/Paris",
                "blocks": [
                    {
                        "role": "user",
                        "speaker_id": "alice",
                        "speaker_name": "Alice Example",
                        "observed_at": 1_694_045_100_000,
                        "content": "I moved.",
                    }
                ],
            },
        )
    }
    payload = json.loads(review_input(snapshot, provenance))
    assert payload["requested_provenance"] == {
        "memory-id": [
            {
                "episode_id": "episode-id",
                "source_started_at": "2023-09-07T00:00:00Z (Thursday)",
                "source_ended_at": "2023-09-07T01:00:00Z (Thursday)",
                "source_timezone": "Europe/Paris",
                "blocks": [
                    {
                        "speaker_id": "alice",
                        "content": "I moved.",
                        "observed_at": "2023-09-07T00:05:00Z (Thursday)",
                    }
                ],
            }
        ]
    }


def test_validation_feedback_collapses_repeated_array_errors() -> None:
    malformed = {
        "result": "links",
        "memories": [
            {
                "memory_id": f"id-{index}",
                "subject_ref": "subject-id",
                "type": "direct",
            }
            for index in range(20)
        ],
    }
    try:
        TypeAdapter(LinkingStageOutput).validate_python(malformed)
    except PydanticValidationError as error:
        feedback = validation_feedback(error)
    else:
        raise AssertionError("malformed linking output unexpectedly validated")

    assert "memories[*]" in feedback
    assert feedback.count("direct_assignments: Field required") == 1
    assert "pydantic.dev" not in feedback
    assert len(feedback) <= 2_000


def test_repair_input_contains_only_the_supplied_previous_output() -> None:
    repaired = repair_input(
        '{"input":true}',
        '{"second_marker":true}',
        "Validation errors:\n- links: Field required",
    )

    assert repaired.startswith('{"input":true}')
    assert '{"second_marker":true}' in repaired
    assert "first_marker" not in repaired
    assert "return one complete, corrected JSON object only" in repaired


def test_structured_retry_does_not_accumulate_older_outputs(tmp_path) -> None:
    async def scenario() -> None:
        generation = RepairingGenerationProvider()
        events: list[dict[str, object]] = []
        engine = await FluxFold.open(
            db_path=str(tmp_path / "repair.sqlite3"),
            generation_provider=generation,
            embedding_provider=FakeEmbeddingProvider(),
            event_sink=events.append,
        )
        space = await engine.create_or_open_space("test:repair")
        episode = NormalizedEpisode(
            source_type="test",
            source_key="repair",
            source_sequence=1,
            blocks=(EpisodeBlock("repair:0", 0, Role.USER, "Alice likes hiking."),),
        )

        await engine.add_episode(space.memory_space_id, episode)

        linking_requests = [
            request
            for request in generation.requests
            if request.stage == "subject_linking"
        ]
        assert len(linking_requests) == 3
        assert "first_marker" in linking_requests[1].user_prompt
        assert "second_marker" in linking_requests[2].user_prompt
        assert "first_marker" not in linking_requests[2].user_prompt
        assert all(request.timeout_seconds is None for request in linking_requests)
        linking_call_events = [
            event
            for event in events
            if event["event_type"] == "llm_call" and event["stage"] == "subject_linking"
        ]
        assert [event["result"] for event in linking_call_events] == [
            "failed",
            "failed",
            "success",
        ]
        await engine.close()

    asyncio.run(scenario())
