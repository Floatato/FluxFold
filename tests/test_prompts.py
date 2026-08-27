from __future__ import annotations

import asyncio
import json

from pydantic import TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from fluxfold import EpisodeBlock, FluxFold, NormalizedEpisode, Role
from fluxfold.models import LinkingStageOutput
from fluxfold.prompts import LINKING_SYSTEM, repair_input, validation_feedback
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
                return GenerationResponse(text, 10, "invalid-request")
        return await super().generate(request)


def test_linking_prompt_contains_the_exact_nested_contract() -> None:
    assert '"result":"links"' in LINKING_SYSTEM
    assert '"subject":{"kind":"existing","subject_id"' in LINKING_SYSTEM
    assert '"subject":{"kind":"new","subject_ref"' in LINKING_SYSTEM
    assert "basis is exactly direct or contextual" in LINKING_SYSTEM


def test_validation_feedback_collapses_repeated_array_errors() -> None:
    malformed = {
        "result": "links",
        "new_subjects": [],
        "links": [
            {
                "memory_ref": f"memory_{index}",
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

    assert "links[*]" in feedback
    assert feedback.count("subject: Field required") == 1
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

        await engine.add(space.memory_space_id, episode)

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
