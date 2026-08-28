from __future__ import annotations

import asyncio
import json

from pydantic import TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from fluxfold import EpisodeBlock, FluxFold, NormalizedEpisode, Role
from fluxfold.models import LinkingStageOutput
from fluxfold.prompts import (
    LINKING_SYSTEM,
    REVIEW_SYSTEM,
    SPLIT_SYSTEM,
    repair_input,
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


def test_linking_prompt_contains_the_exact_nested_contract() -> None:
    assert "There are exactly two valid shapes" in LINKING_SYSTEM
    assert "`association_search_results` is null" in LINKING_SYSTEM
    assert '"result":"association_search"' in LINKING_SYSTEM
    assert '"result":"links"' in LINKING_SYSTEM
    assert '"subject":{"kind":"existing","subject_id"' in LINKING_SYSTEM
    assert '"subject":{"kind":"new","subject_ref"' in LINKING_SYSTEM
    assert "`association_search_results` is non-null" in LINKING_SYSTEM
    assert "shape 1 is no longer valid" in LINKING_SYSTEM
    assert "basis is exactly direct or contextual" in LINKING_SYSTEM


def test_linking_prompt_distinguishes_direct_granularity_from_contextual_links() -> (
    None
):
    assert (
        "A direct link is valid only when the subject is a home of the memory"
        in LINKING_SYSTEM
    )
    assert "kind of thing the subject is for" in LINKING_SYSTEM
    assert (
        "never also a direct or contextual link to that coarser subject"
        in LINKING_SYSTEM
    )
    assert "Apply this rule separately to every core subject" in LINKING_SYSTEM
    assert "This restriction is specific to direct links" in LINKING_SYSTEM
    assert "A contextual link is a retrieval bridge" in LINKING_SYSTEM
    assert "too dissimilar for vector recall" in LINKING_SYSTEM
    assert "without treating the affected subject as a home" in LINKING_SYSTEM
    assert "Linking only decides membership" in LINKING_SYSTEM
    assert "Mike had dental implant surgery on 3 May 2024" in LINKING_SYSTEM
    assert "cannot drink alcohol for a month" not in LINKING_SYSTEM
    assert '"Mike\'s dietary preferences", "Mike\'s diet plan"' in LINKING_SYSTEM
    assert '"Mike\'s travel plans", "Mike\'s commute"' in LINKING_SYSTEM
    assert '"Mike\'s evening plans", "Mike\'s sleep schedule"' in LINKING_SYSTEM
    assert "driving licence was suspended" in LINKING_SYSTEM
    assert "night shifts at the hospital" in LINKING_SYSTEM
    assert "the other topic this memory could change" in LINKING_SYSTEM
    assert "better direct home that passive recall missed" in LINKING_SYSTEM


def test_split_prompt_explains_link_basis_against_result_subjects() -> None:
    assert "Input `link_basis` is relative to the original subject" in SPLIT_SYSTEM
    assert "do not copy it" in SPLIT_SYSTEM
    assert "`direct` means the result subject is a home of the memory" in SPLIT_SYSTEM
    assert "kind of thing the subject is for" in SPLIT_SYSTEM
    assert "A contextual link is a retrieval bridge" in SPLIT_SYSTEM
    assert "This restriction is specific to direct links" in SPLIT_SYSTEM
    assert "memories you do not list stay in the original" in SPLIT_SYSTEM
    assert "links to subjects outside this split are preserved" in SPLIT_SYSTEM
    assert "Keep together memories that a later question will need" in SPLIT_SYSTEM


def test_review_prompt_exposes_both_valid_output_shapes() -> None:
    assert "There are exactly two valid shapes" in REVIEW_SYSTEM
    assert '"result":"provenance_request"' in REVIEW_SYSTEM
    assert '"result":"review"' in REVIEW_SYSTEM
    assert "`provenance_may_be_requested` is true" in REVIEW_SYSTEM
    assert "`requested_provenance` is null" in REVIEW_SYSTEM
    assert "`provenance_may_be_requested` is false" in REVIEW_SYSTEM
    assert "shape 1 is no longer valid" in REVIEW_SYSTEM


def test_review_prompt_compiles_sibling_memories() -> None:
    assert "Linking only placed these memories in this subject" in REVIEW_SYSTEM
    assert "rewrite the incomplete memory to name Sweden" in REVIEW_SYSTEM
    assert "rewrite the drinking preference" in REVIEW_SYSTEM
    assert "Normalize parallel instances" in REVIEW_SYSTEM
    assert "classical-music preference" in REVIEW_SYSTEM
    assert "two hops of one later question" in REVIEW_SYSTEM


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
