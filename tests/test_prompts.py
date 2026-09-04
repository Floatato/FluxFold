from __future__ import annotations

import asyncio
import json

from pydantic import TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from fluxfold import EpisodeBlock, FluxFold, NormalizedEpisode, Role
from fluxfold.models import LinkingStageOutput, MemorySnapshot, SubjectSnapshot
from fluxfold.prompts import (
    LINKING_SYSTEM,
    REVIEW_SYSTEM,
    SPLIT_SYSTEM,
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


def test_linking_prompt_contains_the_exact_nested_contract() -> None:
    assert "There are exactly two valid shapes" in LINKING_SYSTEM
    assert '"result":"association_search"' in LINKING_SYSTEM
    assert '"result":"links"' in LINKING_SYSTEM
    assert '"subject":{"kind":"existing","subject_id"' in LINKING_SYSTEM
    assert '"subject":{"kind":"new","subject_ref"' in LINKING_SYSTEM
    assert "up to five times" in LINKING_SYSTEM
    assert "final linking result for every supplied memory" in LINKING_SYSTEM
    assert "provisional" not in LINKING_SYSTEM
    assert "basis is exactly direct or contextual" in LINKING_SYSTEM


def test_linking_prompt_distinguishes_direct_granularity_from_contextual_links() -> (
    None
):
    assert (
        "A direct link is valid only when the subject is a home of the memory"
        in LINKING_SYSTEM
    )
    assert "kind of thing the subject is for" in LINKING_SYSTEM
    assert "never also direct- or contextual-link a coarser parent" in LINKING_SYSTEM
    assert "After resolving every core anchor" in LINKING_SYSTEM
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


def test_linking_prompt_resolves_each_core_anchor_before_creating_subjects() -> None:
    assert "First identify every core anchor in each memory" in LINKING_SYSTEM
    assert "A relationship may have multiple core anchors" in LINKING_SYSTEM
    assert (
        "a fitting subject for one anchor never removes the need to resolve another"
        in LINKING_SYSTEM
    )
    assert "Do not treat a candidate that fits a different anchor" in LINKING_SYSTEM
    assert "create no new subject for that anchor" in LINKING_SYSTEM
    assert "Only when zero candidates are fitting homes" in LINKING_SYSTEM
    assert "an accumulation container, not a summary" in LINKING_SYSTEM
    assert "Fine-grained subjects emerge later through split" in LINKING_SYSTEM
    assert "The fitting Mike candidate does not resolve John" in LINKING_SYSTEM
    assert "do not create `Melanie's family 2022 camping trip`" in LINKING_SYSTEM
    assert "A missing contextual scope never justifies creating" in LINKING_SYSTEM


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


def test_split_prompt_defines_outcome_order_and_result_size_bounds() -> None:
    assert "Every new subject holds three to twenty distinct memories" in SPLIT_SYSTEM
    assert "Choose the result by this order" in SPLIT_SYSTEM
    assert "no residual memory needs the original broad subject" in SPLIT_SYSTEM
    assert "Move a non-empty proper subset" in SPLIT_SYSTEM
    assert "leave at least one memory in the original" in SPLIT_SYSTEM
    assert "no coherent group reaches three memories" in SPLIT_SYSTEM
    assert "never force an outlier into a group" in SPLIT_SYSTEM


def test_review_prompt_exposes_both_valid_output_shapes() -> None:
    assert "There are exactly two valid shapes" in REVIEW_SYSTEM
    assert '"result":"provenance_request"' in REVIEW_SYSTEM
    assert '"result":"review"' in REVIEW_SYSTEM
    assert "`requested_provenance` is null" in REVIEW_SYSTEM
    assert "`requested_provenance` is non-null" in REVIEW_SYSTEM
    assert "shape 1 is no longer valid" in REVIEW_SYSTEM


def test_review_prompt_compiles_sibling_memories() -> None:
    assert "Linking only placed these memories in this subject" in REVIEW_SYSTEM
    assert "rewrite the incomplete memory to name Sweden" in REVIEW_SYSTEM
    assert "rewrite the drinking preference" in REVIEW_SYSTEM
    assert "Normalize parallel instances" in REVIEW_SYSTEM
    assert "classical-music preference" in REVIEW_SYSTEM
    assert "two hops of one later question" in REVIEW_SYSTEM


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
            "messages": [{"speaker_id": "alice", "content": "Alice likes hiking."}],
        }
    }

    snapshot = SubjectSnapshot(
        "subject-id",
        "Alice's hiking",
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
        "last_mentioned_at",
        "link_basis",
        "provenance_episode_ids",
    }
    split = json.loads(split_input(snapshot))
    assert set(split["subject"]) == {"name", "memories"}
    assert "provenance_episode_ids" not in split["subject"]["memories"][0]
    assert json.loads(summary_refresh_input(snapshot)) == {
        "subject": {
            "name": "Alice's hiking",
            "memories": [
                {
                    "content": "Alice likes hiking.",
                    "last_mentioned_at": "2023-09-07T00:00:00Z (Thursday)",
                }
            ],
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
                "blocks": [{"speaker_id": "alice", "content": "I moved."}],
            }
        ]
    }


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
