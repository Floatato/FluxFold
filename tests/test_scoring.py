from __future__ import annotations

from benchmarks.scoring import parse_judge_response

from fluxfold.models import (
    RankedMemoryRef,
    RankedSubjectRef,
    SearchMemory,
    SearchResult,
    SearchSubject,
)


def test_parse_official_judge_responses() -> None:
    assert parse_judge_response("longmemeval", "Yes") == (True, "Yes")
    assert parse_judge_response("longmemeval", "no") == (False, "no")
    assert parse_judge_response("locomo_refined", '{"label": "CORRECT"}')[0] is True
    assert parse_judge_response("locomo_refined", '{"label": "WRONG"}')[0] is False
    wrapped = '```json\n{"label": "CORRECT"}\n```'
    assert parse_judge_response("locomo_refined", wrapped)[0] is True


def test_search_result_keeps_source_time_out_of_rendered_memory() -> None:
    result = SearchResult(
        query="when",
        subjects=(SearchSubject("s1", "Caroline", None, 1.0),),
        memories=(
            SearchMemory(
                "m1", "Caroline went to the support group.", 1683417600000, 1.0
            ),
        ),
        links=(),
        subject_channel=(RankedSubjectRef("s1", 1.0, ("m1",)),),
        memory_channel=(),
    )
    rendered = result.render()
    assert "Memory: Caroline went to the support group." in rendered
    assert "7 May 2023" not in rendered
    assert result.memories[0].latest_source_at == 1683417600000


def test_search_result_globally_deduplicates_and_regroups_memories() -> None:
    result = SearchResult(
        query="query",
        subjects=(
            SearchSubject("direct", "Direct", "Direct summary", 0.8),
            SearchSubject("best", "Best", "Hidden summary", 0.9),
            SearchSubject("loser", "Loser", None, 0.7),
            SearchSubject("winner", "Winner", None, 0.75),
        ),
        memories=(
            SearchMemory("shared", "Shared memory", None, 0.6),
            SearchMemory("other", "Other memory", None, 0.7),
            SearchMemory("highest", "Highest memory", None, 0.95),
        ),
        links=(),
        subject_channel=(RankedSubjectRef("direct", 0.8, ("shared",)),),
        memory_channel=(
            RankedMemoryRef("shared", 0.6, ("best",)),
            RankedMemoryRef("other", 0.7, ("loser", "winner")),
            RankedMemoryRef("highest", 0.95, ("best",)),
        ),
    )

    groups = result.displayed_groups()
    assert [group.subject.subject_id for group in groups] == [
        "best",
        "direct",
        "winner",
    ]
    assert groups[0].subject.summary is None
    assert [memory.memory_id for memory in groups[0].memories] == [
        "highest",
        "shared",
    ]
    assert groups[1].memories == ()
    assert [memory.memory_id for memory in groups[2].memories] == ["other"]

    rendered = result.render()
    assert rendered.count("Memory: Shared memory") == 1
    assert rendered.count("Memory: Other memory") == 1
    assert rendered.index("Memory: Highest memory") < rendered.index(
        "Memory: Shared memory"
    )
    assert "- Subject: Direct\n  Summary: Direct summary" in rendered
    assert "Hidden summary" not in rendered
    assert "Subject: Loser" not in rendered
