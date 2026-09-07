from __future__ import annotations

from benchmarks.scoring import parse_judge_response

from fluxfold.models import RankedSubjectRef, SearchMemory, SearchResult, SearchSubject


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
        subjects=(SearchSubject("s1", "Caroline", None),),
        memories=(
            SearchMemory("m1", "Caroline went to the support group.", 1683417600000),
        ),
        links=(),
        subject_channel=(RankedSubjectRef("s1", 1.0, ("m1",)),),
        memory_channel=(),
    )
    rendered = result.render()
    assert "Memory: Caroline went to the support group." in rendered
    assert "7 May 2023" not in rendered
    assert result.memories[0].latest_source_at == 1683417600000
