from __future__ import annotations

from benchmarks.adapters import BenchmarkQuestion
from benchmarks.runner import _answer_user_prompt
from benchmarks.scoring import (
    locomo_judge_prompt,
    longmemeval_judge_prompt,
    parse_judge_response,
)

from fluxfold.models import RankedSubjectRef, SearchMemory, SearchResult, SearchSubject


def test_longmemeval_judge_prompt_matches_official_template() -> None:
    question = BenchmarkQuestion(
        question_id="q-1",
        space_key="longmemeval:q-1",
        question="What does Alice like?",
        answer="hiking",
        category="single-session-user",
    )
    prompt = longmemeval_judge_prompt(question, "hiking", "She likes hiking.")
    assert "Answer yes or no only." in prompt
    assert "Question: What does Alice like?" in prompt
    assert "Correct Answer: hiking" in prompt
    assert "Model Response: She likes hiking." in prompt


def test_longmemeval_abstention_and_temporal_prompts() -> None:
    abstention = BenchmarkQuestion(
        "q-abs",
        "longmemeval:q-abs",
        "Did Bob move?",
        "No evidence.",
        "abstention",
    )
    prompt = longmemeval_judge_prompt(abstention, "No evidence.", "Not enough info.")
    assert "unanswerable" in prompt
    temporal = BenchmarkQuestion(
        "q-t",
        "longmemeval:q-t",
        "How many days?",
        "18 days",
        "temporal-reasoning",
    )
    assert "off-by-one" in longmemeval_judge_prompt(temporal, "18 days", "19 days")


def test_locomo_judge_prompt_is_official_refined() -> None:
    prompt = locomo_judge_prompt(
        "When did Caroline go?",
        "7 May 2023",
        "Caroline went on 7 May 2023.",
    )
    assert "Inclusion + Non-contradiction" in prompt
    assert "Do NOT convert relative ↔ absolute." in prompt
    assert "Question: When did Caroline go?" in prompt
    assert "Gold answer: 7 May 2023" in prompt
    assert '"label": "CORRECT" or "WRONG"' in prompt


def test_parse_official_judge_responses() -> None:
    assert parse_judge_response("longmemeval", "Yes") == (True, "Yes")
    assert parse_judge_response("longmemeval", "no") == (False, "no")
    assert parse_judge_response("locomo_refined", '{"label": "CORRECT"}')[0] is True
    assert parse_judge_response("locomo_refined", '{"label": "WRONG"}')[0] is False
    wrapped = '```json\n{"label": "CORRECT"}\n```'
    assert parse_judge_response("locomo_refined", wrapped)[0] is True


def test_locomo_answer_prompt_asks_for_short_phrases() -> None:
    prompt = _answer_user_prompt(
        "locomo_refined",
        "When did Caroline go to the LGBTQ support group?",
        None,
        "Memory [7 May 2023]: Caroline went to the LGBTQ support group yesterday.",
    )
    assert "short phrase" in prompt
    assert "exact words" in prompt
    assert "7 May 2023" in prompt
    assert "Not mentioned" in prompt
    assert "Question: When did Caroline go to the LGBTQ support group?" in prompt


def test_longmemeval_answer_prompt_includes_current_date() -> None:
    prompt = _answer_user_prompt(
        "longmemeval",
        "What does Alice like?",
        "1 July 2023",
        "Memory: Alice likes hiking.",
    )
    assert "Current Date: 1 July 2023" in prompt
    assert "Question: What does Alice like?" in prompt
    assert "insufficient" in prompt


def test_search_result_render_includes_memory_dates() -> None:
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
    assert "Memory [7 May 2023]: Caroline went to the support group." in rendered
