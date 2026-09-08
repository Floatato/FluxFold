from __future__ import annotations

import asyncio
import json

from benchmarks.adapters import load_longmemeval
from benchmarks.artifacts import (
    ArtifactWriter,
    RunPaths,
    append_jsonl,
    write_search_artifacts,
)
from benchmarks.runner import answer_run, build_run, score_run

from fluxfold import FluxFoldConfig
from tests.fakes import FakeEmbeddingProvider, FakeGenerationProvider


def _alice_record(question_id: str) -> dict[str, object]:
    return {
        "question_id": question_id,
        "question_type": "single-session-user",
        "question": "What does Alice like?",
        "answer": "Alice likes hiking.",
        "haystack_session_ids": ["s-1", "s-2", "s-3"],
        "haystack_dates": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "haystack_sessions": [
            [{"role": "user", "content": "Alice likes hiking."}],
            [{"role": "user", "content": "Alice owns hiking boots."}],
            [{"role": "user", "content": "Alice hikes on weekends."}],
        ],
    }


def _search_config() -> FluxFoldConfig:
    return FluxFoldConfig().with_overrides(
        subject_candidate_min_similarity=-1.0,
        memory_candidate_min_similarity=-1.0,
        search_subject_min_similarity=-1.0,
        search_memory_min_similarity=-1.0,
    )


def _patch_providers(monkeypatch, generation: FakeGenerationProvider) -> None:
    monkeypatch.setenv("FLUXFOLD_BUILD_MODEL", "fake-generation")
    monkeypatch.setenv("FLUXFOLD_ANSWER_MODEL", "fake-generation")
    monkeypatch.setenv("FLUXFOLD_SCORE_MODEL", "fake-generation")
    monkeypatch.setattr(
        "benchmarks.runner.generation_provider",
        lambda ignored_config, **_kwargs: generation,
    )
    monkeypatch.setattr(
        "benchmarks.runner.embedding_provider",
        lambda ignored_config: FakeEmbeddingProvider(),
    )


def test_build_metrics_distinguish_successful_and_failed_llm_calls(tmp_path) -> None:
    writer = ArtifactWriter(RunPaths(tmp_path / "metrics"))
    writer.event(
        {
            "event_type": "llm_call",
            "result": "success",
            "total_tokens": 10,
            "input_tokens": 7,
            "output_tokens": 3,
        }
    )
    writer.event(
        {
            "event_type": "llm_call",
            "result": "failed",
            "total_tokens": 7,
            "input_tokens": 5,
            "output_tokens": 2,
        }
    )

    assert writer.build_metrics() == {
        "successful_llm_call_count": 1,
        "failed_llm_call_count": 1,
        "build_llm_input_tokens": 12,
        "build_llm_output_tokens": 5,
        "build_llm_total_tokens": 17,
        "terminal_failure_count": 0,
    }


def test_build_answer_score_produces_results(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "longmemeval.json"
    dataset_path.write_text(json.dumps([_alice_record("q-1")]), encoding="utf-8")
    spaces = load_longmemeval(dataset_path)
    paths = RunPaths(tmp_path / "run")
    config = _search_config()
    generation = FakeGenerationProvider(extraction_delay_seconds=0.01)
    _patch_providers(monkeypatch, generation)
    monkeypatch.setattr("fluxfold.engine.random.random", lambda: 0.0)
    first_elapsed = 0.0

    async def scenario() -> None:
        nonlocal first_elapsed
        await build_run(
            dataset="longmemeval",
            spaces=spaces,
            run_paths=paths,
            data_paths=(str(dataset_path),),
            mode="sample",
            config=config,
        )
        first_summary = json.loads(paths.build_summary.read_text(encoding="utf-8"))
        first_elapsed = float(first_summary["elapsed_seconds"])
        assert first_elapsed > 0
        previous_checkpoint = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
        paths.checkpoint.write_text(
            json.dumps(
                {
                    "completed_space_ids": [],
                    "last_episode_by_space": {spaces[0].source_id: 1},
                    "elapsed_seconds": previous_checkpoint["elapsed_seconds"],
                }
            ),
            encoding="utf-8",
        )
        await build_run(
            dataset="longmemeval",
            spaces=spaces,
            run_paths=paths,
            data_paths=(str(dataset_path),),
            mode="sample",
            config=config,
        )
        later_config = config.with_overrides(subject_split_memory_count_threshold=12)
        assert later_config.signature != config.signature
        await answer_run(
            dataset="longmemeval",
            spaces=spaces,
            run_paths=paths,
            config=later_config,
        )
        await score_run(
            dataset="longmemeval",
            spaces=spaces,
            run_paths=paths,
            config=later_config,
        )

    asyncio.run(scenario())
    assert paths.database.is_file()
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert manifest["build_model"] == "fake-generation"
    prediction = json.loads(paths.predictions.read_text(encoding="utf-8"))
    assert prediction == {"question_id": "q-1", "hypothesis": "Alice likes hiking."}
    search_record = json.loads(paths.search_records.read_text(encoding="utf-8"))
    assert search_record["query"] == "What does Alice like?"
    assert "search_result" not in search_record
    assert search_record["answer_llm_input_tokens"] == 6
    assert len(search_record["groups"]) == 1
    assert search_record["groups"][0]["route"] == "direct subject hit"
    assert all(
        memory["route"] == "direct memory hit"
        for memory in search_record["groups"][0]["memories"]
    )
    search_summary = json.loads(paths.search_summary.read_text(encoding="utf-8"))
    assert search_summary["answer_llm_input_tokens_per_query"] == {
        "max": 6,
        "mean": 6.0,
        "min": 6,
    }
    assert search_summary["returned_per_query"] == {
        "memories_mean": 3.0,
        "subjects_mean": 1.0,
        "summaries_mean": 1.0,
    }
    search_audit = paths.search_audit.read_text(encoding="utf-8")
    assert "## Query 1" in search_audit
    assert "- Displayed: 1 subjects · 1 summaries · 3 memories" in search_audit
    assert "- Retrieval route: `direct subject hit`" in search_audit
    assert "subject_id" not in search_audit
    answer_samples = paths.answer_llm_input_samples.read_text(encoding="utf-8")
    assert answer_samples.count("## Sample") == 1
    assert "### System prompt\n\n_(none)_" in answer_samples
    assert "What does Alice like?" in answer_samples
    assert "Retrieved memories:" in answer_samples
    summary = json.loads(paths.score_summary.read_text(encoding="utf-8"))
    assert summary["overall"]["accuracy"] == 1.0
    assert summary["score_model"] == "fake-generation"
    assert paths.score_markdown.is_file()
    assert not (paths.root / "build_audit.md").exists()
    assert paths.llm_io_samples.is_file()
    sample_text = paths.llm_io_samples.read_text(encoding="utf-8")
    assert "## extract · sample 1" in sample_text
    assert "### System prompt" not in sample_text
    assert generation.max_active_extractions == 1
    bank_text = paths.memory_bank.read_text(encoding="utf-8")
    assert bank_text.count("# Memory bank") == 1
    assert "after build completed" in bank_text
    assert f"## `{spaces[0].space_key}`" in bank_text
    assert "### Alice's hiking" in bank_text
    assert "- Alice likes hiking." in bank_text
    assert "- Alice owns hiking boots." in bank_text
    assert "- Alice hikes on weekends." in bank_text
    build_summary = json.loads(paths.build_summary.read_text(encoding="utf-8"))
    assert build_summary["successful_llm_call_count"] > 0
    assert build_summary["failed_llm_call_count"] == 0
    assert build_summary["build_llm_input_tokens"] > 0
    assert build_summary["build_llm_output_tokens"] > 0
    assert build_summary["build_llm_total_tokens"] > 0
    assert build_summary["elapsed_seconds"] >= first_elapsed
    assert "write_llm_total_tokens" not in build_summary
    assert "write_llm_calls" not in build_summary
    space_summary = build_summary["spaces"][spaces[0].source_id]
    assert space_summary["active_subjects"] == 1
    assert space_summary["episode_count"] == 3
    assert space_summary["direct_links"] == 3
    assert space_summary["contextual_links"] == 0
    assert space_summary["max_links_per_memory"] == 1
    assert space_summary["max_direct_links_per_memory"] == 1
    assert space_summary["max_contextual_links_per_memory"] == 0
    assert space_summary["max_memories_per_subject"] == 3
    assert space_summary["max_provenance_per_memory"] == 1
    assert space_summary["mean_links_per_memory"] == 1.0
    assert space_summary["mean_memories_per_subject"] == 3.0
    assert space_summary["retired_memories"] == 0
    assert space_summary["retired_subjects"] == 0
    assert space_summary["rewritten_memories"] == 0
    assert space_summary["subjects"] == [
        {
            "contextual_links": 0,
            "direct_links": 3,
            "name": "Alice's hiking",
        }
    ]


def test_build_resume_ignores_config_signature(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "longmemeval.json"
    dataset_path.write_text(json.dumps([_alice_record("q-1")]), encoding="utf-8")
    spaces = load_longmemeval(dataset_path)
    paths = RunPaths(tmp_path / "run")
    config = _search_config()
    generation = FakeGenerationProvider(extraction_delay_seconds=0.01)
    _patch_providers(monkeypatch, generation)

    async def scenario() -> None:
        await build_run(
            dataset="longmemeval",
            spaces=spaces,
            run_paths=paths,
            data_paths=(str(dataset_path),),
            mode="sample",
            config=config,
        )
        original = json.loads(paths.manifest.read_text(encoding="utf-8"))
        later_config = config.with_overrides(benchmark_search_concurrency=1)
        assert later_config.signature != config.signature
        await build_run(
            dataset="longmemeval",
            spaces=spaces,
            run_paths=paths,
            data_paths=(str(dataset_path),),
            mode="sample",
            config=later_config,
        )
        assert json.loads(paths.manifest.read_text(encoding="utf-8")) == original

    asyncio.run(scenario())


def test_answer_run_skips_existing_predictions(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "longmemeval.json"
    dataset_path.write_text(
        json.dumps([_alice_record("q-1"), _alice_record("q-2")]),
        encoding="utf-8",
    )
    spaces = load_longmemeval(dataset_path)
    paths = RunPaths(tmp_path / "run")
    config = _search_config()
    generation = FakeGenerationProvider(extraction_delay_seconds=0.01)
    _patch_providers(monkeypatch, generation)

    async def scenario() -> None:
        await build_run(
            dataset="longmemeval",
            spaces=spaces,
            run_paths=paths,
            data_paths=(str(dataset_path),),
            mode="sample",
            config=config,
        )
        append_jsonl(
            paths.predictions,
            {"question_id": "q-1", "hypothesis": "kept existing answer"},
        )
        generation.requests.clear()
        await answer_run(
            dataset="longmemeval",
            spaces=spaces,
            run_paths=paths,
            config=config,
        )
        paths.search_summary.unlink()
        paths.search_audit.unlink()
        paths.answer_llm_input_samples.unlink()
        await answer_run(
            dataset="longmemeval",
            spaces=spaces,
            run_paths=paths,
            config=config,
        )

    asyncio.run(scenario())
    predictions = [
        json.loads(line)
        for line in paths.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert predictions[0] == {
        "hypothesis": "kept existing answer",
        "question_id": "q-1",
    }
    assert {item["question_id"] for item in predictions} == {"q-1", "q-2"}
    assert [
        request.stage
        for request in generation.requests
        if request.stage == "benchmark_answer"
    ] == ["benchmark_answer"]
    assert paths.search_summary.is_file()
    assert paths.search_audit.is_file()
    assert paths.answer_llm_input_samples.is_file()


def test_search_summary_uses_only_the_requested_aggregates(tmp_path) -> None:
    paths = RunPaths(tmp_path / "run")
    records: list[dict[str, object]] = [
        {
            "question_id": "q-1",
            "query_number": 1,
            "query": "First question?",
            "search_latency_seconds": 1.0,
            "answer_llm_input_tokens": 10,
            "rendered_context_chars": 20,
            "answer_input_sample": {
                "system_prompt": None,
                "user_prompt": "First prompt",
            },
            "groups": [
                {
                    "name": "Higher",
                    "summary": "abc",
                    "similarity": 0.8,
                    "route": "direct subject hit",
                    "memories": [
                        {
                            "content": "1234",
                            "similarity": 0.7,
                            "route": "direct memory hit",
                        }
                    ],
                },
                {
                    "name": "Lower",
                    "summary": None,
                    "similarity": 0.5,
                    "route": "attached through memory",
                    "memories": [
                        {
                            "content": "12",
                            "similarity": 0.4,
                            "route": "attached through subject",
                        }
                    ],
                },
            ],
        },
        {
            "question_id": "q-2",
            "query_number": 2,
            "query": "Second question?",
            "search_latency_seconds": 3.0,
            "answer_llm_input_tokens": 20,
            "rendered_context_chars": 40,
            "answer_input_sample": {
                "system_prompt": "System",
                "user_prompt": "Second prompt",
            },
            "groups": [],
        },
    ]

    write_search_artifacts(paths, records)
    summary = json.loads(paths.search_summary.read_text(encoding="utf-8"))

    assert set(summary) == {
        "answer_llm_input_tokens_per_query",
        "memory_content_chars_per_query",
        "memory_retrieval_similarity",
        "rendered_context_chars_per_query",
        "returned_per_query",
        "search_latency_seconds",
        "subject_retrieval_similarity",
        "summary_chars_per_query",
    }
    assert summary["search_latency_seconds"] == {
        "max": 3.0,
        "mean": 2.0,
        "p50": 2.0,
        "p90": 2.8,
    }
    assert summary["summary_chars_per_query"] == {
        "max": 3,
        "mean": 1.5,
        "min": 0,
    }
    assert summary["memory_content_chars_per_query"] == {
        "max": 6,
        "mean": 3.0,
        "min": 0,
    }
    assert summary["rendered_context_chars_per_query"] == {
        "max": 40,
        "mean": 30.0,
        "min": 20,
    }
    assert summary["returned_per_query"] == {
        "memories_mean": 1.0,
        "subjects_mean": 1.0,
        "summaries_mean": 0.5,
    }
    assert summary["subject_retrieval_similarity"] == {
        "per_query_top1": {"max": 0.8, "mean": 0.8, "min": 0.8},
        "per_query_mean": {"max": 0.65, "mean": 0.65, "min": 0.65},
        "per_query_min": {"max": 0.5, "mean": 0.5, "min": 0.5},
    }
