from __future__ import annotations

import asyncio
import json

from benchmarks.adapters import load_longmemeval
from benchmarks.artifacts import ArtifactWriter, RunPaths
from benchmarks.runner import answer_run, build_run, score_run

from fluxfold import FluxFoldConfig
from tests.fakes import FakeEmbeddingProvider, FakeGenerationProvider


def test_build_metrics_distinguish_successful_and_failed_llm_calls(tmp_path) -> None:
    writer = ArtifactWriter(RunPaths(tmp_path / "metrics"))
    writer.event(
        {
            "event_type": "llm_call",
            "result": "success",
            "total_tokens": 10,
        }
    )
    writer.event({"event_type": "llm_call", "result": "failed", "total_tokens": 7})

    assert writer.build_metrics() == {
        "successful_llm_call_count": 1,
        "failed_llm_call_count": 1,
        "write_llm_total_tokens": 17,
        "terminal_failure_count": 0,
    }


def test_build_answer_score_produces_results(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "longmemeval.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q-1",
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
            ]
        ),
        encoding="utf-8",
    )
    spaces = load_longmemeval(dataset_path)
    paths = RunPaths(tmp_path / "run")
    config = FluxFoldConfig().with_overrides(
        subject_candidate_min_similarity=-1.0,
        memory_candidate_min_similarity=-1.0,
        search_subject_min_similarity=-1.0,
        search_memory_min_similarity=-1.0,
    )
    monkeypatch.setenv("FLUXFOLD_BUILD_MODEL", "fake-generation")
    monkeypatch.setenv("FLUXFOLD_ANSWER_MODEL", "fake-generation")
    monkeypatch.setenv("FLUXFOLD_SCORE_MODEL", "fake-generation")
    generation = FakeGenerationProvider(extraction_delay_seconds=0.01)
    monkeypatch.setattr(
        "benchmarks.runner.generation_provider",
        lambda ignored_config, **_kwargs: generation,
    )
    monkeypatch.setattr(
        "benchmarks.runner.embedding_provider",
        lambda ignored_config: FakeEmbeddingProvider(),
    )

    async def scenario() -> None:
        await build_run(
            dataset="longmemeval",
            spaces=spaces,
            run_paths=paths,
            data_paths=(str(dataset_path),),
            mode="sample",
            config=config,
        )
        event_count = len(paths.events.read_text(encoding="utf-8").splitlines())
        paths.checkpoint.write_text(
            json.dumps(
                {
                    "completed_space_ids": [],
                    "last_episode_by_space": {spaces[0].source_id: 1},
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
        resumed_events = [
            json.loads(line)
            for line in paths.events.read_text(encoding="utf-8").splitlines()[
                event_count:
            ]
        ]
        assert [
            event["source_sequence"]
            for event in resumed_events
            if event["event_type"] == "episode_processing_started"
        ] == [2]
        await answer_run(
            dataset="longmemeval", spaces=spaces, run_paths=paths, config=config
        )
        await score_run(
            dataset="longmemeval", spaces=spaces, run_paths=paths, config=config
        )

    asyncio.run(scenario())
    assert paths.database.is_file()
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert manifest["build_model"] == "fake-generation"
    prediction = json.loads(paths.predictions.read_text(encoding="utf-8"))
    assert prediction == {"question_id": "q-1", "hypothesis": "Alice likes hiking."}
    summary = json.loads(paths.score_summary.read_text(encoding="utf-8"))
    assert summary["overall"]["accuracy"] == 1.0
    assert summary["score_model"] == "fake-generation"
    assert paths.score_markdown.is_file()
    assert generation.max_active_extractions > 1
    build_summary = json.loads(paths.build_summary.read_text(encoding="utf-8"))
    assert build_summary["successful_llm_call_count"] > 0
    assert build_summary["failed_llm_call_count"] == 0
    assert "write_llm_calls" not in build_summary
