from __future__ import annotations

import asyncio
import json

from benchmarks.adapters import load_longmemeval
from benchmarks.artifacts import RunPaths
from benchmarks.runner import answer_run, build_run, score_run

from fluxfold import FluxFoldConfig
from tests.fakes import FakeEmbeddingProvider, FakeGenerationProvider


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
    monkeypatch.setenv("FLUXFOLD_GENERATION_MODEL", "fake-generation")
    generation = FakeGenerationProvider(extraction_delay_seconds=0.01)
    monkeypatch.setattr(
        "benchmarks.runner.generation_provider",
        lambda ignored_config: generation,
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
        await answer_run(
            dataset="longmemeval", spaces=spaces, run_paths=paths, config=config
        )
        await score_run(
            dataset="longmemeval", spaces=spaces, run_paths=paths, config=config
        )

    asyncio.run(scenario())
    assert paths.database.is_file()
    prediction = json.loads(paths.predictions.read_text(encoding="utf-8"))
    assert prediction == {"question_id": "q-1", "hypothesis": "Alice likes hiking."}
    summary = json.loads(paths.score_summary.read_text(encoding="utf-8"))
    assert summary["overall"]["accuracy"] == 1.0
    assert paths.score_markdown.is_file()
    assert generation.max_active_extractions > 1
