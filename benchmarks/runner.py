"""Three-stage benchmark build, answer, and scoring runners."""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Awaitable
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from benchmarks.adapters import BenchmarkSpace, load_locomo, load_longmemeval
from benchmarks.artifacts import ArtifactWriter, RunPaths, append_jsonl
from benchmarks.runtime import (
    dataset_hash,
    embedding_provider,
    generation_model_id,
    generation_provider,
)
from benchmarks.scoring import score_predictions
from fluxfold.config import FluxFoldConfig
from fluxfold.engine import FluxFold
from fluxfold.errors import ErrorClass, ProviderError, StageFailure, ValidationError
from fluxfold.models import AddResult, NormalizedEpisode
from fluxfold.providers import GenerationRequest


async def build_run(
    *,
    dataset: str,
    spaces: tuple[BenchmarkSpace, ...],
    run_paths: RunPaths,
    data_paths: tuple[str, ...],
    mode: str,
    config: FluxFoldConfig,
) -> None:
    writer = ArtifactWriter(run_paths)
    generation = generation_provider(config)
    embedding = embedding_provider(config)
    selected_ids = [space.source_id for space in spaces]
    existing_manifest = (
        json.loads(run_paths.manifest.read_text(encoding="utf-8"))
        if run_paths.manifest.exists()
        else None
    )
    run_id = (
        str(existing_manifest["run_id"])
        if existing_manifest is not None
        else str(uuid4())
    )
    manifest = {
        "manifest_version": 1,
        "run_id": run_id,
        "dataset": dataset,
        "mode": mode,
        "data_paths": list(data_paths),
        "dataset_hash": dataset_hash(data_paths),
        "selected_space_ids": selected_ids,
        "config_signature": config.signature,
        "generation_model": generation.model_id,
        "embedding_model": asdict(embedding.model_info),
        "benchmark_seed": config.benchmark_seed,
        "source_timezone_convention": "UTC",
    }
    if run_paths.manifest.exists():
        if existing_manifest != manifest:
            raise ValidationError("run manifest differs from the existing build")
    else:
        writer.write_json(run_paths.manifest, manifest)
    writer.run_id = run_id
    checkpoint = _load_checkpoint(run_paths.checkpoint)
    completed = set(checkpoint["completed_space_ids"])
    last_episode_by_space = dict(checkpoint["last_episode_by_space"])
    engine = await FluxFold.open(
        db_path=str(run_paths.database),
        generation_provider=generation,
        embedding_provider=embedding,
        config=config,
        event_sink=writer.event,
        benchmark_seed=config.benchmark_seed,
    )
    semaphore = asyncio.Semaphore(config.benchmark_memory_space_build_concurrency)
    summary: dict[str, dict[str, int | float]] = {}
    completed_lock = asyncio.Lock()

    async def build_space(space: BenchmarkSpace) -> None:
        if space.source_id in completed:
            memory_space = await engine.create_or_open_space(space.space_key)
            summary[space.source_id] = engine.space_statistics(
                memory_space.memory_space_id
            )
            return
        async with semaphore:

            async def work() -> None:
                memory_space = await engine.create_or_open_space(space.space_key)
                writer.event(
                    {
                        "event_type": "memory_space_build_started",
                        "severity": "info",
                        "timestamp_ms": int(time.time() * 1000),
                        "space_key": space.space_key,
                        "memory_space_id": memory_space.memory_space_id,
                    }
                )
                extraction_semaphore = asyncio.Semaphore(
                    config.benchmark_extraction_concurrency_per_space
                )

                async def prepare(episode: NormalizedEpisode) -> object:
                    async with extraction_semaphore:
                        return await engine._prepare_add(
                            memory_space.memory_space_id, episode
                        )

                preparation_tasks = [
                    asyncio.create_task(prepare(episode)) for episode in space.episodes
                ]
                try:
                    for episode, preparation_task in zip(
                        space.episodes, preparation_tasks, strict=True
                    ):
                        writer.event(
                            {
                                "event_type": "episode_processing_started",
                                "severity": "info",
                                "timestamp_ms": int(time.time() * 1000),
                                "space_key": space.space_key,
                                "memory_space_id": memory_space.memory_space_id,
                                "source_sequence": episode.source_sequence,
                                "source_key": episode.source_key,
                            }
                        )
                        first_attempt = _finish_prepared_add(engine, preparation_task)
                        result, failure = await _add_with_item_retry(
                            engine=engine,
                            memory_space_id=memory_space.memory_space_id,
                            episode=episode,
                            writer=writer,
                            config=config,
                            first_attempt=first_attempt,
                        )
                        if result is None:
                            assert failure is not None
                            writer.audit_episode_failure(
                                space.space_key,
                                memory_space.memory_space_id,
                                episode,
                                failure.error_class.value,
                                failure.message,
                            )
                        else:
                            writer.audit_episode(
                                space.space_key,
                                memory_space.memory_space_id,
                                episode,
                                result,
                            )
                        async with completed_lock:
                            last_episode_by_space[space.source_id] = (
                                episode.source_sequence
                            )
                            _write_build_checkpoint(
                                writer,
                                run_paths,
                                completed,
                                last_episode_by_space,
                            )
                finally:
                    for preparation_task in preparation_tasks:
                        if not preparation_task.done():
                            preparation_task.cancel()
                    await asyncio.gather(*preparation_tasks, return_exceptions=True)
                summary[space.source_id] = engine.space_statistics(
                    memory_space.memory_space_id
                )
                async with completed_lock:
                    completed.add(space.source_id)
                    _write_build_checkpoint(
                        writer,
                        run_paths,
                        completed,
                        last_episode_by_space,
                    )
                writer.event(
                    {
                        "event_type": "memory_space_build_completed",
                        "severity": "info",
                        "timestamp_ms": int(time.time() * 1000),
                        "space_key": space.space_key,
                        "memory_space_id": memory_space.memory_space_id,
                    }
                )

            await asyncio.wait_for(
                work(), timeout=config.benchmark_memory_space_build_timeout_seconds
            )

    try:
        await asyncio.gather(*(build_space(space) for space in spaces))
        writer.write_json(
            run_paths.build_summary,
            {
                "dataset": dataset,
                "spaces": {key: value for key, value in sorted(summary.items())},
                **writer.build_metrics(),
            },
        )
    finally:
        await engine.close()


async def answer_run(
    *,
    dataset: str,
    spaces: tuple[BenchmarkSpace, ...],
    run_paths: RunPaths,
    config: FluxFoldConfig,
) -> None:
    _validate_manifest(run_paths, dataset, spaces, config)
    _remove_if_exists(run_paths.predictions)
    _remove_if_exists(run_paths.search_results)
    generation = generation_provider(config)
    embedding = embedding_provider(config)
    engine = await FluxFold.open(
        db_path=str(run_paths.database),
        generation_provider=generation,
        embedding_provider=embedding,
        config=config,
        benchmark_seed=config.benchmark_seed,
    )
    semaphore = asyncio.Semaphore(config.benchmark_search_concurrency)
    opened_spaces = {
        space.space_key: await engine.create_or_open_space(space.space_key)
        for space in spaces
    }

    async def answer_one(
        space: BenchmarkSpace, question_index: int
    ) -> tuple[dict[str, object], dict[str, object]]:
        question = space.questions[question_index]
        memory_space = opened_spaces[space.space_key]

        async def work() -> tuple[dict[str, object], dict[str, object]]:
            async with semaphore:
                started = time.perf_counter()
                result = await engine.search(
                    memory_space.memory_space_id, question.question
                )
                search_latency = time.perf_counter() - started
                date_context = (
                    f"\nQuestion date: {question.question_date}"
                    if question.question_date
                    else ""
                )
                prompt = (
                    "Answer the question using only the FluxFold search result. Be concise and include every required fact. "
                    "If the result is insufficient, explicitly say that the available information is insufficient."
                    f"{date_context}\nQuestion: {question.question}\n\nFluxFold search result:\n{result.render()}"
                )
                response = await generation.generate(
                    GenerationRequest(
                        stage="benchmark_answer",
                        system_prompt="You answer long-term memory questions from retrieved evidence only.",
                        user_prompt=prompt,
                        temperature=0.0,
                        timeout_seconds=config.benchmark_search_sample_timeout_seconds,
                        seed=config.benchmark_seed,
                    )
                )
                prediction = (
                    {"question_id": question.question_id, "hypothesis": response.text}
                    if dataset == "longmemeval"
                    else {
                        "qa_id": question.question_id,
                        "predicted_answer": response.text,
                    }
                )
                search_record = {
                    "question_id": question.question_id,
                    "space_key": space.space_key,
                    "search_latency_seconds": search_latency,
                    "search_result": asdict(result),
                }
                return prediction, search_record

        return await asyncio.wait_for(
            work(), timeout=config.benchmark_search_sample_timeout_seconds
        )

    try:
        tasks = [
            asyncio.create_task(answer_one(space, question_index))
            for space in spaces
            for question_index in range(len(space.questions))
        ]
        for task in tasks:
            prediction, search_record = await task
            append_jsonl(run_paths.predictions, prediction)
            append_jsonl(run_paths.search_results, search_record)
    finally:
        await engine.close()


async def score_run(
    *,
    dataset: str,
    spaces: tuple[BenchmarkSpace, ...],
    run_paths: RunPaths,
    config: FluxFoldConfig,
) -> None:
    _validate_manifest(run_paths, dataset, spaces, config)
    predictions = _load_predictions(run_paths.predictions, dataset)
    questions = tuple(question for space in spaces for question in space.questions)
    generation = generation_provider(config)
    try:
        scores, summary = await score_predictions(
            dataset=dataset,
            questions=questions,
            predictions=predictions,
            provider=generation,
            concurrency=config.benchmark_search_concurrency,
            timeout_seconds=config.benchmark_search_sample_timeout_seconds,
            seed=config.benchmark_seed,
        )
    finally:
        await generation.close()
    _remove_if_exists(run_paths.scores)
    for score in scores:
        append_jsonl(run_paths.scores, score)
    writer = ArtifactWriter(run_paths)
    summary["generation_model"] = generation_model_id()
    writer.write_json(run_paths.score_summary, summary)
    run_paths.score_markdown.write_text(_summary_markdown(summary), encoding="utf-8")


def load_dataset_from_manifest(
    run_paths: RunPaths,
) -> tuple[str, tuple[BenchmarkSpace, ...]]:
    manifest = json.loads(run_paths.manifest.read_text(encoding="utf-8"))
    dataset = str(manifest["dataset"])
    paths = tuple(str(item) for item in manifest["data_paths"])
    all_spaces = _load_dataset(dataset, paths)
    selected = set(str(item) for item in manifest["selected_space_ids"])
    spaces = tuple(space for space in all_spaces if space.source_id in selected)
    if len(spaces) != len(selected):
        raise ValidationError("manifest selection no longer exists in the dataset")
    return dataset, spaces


def _load_dataset(dataset: str, paths: tuple[str, ...]) -> tuple[BenchmarkSpace, ...]:
    if dataset == "longmemeval":
        if len(paths) != 1:
            raise ValidationError("LongMemEval requires one data path")
        return load_longmemeval(paths[0])
    if len(paths) != 2:
        raise ValidationError("LoCoMo requires conversations and questions paths")
    return load_locomo(paths[0], paths[1])


def _validate_manifest(
    run_paths: RunPaths,
    dataset: str,
    spaces: tuple[BenchmarkSpace, ...],
    config: FluxFoldConfig,
) -> None:
    manifest = json.loads(run_paths.manifest.read_text(encoding="utf-8"))
    if manifest["dataset"] != dataset:
        raise ValidationError("dataset does not match build manifest")
    if manifest["config_signature"] != config.signature:
        raise ValidationError("configuration does not match build manifest")
    if manifest["generation_model"] != generation_model_id():
        raise ValidationError("generation model does not match build manifest")
    if list(manifest["selected_space_ids"]) != [space.source_id for space in spaces]:
        raise ValidationError("space selection does not match build manifest")


def _load_predictions(path: Path, dataset: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        key_field = "question_id" if dataset == "longmemeval" else "qa_id"
        value_field = "hypothesis" if dataset == "longmemeval" else "predicted_answer"
        question_id = str(item[key_field])
        if question_id in output:
            raise ValidationError(f"duplicate prediction: {question_id}")
        output[question_id] = str(item[value_field])
    return output


async def _add_with_item_retry(
    *,
    engine: FluxFold,
    memory_space_id: str,
    episode: NormalizedEpisode,
    writer: ArtifactWriter,
    config: FluxFoldConfig,
    first_attempt: Awaitable[AddResult] | None = None,
) -> tuple[AddResult | None, StageFailure | None]:
    transient = {
        ErrorClass.TRANSIENT_TRANSPORT,
        ErrorClass.RATE_LIMITED,
        ErrorClass.SERVICE_UNAVAILABLE,
    }
    terminal_item = {
        ErrorClass.CONTEXT_OVERFLOW,
        ErrorClass.POLICY_REJECTED,
        ErrorClass.INVALID_STRUCTURED_OUTPUT,
        ErrorClass.INCOMPLETE_OUTPUT,
        ErrorClass.STAGE_DEADLINE_EXCEEDED,
    }
    for retry_index in range(config.benchmark_failure_max_retries + 1):
        try:
            if retry_index == 0 and first_attempt is not None:
                return await first_attempt, None
            return await engine.add(memory_space_id, episode), None
        except (ProviderError, StageFailure) as error:
            failure = (
                error
                if isinstance(error, StageFailure)
                else StageFailure("embedding", error.error_class, error.message)
            )
            if failure.error_class in terminal_item:
                engine.record_episode_terminal_failure(
                    memory_space_id, episode, failure
                )
                writer.event(
                    {
                        "event_type": "episode_terminal_failure",
                        "severity": "error",
                        "timestamp_ms": int(time.time() * 1000),
                        "memory_space_id": memory_space_id,
                        "source_sequence": episode.source_sequence,
                        "error_class": failure.error_class.value,
                        "reason": failure.message,
                    }
                )
                return None, failure
            if failure.error_class not in transient:
                writer.event(
                    {
                        "event_type": "memory_space_build_blocked",
                        "severity": "error",
                        "timestamp_ms": int(time.time() * 1000),
                        "memory_space_id": memory_space_id,
                        "source_sequence": episode.source_sequence,
                        "error_class": failure.error_class.value,
                        "reason": failure.message,
                    }
                )
                if isinstance(error, StageFailure):
                    raise
                raise failure from error
            if retry_index >= config.benchmark_failure_max_retries:
                writer.event(
                    {
                        "event_type": "memory_space_build_paused",
                        "severity": "error",
                        "timestamp_ms": int(time.time() * 1000),
                        "memory_space_id": memory_space_id,
                        "source_sequence": episode.source_sequence,
                        "error_class": failure.error_class.value,
                        "reason": failure.message,
                    }
                )
                raise
            writer.event(
                {
                    "event_type": "benchmark_item_retry",
                    "severity": "warning",
                    "timestamp_ms": int(time.time() * 1000),
                    "memory_space_id": memory_space_id,
                    "source_sequence": episode.source_sequence,
                    "attempt": retry_index + 2,
                    "error_class": failure.error_class.value,
                }
            )
            await asyncio.sleep(
                random.uniform(
                    0,
                    config.retry_initial_seconds * config.retry_multiplier**retry_index,
                )
            )
    raise AssertionError("unreachable benchmark item retry state")


async def _finish_prepared_add(
    engine: FluxFold, preparation_task: asyncio.Task[object]
) -> AddResult:
    prepared = await preparation_task
    return await engine._commit_prepared_add(prepared)  # type: ignore[arg-type]


def _load_checkpoint(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"completed_space_ids": [], "last_episode_by_space": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        "completed_space_ids": [
            str(item) for item in value.get("completed_space_ids", [])
        ],
        "last_episode_by_space": {
            str(key): int(sequence)
            for key, sequence in value.get("last_episode_by_space", {}).items()
        },
    }


def _write_build_checkpoint(
    writer: ArtifactWriter,
    run_paths: RunPaths,
    completed: set[str],
    last_episode_by_space: dict[str, int],
) -> None:
    writer.write_json(
        run_paths.checkpoint,
        {
            "completed_space_ids": sorted(completed),
            "last_episode_by_space": {
                key: value for key, value in sorted(last_episode_by_space.items())
            },
        },
    )


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _summary_markdown(summary: dict[str, object]) -> str:
    overall = dict(summary["overall"])
    lines = [
        f"# {summary['dataset']} score",
        "",
        f"Evaluator: `{summary['evaluator']}`",
        f"Generation model: `{summary['generation_model']}`",
        "",
        "| Category | Count | Accuracy | F1 | BLEU-1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    categories = dict(summary["by_category"])
    for category, raw_metrics in sorted(categories.items()):
        metrics = dict(raw_metrics)
        lines.append(
            f"| {category} | {metrics['count']} | {float(metrics['accuracy']):.4f} | "
            f"{float(metrics['f1']):.4f} | {float(metrics['bleu1']):.4f} |"
        )
    lines.append(
        f"| **Overall** | {overall['count']} | {float(overall['accuracy']):.4f} | "
        f"{float(overall['f1']):.4f} | {float(overall['bleu1']):.4f} |"
    )
    return "\n".join(lines) + "\n"
