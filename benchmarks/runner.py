"""Three-stage benchmark build, answer, and scoring runners."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from benchmarks.adapters import BenchmarkSpace, load_locomo, load_longmemeval
from benchmarks.artifacts import ArtifactWriter, RunPaths, append_jsonl
from benchmarks.runtime import (
    dataset_hash,
    embedding_provider,
    generation_provider,
)
from benchmarks.scoring import score_predictions
from fluxfold.config import FluxFoldConfig
from fluxfold.engine import FluxFold
from fluxfold.errors import ErrorClass, StageFailure, ValidationError
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
    generation = generation_provider(config, stage="build")
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
        "build_model": generation.model_id,
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
    prior_elapsed = float(checkpoint["elapsed_seconds"])
    started = time.monotonic()
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

    def elapsed_seconds() -> float:
        return round(prior_elapsed + (time.monotonic() - started), 3)

    def persist_progress() -> None:
        _write_build_checkpoint(
            writer,
            run_paths,
            completed,
            last_episode_by_space,
            elapsed_seconds=elapsed_seconds(),
        )
        writer.write_json(
            run_paths.build_summary,
            {
                "dataset": dataset,
                "spaces": {key: value for key, value in sorted(summary.items())},
                **writer.build_metrics(),
                "elapsed_seconds": elapsed_seconds(),
            },
        )

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
                last_completed_sequence = int(
                    last_episode_by_space.get(space.source_id, -1)
                )
                pending_episodes = tuple(
                    episode
                    for episode in space.episodes
                    if episode.source_sequence > last_completed_sequence
                )
                for episode in pending_episodes:
                    result, failure = await _finish_add_item(
                        engine=engine,
                        memory_space_id=memory_space.memory_space_id,
                        episode=episode,
                        writer=writer,
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
                        last_episode_by_space[space.source_id] = episode.source_sequence
                        if result is not None:
                            _write_memory_bank(
                                writer,
                                engine,
                                updated_after=(
                                    f"episode `{episode.source_key}` in "
                                    f"`{space.space_key}` (source sequence "
                                    f"{episode.source_sequence})"
                                ),
                            )
                        persist_progress()
                summary[space.source_id] = engine.space_statistics(
                    memory_space.memory_space_id
                )
                async with completed_lock:
                    completed.add(space.source_id)
                    persist_progress()
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
        _write_memory_bank(writer, engine, updated_after="build completed")
    finally:
        persist_progress()
        await engine.close()


async def answer_run(
    *,
    dataset: str,
    spaces: tuple[BenchmarkSpace, ...],
    run_paths: RunPaths,
    config: FluxFoldConfig,
) -> None:
    _validate_manifest(run_paths, dataset, spaces)
    completed = _completed_question_ids(run_paths.predictions, dataset)
    pending = [
        (space, question_index)
        for space in spaces
        for question_index in range(len(space.questions))
        if space.questions[question_index].question_id not in completed
    ]
    if not pending:
        return
    generation = generation_provider(config, stage="answer")
    embedding = embedding_provider(config)
    engine = await FluxFold.open(
        db_path=str(run_paths.database),
        generation_provider=generation,
        embedding_provider=embedding,
        config=config,
        benchmark_seed=config.benchmark_seed,
    )
    semaphore = asyncio.Semaphore(config.benchmark_search_concurrency)
    write_lock = asyncio.Lock()
    opened_spaces = {
        space.space_key: await engine.create_or_open_space(space.space_key)
        for space in spaces
    }

    async def answer_one(space: BenchmarkSpace, question_index: int) -> None:
        question = space.questions[question_index]
        memory_space = opened_spaces[space.space_key]

        async def work() -> tuple[dict[str, object], dict[str, object]]:
            started = time.perf_counter()
            result = await engine.search(
                memory_space.memory_space_id, question.question
            )
            search_latency = time.perf_counter() - started
            prompt = _answer_user_prompt(
                dataset, question.question, question.question_date, result.render()
            )
            response = await generation.generate(
                GenerationRequest(
                    stage="benchmark_answer",
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

        async with semaphore:
            prediction, search_record = await asyncio.wait_for(
                work(), timeout=config.benchmark_search_sample_timeout_seconds
            )
        async with write_lock:
            append_jsonl(run_paths.predictions, prediction)
            append_jsonl(run_paths.search_results, search_record)

    try:
        await asyncio.gather(
            *(answer_one(space, question_index) for space, question_index in pending)
        )
    finally:
        await engine.close()


async def score_run(
    *,
    dataset: str,
    spaces: tuple[BenchmarkSpace, ...],
    run_paths: RunPaths,
    config: FluxFoldConfig,
) -> None:
    _validate_manifest(run_paths, dataset, spaces)
    predictions = _load_predictions(run_paths.predictions, dataset)
    questions = tuple(question for space in spaces for question in space.questions)
    generation = generation_provider(config, stage="score")
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
    summary["score_model"] = generation.model_id
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
) -> None:
    manifest = json.loads(run_paths.manifest.read_text(encoding="utf-8"))
    if manifest["dataset"] != dataset:
        raise ValidationError("dataset does not match build manifest")
    if list(manifest["selected_space_ids"]) != [space.source_id for space in spaces]:
        raise ValidationError("space selection does not match build manifest")


def _load_predictions(path: Path, dataset: str) -> dict[str, str]:
    output: dict[str, str] = {}
    key_field, value_field = _prediction_fields(dataset)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        question_id = str(item[key_field])
        if question_id in output:
            raise ValidationError(f"duplicate prediction: {question_id}")
        output[question_id] = str(item[value_field])
    return output


def _completed_question_ids(path: Path, dataset: str) -> set[str]:
    if not path.exists():
        return set()
    return set(_load_predictions(path, dataset))


def _prediction_fields(dataset: str) -> tuple[str, str]:
    if dataset == "longmemeval":
        return "question_id", "hypothesis"
    return "qa_id", "predicted_answer"


async def _finish_add_item(
    *,
    engine: FluxFold,
    memory_space_id: str,
    episode: NormalizedEpisode,
    writer: ArtifactWriter,
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
    }
    try:
        return await engine.add_episode(memory_space_id, episode), None
    except StageFailure as failure:
        if failure.error_class in terminal_item:
            return None, failure
        event_type = (
            "memory_space_build_paused"
            if failure.error_class in transient
            else "memory_space_build_blocked"
        )
        writer.event(
            {
                "event_type": event_type,
                "severity": "error",
                "timestamp_ms": int(time.time() * 1000),
                "memory_space_id": memory_space_id,
                "source_sequence": episode.source_sequence,
                "error_class": failure.error_class.value,
                "reason": failure.message,
            }
        )
        raise


def _load_checkpoint(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "completed_space_ids": [],
            "last_episode_by_space": {},
            "elapsed_seconds": 0.0,
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        "completed_space_ids": [
            str(item) for item in value.get("completed_space_ids", [])
        ],
        "last_episode_by_space": {
            str(key): int(sequence)
            for key, sequence in value.get("last_episode_by_space", {}).items()
        },
        "elapsed_seconds": float(value.get("elapsed_seconds") or 0.0),
    }


def _write_memory_bank(
    writer: ArtifactWriter, engine: FluxFold, *, updated_after: str
) -> None:
    writer.write_memory_bank(engine.memory_bank(), updated_after=updated_after)


def _write_build_checkpoint(
    writer: ArtifactWriter,
    run_paths: RunPaths,
    completed: set[str],
    last_episode_by_space: dict[str, int],
    *,
    elapsed_seconds: float,
) -> None:
    writer.write_json(
        run_paths.checkpoint,
        {
            "completed_space_ids": sorted(completed),
            "last_episode_by_space": {
                key: value for key, value in sorted(last_episode_by_space.items())
            },
            "elapsed_seconds": elapsed_seconds,
        },
    )


def _answer_user_prompt(
    dataset: str, question: str, question_date: str | None, search_text: str
) -> str:
    if dataset == "longmemeval":
        date_line = f"Current Date: {question_date}\n" if question_date else ""
        return (
            "I will give you retrieved memories. Please answer the question based on "
            "the relevant memories. Include every required fact. If the memories are "
            "insufficient, say that the available information is insufficient.\n\n"
            f"Retrieved memories:\n\n{search_text}\n\n"
            f"{date_line}Question: {question}\nAnswer:"
        )
    return (
        "Based on the retrieved memories, write an answer in the form of a short "
        "phrase. Answer with exact words from the memories whenever possible. Do not "
        'add extra facts, explanations, or hedging words such as "around" or '
        '"about".\n'
        "Time:\n"
        '- For "when" questions, answer with a date or relative time at the same '
        "granularity as the evidence. Format calendar days as D Month YYYY (for "
        'example "7 May 2023", not "07 May 2023" or "2023-05-07").\n'
        '- If a memory uses a relative expression such as "yesterday", "last week", '
        '"next month", or "the week before", resolve it against the date shown on '
        "that memory when the question asks when an event happened.\n"
        '- For "how long" or "how long ago" questions, copy the duration or relative '
        "form from the memories. Do not convert relative forms into calendar dates or "
        "the reverse unless the memories already state that form.\n"
        "If the memories contain the answer, output it even if you must connect more "
        'than one memory. Output "Not mentioned" only when the memories truly lack '
        "the asked information.\n\n"
        f"Question: {question}\n\nRetrieved memories:\n{search_text}\n\nShort answer:"
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
        f"Score model: `{summary['score_model']}`",
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
