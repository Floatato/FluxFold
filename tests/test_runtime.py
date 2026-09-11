from __future__ import annotations

import asyncio
import os
from argparse import Namespace
from pathlib import Path

import pytest
from benchmarks.cli import (
    _build_selection,
    _existing_data_file,
    _parser,
    _runtime_config,
)
from benchmarks.runtime import (
    DEFAULT_LOCOMO_CONVERSATIONS_PATH,
    DEFAULT_LOCOMO_QUESTIONS_PATH,
    DEFAULT_LONGMEMEVAL_PATH,
    PROJECT_ROOT,
    embedding_provider,
    generation_provider,
    load_env_file,
)

from fluxfold.config import FluxFoldConfig
from fluxfold.errors import ValidationError
from fluxfold.providers import (
    LocalMiniLMEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)


def test_default_dataset_paths_are_under_data() -> None:
    assert DEFAULT_LONGMEMEVAL_PATH == (
        PROJECT_ROOT / "data" / "LongMemEval" / "data" / "longmemeval_s_cleaned.json"
    )
    assert DEFAULT_LOCOMO_CONVERSATIONS_PATH == (
        PROJECT_ROOT
        / "data"
        / "LoCoMo_refined"
        / "data"
        / "public"
        / "conversations.jsonl"
    )
    assert DEFAULT_LOCOMO_QUESTIONS_PATH == (
        PROJECT_ROOT / "data" / "LoCoMo_refined" / "data" / "public" / "questions.jsonl"
    )


def test_load_env_file_sets_missing_keys(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "FLUXFOLD_TEST_A=one\n"
        'FLUXFOLD_TEST_B="Instruct: x\\nQuery:{text}"\n'
        "FLUXFOLD_TEST_C=plain\\nQuery:{text}\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("FLUXFOLD_TEST_A", raising=False)
    monkeypatch.delenv("FLUXFOLD_TEST_B", raising=False)
    monkeypatch.delenv("FLUXFOLD_TEST_C", raising=False)
    load_env_file(path)
    assert os.environ["FLUXFOLD_TEST_A"] == "one"
    assert os.environ["FLUXFOLD_TEST_B"] == "Instruct: x\nQuery:{text}"
    assert os.environ["FLUXFOLD_TEST_C"] == "plain\nQuery:{text}"


def test_load_env_file_does_not_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FLUXFOLD_TEST_A", "existing")
    path = tmp_path / ".env"
    path.write_text("FLUXFOLD_TEST_A=fromfile\n", encoding="utf-8")
    load_env_file(path)
    assert os.environ["FLUXFOLD_TEST_A"] == "existing"


def test_load_env_file_skips_missing_path(tmp_path: Path) -> None:
    load_env_file(tmp_path / "absent.env")


def test_generation_provider_reads_stage_env(monkeypatch) -> None:
    monkeypatch.setenv("FLUXFOLD_BUILD_MODEL", "build-model")
    monkeypatch.setenv("FLUXFOLD_BUILD_API_KEY", "build-key")
    monkeypatch.setenv("FLUXFOLD_BUILD_BASE_URL", "https://build.example/v1")
    monkeypatch.setenv("FLUXFOLD_ANSWER_MODEL", "answer-model")
    monkeypatch.setenv("FLUXFOLD_ANSWER_API_KEY", "answer-key")
    monkeypatch.setenv("FLUXFOLD_ANSWER_BASE_URL", "https://answer.example/v1")
    monkeypatch.setenv("FLUXFOLD_SCORE_MODEL", "score-model")
    monkeypatch.setenv("FLUXFOLD_SCORE_API_KEY", "score-key")
    monkeypatch.setenv("FLUXFOLD_SCORE_BASE_URL", "https://score.example/v1")
    config = FluxFoldConfig()
    build = generation_provider(config, stage="build")
    answer = generation_provider(config, stage="answer")
    score = generation_provider(config, stage="score")
    assert build.model_id == "build-model"
    assert answer.model_id == "answer-model"
    assert score.model_id == "score-model"
    assert build._client.max_retries == 5
    assert build._client.timeout.connect == 30.0
    assert build._client.timeout.read == 600.0
    assert build._client.timeout.write == 600.0
    assert build._client.timeout.pool == 30.0
    asyncio.run(build.close())
    asyncio.run(answer.close())
    asyncio.run(score.close())


def test_generation_provider_requires_stage_model(monkeypatch) -> None:
    monkeypatch.delenv("FLUXFOLD_BUILD_MODEL", raising=False)
    with pytest.raises(ValidationError, match="FLUXFOLD_BUILD_MODEL"):
        generation_provider(FluxFoldConfig(), stage="build")


def test_embedding_provider_defaults_to_local(monkeypatch) -> None:
    monkeypatch.delenv("FLUXFOLD_EMBEDDING_PROVIDER", raising=False)
    provider = embedding_provider(FluxFoldConfig())
    assert isinstance(provider, LocalMiniLMEmbeddingProvider)


def test_embedding_provider_supports_openai_compatible(monkeypatch) -> None:
    monkeypatch.setenv("FLUXFOLD_EMBEDDING_PROVIDER", "openai-compatible")
    monkeypatch.setenv("FLUXFOLD_EMBEDDING_MODEL", "embedding-model")
    monkeypatch.setenv("FLUXFOLD_EMBEDDING_DIMENSION", "12")
    provider = embedding_provider(FluxFoldConfig())
    assert isinstance(provider, OpenAICompatibleEmbeddingProvider)
    assert provider.model_info.model == "embedding-model"
    assert provider.model_info.dimension == 12


def test_embedding_provider_rejects_unknown_kind(monkeypatch) -> None:
    monkeypatch.setenv("FLUXFOLD_EMBEDDING_PROVIDER", "unknown")
    with pytest.raises(ValidationError, match="FLUXFOLD_EMBEDDING_PROVIDER"):
        embedding_provider(FluxFoldConfig())


def test_missing_dataset_file_explains_setup(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    with pytest.raises(ValidationError, match="setup-dev.sh"):
        _existing_data_file(str(missing), "LongMemEval-S dataset")


def test_build_selection_rejects_missing_longmemeval(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="LongMemEval-S dataset is missing"):
        _build_selection(
            Namespace(
                dataset="longmemeval",
                data_path=str(tmp_path / "missing.json"),
                conversations_path=None,
                questions_path=None,
                select=[],
            ),
            sample=True,
        )


def test_build_cli_overrides_memory_space_build_concurrency(tmp_path: Path) -> None:
    parser = _parser("build", sample=False)
    arguments = parser.parse_args(
        [
            "--dataset",
            "longmemeval",
            "--memory-space-build-concurrency",
            "4",
        ]
    )
    config = _runtime_config(arguments, stage="build")
    assert config.benchmark_memory_space_build_concurrency == 4
    toml_path = tmp_path / "fluxfold.toml"
    toml_path.write_text(
        "[fluxfold]\nbenchmark_memory_space_build_concurrency = 2\n",
        encoding="utf-8",
    )
    arguments = parser.parse_args(
        [
            "--dataset",
            "longmemeval",
            "--config",
            str(toml_path),
            "--memory-space-build-concurrency",
            "3",
        ]
    )
    config = _runtime_config(arguments, stage="build")
    assert config.benchmark_memory_space_build_concurrency == 3
    arguments = parser.parse_args(
        ["--dataset", "longmemeval", "--config", str(toml_path)]
    )
    config = _runtime_config(arguments, stage="build")
    assert config.benchmark_memory_space_build_concurrency == 2


def test_build_cli_rejects_non_positive_memory_space_build_concurrency() -> None:
    parser = _parser("build", sample=False)
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--dataset", "longmemeval", "--memory-space-build-concurrency", "0"]
        )
