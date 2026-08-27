"""Shared command-line plumbing for the six benchmark stage scripts."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from benchmarks.adapters import (
    BenchmarkSpace,
    load_locomo,
    load_longmemeval,
    select_sample,
)
from benchmarks.artifacts import RunPaths
from benchmarks.runner import (
    answer_run,
    build_run,
    load_dataset_from_manifest,
    score_run,
)
from benchmarks.runtime import (
    DEFAULT_LOCOMO_CONVERSATIONS_PATH,
    DEFAULT_LOCOMO_QUESTIONS_PATH,
    DEFAULT_LONGMEMEVAL_PATH,
    load_config,
    load_project_env,
)
from fluxfold.errors import ValidationError


def main(stage: str, *, sample: bool, argv: Sequence[str] | None = None) -> None:
    load_project_env()
    parser = _parser(stage, sample=sample)
    arguments = parser.parse_args(argv)
    config = load_config(arguments.config)
    run_paths = RunPaths(Path(arguments.run_dir).resolve())
    if stage == "build":
        dataset, spaces, data_paths = _build_selection(arguments, sample=sample)
        asyncio.run(
            build_run(
                dataset=dataset,
                spaces=spaces,
                run_paths=run_paths,
                data_paths=data_paths,
                mode="sample" if sample else "full",
                config=config,
            )
        )
        return
    if not run_paths.manifest.is_file():
        raise ValidationError(f"build manifest does not exist: {run_paths.manifest}")
    manifest = json.loads(run_paths.manifest.read_text(encoding="utf-8"))
    expected_mode = "sample" if sample else "full"
    if manifest.get("mode") != expected_mode:
        raise ValidationError(
            f"{expected_mode} stage cannot consume a {manifest.get('mode')} run"
        )
    dataset, spaces = load_dataset_from_manifest(run_paths)
    if stage == "answer":
        asyncio.run(
            answer_run(
                dataset=dataset,
                spaces=spaces,
                run_paths=run_paths,
                config=config,
            )
        )
    else:
        asyncio.run(
            score_run(
                dataset=dataset,
                spaces=spaces,
                run_paths=run_paths,
                config=config,
            )
        )


def _parser(stage: str, *, sample: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"FluxFold {stage} stage ({'sample' if sample else 'full'} run)."
    )
    parser.add_argument("--run-dir", required=True, help="Run artifact directory.")
    parser.add_argument("--config", help="Optional TOML configuration file.")
    if stage != "build":
        return parser
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("longmemeval", "locomo_refined"),
    )
    parser.add_argument(
        "--data-path",
        help="LongMemEval-S JSON or JSONL file. Defaults to the cloned dataset under data/.",
    )
    parser.add_argument(
        "--conversations-path",
        help="LoCoMo_refined conversations file. Defaults to the cloned dataset under data/.",
    )
    parser.add_argument(
        "--questions-path",
        help="LoCoMo_refined questions file. Defaults to the cloned dataset under data/.",
    )
    if sample:
        parser.add_argument(
            "--select",
            action="append",
            default=[],
            help="Question ID (LongMemEval) or sample ID/index (LoCoMo); repeatable.",
        )
    return parser


def _build_selection(
    arguments: argparse.Namespace, *, sample: bool
) -> tuple[str, tuple[BenchmarkSpace, ...], tuple[str, ...]]:
    dataset = str(arguments.dataset)
    if dataset == "longmemeval":
        data_paths = (
            str(
                _existing_data_file(
                    arguments.data_path or str(DEFAULT_LONGMEMEVAL_PATH),
                    "LongMemEval-S dataset",
                )
            ),
        )
        spaces = load_longmemeval(data_paths[0])
    else:
        data_paths = (
            str(
                _existing_data_file(
                    arguments.conversations_path
                    or str(DEFAULT_LOCOMO_CONVERSATIONS_PATH),
                    "LoCoMo_refined conversations",
                )
            ),
            str(
                _existing_data_file(
                    arguments.questions_path or str(DEFAULT_LOCOMO_QUESTIONS_PATH),
                    "LoCoMo_refined questions",
                )
            ),
        )
        spaces = load_locomo(*data_paths)
    if sample:
        spaces = select_sample(dataset, spaces, tuple(arguments.select))
    return dataset, spaces, data_paths


def _existing_data_file(path: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValidationError(
            f"{label} is missing: {resolved}. Run ./scripts/setup-dev.sh to clone datasets."
        )
    return resolved
