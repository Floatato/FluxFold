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
from benchmarks.runtime import load_config
from fluxfold.errors import ValidationError


def main(stage: str, *, sample: bool, argv: Sequence[str] | None = None) -> None:
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
    parser.add_argument("--data-path", help="LongMemEval-S JSON or JSONL file.")
    parser.add_argument(
        "--conversations-path", help="LoCoMo_refined conversations file."
    )
    parser.add_argument("--questions-path", help="LoCoMo_refined questions file.")
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
        if not arguments.data_path:
            raise ValidationError("--data-path is required for LongMemEval")
        data_paths = (str(Path(arguments.data_path).resolve()),)
        spaces = load_longmemeval(data_paths[0])
    else:
        if not arguments.conversations_path or not arguments.questions_path:
            raise ValidationError(
                "--conversations-path and --questions-path are required for LoCoMo_refined"
            )
        data_paths = (
            str(Path(arguments.conversations_path).resolve()),
            str(Path(arguments.questions_path).resolve()),
        )
        spaces = load_locomo(*data_paths)
    if sample:
        spaces = select_sample(dataset, spaces, tuple(arguments.select))
    return dataset, spaces, data_paths
