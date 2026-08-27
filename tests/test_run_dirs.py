from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import pytest
from benchmarks.cli import _resolve_run_dir
from benchmarks.run_dirs import (
    allocate_run_dir,
    format_run_dir_name,
    latest_run_dir,
    parse_run_dir_name,
)

from fluxfold.errors import ValidationError


def _write_manifest(path: Path, *, dataset: str, mode: str) -> None:
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(
        json.dumps({"dataset": dataset, "mode": mode}),
        encoding="utf-8",
    )


def test_format_matches_month_day_clock_and_seq() -> None:
    when = datetime(2026, 8, 27, 21, 2)
    assert format_run_dir_name("longmemeval", when, 1) == "longmemeval_8.27_21:02_1"
    assert format_run_dir_name("longmemeval", when, 2) == "longmemeval_8.27_21:02_2"
    assert (
        format_run_dir_name("locomo_refined", datetime(2026, 1, 5, 9, 7), 1)
        == "locomo_refined_1.5_09:07_1"
    )


def test_parse_run_dir_name_round_trip() -> None:
    name = "locomo_refined_8.27_21:02_2"
    parsed = parse_run_dir_name(name)
    assert parsed == ("locomo_refined", (8, 27, 21, 2), 2)
    assert parse_run_dir_name("locomo-sample") is None
    assert parse_run_dir_name("longmemeval_8.27_21:02") is None


def test_allocate_run_dir_increments_seq_in_the_same_minute(tmp_path: Path) -> None:
    when = datetime(2026, 8, 27, 21, 2)
    first = allocate_run_dir("longmemeval", runs_root=tmp_path, when=when)
    first.mkdir()
    second = allocate_run_dir("longmemeval", runs_root=tmp_path, when=when)
    assert first.name == "longmemeval_8.27_21:02_1"
    assert second.name == "longmemeval_8.27_21:02_2"


def test_latest_run_dir_selects_same_dataset_and_mode(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path / "longmemeval_8.27_20:00_1", dataset="longmemeval", mode="sample"
    )
    _write_manifest(
        tmp_path / "longmemeval_8.27_21:02_1", dataset="longmemeval", mode="sample"
    )
    _write_manifest(
        tmp_path / "longmemeval_8.27_21:02_2", dataset="longmemeval", mode="full"
    )
    _write_manifest(
        tmp_path / "locomo_refined_8.27_21:03_1",
        dataset="locomo_refined",
        mode="sample",
    )
    (tmp_path / "longmemeval_8.27_21:04_1").mkdir()
    latest = latest_run_dir("longmemeval", mode="sample", runs_root=tmp_path)
    assert latest.name == "longmemeval_8.27_21:02_1"


def test_latest_run_dir_prefers_higher_seq_at_same_stamp(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path / "longmemeval_8.27_21:02_1", dataset="longmemeval", mode="sample"
    )
    _write_manifest(
        tmp_path / "longmemeval_8.27_21:02_3", dataset="longmemeval", mode="sample"
    )
    latest = latest_run_dir("longmemeval", mode="sample", runs_root=tmp_path)
    assert latest.name == "longmemeval_8.27_21:02_3"


def test_latest_run_dir_errors_when_none_match(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path / "longmemeval_8.27_21:02_1", dataset="longmemeval", mode="full"
    )
    with pytest.raises(ValidationError, match="no sample run directory"):
        latest_run_dir("longmemeval", mode="sample", runs_root=tmp_path)


def test_resolve_run_dir_uses_explicit_path(tmp_path: Path) -> None:
    target = tmp_path / "custom"
    path = _resolve_run_dir(
        "answer",
        Namespace(run_dir=str(target), dataset="longmemeval"),
        sample=True,
    )
    assert path == target.resolve()


def test_resolve_run_dir_requires_dataset_when_omitted() -> None:
    with pytest.raises(ValidationError, match="--dataset is required"):
        _resolve_run_dir(
            "answer",
            Namespace(run_dir=None, dataset=None),
            sample=True,
        )
