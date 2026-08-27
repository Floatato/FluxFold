"""Default benchmark run-directory allocation and lookup."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from benchmarks.runtime import PROJECT_ROOT
from fluxfold.errors import ValidationError

DATASETS = ("longmemeval", "locomo_refined")
RUNS_ROOT = PROJECT_ROOT / "runs"

_STAMP = re.compile(r"^_(\d{1,2})\.(\d{1,2})_(\d{2}):(\d{2})_(\d+)$")


def allocate_run_dir(
    dataset: str,
    *,
    runs_root: Path | None = None,
    when: datetime | None = None,
) -> Path:
    root = runs_root if runs_root is not None else RUNS_ROOT
    moment = when if when is not None else datetime.now()
    prefix = _stamp_prefix(dataset, moment)
    seq = 1
    while True:
        candidate = root / f"{prefix}{seq}"
        if not candidate.exists():
            return candidate
        seq += 1


def latest_run_dir(
    dataset: str,
    *,
    mode: str,
    runs_root: Path | None = None,
) -> Path:
    root = runs_root if runs_root is not None else RUNS_ROOT
    matches: list[tuple[tuple[int, int, int, int, int], Path]] = []
    if root.is_dir():
        for path in root.iterdir():
            parsed = parse_run_dir_name(path.name)
            if parsed is None:
                continue
            parsed_dataset, stamp, seq = parsed
            if parsed_dataset != dataset or not path.is_dir():
                continue
            manifest_path = path / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("dataset") != dataset or manifest.get("mode") != mode:
                continue
            matches.append(((*stamp, seq), path))
    if not matches:
        raise ValidationError(
            f"no {mode} run directory found for {dataset} under {root}"
        )
    matches.sort(key=lambda item: item[0])
    return matches[-1][1]


def parse_run_dir_name(
    name: str,
) -> tuple[str, tuple[int, int, int, int], int] | None:
    for dataset in DATASETS:
        if not name.startswith(f"{dataset}_"):
            continue
        matched = _STAMP.fullmatch(name[len(dataset) :])
        if matched is None:
            return None
        month, day, hour, minute, seq = (int(part) for part in matched.groups())
        return dataset, (month, day, hour, minute), seq
    return None


def format_run_dir_name(dataset: str, when: datetime, seq: int) -> str:
    return f"{_stamp_prefix(dataset, when)}{seq}"


def display_run_dir(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _stamp_prefix(dataset: str, when: datetime) -> str:
    return f"{dataset}_{when.month}.{when.day}_{when.hour:02d}:{when.minute:02d}_"
