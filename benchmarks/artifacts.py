"""Atomic benchmark artifacts and human-readable build logging."""

from __future__ import annotations

import json
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fluxfold.engine import LLM_IO_SAMPLE_QUOTAS
from fluxfold.models import MemoryBankSpace

_LLM_IO_SAMPLE_HEADING = re.compile(r"^## ([a-z_]+) · sample (\d+)\s*$")
_LLM_IO_SAMPLE_HEADER = """# LLM I/O samples

Each LLM call that returns model output has an independent 10% chance of being sampled. Each sample contains only the user prompt and model output.

Quotas: extract, link (no `association_search`), review, split, and summary ×10; link with `association_search` ×5. For a link agent loop that uses `association_search`, only the final linking decision is eligible.

"""


@dataclass(frozen=True, slots=True)
class RunPaths:
    root: Path

    @property
    def database(self) -> Path:
        return self.root / "fluxfold.sqlite3"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def events(self) -> Path:
        return self.root / "build_events.jsonl"

    @property
    def llm_io_samples(self) -> Path:
        return self.root / "llm_io_samples.md"

    @property
    def memory_bank(self) -> Path:
        return self.root / "memory_bank.md"

    @property
    def checkpoint(self) -> Path:
        return self.root / "build_checkpoint.json"

    @property
    def build_summary(self) -> Path:
        return self.root / "build_summary.json"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions.jsonl"

    @property
    def search_records(self) -> Path:
        return self.root / "search_records.jsonl"

    @property
    def search_summary(self) -> Path:
        return self.root / "search_summary.json"

    @property
    def search_audit(self) -> Path:
        return self.root / "search_audit.md"

    @property
    def answer_llm_input_samples(self) -> Path:
        return self.root / "answer_llm_input_samples.md"

    @property
    def scores(self) -> Path:
        return self.root / "scores.jsonl"

    @property
    def score_summary(self) -> Path:
        return self.root / "score_summary.json"

    @property
    def score_markdown(self) -> Path:
        return self.root / "score_summary.md"


class ArtifactWriter:
    def __init__(self, paths: RunPaths) -> None:
        self.paths = paths
        paths.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.run_id: str | None = None
        self._successful_llm_calls = 0
        self._failed_llm_calls = 0
        self._llm_input_tokens = 0
        self._llm_output_tokens = 0
        self._llm_total_tokens = 0
        self._terminal_failures = 0
        self._llm_io_sample_counts = _llm_io_sample_counts(paths.llm_io_samples)
        if paths.events.exists():
            for line in paths.events.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._count_event(json.loads(line))

    def event(self, event: dict[str, object]) -> None:
        event_type = str(event.get("event_type", ""))
        if event_type == "llm_io_sample":
            self._accept_llm_io_sample(event)
            return
        with self._lock:
            if self.run_id is not None:
                event = {"run_id": self.run_id, **event}
            self._count_event(event)
            with self.paths.events.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
                )

    def _accept_llm_io_sample(self, event: dict[str, object]) -> None:
        kind = str(event["kind"])
        with self._lock:
            if self._llm_io_sample_counts[kind] >= LLM_IO_SAMPLE_QUOTAS[kind]:
                return
            self._llm_io_sample_counts[kind] += 1
            index = self._llm_io_sample_counts[kind]
            path = self.paths.llm_io_samples
            if not path.exists():
                path.write_text(_LLM_IO_SAMPLE_HEADER, encoding="utf-8")
            with path.open("a", encoding="utf-8") as handle:
                handle.write(
                    _render_llm_io_sample(
                        kind,
                        index,
                        str(event["user_prompt"]),
                        str(event["output"]),
                    )
                )

    def build_metrics(self) -> dict[str, int]:
        with self._lock:
            return {
                "successful_llm_call_count": self._successful_llm_calls,
                "failed_llm_call_count": self._failed_llm_calls,
                "build_llm_input_tokens": self._llm_input_tokens,
                "build_llm_output_tokens": self._llm_output_tokens,
                "build_llm_total_tokens": self._llm_total_tokens,
                "terminal_failure_count": self._terminal_failures,
            }

    def _count_event(self, event: dict[str, object]) -> None:
        if event.get("event_type") == "llm_call":
            self._llm_input_tokens += int(event.get("input_tokens", 0))
            self._llm_output_tokens += int(event.get("output_tokens", 0))
            self._llm_total_tokens += int(event.get("total_tokens", 0))
            if event.get("result") == "success":
                self._successful_llm_calls += 1
            elif event.get("result") == "failed":
                self._failed_llm_calls += 1
        if event.get("event_type") in {
            "episode_terminal_failure",
            "maintenance_terminal_failure",
            "qa_terminal_failure",
        }:
            self._terminal_failures += 1

    def write_memory_bank(
        self,
        spaces: tuple[MemoryBankSpace, ...],
        *,
        updated_after: str,
    ) -> None:
        text = _render_memory_bank(spaces, updated_after=updated_after)
        with self._lock:
            temporary = self.paths.memory_bank.with_suffix(
                self.paths.memory_bank.suffix + ".tmp"
            )
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(self.paths.memory_bank)

    def write_json(self, path: Path, value: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def append_jsonl(path: Path, value: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_search_artifacts(paths: RunPaths, records: list[dict[str, object]]) -> None:
    ordered = sorted(records, key=lambda record: int(record["query_number"]))
    ArtifactWriter(paths).write_json(paths.search_summary, _search_summary(ordered))
    _write_text(paths.search_audit, _render_search_audit(ordered))
    _write_text(
        paths.answer_llm_input_samples,
        _render_answer_llm_input_samples(ordered),
    )


def _search_summary(records: list[dict[str, object]]) -> dict[str, object]:
    latencies = [float(record["search_latency_seconds"]) for record in records]
    answer_tokens = [int(record["answer_llm_input_tokens"]) for record in records]
    summary_chars: list[int] = []
    memory_chars: list[int] = []
    context_chars: list[int] = []
    subject_counts: list[int] = []
    summary_counts: list[int] = []
    memory_counts: list[int] = []
    subject_similarities: list[list[float]] = []
    memory_similarities: list[list[float]] = []

    for record in records:
        groups = list(record["groups"])
        summaries = [
            str(group["summary"]) for group in groups if group["summary"] is not None
        ]
        memories = [memory for group in groups for memory in list(group["memories"])]
        summary_chars.append(sum(len(summary) for summary in summaries))
        memory_chars.append(sum(len(str(memory["content"])) for memory in memories))
        context_chars.append(int(record["rendered_context_chars"]))
        subject_counts.append(len(groups))
        summary_counts.append(len(summaries))
        memory_counts.append(len(memories))
        subject_similarities.append([float(group["similarity"]) for group in groups])
        memory_similarities.append([float(memory["similarity"]) for memory in memories])

    return {
        "search_latency_seconds": {
            "mean": _mean(latencies),
            "p50": _percentile(latencies, 0.5),
            "p90": _percentile(latencies, 0.9),
            "max": max(latencies, default=None),
        },
        "answer_llm_input_tokens_per_query": _mean_min_max(answer_tokens),
        "summary_chars_per_query": _mean_min_max(summary_chars),
        "memory_content_chars_per_query": _mean_min_max(memory_chars),
        "rendered_context_chars_per_query": _mean_min_max(context_chars),
        "subject_retrieval_similarity": _retrieval_similarity_summary(
            subject_similarities
        ),
        "memory_retrieval_similarity": _retrieval_similarity_summary(
            memory_similarities
        ),
        "returned_per_query": {
            "subjects_mean": _mean(subject_counts),
            "summaries_mean": _mean(summary_counts),
            "memories_mean": _mean(memory_counts),
        },
    }


def _retrieval_similarity_summary(
    per_query: list[list[float]],
) -> dict[str, object]:
    nonempty = [values for values in per_query if values]
    return {
        "per_query_top1": _mean_min_max([max(values) for values in nonempty]),
        "per_query_mean": _mean_min_max(
            [sum(values) / len(values) for values in nonempty]
        ),
        "per_query_min": _mean_min_max([min(values) for values in nonempty]),
    }


def _mean_min_max(values: list[int] | list[float]) -> dict[str, object]:
    return {
        "mean": _mean(values),
        "max": max(values, default=None),
        "min": min(values, default=None),
    }


def _mean(values: list[int] | list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _render_search_audit(records: list[dict[str, object]]) -> str:
    lines = ["# Search audit", ""]
    for record in records:
        groups = list(record["groups"])
        summary_count = sum(group["summary"] is not None for group in groups)
        memory_count = sum(len(list(group["memories"])) for group in groups)
        lines.extend(
            [
                f"## Query {record['query_number']}",
                "",
                "**Question**",
                "",
                *_blockquote(str(record["query"])),
                "",
                (
                    f"- Displayed: {len(groups)} subjects · {summary_count} "
                    f"summaries · {memory_count} memories"
                ),
                "",
            ]
        )
        for index, group in enumerate(groups, start=1):
            lines.extend(
                [
                    f"### {index}. {_heading_text(str(group['name']))}",
                    "",
                    f"- Subject name similarity: `{float(group['similarity']):.4f}`",
                    f"- Retrieval route: `{group['route']}`",
                ]
            )
            if group["summary"] is not None:
                lines.extend(_field_lines("Summary", str(group["summary"])))
            lines.extend(["", "#### Memories", ""])
            for memory_index, memory in enumerate(list(group["memories"]), start=1):
                lines.extend(
                    [
                        (
                            f"{memory_index}. Similarity "
                            f"`{float(memory['similarity']):.4f}` · "
                            f"{memory['route']}"
                        ),
                        *_blockquote(str(memory["content"]), indent="   "),
                        "",
                    ]
                )
    return "\n".join(lines).rstrip() + "\n"


def _render_answer_llm_input_samples(records: list[dict[str, object]]) -> str:
    lines = ["# Answer LLM input samples", ""]
    samples = [
        record["answer_input_sample"]
        for record in records
        if "answer_input_sample" in record
    ][:3]
    for index, sample in enumerate(samples, start=1):
        lines.extend([f"## Sample {index}", "", "### System prompt", ""])
        system_prompt = sample["system_prompt"]
        if system_prompt is None:
            lines.extend(["_(none)_", ""])
        else:
            lines.extend([_markdown_fence(str(system_prompt), ""), ""])
        lines.extend(
            [
                "### User prompt",
                "",
                _markdown_fence(str(sample["user_prompt"]), ""),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _field_lines(name: str, value: str) -> list[str]:
    parts = value.splitlines() or [""]
    return [f"- {name}: {parts[0]}", *(f"  {part}" for part in parts[1:])]


def _blockquote(value: str, *, indent: str = "") -> list[str]:
    return [f"{indent}> {line}" for line in (value.splitlines() or [""])]


def _write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _render_memory_bank(
    spaces: tuple[MemoryBankSpace, ...],
    *,
    updated_after: str,
) -> str:
    lines = [
        "# Memory bank",
        "",
        f"Active subjects and memories after {updated_after}.",
        "",
    ]
    if not spaces:
        lines.extend(["_(no memory spaces)_", ""])
        return "\n".join(lines)
    for space in spaces:
        lines.extend([f"## `{space.space_key}`", ""])
        if not space.subjects:
            lines.extend(["_(no active subjects)_", ""])
            continue
        for subject in space.subjects:
            lines.extend([f"### {_heading_text(subject.name)}", ""])
            if subject.summary:
                lines.extend([subject.summary, ""])
            if not subject.memory_contents:
                lines.extend(["_(no active memories)_", ""])
                continue
            for content in subject.memory_contents:
                lines.append(_bullet(content))
            lines.append("")
    return "\n".join(lines)


def _heading_text(value: str) -> str:
    return " ".join(value.split()) or "(unnamed)"


def _bullet(text: str) -> str:
    lines = text.splitlines() or [""]
    return "\n".join([f"- {lines[0]}", *(f"  {line}" for line in lines[1:])])


def _llm_io_sample_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _LLM_IO_SAMPLE_HEADING.match(line)
        if match is not None:
            counts[match.group(1)] += 1
    return counts


def _render_llm_io_sample(
    kind: str,
    index: int,
    user_prompt: str,
    output: str,
) -> str:
    lines = [f"## {kind} · sample {index}", ""]
    lines.extend(_prompt_section("### User prompt", user_prompt))
    lines.extend(_prompt_section("### Model output", output))
    lines.extend(["---", "", ""])
    return "\n".join(lines)


def _prompt_section(heading: str, text: str) -> list[str]:
    rendered, language = _pretty_json_if_possible(text)
    return [heading, "", _markdown_fence(rendered, language), ""]


def _pretty_json_if_possible(text: str) -> tuple[str, str]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return text, ""
    return json.dumps(value, ensure_ascii=False, indent=2), "json"


def _markdown_fence(text: str, language: str) -> str:
    longest = 0
    current = 0
    for char in text:
        if char == "`":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    ticks = "`" * max(3, longest + 1)
    body = text if text.endswith("\n") else f"{text}\n"
    opener = f"{ticks}{language}\n" if language else f"{ticks}\n"
    return f"{opener}{body}{ticks}"
