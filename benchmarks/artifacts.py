"""Atomic benchmark artifacts and human-readable build logging."""

from __future__ import annotations

import json
import re
import threading
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fluxfold.engine import LLM_IO_SAMPLE_QUOTAS
from fluxfold.models import AddResult, NormalizedEpisode

_LLM_IO_SAMPLE_HEADING = re.compile(r"^## ([a-z_]+) · sample (\d+)\s*$")
_LLM_IO_SAMPLE_HEADER = """# LLM I/O samples

Sampled first-attempt structured-output successes: the complete system prompt, user prompt, and model output.

Quotas: extract, link (no `association_search`), review (no `provenance_viewed`), split, and summary ×2. Two-round `association_search` and `provenance_viewed` paths are sampled once each if they occur.

"""
_TWO_ROUND_TITLES = {
    "link_association_search": (
        "Round 1 — association_search request",
        "Round 2 — final linking decision",
    ),
    "review_provenance": (
        "Round 1 — provenance request",
        "Round 2 — final review decision",
    ),
}


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
    def audit(self) -> Path:
        return self.root / "build_audit.md"

    @property
    def llm_io_samples(self) -> Path:
        return self.root / "llm_io_samples.md"

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
    def search_results(self) -> Path:
        return self.root / "search_results.jsonl"

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
        self._pending_audit: dict[str, list[dict[str, object]]] = {}
        self._successful_llm_calls = 0
        self._failed_llm_calls = 0
        self._llm_tokens = 0
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
            if event_type.startswith("audit_"):
                memory_space_id = str(event["memory_space_id"])
                self._pending_audit.setdefault(memory_space_id, []).append(event)
                return
            if self.run_id is not None:
                event = {"run_id": self.run_id, **event}
            self._count_event(event)
            with self.paths.events.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
                )

    def _accept_llm_io_sample(self, event: dict[str, object]) -> None:
        kind = str(event["kind"])
        rounds = event["rounds"]
        assert isinstance(rounds, list)
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
                        tuple(rounds),
                        run_id=self.run_id,
                        timestamp_ms=event.get("timestamp_ms"),
                    )
                )

    def audit_episode(
        self,
        space_key: str,
        memory_space_id: str,
        episode: NormalizedEpisode,
        result: AddResult,
    ) -> None:
        lines = [
            f"## Episode `{episode.source_key}` in `{space_key}`",
            "",
            f"Source sequence: {episode.source_sequence}",
            "",
        ]
        for block in episode.blocks:
            speaker = f" / {block.speaker_name}" if block.speaker_name else ""
            lines.extend(
                [
                    f"### {block.sequence_no}. {block.role.value}{speaker}",
                    "",
                    block.content,
                    "",
                ]
            )
        lines.extend(
            [
                "### Commit result",
                "",
                "```json",
                json.dumps(asdict(result), ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
        with self._lock:
            for decision in self._pending_audit.pop(memory_space_id, []):
                lines.extend(
                    [
                        f"### {str(decision['event_type']).replace('_', ' ').title()}",
                        "",
                        "```json",
                        json.dumps(decision, ensure_ascii=False, indent=2),
                        "```",
                        "",
                    ]
                )
            with self.paths.audit.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(lines))

    def build_metrics(self) -> dict[str, int]:
        with self._lock:
            return {
                "successful_llm_call_count": self._successful_llm_calls,
                "failed_llm_call_count": self._failed_llm_calls,
                "write_llm_total_tokens": self._llm_tokens,
                "terminal_failure_count": self._terminal_failures,
            }

    def _count_event(self, event: dict[str, object]) -> None:
        if event.get("event_type") == "llm_call":
            self._llm_tokens += int(event.get("total_tokens", 0))
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

    def audit_episode_failure(
        self,
        space_key: str,
        memory_space_id: str,
        episode: NormalizedEpisode,
        error_class: str,
        reason: str,
    ) -> None:
        lines = [
            f"## Episode `{episode.source_key}` in `{space_key}`",
            "",
            f"Source sequence: {episode.source_sequence}",
            "",
        ]
        for block in episode.blocks:
            speaker = f" / {block.speaker_name}" if block.speaker_name else ""
            lines.extend(
                [
                    f"### {block.sequence_no}. {block.role.value}{speaker}",
                    "",
                    block.content,
                    "",
                ]
            )
        lines.extend(
            [
                "### Terminal failure",
                "",
                f"- Error class: `{error_class}`",
                f"- Reason: {reason}",
                "",
            ]
        )
        with self._lock:
            self._pending_audit.pop(memory_space_id, None)
            with self.paths.audit.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(lines))

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
    rounds: tuple[object, ...],
    *,
    run_id: str | None,
    timestamp_ms: object,
) -> str:
    lines = [f"## {kind} · sample {index}", ""]
    if run_id is not None:
        lines.append(f"- run_id: `{run_id}`")
    if timestamp_ms is not None:
        lines.append(f"- timestamp_ms: `{timestamp_ms}`")
    titles = _TWO_ROUND_TITLES.get(kind)
    for position, round_payload in enumerate(rounds):
        assert isinstance(round_payload, dict)
        if titles is not None:
            lines.extend(["", f"### {titles[position]}", ""])
        request_id = round_payload.get("request_id")
        stage = round_payload.get("stage")
        meta: list[str] = []
        if stage is not None:
            meta.append(f"- stage: `{stage}`")
        if request_id is not None:
            meta.append(f"- request_id: `{request_id}`")
        if meta:
            lines.extend([*meta, ""])
        heading_prefix = "#### " if titles is not None else "### "
        lines.extend(
            _prompt_section(
                f"{heading_prefix}System prompt",
                str(round_payload["system_prompt"]),
            )
        )
        lines.extend(
            _prompt_section(
                f"{heading_prefix}User prompt",
                str(round_payload["user_prompt"]),
            )
        )
        lines.extend(
            _prompt_section(
                f"{heading_prefix}Model output",
                str(round_payload["output"]),
            )
        )
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
