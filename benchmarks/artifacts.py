"""Atomic benchmark artifacts and human-readable build logging."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fluxfold.models import AddResult, NormalizedEpisode


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
        if paths.events.exists():
            for line in paths.events.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._count_event(json.loads(line))

    def event(self, event: dict[str, object]) -> None:
        with self._lock:
            if str(event.get("event_type", "")).startswith("audit_"):
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
