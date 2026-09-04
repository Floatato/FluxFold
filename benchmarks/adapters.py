"""Dataset-only normalization for LongMemEval-S and LoCoMo_refined."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fluxfold.errors import ValidationError
from fluxfold.models import EpisodeBlock, NormalizedEpisode, Role


@dataclass(frozen=True, slots=True)
class BenchmarkQuestion:
    question_id: str
    space_key: str
    question: str
    answer: str | tuple[str, ...]
    category: str
    question_date: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkSpace:
    space_key: str
    source_id: str
    episodes: tuple[NormalizedEpisode, ...]
    questions: tuple[BenchmarkQuestion, ...]


def load_longmemeval(path: str | Path) -> tuple[BenchmarkSpace, ...]:
    records = _load_records(path)
    spaces: list[BenchmarkSpace] = []
    for record_position, raw in enumerate(records):
        question_id = _required_string(raw, "question_id")
        session_ids = _required_list(raw, "haystack_session_ids")
        session_dates = _required_list(raw, "haystack_dates")
        sessions = _required_list(raw, "haystack_sessions")
        if not len(session_ids) == len(session_dates) == len(sessions):
            raise ValidationError(
                f"LongMemEval arrays have different lengths: {question_id}"
            )
        episodes: list[NormalizedEpisode] = []
        for index, (raw_id, raw_date, raw_session) in enumerate(
            zip(session_ids, session_dates, sessions, strict=True)
        ):
            if not isinstance(raw_session, list):
                raise ValidationError(
                    f"LongMemEval session is not a list: {question_id}/{index}"
                )
            source_key = f"{index}:{raw_id}"
            blocks: list[EpisodeBlock] = []
            for message_index, message in enumerate(raw_session):
                if not isinstance(message, dict):
                    raise ValidationError("LongMemEval message is not an object")
                role = _role(message.get("role"))
                content = _required_string(message, "content")
                blocks.append(
                    EpisodeBlock(
                        block_id=f"longmemeval:{question_id}:{index}:{message_index}",
                        sequence_no=message_index,
                        role=role,
                        content=content,
                        message_phase="final" if role is Role.ASSISTANT else None,
                    )
                )
            source_time = _parse_utc_milliseconds(str(raw_date))
            episodes.append(
                NormalizedEpisode(
                    source_type="longmemeval",
                    source_key=source_key,
                    source_sequence=index,
                    blocks=tuple(blocks),
                    source_started_at=source_time,
                    source_timezone="UTC",
                )
            )
        question = BenchmarkQuestion(
            question_id=question_id,
            space_key=f"longmemeval:{question_id}",
            question=_required_string(raw, "question"),
            answer=_answer_value(raw.get("answer")),
            category=(
                "abstention"
                if question_id.endswith("_abs")
                else _required_string(raw, "question_type")
            ),
            question_date=str(raw.get("question_date") or "") or None,
        )
        spaces.append(
            BenchmarkSpace(
                question.space_key, question_id, tuple(episodes), (question,)
            )
        )
    return tuple(spaces)


def load_locomo(
    conversations_path: str | Path, questions_path: str | Path
) -> tuple[BenchmarkSpace, ...]:
    conversations = _load_records(conversations_path)
    questions = _load_records(questions_path)
    questions_by_sample: dict[str, list[BenchmarkQuestion]] = {}
    for raw in questions:
        sample_id = _required_string(raw, "sample_id")
        questions_by_sample.setdefault(sample_id, []).append(
            BenchmarkQuestion(
                question_id=_required_string(raw, "qa_id"),
                space_key=f"locomo_refined:{sample_id}",
                question=_required_string(raw, "question"),
                answer=_answer_value(raw.get("answer")),
                category=_required_string(raw, "category"),
            )
        )
    spaces: list[BenchmarkSpace] = []
    for conversation_position, raw in enumerate(conversations):
        sample_id = _required_string(raw, "sample_id")
        speaker_a = _required_string(raw, "speaker_a")
        speaker_b = _required_string(raw, "speaker_b")
        raw_sessions = _required_list(raw, "sessions")
        episodes: list[NormalizedEpisode] = []
        for fallback_index, session in enumerate(raw_sessions):
            if not isinstance(session, dict):
                raise ValidationError("LoCoMo session is not an object")
            session_index = int(session.get("session_index", fallback_index))
            messages = _required_list(session, "messages")
            blocks: list[EpisodeBlock] = []
            for message_index, message in enumerate(messages):
                if not isinstance(message, dict):
                    raise ValidationError("LoCoMo message is not an object")
                speaker = _required_string(message, "speaker")
                role = _locomo_role(message.get("role"), speaker, speaker_a, speaker_b)
                content = _required_string(message, "text")
                caption = str(message.get("blip_caption") or "").strip()
                if caption:
                    content += f"\n[Dataset-provided image description: {caption}]"
                metadata: dict[str, object] = {
                    "dia_id": message.get("dia_id"),
                    "message_index": message_index,
                }
                images = message.get("images")
                if images:
                    metadata["image_urls"] = images
                blocks.append(
                    EpisodeBlock(
                        block_id=f"locomo:{sample_id}:{session_index}:{message_index}",
                        sequence_no=message_index,
                        role=role,
                        content=content,
                        message_phase="final" if role is Role.ASSISTANT else None,
                        speaker_id=speaker,
                        speaker_name=speaker,
                        metadata=metadata,
                    )
                )
            date_value = session.get("date_time")
            source_time = (
                _parse_utc_milliseconds(str(date_value))
                if date_value not in (None, "")
                else None
            )
            episodes.append(
                NormalizedEpisode(
                    source_type="locomo_refined",
                    source_key=f"{sample_id}:{session_index}",
                    source_sequence=session_index,
                    blocks=tuple(blocks),
                    source_started_at=source_time,
                    source_timezone="UTC" if source_time is not None else None,
                )
            )
        episodes.sort(key=lambda item: item.source_sequence)
        sequences = [item.source_sequence for item in episodes]
        if not sequences or sequences != list(
            range(sequences[0], sequences[0] + len(sequences))
        ):
            raise ValidationError(
                f"LoCoMo session indexes are not contiguous: {sample_id}"
            )
        spaces.append(
            BenchmarkSpace(
                f"locomo_refined:{sample_id}",
                sample_id,
                tuple(episodes),
                tuple(questions_by_sample.get(sample_id, [])),
            )
        )
    if {space.source_id for space in spaces} != set(questions_by_sample):
        raise ValidationError(
            "LoCoMo conversations and questions have different sample IDs"
        )
    return tuple(spaces)


def select_sample(
    dataset: str,
    spaces: tuple[BenchmarkSpace, ...],
    selectors: tuple[str, ...],
) -> tuple[BenchmarkSpace, ...]:
    if dataset == "locomo_refined":
        if len(selectors) != 1:
            raise ValidationError(
                "LoCoMo sample mode requires exactly one --select value"
            )
        selector = selectors[0]
        matches = [
            space
            for index, space in enumerate(spaces)
            if selector in {space.source_id, str(index)}
        ]
        if len(matches) != 1:
            raise ValidationError(f"LoCoMo conversation not found: {selector}")
        return tuple(matches)
    if selectors:
        wanted = set(selectors)
        selected = tuple(
            space for space in spaces if space.questions[0].question_id in wanted
        )
        if len(selected) != len(wanted):
            found = {space.questions[0].question_id for space in selected}
            raise ValidationError(
                f"LongMemEval question IDs not found: {sorted(wanted - found)}"
            )
        return selected
    by_category: dict[str, BenchmarkSpace] = {}
    for space in spaces:
        by_category.setdefault(space.questions[0].category, space)
    required = {
        "abstention",
        "knowledge-update",
        "multi-session",
        "single-session-assistant",
        "single-session-preference",
        "single-session-user",
        "temporal-reasoning",
    }
    if set(by_category) < required:
        raise ValidationError(
            "LongMemEval data does not contain all seven sample categories"
        )
    return tuple(by_category[category] for category in sorted(required))


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(value, dict) and isinstance(value.get("data"), list):
        value = value["data"]
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValidationError(
            f"dataset must contain a list or JSONL of objects: {path}"
        )
    return value


def _required_string(value: dict[str, Any], field: str) -> str:
    result = value.get(field)
    if not isinstance(result, str):
        raise ValidationError(f"missing string field: {field}")
    return result


def _required_list(value: dict[str, Any], field: str) -> list[Any]:
    result = value.get(field)
    if not isinstance(result, list):
        raise ValidationError(f"missing list field: {field}")
    return result


def _answer_value(value: Any) -> str | tuple[str, ...]:
    if isinstance(value, str):
        return value
    if type(value) is int:
        return str(value)
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, str) for item in value)
    ):
        return tuple(value)
    raise ValidationError(
        "benchmark answer must be a string, integer, or non-empty string list"
    )


def _role(value: object) -> Role:
    if value == "user":
        return Role.USER
    if value == "assistant":
        return Role.ASSISTANT
    raise ValidationError(f"unknown dataset role: {value}")


def _locomo_role(value: object, speaker: str, speaker_a: str, speaker_b: str) -> Role:
    if value in {"user", "assistant"}:
        return _role(value)
    if speaker == speaker_a:
        return Role.USER
    if speaker == speaker_b:
        return Role.ASSISTANT
    raise ValidationError(f"unknown LoCoMo speaker: {speaker}")


def _parse_utc_milliseconds(value: str) -> int:
    stripped = value.strip()
    formats = (
        "%Y/%m/%d (%a) %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%I:%M %p on %d %B, %Y",
    )
    for candidate in (stripped, stripped.upper()):
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return int(parsed.astimezone(UTC).timestamp() * 1000)
        except ValueError:
            pass
        for format_string in formats:
            try:
                parsed = datetime.strptime(candidate, format_string).replace(tzinfo=UTC)
                return int(parsed.timestamp() * 1000)
            except ValueError:
                continue
    raise ValidationError(f"unrecognized source datetime: {value}")
