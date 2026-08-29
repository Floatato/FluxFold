"""Typed public values and provider structured-output models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


def utc_milliseconds() -> int:
    """Return current UTC time as Unix milliseconds."""

    return int(datetime.now(tz=UTC).timestamp() * 1000)


def _search_memory_date(latest_source_at: int | None) -> str | None:
    if latest_source_at is None:
        return None
    moment = datetime.fromtimestamp(latest_source_at / 1000, tz=UTC)
    return f"{moment.day} {moment.strftime('%B %Y')}"


def canonical_json(value: object) -> str:
    """Serialize a JSON-compatible value deterministically."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def text_hash(value: str) -> str:
    """Return a lowercase hexadecimal SHA-256 digest."""

    return sha256(value.encode("utf-8")).hexdigest()


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class EpisodeBlock:
    """One normalized user or assistant message."""

    block_id: str
    sequence_no: int
    role: Role
    content: str
    message_phase: str | None = None
    speaker_id: str | None = None
    speaker_name: str | None = None
    observed_at: int | None = None
    metadata: dict[str, object] | None = None
    preprocessor_version: str | None = None

    def canonical_value(self) -> dict[str, object]:
        """Return the block representation included in episode identity."""

        return {
            "block_id": self.block_id,
            "content": self.content,
            "message_phase": self.message_phase,
            "metadata": self.metadata,
            "observed_at": self.observed_at,
            "preprocessor_version": self.preprocessor_version,
            "role": self.role.value,
            "sequence_no": self.sequence_no,
            "speaker_id": self.speaker_id,
            "speaker_name": self.speaker_name,
        }


@dataclass(frozen=True, slots=True)
class NormalizedEpisode:
    """A complete immutable dataset session accepted by the experimental API."""

    source_type: str
    source_key: str
    source_sequence: int
    blocks: tuple[EpisodeBlock, ...]
    payload_version: str = "1"
    source_started_at: int | None = None
    source_ended_at: int | None = None
    source_timezone: str | None = None

    def canonical_value(self) -> dict[str, object]:
        """Return the entire stable source payload."""

        return {
            "blocks": [block.canonical_value() for block in self.blocks],
            "payload_version": self.payload_version,
            "source_ended_at": self.source_ended_at,
            "source_key": self.source_key,
            "source_sequence": self.source_sequence,
            "source_started_at": self.source_started_at,
            "source_timezone": self.source_timezone,
            "source_type": self.source_type,
        }

    @property
    def content_hash(self) -> str:
        return text_hash(canonical_json(self.canonical_value()))

    @property
    def source_chars(self) -> int:
        return sum(len(block.content) for block in self.blocks)


@dataclass(frozen=True, slots=True)
class MemorySpace:
    memory_space_id: str
    space_key: str
    created_at: int


@dataclass(frozen=True, slots=True)
class EmbeddingRebuildResult:
    memory_space_id: str
    model_signature_id: str
    memories_embedded: int
    subjects_embedded: int


@dataclass(frozen=True, slots=True)
class MaintenanceResult:
    subject_id: str
    operation: Literal[
        "review", "full_split", "partial_split", "defer_split", "summary_refresh"
    ]
    operation_id: str | None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class AddResult:
    episode_id: str
    operation_id: str | None
    replayed: bool
    memories_created: int
    subjects_created: int
    links_created: int
    maintenance: tuple[MaintenanceResult, ...] = ()


@dataclass(frozen=True, slots=True)
class SearchSubject:
    subject_id: str
    name: str
    summary: str | None


@dataclass(frozen=True, slots=True)
class SearchMemory:
    memory_id: str
    content: str
    latest_source_at: int | None


@dataclass(frozen=True, slots=True)
class SearchLink:
    subject_id: str
    memory_id: str
    basis: Literal["direct", "contextual"]


@dataclass(frozen=True, slots=True)
class RankedSubjectRef:
    subject_id: str
    similarity: float
    attached_memory_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RankedMemoryRef:
    memory_id: str
    similarity: float
    attached_subject_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SearchResult:
    query: str
    subjects: tuple[SearchSubject, ...]
    memories: tuple[SearchMemory, ...]
    links: tuple[SearchLink, ...]
    subject_channel: tuple[RankedSubjectRef, ...]
    memory_channel: tuple[RankedMemoryRef, ...]

    def render(self) -> str:
        """Render one deduplicated group per subject in retrieval order."""

        memories = {item.memory_id: item for item in self.memories}
        grouped_memory_ids: dict[str, list[str]] = {
            subject.subject_id: [] for subject in self.subjects
        }
        for hit in self.subject_channel:
            grouped_memory_ids[hit.subject_id].extend(hit.attached_memory_ids)
        for memory_hit in self.memory_channel:
            for subject_id in memory_hit.attached_subject_ids:
                grouped_memory_ids[subject_id].append(memory_hit.memory_id)

        lines = ["Subject groups:"]
        for subject in self.subjects:
            lines.append(f"- Subject: {subject.name}")
            if subject.summary is not None:
                lines.append(f"  Summary: {subject.summary}")
            seen: set[str] = set()
            for memory_id in grouped_memory_ids[subject.subject_id]:
                if memory_id not in seen:
                    seen.add(memory_id)
                    memory = memories[memory_id]
                    dated = _search_memory_date(memory.latest_source_at)
                    prefix = f"  Memory [{dated}]: " if dated else "  Memory: "
                    lines.append(f"{prefix}{memory.content}")
        return "\n".join(lines)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExtractedMemory(StrictModel):
    content: str


class MemoriesExtraction(StrictModel):
    result: Literal["memories"]
    memories: list[ExtractedMemory] = Field(min_length=1)


class NoValuableMemory(StrictModel):
    result: Literal["no_valuable_memory"]
    reason: str


ExtractionOutput = Annotated[
    MemoriesExtraction | NoValuableMemory, Field(discriminator="result")
]


class NewSubjectOutput(StrictModel):
    subject_ref: str
    name: str


class ExistingSubjectTarget(StrictModel):
    kind: Literal["existing"]
    subject_id: str


class NewSubjectTarget(StrictModel):
    kind: Literal["new"]
    subject_ref: str


SubjectTarget = Annotated[
    ExistingSubjectTarget | NewSubjectTarget, Field(discriminator="kind")
]


class LinkingLinkOutput(StrictModel):
    memory_ref: str
    subject: SubjectTarget
    basis: Literal["direct", "contextual"]


class LinkingOutput(StrictModel):
    result: Literal["links"] = "links"
    new_subjects: list[NewSubjectOutput]
    links: list[LinkingLinkOutput]


class AssociationSearchOutput(StrictModel):
    result: Literal["association_search"]
    query: str


LinkingStageOutput = Annotated[
    LinkingOutput | AssociationSearchOutput, Field(discriminator="result")
]


class KeepContent(StrictModel):
    action: Literal["keep"]


class ReplaceContent(StrictModel):
    action: Literal["replace"]
    content: str


ContentChange = Annotated[KeepContent | ReplaceContent, Field(discriminator="action")]


class KeepProvenance(StrictModel):
    action: Literal["keep"]


class ReplaceProvenance(StrictModel):
    action: Literal["replace"]
    episode_ids: list[str]


ProvenanceChange = Annotated[
    KeepProvenance | ReplaceProvenance, Field(discriminator="action")
]


class MemoryUpdateOutput(StrictModel):
    memory_id: str
    content_change: ContentChange
    provenance_change: ProvenanceChange


class ProvenanceRequestOutput(StrictModel):
    result: Literal["provenance_request"]
    memory_ids: list[str]


class ReviewOutput(StrictModel):
    result: Literal["review"]
    updates: list[MemoryUpdateOutput]
    retirements: list[str]


class SummaryRefreshOutput(StrictModel):
    result: Literal["summary_refresh"]
    summary: str


ReviewStageOutput = Annotated[
    ProvenanceRequestOutput | ReviewOutput, Field(discriminator="result")
]


class SplitLinkOutput(StrictModel):
    memory_id: str
    basis: Literal["direct", "contextual"]


class SplitSubjectOutput(StrictModel):
    subject_ref: str
    name: str
    links: list[SplitLinkOutput]


class FullSplitOutput(StrictModel):
    result: Literal["full_split"]
    subjects: list[SplitSubjectOutput]


class PartialSplitOutput(StrictModel):
    result: Literal["partial_split"]
    new_subjects: list[SplitSubjectOutput]


class DeferSplitOutput(StrictModel):
    result: Literal["defer_split"]
    reason: str


SplitStageOutput = Annotated[
    FullSplitOutput | PartialSplitOutput | DeferSplitOutput,
    Field(discriminator="result"),
]


@dataclass(frozen=True, slots=True)
class CandidateMemory:
    memory_id: str
    content: str
    latest_source_at: int | None
    similarity: float
    attached_subject_id: str | None = None
    attached_subject_name: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateSubject:
    subject_id: str
    name: str
    similarity: float
    attached_memory_id: str | None = None
    attached_memory_content: str | None = None


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    memory_id: str
    content: str
    latest_source_at: int | None
    provenance_episode_ids: tuple[str, ...]
    link_basis: Literal["direct", "contextual"] | None = None


@dataclass(frozen=True, slots=True)
class SubjectSnapshot:
    subject_id: str
    name: str
    summary: str | None
    new_memory_count: int
    summary_revision: int
    memories: tuple[MemorySnapshot, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class MemoryBankSubject:
    name: str
    summary: str | None
    memory_contents: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MemoryBankSpace:
    space_key: str
    subjects: tuple[MemoryBankSubject, ...]
