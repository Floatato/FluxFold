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
    similarity: float


@dataclass(frozen=True, slots=True)
class SearchMemory:
    memory_id: str
    content: str
    latest_source_at: int | None
    similarity: float


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
class SearchSubjectGroup:
    subject: SearchSubject
    memories: tuple[SearchMemory, ...]


@dataclass(frozen=True, slots=True)
class SearchResult:
    query: str
    subjects: tuple[SearchSubject, ...]
    memories: tuple[SearchMemory, ...]
    links: tuple[SearchLink, ...]
    subject_channel: tuple[RankedSubjectRef, ...]
    memory_channel: tuple[RankedMemoryRef, ...]

    def displayed_groups(self) -> tuple[SearchSubjectGroup, ...]:
        """Merge both retrieval channels into globally deduplicated groups."""

        subjects = {item.subject_id: item for item in self.subjects}
        memories = {item.memory_id: item for item in self.memories}
        direct_subject_ids = {item.subject_id for item in self.subject_channel}
        candidate_subject_ids: dict[str, set[str]] = {}

        for hit in self.subject_channel:
            for memory_id in hit.attached_memory_ids:
                candidate_subject_ids.setdefault(memory_id, set()).add(hit.subject_id)
        for memory_hit in self.memory_channel:
            for subject_id in memory_hit.attached_subject_ids:
                candidate_subject_ids.setdefault(memory_hit.memory_id, set()).add(
                    subject_id
                )

        grouped_memory_ids: dict[str, list[str]] = {
            subject_id: [] for subject_id in direct_subject_ids
        }
        for memory_id, candidates in candidate_subject_ids.items():
            if memory_id not in memories:
                continue
            eligible = [
                subject_id for subject_id in candidates if subject_id in subjects
            ]
            if not eligible:
                continue
            winner = min(
                eligible,
                key=lambda subject_id: (
                    -subjects[subject_id].similarity,
                    subject_id,
                ),
            )
            grouped_memory_ids.setdefault(winner, []).append(memory_id)

        ordered_subject_ids = sorted(
            grouped_memory_ids,
            key=lambda subject_id: (-subjects[subject_id].similarity, subject_id),
        )
        return tuple(
            SearchSubjectGroup(
                (
                    subjects[subject_id]
                    if subject_id in direct_subject_ids
                    or subjects[subject_id].summary is None
                    else SearchSubject(
                        subject_id,
                        subjects[subject_id].name,
                        None,
                        subjects[subject_id].similarity,
                    )
                ),
                tuple(
                    memories[memory_id]
                    for memory_id in sorted(
                        grouped_memory_ids[subject_id],
                        key=lambda memory_id: (
                            -memories[memory_id].similarity,
                            memory_id,
                        ),
                    )
                ),
            )
            for subject_id in ordered_subject_ids
        )

    def render(self) -> str:
        """Render the globally deduplicated subject groups shown to the LLM."""

        lines = ["Subject groups:"]
        for group in self.displayed_groups():
            subject = group.subject
            lines.append(f"- Subject: {subject.name}")
            if subject.summary is not None:
                lines.append(f"  Summary: {subject.summary}")
            for memory in group.memories:
                lines.append(f"  Memory: {memory.content}")
        return "\n".join(lines)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExtractedMemory(StrictModel):
    content: str
    anchors: list[str] = Field(min_length=1)


class MemoriesExtraction(StrictModel):
    result: Literal["memories"]
    memories: list[ExtractedMemory] = Field(min_length=1)


class NoValuableMemory(StrictModel):
    result: Literal["no_valuable_memory"]
    reason: str


ExtractionOutput = Annotated[
    MemoriesExtraction | NoValuableMemory, Field(discriminator="result")
]


class DirectAssignment(StrictModel):
    anchor: str
    subject: str


class MemoryLinkingOutput(StrictModel):
    memory_id: str
    direct_assignments: list[DirectAssignment]
    contextual_subjects: list[str]


class LinkingOutput(StrictModel):
    result: Literal["links"] = "links"
    memories: list[MemoryLinkingOutput]


class NameResolution(StrictModel):
    proposed_name: str
    canonical_name: str


class NameResolutionOutput(StrictModel):
    resolutions: list[NameResolution]


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
