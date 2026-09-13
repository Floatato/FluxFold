"""Deterministic provider doubles used by integration tests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence

import numpy as np

from fluxfold.providers import (
    EmbeddingModelInfo,
    EmbeddingResponse,
    GenerationRequest,
    GenerationResponse,
)


class FakeGenerationProvider:
    def __init__(
        self,
        *,
        benchmark_answer: str = "Alice likes hiking.",
        split_result: str | None = None,
        split_missing_direct_first: bool = False,
        split_oversized_first: bool = False,
        association_search_once: bool = False,
        review_mode: str = "keep",
        extraction_delay_seconds: float = 0.0,
        answer_delay_seconds: float = 0.0,
        always_new_subject: bool = False,
    ) -> None:
        self.benchmark_answer = benchmark_answer
        self.split_result = split_result
        self.split_missing_direct_first = split_missing_direct_first
        self.split_oversized_first = split_oversized_first
        self.association_search_once = association_search_once
        self.review_mode = review_mode
        self.extraction_delay_seconds = extraction_delay_seconds
        self.answer_delay_seconds = answer_delay_seconds
        self.always_new_subject = always_new_subject
        self.requests: list[GenerationRequest] = []
        self._association_requested = False
        self._illegal_split_emitted = False
        self._oversized_split_emitted = False
        self.active_extractions = 0
        self.max_active_extractions = 0

    @property
    def model_id(self) -> str:
        return "fake-generation"

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        initial = (
            _original_json(request.user_prompt)
            if request.stage not in {"benchmark_answer", "benchmark_judge"}
            else {}
        )
        if "name_resolution" in initial:
            return GenerationResponse(
                text=json.dumps(
                    {
                        "resolutions": [
                            {
                                "proposed_name": item["proposed_name"],
                                "canonical_name": item["proposed_name"],
                            }
                            for item in initial["name_resolution"]["items"]
                        ]
                    }
                ),
                input_tokens=6,
                output_tokens=4,
                total_tokens=10,
            )
        if request.stage == "memory_extraction":
            self.active_extractions += 1
            self.max_active_extractions = max(
                self.max_active_extractions, self.active_extractions
            )
            try:
                await asyncio.sleep(self.extraction_delay_seconds)
                payload = _original_json(request.user_prompt)
                messages = payload["episode"]["messages"]
                content = str(messages[-1]["content"])
                text = (
                    json.dumps(
                        {
                            "result": "no_valuable_memory",
                            "reason": "The episode contains no durable information.",
                        }
                    )
                    if content == "NO_MEMORY"
                    else json.dumps(
                        {
                            "result": "memories",
                            "memories": [
                                {
                                    "content": content,
                                    "anchors": [
                                        content if self.always_new_subject else "Alice"
                                    ],
                                }
                            ],
                        }
                    )
                )
            finally:
                self.active_extractions -= 1
        elif request.stage == "subject_linking":
            payload = _original_json(request.user_prompt)
            if self.association_search_once and not self._association_requested:
                self._association_requested = True
                return GenerationResponse(
                    text=json.dumps(
                        {"result": "association_search", "query": "Alice activities"}
                    ),
                    input_tokens=6,
                    output_tokens=4,
                    total_tokens=10,
                    request_id="fake-request",
                )
            memories = payload["new_memories"]
            candidates = payload["candidates"]
            text = json.dumps(
                {
                    "result": "links",
                    "memories": [
                        {
                            "memory_id": memory["memory_id"],
                            "direct_assignments": [
                                {
                                    "anchor": anchor,
                                    "subject": next(
                                        candidate["name"]
                                        for candidate in candidates
                                        if anchor in candidate["anchors"]
                                        and (
                                            not self.always_new_subject
                                            or candidate["name"] == anchor
                                        )
                                    ),
                                }
                                for anchor in memory["anchors"]
                            ],
                            "contextual_subjects": [],
                        }
                        for memory in memories
                    ],
                }
            )
        elif request.stage == "subject_review":
            payload = _original_json(request.user_prompt)
            memories = payload["subject"]["memories"]
            if (
                self.review_mode == "provenance_request"
                and payload["requested_provenance"] is None
            ):
                text = json.dumps(
                    {
                        "result": "provenance_request",
                        "memory_ids": [memory["memory_id"] for memory in memories],
                    }
                )
            elif self.review_mode == "update_retire":
                text = json.dumps(
                    {
                        "result": "review",
                        "updates": [
                            {
                                "memory_id": memories[0]["memory_id"],
                                "content_change": {
                                    "action": "replace",
                                    "content": "Alice enjoys hiking.",
                                },
                                "provenance_change": {"action": "keep"},
                            }
                        ],
                        "retirements": [memories[1]["memory_id"]],
                    }
                )
            else:
                text = json.dumps(
                    {
                        "result": "review",
                        "updates": [],
                        "retirements": [],
                    }
                )
        elif request.stage == "subject_summary_refresh":
            payload = _original_json(request.user_prompt)
            memories = payload["subject"]["memories"]
            text = json.dumps(
                {
                    "result": "summary_refresh",
                    "summary": " ".join(memory["content"] for memory in memories),
                }
            )
        elif request.stage == "subject_split":
            payload = _original_json(request.user_prompt)
            subject = payload["subject"]
            memories = subject["memories"]
            memory_ids = [memory["memory_id"] for memory in memories]
            if (
                self.split_missing_direct_first
                and not self._illegal_split_emitted
                and len(memory_ids) >= 3
            ):
                self._illegal_split_emitted = True
                links = [
                    {
                        "memory_id": memory_id,
                        "basis": "direct" if index == 0 else "contextual",
                    }
                    for index, memory_id in enumerate(memory_ids)
                ]
                text = json.dumps(
                    {
                        "result": "full_split",
                        "subjects": [
                            {
                                "name": "Alice's hiking trails",
                                "links": links,
                            },
                            {
                                "name": "Alice's hiking equipment",
                                "links": links,
                            },
                        ],
                    }
                )
            elif self.split_oversized_first and not self._oversized_split_emitted:
                self._oversized_split_emitted = True
                links = [
                    {"memory_id": memory_id, "basis": "direct"}
                    for memory_id in memory_ids
                ]
                text = json.dumps(
                    {
                        "result": "full_split",
                        "subjects": [
                            {
                                "name": "Alice's hiking trails",
                                "links": links,
                            },
                            {
                                "name": "Alice's hiking equipment",
                                "links": links,
                            },
                        ],
                    }
                )
            elif self.split_result == "full_split":
                first_ids = memory_ids if len(memory_ids) == 3 else memory_ids[:3]
                second_ids = memory_ids if len(memory_ids) == 3 else memory_ids[-3:]
                text = json.dumps(
                    {
                        "result": "full_split",
                        "subjects": [
                            {
                                "name": "Alice's hiking trails",
                                "links": [
                                    {"memory_id": memory_id, "basis": "direct"}
                                    for memory_id in first_ids
                                ],
                            },
                            {
                                "name": "Alice's hiking equipment",
                                "links": [
                                    {"memory_id": memory_id, "basis": "direct"}
                                    for memory_id in second_ids
                                ],
                            },
                        ],
                    }
                )
            elif self.split_result == "partial_split":
                text = json.dumps(
                    {
                        "result": "partial_split",
                        "new_subjects": [
                            {
                                "name": "Alice's hiking equipment",
                                "links": [
                                    {"memory_id": memory_id, "basis": "direct"}
                                    for memory_id in memory_ids[:3]
                                ],
                            }
                        ],
                    }
                )
            else:
                text = json.dumps(
                    {
                        "result": "defer_split",
                        "reason": "The memories do not form legal semantic groups.",
                    }
                )
        elif request.stage == "benchmark_answer":
            await asyncio.sleep(self.answer_delay_seconds)
            text = self.benchmark_answer
        elif request.stage == "benchmark_judge":
            text = '{"label": "CORRECT"}' if '"label"' in request.user_prompt else "yes"
        else:
            raise AssertionError(f"unexpected generation stage: {request.stage}")
        return GenerationResponse(
            text=text,
            input_tokens=6,
            output_tokens=4,
            total_tokens=10,
            request_id="fake-request",
        )

    async def close(self) -> None:
        return None


class FakeEmbeddingProvider:
    def __init__(self, *, revision: str = "1") -> None:
        self._model_info = EmbeddingModelInfo(
            provider="fake",
            model="hash-vector",
            revision=revision,
            dimension=8,
        )

    @property
    def model_info(self) -> EmbeddingModelInfo:
        return self._model_info

    async def embed(
        self,
        texts: Sequence[str],
        *,
        input_type: str,
        timeout_seconds: float,
    ) -> EmbeddingResponse:
        del input_type, timeout_seconds
        vectors: list[np.ndarray] = []
        for text in texts:
            vector = np.zeros(self._model_info.dimension, dtype="<f4")
            for byte in text.lower().encode("utf-8"):
                vector[byte % self._model_info.dimension] += 1.0
            norm = float(np.linalg.norm(vector))
            if norm == 0:
                vector[0] = 1.0
            else:
                vector /= norm
            vectors.append(vector)
        return EmbeddingResponse(tuple(vectors), "fake-embedding-request")

    async def close(self) -> None:
        return None


def _original_json(prompt: str) -> dict[str, object]:
    original = prompt.split(
        "\n\nYour immediately previous response failed validation.", 1
    )[0]
    value = json.loads(original)
    assert isinstance(value, dict)
    return value
