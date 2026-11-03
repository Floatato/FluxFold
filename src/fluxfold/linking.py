"""Entity-local Choice routing followed by independent relationship judgments."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from fluxfold.config import FluxFoldConfig
from fluxfold.decisions import DecisionProvider, DecisionQuestion
from fluxfold.errors import ErrorClass, ProviderError, StageFailure, ValidationError
from fluxfold.models import DirectAssignment, LinkingOutput, MemoryLinkingOutput
from fluxfold.names import normalized_name

DIRECT_INSTRUCTIONS = """Select the domain that best organizes this fact for the specified entity.
Choose the most specific suitable scope, using the supplied ancestor relationships.
A narrower domain is appropriate only when the fact actually belongs to its scope.
Incidental mentions do not establish organizational ownership."""
RELATED_INSTRUCTIONS = """Is the fact concretely related to the domain's scope such that discovering
information in that scope should also surface this fact? Relationships include constraints,
influence, causation, updates, and explanation in either direction. Shared identity or a
vague topical association alone is insufficient. Use supported facts and ordinary common
sense, without inventing circumstances. For example, a suspended driving license affects a
planned road trip; a food allergy constrains food preferences; liking photography alone does
not establish relevance to a particular trip."""


class Linker:
    def __init__(
        self,
        provider: DecisionProvider | None,
        config: FluxFoldConfig,
        emit: Callable[..., None],
        context: dict[str, str],
    ) -> None:
        self.provider = provider
        self.config = config
        self.emit = emit
        self.context = context

    async def ask(
        self, stage: str, questions: dict[str, DecisionQuestion]
    ) -> dict[str, dict[str, Any]]:
        if not questions:
            return {}
        if self.provider is None:
            raise StageFailure(
                stage,
                ErrorClass.AUTHENTICATION_OR_CONFIGURATION,
                "A decision_provider is required for Jev linking",
            )
        for attempt in range(self.config.provider_max_retries + 1):
            call_id = str(uuid4())
            started = time.monotonic()
            self.emit(
                "link_request",
                **self.context,
                call_id=call_id,
                stage=stage,
                attempt=attempt + 1,
                model=self.provider.model_id,
                state={"items": {key: q.state for key, q in questions.items()}},
                questions={key: q.wire(key) for key, q in questions.items()},
            )
            response = None
            try:
                response = await self.provider.decide(questions)
                self.emit(
                    "link_response",
                    **self.context,
                    call_id=call_id,
                    stage=stage,
                    model=response.model,
                    request_id=response.request_id,
                    answers=response.answers,
                )
                validate_answers(questions, response.answers)
            except (ProviderError, ValidationError) as error:
                kind = (
                    error.error_class
                    if isinstance(error, ProviderError)
                    else ErrorClass.INVALID_STRUCTURED_OUTPUT
                )
                self.emit(
                    "llm_call",
                    **self.context,
                    call_id=call_id,
                    stage=stage,
                    attempt=attempt + 1,
                    result="failed",
                    question_count=len(questions),
                    elapsed_seconds=time.monotonic() - started,
                    input_tokens=response.input_tokens if response else 0,
                    output_tokens=response.output_tokens if response else 0,
                    total_tokens=response.input_tokens + response.output_tokens
                    if response
                    else 0,
                    error_class=kind.value,
                    reason=str(error),
                )
                self.emit(
                    "link_request_failed",
                    **self.context,
                    call_id=call_id,
                    stage=stage,
                    error_class=kind.value,
                    reason=str(error),
                )
                if kind == ErrorClass.CONTEXT_OVERFLOW and len(questions) > 1:
                    items = list(questions.items())
                    middle = len(items) // 2
                    left = await self.ask(stage, dict(items[:middle]))
                    right = await self.ask(stage, dict(items[middle:]))
                    return left | right
                if (
                    kind
                    not in {
                        ErrorClass.TRANSIENT_TRANSPORT,
                        ErrorClass.RATE_LIMITED,
                        ErrorClass.SERVICE_UNAVAILABLE,
                        ErrorClass.INVALID_STRUCTURED_OUTPUT,
                        ErrorClass.INCOMPLETE_OUTPUT,
                    }
                    or attempt == self.config.provider_max_retries
                ):
                    raise StageFailure(stage, kind, str(error)) from error
                delay = (
                    error.retry_after_seconds
                    if isinstance(error, ProviderError)
                    else None
                )
                await asyncio.sleep(
                    delay if delay is not None else min(0.5 * 2**attempt, 5.0)
                )
                continue
            self.emit(
                "llm_call",
                **self.context,
                call_id=call_id,
                stage=stage,
                attempt=attempt + 1,
                result="success",
                model=response.model,
                request_id=response.request_id,
                question_count=len(questions),
                elapsed_seconds=time.monotonic() - started,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                total_tokens=response.input_tokens + response.output_tokens,
            )
            return response.answers
        raise AssertionError("unreachable decision retry state")

    async def link(
        self, memories: list[dict[str, Any]], ancestors: dict[str, set[str]]
    ) -> LinkingOutput:
        self.emit(
            "link_candidates",
            **self.context,
            memories=memories,
            ancestors={name: sorted(parents) for name, parents in ancestors.items()},
            contextual_probability_threshold=self.config.contextual_probability_threshold,
            contextual_domain_max_per_entity=self.config.contextual_domain_max_per_entity,
        )
        questions: dict[str, DecisionQuestion] = {}
        groups: list[dict[str, Any]] = []
        for memory in memories:
            for entity, candidates in memory["candidates"].items():
                names = tuple(item["name"] for item in candidates)
                key = f"q{len(groups)}"
                group: dict[str, Any] = {
                    "memory_id": memory["memory_id"],
                    "entity": entity,
                    "candidates": candidates,
                    "question_id": key,
                }
                groups.append(group)
                if len(names) == 1:
                    group.update(
                        direct=names[0],
                        selection="single_candidate",
                        probabilities=None,
                    )
                else:
                    questions[key] = DecisionQuestion(
                        "choice",
                        {
                            "fact": memory["content"],
                            "entity": entity,
                            "domains": list(names),
                            "ancestors": {
                                name: sorted(ancestors.get(name, set()) & set(names))
                                for name in names
                            },
                        },
                        DIRECT_INSTRUCTIONS,
                        names,
                    )
        answers = await self.ask("domain_linking_direct", questions)
        for group in groups:
            if group["question_id"] in answers:
                answer = answers[group["question_id"]]
                group.update(
                    direct=answer["choice"],
                    selection="choice",
                    probabilities=answer["probabilities"],
                    confidence=answer.get("confidence"),
                )
        self.emit(
            "link_direct_decisions",
            **self.context,
            groups=[dict(group) for group in groups],
        )
        direct: dict[str, set[str]] = {}
        for group in groups:
            direct.setdefault(group["memory_id"], set()).add(group["direct"])
        content = {item["memory_id"]: item["content"] for item in memories}
        related: dict[str, DecisionQuestion] = {}
        pair_keys: dict[tuple[str, str], str] = {}
        for group in groups:
            excluded_ancestors = ancestors.get(group["direct"], set())
            group["contextual"] = []
            for candidate in group["candidates"]:
                name = candidate["name"]
                item: dict[str, Any] = {"domain": name}
                group["contextual"].append(item)
                if name in direct[group["memory_id"]]:
                    item["reason"] = "direct_target"
                elif name in excluded_ancestors:
                    item["reason"] = "ancestor_of_direct"
                else:
                    pair = (group["memory_id"], name)
                    if pair not in pair_keys:
                        key = f"q{len(related)}"
                        pair_keys[pair] = key
                        related[key] = DecisionQuestion(
                            "noul",
                            {"fact": content[pair[0]], "domain": name},
                            RELATED_INSTRUCTIONS,
                        )
                    item["question_id"] = pair_keys[pair]
        relevance = await self.ask("domain_linking_contextual", related)
        for group in groups:
            eligible = []
            for item in group["contextual"]:
                if "question_id" in item:
                    item["probability"] = relevance[item["question_id"]]["noul"]
                    item["reason"] = "below_threshold"
                    if (
                        item["probability"]
                        > self.config.contextual_probability_threshold
                    ):
                        eligible.append(item)
            eligible.sort(
                key=lambda item: (-item["probability"], normalized_name(item["domain"]))
            )
            for index, item in enumerate(eligible):
                item["reason"] = (
                    "selected"
                    if index < self.config.contextual_domain_max_per_entity
                    else "entity_limit"
                )
        results = []
        for memory in memories:
            own = [
                group for group in groups if group["memory_id"] == memory["memory_id"]
            ]
            result = MemoryLinkingOutput(
                memory_id=memory["memory_id"],
                direct_assignments=[
                    DirectAssignment(entity=g["entity"], domain=g["direct"])
                    for g in own
                ],
                contextual_domains=sorted(
                    {
                        item["domain"]
                        for g in own
                        for item in g["contextual"]
                        if item["reason"] == "selected"
                    },
                    key=normalized_name,
                ),
            )
            results.append(result)
            self.emit(
                "link_decision",
                **self.context,
                memory_id=memory["memory_id"],
                content=memory["content"],
                groups=own,
                final=result.model_dump(),
                status="prepared",
            )
        self.emit(
            "link_batch_summary",
            **self.context,
            direct_question_count=len(questions),
            contextual_question_count=len(related),
            single_candidate_count=len(groups) - len(questions),
            direct_link_count=sum(
                len({a.domain for a in r.direct_assignments}) for r in results
            ),
            contextual_link_count=sum(len(r.contextual_domains) for r in results),
            noul_probabilities=[answer["noul"] for answer in relevance.values()],
            choice_margins=[
                sorted(answer["probabilities"].values(), reverse=True)[0]
                - sorted(answer["probabilities"].values(), reverse=True)[1]
                for answer in answers.values()
            ],
            entity_limit_count=sum(
                item["reason"] == "entity_limit"
                for g in groups
                for item in g["contextual"]
            ),
        )
        return LinkingOutput(memories=results)


def validate_answers(
    questions: dict[str, DecisionQuestion], answers: dict[str, dict[str, Any]]
) -> None:
    if answers.keys() != questions.keys():
        raise ValidationError("return exactly one answer for every supplied question")
    for key, question in questions.items():
        answer = answers[key]
        if answer.get("type") != question.kind:
            raise ValidationError("decision answer has the wrong type")
        if question.kind == "noul":
            values: list[Any] = [answer.get("noul")]
        else:
            probabilities = answer.get("probabilities")
            if not isinstance(probabilities, dict) or set(probabilities) != set(
                question.options
            ):
                raise ValidationError(
                    "choice probabilities must cover exactly the supplied options"
                )
            if answer.get("choice") not in question.options:
                raise ValidationError("choice target was not supplied")
            values = list(probabilities.values())
        if any(
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in values
        ):
            raise ValidationError(
                "decision probabilities must be finite numbers in [0, 1]"
            )
        if question.kind == "choice":
            if not math.isclose(sum(values), 1, abs_tol=1e-4):
                raise ValidationError("choice probabilities must sum to one")
            if answer["probabilities"][answer["choice"]] != max(values):
                raise ValidationError("choice must select a highest probability option")
