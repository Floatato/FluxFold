"""Clean-room benchmark scoring compatible with the documented protocols.

LongMemEval's evaluator behavior is based on its MIT-licensed public protocol:
https://github.com/xiaowu0162/LongMemEval

LoCoMo_refined behavior is independently implemented from its public metric
description. This module does not copy its CC BY-NC source or judge prompt.
"""

from __future__ import annotations

import asyncio
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence

from benchmarks.adapters import BenchmarkQuestion
from fluxfold.providers import GenerationProvider, GenerationRequest


async def score_predictions(
    *,
    dataset: str,
    questions: Sequence[BenchmarkQuestion],
    predictions: dict[str, str],
    provider: GenerationProvider,
    concurrency: int,
    timeout_seconds: float,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def score_one(question: BenchmarkQuestion) -> dict[str, object]:
        async with semaphore:
            prediction = _normalize_prediction(
                predictions.get(question.question_id, "")
            )
            candidates = (
                question.answer
                if isinstance(question.answer, tuple)
                else (question.answer,)
            )
            candidate_results: list[dict[str, object]] = []
            for candidate in candidates:
                f1 = (
                    token_f1(prediction, candidate)
                    if dataset == "locomo_refined"
                    else 0.0
                )
                bleu = (
                    bleu1(prediction, candidate) if dataset == "locomo_refined" else 0.0
                )
                if prediction.strip() == candidate.strip():
                    correct = True
                    reason = "Exact match."
                elif not prediction.strip():
                    correct = False
                    reason = "Prediction is empty."
                else:
                    prompt = (
                        _longmemeval_judge_prompt(question, candidate, prediction)
                        if dataset == "longmemeval"
                        else _locomo_judge_prompt(question, candidate, prediction)
                    )
                    response = await provider.generate(
                        GenerationRequest(
                            stage="benchmark_judge",
                            system_prompt="Follow the evaluation rubric exactly and answer YES or NO, followed by one short reason.",
                            user_prompt=prompt,
                            temperature=0.0,
                            timeout_seconds=timeout_seconds,
                            seed=seed,
                        )
                    )
                    normalized = response.text.strip()
                    correct = normalized.upper().startswith("YES")
                    reason = normalized
                candidate_results.append(
                    {
                        "reference": candidate,
                        "llm_score": 1.0 if correct else 0.0,
                        "f1_score": f1,
                        "bleu_score": bleu,
                        "reason": reason,
                    }
                )
            best = max(
                candidate_results,
                key=lambda item: (
                    float(item["llm_score"]),
                    float(item["f1_score"]),
                    float(item["bleu_score"]),
                ),
            )
            return {
                "question_id": question.question_id,
                "category": question.category,
                "question": question.question,
                "prediction": prediction,
                "matched_answer": best["reference"],
                "llm_score": best["llm_score"],
                "f1_score": best["f1_score"],
                "bleu_score": best["bleu_score"],
                "judge_reason": best["reason"],
            }

    scores = list(
        await asyncio.gather(*(score_one(question) for question in questions))
    )
    summary = summarize_scores(dataset, scores)
    return scores, summary


def summarize_scores(
    dataset: str, scores: Sequence[dict[str, object]]
) -> dict[str, object]:
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for score in scores:
        grouped[str(score["category"])].append(score)

    def metrics(items: Sequence[dict[str, object]]) -> dict[str, float | int]:
        count = len(items)
        if count == 0:
            return {"count": 0, "accuracy": 0.0, "f1": 0.0, "bleu1": 0.0}
        return {
            "count": count,
            "accuracy": sum(float(item["llm_score"]) for item in items) / count,
            "f1": sum(float(item["f1_score"]) for item in items) / count,
            "bleu1": sum(float(item["bleu_score"]) for item in items) / count,
        }

    return {
        "dataset": dataset,
        "evaluator": "fluxfold_reimplementation",
        "overall": metrics(scores),
        "by_category": {
            category: metrics(items) for category, items in sorted(grouped.items())
        },
    }


def token_f1(prediction: str, reference: str) -> float:
    predicted = _tokens(prediction)
    expected = _tokens(reference)
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def bleu1(prediction: str, reference: str) -> float:
    predicted = _tokens(prediction)
    expected = _tokens(reference)
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    precision = overlap / len(predicted)
    if precision == 0:
        return 0.0
    penalty = (
        math.exp(1 - len(expected) / len(predicted))
        if len(predicted) < len(expected)
        else 1.0
    )
    return penalty * precision


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"\w+|[^\w\s]", text, re.UNICODE)]


def _normalize_prediction(text: str) -> str:
    cleaned = text.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[1].strip()
    marker = "final answer:"
    marker_index = cleaned.lower().rfind(marker)
    if marker_index >= 0:
        cleaned = cleaned[marker_index + len(marker) :].strip()
    boxed_start = max(cleaned.rfind(r"\boxed{"), cleaned.rfind(r"\box{"))
    if boxed_start >= 0:
        opening = cleaned.find("{", boxed_start)
        depth = 0
        for index in range(opening, len(cleaned)):
            if cleaned[index] == "{":
                depth += 1
            elif cleaned[index] == "}":
                depth -= 1
                if depth == 0:
                    return cleaned[opening + 1 : index].strip()
        return cleaned[opening + 1 :].strip()
    return cleaned


def _longmemeval_judge_prompt(
    question: BenchmarkQuestion, reference: str, prediction: str
) -> str:
    if question.category == "abstention":
        rule = "The question is unanswerable. Accept only if the response identifies the available information as insufficient."
    elif question.category == "temporal-reasoning":
        rule = "Require the complete answer, but accept a one-unit error in a requested number of days, weeks, or months."
    elif question.category == "knowledge-update":
        rule = "Accept when the required updated answer is present, even if an earlier state is also mentioned."
    elif question.category == "single-session-preference":
        rule = "Accept when the response correctly recalls and uses the personal information; it need not reproduce every rubric point."
    else:
        rule = "Require the complete correct answer or equivalent reasoning; reject answers containing only a required subset."
    return f"{rule}\nQuestion: {question.question}\nReference: {reference}\nResponse: {prediction}\nVerdict:"


def _locomo_judge_prompt(
    question: BenchmarkQuestion, reference: str, prediction: str
) -> str:
    return (
        "Judge whether the response is inclusive without contradiction, complete without overreach. "
        "It must cover all required reference information, preserve subject/object relations and strict temporal granularity, "
        "and contain no unsupported additions that change the answer. Do not convert relative time to absolute time or vice versa "
        "unless that exact granularity is present in the reference.\n"
        f"Question: {question.question}\nReference: {reference}\nResponse: {prediction}\nVerdict:"
    )
