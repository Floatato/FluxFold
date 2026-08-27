"""Official-protocol scoring for LongMemEval and LoCoMo_refined.

LongMemEval judge templates are the public MIT-licensed prompts from
https://github.com/xiaowu0162/LongMemEval (`src/evaluation/evaluate_qa.py`).

LoCoMo_refined uses the official `refined` judge prompt from
https://github.com/mem-eval-suite/LoCoMo_refined (`src/llm_judge.py`), plus
token F1 and BLEU-1. That judge prompt is CC BY-NC 4.0 and is used only for
benchmark scoring.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence

from benchmarks.adapters import BenchmarkQuestion
from fluxfold.providers import GenerationProvider, GenerationRequest

_LOCOMO_REFINED_JUDGE_PROMPT = """Your task is to label an answer as ’CORRECT’ or ’WRONG’ given:
(1) a question,
(2) a gold (ground truth) answer,
(3) a generated answer.

Core principle — Inclusion + Non-contradiction
- Be GENEROUS: if the generated answer clearly includes the gold’s key content (or a clear paraphrase of the same content) and does not contradict it, mark CORRECT — even if extra details are added.
- Mark WRONG only when the generated answer does not include the gold’s content, changes it, or contradicts it.

TIME (strict granularity; relative form equivalence; no calendar math)
- Granularity must match exactly: HOUR↔HOUR, DAY↔DAY, MONTH↔MONTH, YEAR↔YEAR.
  Do not answer a gold at a different time unit — even if the numeric value overlaps. Do not answer a month-level gold with a specific day, nor a year with a specific month/day/hour, etc.
  (e.g., gold = "July 26, 2019" [DAY]; generated = "2019-07-26 08:09:17" [includes Second] → WRONG)
- Do NOT convert relative ↔ absolute. If the gold uses a relative time expression, the generated answer must also use a relative form (or a clear paraphrase of that same form), not a computed date/range.
- Treat harmless modifiers in relative forms (e.g., “the/last/previous/just prior”) as equivalent when both the anchor date and the time unit are the same.

- Lists of DISTINCT facts:
- If the gold answer lists multiple distinct facts (joined by "and", commas, or slashes), the generated answer must cover **all** of them.
- Extra non-contradictory items **generally count as WRONG**.
    - Example: gold = A, B, C ; gen = A, B, C → CORRECT
    - Example: gold = A, B, C ; gen = A, B, C, D → WRONG
- Exception: If a gold element is elaborated or split into finer details in the generated answer (e.g., C → C, C′), it is still considered CORRECT.

Preference/Benefit Questions (e.g., "what X likes/values most")
- If gold lists multiple reasons/aspects, the generated answer only needs to include **any one** of them without contradiction to be CORRECT.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label":

```json
{{
    "label": "CORRECT" or "WRONG"
}}
```
"""


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
                        longmemeval_judge_prompt(question, candidate, prediction)
                        if dataset == "longmemeval"
                        else locomo_judge_prompt(
                            question.question, candidate, prediction
                        )
                    )
                    response = await provider.generate(
                        GenerationRequest(
                            stage="benchmark_judge",
                            user_prompt=prompt,
                            temperature=0.0,
                            timeout_seconds=timeout_seconds,
                            seed=seed,
                        )
                    )
                    correct, reason = parse_judge_response(dataset, response.text)
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
        "evaluator": (
            "longmemeval_official"
            if dataset == "longmemeval"
            else "locomo_refined_official"
        ),
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


def longmemeval_judge_prompt(
    question: BenchmarkQuestion, reference: str, prediction: str
) -> str:
    return _longmemeval_anscheck_prompt(
        question.category,
        question.question,
        reference,
        prediction,
        abstention=question.category == "abstention",
    )


def locomo_judge_prompt(question: str, reference: str, prediction: str) -> str:
    return _LOCOMO_REFINED_JUDGE_PROMPT.format(
        question=question,
        gold_answer=reference,
        generated_answer=prediction,
    )


def parse_judge_response(dataset: str, text: str) -> tuple[bool, str]:
    raw = text.strip()
    if dataset == "longmemeval":
        return "yes" in raw.lower(), raw
    try:
        payload = json.loads(_extract_json_object(raw))
    except (json.JSONDecodeError, ValueError):
        return False, raw
    return str(payload.get("label", "")).strip().upper() == "CORRECT", raw


def _longmemeval_anscheck_prompt(
    task: str,
    question: str,
    answer: str,
    response: str,
    *,
    abstention: bool,
) -> str:
    # Verbatim templates from LongMemEval src/evaluation/evaluate_qa.py.
    if not abstention:
        if task in ["single-session-user", "single-session-assistant", "multi-session"]:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        if task == "temporal-reasoning":
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        if task == "knowledge-update":
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        if task == "single-session-preference":
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        raise NotImplementedError(task)
    template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
    return template.format(question, answer, response)


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


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
    in_string = False
    escape = False
    depth = 0
    start: int | None = None
    for index, ch in enumerate(stripped):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = index
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                return stripped[start : index + 1]
    raise ValueError("failed to extract JSON object from judge response")
