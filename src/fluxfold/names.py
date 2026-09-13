"""Name normalization and write-side duplicate candidate scoring."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np


def normalized_name(name: str) -> str:
    """Normalize case and whitespace without removing identity-bearing punctuation."""
    return " ".join(name.split()).casefold()


@dataclass(frozen=True, slots=True)
class NameEntry:
    object_id: str
    name: str
    vector: np.ndarray


def bm25_scores(query: str, names: list[str]) -> list[float]:
    """Score names using BM25 (k1=1.2, b=0.75) and Unicode word tokens."""
    tokens = [re.findall(r"\w+", normalized_name(name)) for name in names]
    if not tokens:
        return []
    mean_length = sum(map(len, tokens)) / len(tokens)
    if mean_length == 0:
        return [0.0] * len(tokens)
    frequencies = Counter(word for document in tokens for word in set(document))
    query_words = set(re.findall(r"\w+", normalized_name(query)))
    scores = []
    for document in tokens:
        counts = Counter(document)
        score = 0.0
        for word in query_words:
            frequency = counts[word]
            if frequency:
                inverse_frequency = math.log(
                    1
                    + (len(tokens) - frequencies[word] + 0.5)
                    / (frequencies[word] + 0.5)
                )
                score += (
                    inverse_frequency
                    * frequency
                    * 2.2
                    / (frequency + 1.2 * (0.25 + 0.75 * len(document) / mean_length))
                )
        scores.append(score)
    return scores
