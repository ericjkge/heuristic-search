"""Official-style MuSiQue answer metrics: exact match and token F1.

Follows the SQuAD normalization used by MuSiQue's evaluator
(github.com/stonybrooknlp/musique): lowercase, strip punctuation and
articles, collapse whitespace. Scores are the max over the gold answer
plus its answer_aliases.
"""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score(prediction: str, gold: str, aliases: list[str] | None = None) -> dict[str, float]:
    """Max EM/F1 over the gold answer and its aliases."""
    golds = [gold] + list(aliases or [])
    return {
        "em": max(exact_match(prediction, g) for g in golds),
        "f1": max(f1(prediction, g) for g in golds),
    }
