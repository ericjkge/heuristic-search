"""
Selection primitives for math QD search.
Ported and adapted from src/method/qd_search.py (ZebraLogic) with:
- Float-score satisfaction threshold (0.8) instead of binary >= 1.0
- truncate_population for bounded population size
"""

from __future__ import annotations

import math

from src.math.common.result import MathCandidate
from src.math.common.verifiers import aggregate_score, get_verifier_vector

_SATISFY_THRESHOLD = 0.8  # score >= this → "satisfied" for diversity/complementarity math


def _quality(scores: list[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0


def _satisfied_set(scores: list[float]) -> set[int]:
    """Indices where score >= threshold. Uses 0.8 not 1.0 — LLM judges rarely give perfect 1.0."""
    return {i for i, s in enumerate(scores) if s >= _SATISFY_THRESHOLD}


def _l2_dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def select_expansion(
    candidates: list[MathCandidate],
    num_expansions: int,
    verifier_names: list[str],
    quality_weight: float = 1.0,
    diversity_weight: float = 0.3,
) -> list[int]:
    """
    Greedy selection of num_expansions candidate indices for expansion.
    priority = quality_weight * sum(scores) + diversity_weight * new_coverage
    new_coverage = # verifiers satisfied by this candidate not yet covered by any selected candidate.
    """
    if not candidates:
        return []
    num_expansions = min(num_expansions, len(candidates))

    score_vecs = [get_verifier_vector(c.verifier_scores, verifier_names) for c in candidates]
    covered: set[int] = set()
    selected: list[int] = []

    remaining = list(range(len(candidates)))
    for _ in range(num_expansions):
        best_idx, best_priority = -1, -1.0
        for i in remaining:
            sat = _satisfied_set(score_vecs[i])
            new_coverage = len(sat - covered)
            total_score = sum(score_vecs[i])
            priority = quality_weight * total_score + diversity_weight * new_coverage
            if priority > best_priority:
                best_priority = priority
                best_idx = i
        if best_idx == -1:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        covered |= _satisfied_set(score_vecs[best_idx])

    return selected


def select_combination(
    candidates: list[MathCandidate],
    num_combinations: int,
    verifier_names: list[str],
) -> list[tuple[int, int]]:
    """
    Select num_combinations most complementary index pairs.
    complementarity(a, b) = |sat(a) ∪ sat(b)| - max(|sat(a)|, |sat(b)|)
    """
    if len(candidates) < 2:
        return []

    score_vecs = [get_verifier_vector(c.verifier_scores, verifier_names) for c in candidates]
    pairs: list[tuple[float, int, int]] = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            sat_i = _satisfied_set(score_vecs[i])
            sat_j = _satisfied_set(score_vecs[j])
            gain = len(sat_i | sat_j) - max(len(sat_i), len(sat_j))
            pairs.append((gain, i, j))

    pairs.sort(key=lambda g: (-g[0], g[1], g[2]))
    return [(i, j) for _, i, j in pairs[:num_combinations]]


def truncate_population(
    candidates: list[MathCandidate],
    population_size: int,
    verifier_names: list[str],
    diversity_weight: float = 0.3,
) -> list[MathCandidate]:
    """
    Greedy frontier: keep at most population_size candidates.
    Add highest-scoring first, then greedily add by:
      score(cand) + diversity_weight * min_L2_distance_to_already_selected
    """
    if not candidates or population_size <= 0:
        return []
    if len(candidates) <= population_size:
        return list(candidates)

    score_vecs = [get_verifier_vector(c.verifier_scores, verifier_names) for c in candidates]
    scores = [_quality(v) for v in score_vecs]

    # Seed with the highest-scoring candidate
    best_start = max(range(len(candidates)), key=lambda i: scores[i])
    selected_idx = [best_start]
    remaining = [i for i in range(len(candidates)) if i != best_start]

    while len(selected_idx) < population_size and remaining:
        best_i, best_val = -1, -1.0
        for i in remaining:
            min_dist = min(_l2_dist(score_vecs[i], score_vecs[s]) for s in selected_idx)
            val = scores[i] + diversity_weight * min_dist
            if val > best_val:
                best_val = val
                best_i = i
        if best_i == -1:
            break
        selected_idx.append(best_i)
        remaining.remove(best_i)

    return [candidates[i] for i in selected_idx]
