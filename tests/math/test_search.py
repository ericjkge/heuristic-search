"""Tests for selection primitives. No LLM calls — uses mock MathCandidates."""

import pytest
from src.math.common.result import MathCandidate, VerifierResult
from src.math.method.selection import select_combination, select_expansion, truncate_population

NAMES = ["v1", "v2", "v3"]


def _cand(cid: str, scores: list[float]) -> MathCandidate:
    vr = {
        n: VerifierResult(name=n, score=s, passed=None, feedback="", is_hard=False)
        for n, s in zip(NAMES, scores)
    }
    return MathCandidate(
        candidate_id=cid,
        problem_id="p1",
        solution_text="",
        extracted_answer=None,
        verifier_scores=vr,
        parent_ids=[],
        search_depth=0,
        operation="seed",
        generation_metadata={},
    )


class TestSelectExpansion:
    def test_returns_requested_count(self):
        cands = [_cand(f"c{i}", [0.5, 0.5, 0.5]) for i in range(5)]
        result = select_expansion(cands, 3, NAMES)
        assert len(result) == 3

    def test_no_duplicates(self):
        cands = [_cand(f"c{i}", [0.5, 0.5, 0.5]) for i in range(5)]
        result = select_expansion(cands, 4, NAMES)
        assert len(result) == len(set(result))

    def test_prefers_high_quality_when_no_diversity(self):
        cands = [
            _cand("lo", [0.1, 0.1, 0.1]),
            _cand("hi", [0.9, 0.9, 0.9]),
        ]
        result = select_expansion(cands, 1, NAMES, quality_weight=1.0, diversity_weight=0.0)
        assert result == [1]

    def test_empty_candidates(self):
        assert select_expansion([], 3, NAMES) == []

    def test_num_expansions_larger_than_pool(self):
        cands = [_cand("c0", [0.5, 0.5, 0.5])]
        result = select_expansion(cands, 10, NAMES)
        assert len(result) == 1

    def test_diversity_selects_complementary(self):
        # c0 covers v1, c1 covers v2 — diversity should prefer picking both
        cands = [
            _cand("c0", [0.9, 0.0, 0.0]),
            _cand("c1", [0.0, 0.9, 0.0]),
        ]
        result = select_expansion(cands, 2, NAMES, quality_weight=1.0, diversity_weight=1.0)
        assert set(result) == {0, 1}


class TestSelectCombination:
    def test_picks_complementary_pairs(self):
        # a covers v1, b covers v2 -> complementarity = 1
        # a and c both cover v1 -> complementarity = 0
        cands = [
            _cand("a", [0.9, 0.1, 0.1]),
            _cand("b", [0.1, 0.9, 0.1]),
            _cand("c", [0.9, 0.1, 0.1]),
        ]
        pairs = select_combination(cands, 1, NAMES)
        assert pairs == [(0, 1)]

    def test_returns_requested_count(self):
        cands = [_cand(f"c{i}", [float(i % 2), float((i + 1) % 2), 0.5]) for i in range(5)]
        pairs = select_combination(cands, 2, NAMES)
        assert len(pairs) <= 2

    def test_empty(self):
        assert select_combination([], 1, NAMES) == []

    def test_single_candidate(self):
        cands = [_cand("c0", [0.9, 0.9, 0.9])]
        assert select_combination(cands, 1, NAMES) == []

    def test_indices_ordered(self):
        cands = [_cand(f"c{i}", [0.5] * 3) for i in range(4)]
        pairs = select_combination(cands, 3, NAMES)
        for i, j in pairs:
            assert i < j


class TestTruncatePopulation:
    def test_truncates_to_size(self):
        cands = [_cand(f"c{i}", [i / 9] * 3) for i in range(10)]
        result = truncate_population(cands, 4, NAMES)
        assert len(result) == 4

    def test_keeps_best_first(self):
        cands = [
            _cand("lo", [0.1, 0.1, 0.1]),
            _cand("hi", [0.9, 0.9, 0.9]),
        ]
        result = truncate_population(cands, 1, NAMES)
        assert len(result) == 1
        assert result[0].candidate_id == "hi"

    def test_no_truncation_needed(self):
        cands = [_cand(f"c{i}", [0.5] * 3) for i in range(3)]
        result = truncate_population(cands, 10, NAMES)
        assert len(result) == 3

    def test_empty_input(self):
        assert truncate_population([], 5, NAMES) == []

    def test_zero_size(self):
        cands = [_cand("c0", [0.5] * 3)]
        assert truncate_population(cands, 0, NAMES) == []

    def test_diversity_brings_in_diverse_candidate(self):
        # hi has best score, lo_diverse covers different verifiers
        # With diversity_weight > 0, lo_diverse should be selected over lo_similar
        cands = [
            _cand("hi",         [0.9, 0.9, 0.9]),
            _cand("lo_similar", [0.8, 0.8, 0.8]),  # close to hi in verifier space
            _cand("lo_diverse", [0.1, 0.1, 0.9]),  # far from hi in verifier space
        ]
        result = truncate_population(cands, 2, NAMES, diversity_weight=1.0)
        ids = {c.candidate_id for c in result}
        assert "hi" in ids
        assert "lo_diverse" in ids
