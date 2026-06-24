"""Tests for verifier utilities. No LLM calls — tests pure-logic functions only."""

import pytest
from src.math.common.result import VerifierResult
from src.math.common.verifiers import (
    ALL_VERIFIERS,
    SOFT_VERIFIERS,
    aggregate_score,
    get_failed_verifiers,
    get_verifier_vector,
)


def _vr(name: str, score: float, is_hard: bool = False) -> VerifierResult:
    return VerifierResult(name=name, score=score, passed=None, feedback="", is_hard=is_hard)


class TestAggregateScore:
    def test_mean(self):
        results = {"a": _vr("a", 0.8), "b": _vr("b", 0.4)}
        assert aggregate_score(results) == pytest.approx(0.6)

    def test_empty(self):
        assert aggregate_score({}) == 0.0

    def test_perfect(self):
        results = {n: _vr(n, 1.0) for n in SOFT_VERIFIERS}
        assert aggregate_score(results) == pytest.approx(1.0)

    def test_hard_verifier_included_in_mean(self):
        results = {
            "answer_format_range": _vr("answer_format_range", 1.0, is_hard=True),
            "local_step_validity": _vr("local_step_validity", 0.0),
        }
        assert aggregate_score(results) == pytest.approx(0.5)

    def test_single_verifier(self):
        results = {"x": _vr("x", 0.75)}
        assert aggregate_score(results) == pytest.approx(0.75)


class TestGetVerifierVector:
    def test_order_matches_names(self):
        names = ["b", "a", "c"]
        results = {"a": _vr("a", 0.3), "b": _vr("b", 0.9), "c": _vr("c", 0.5)}
        assert get_verifier_vector(results, names) == pytest.approx([0.9, 0.3, 0.5])

    def test_missing_name_is_zero(self):
        assert get_verifier_vector({}, ["x"]) == pytest.approx([0.0])

    def test_partial_missing(self):
        results = {"a": _vr("a", 0.7)}
        assert get_verifier_vector(results, ["a", "b"]) == pytest.approx([0.7, 0.0])

    def test_all_verifiers_vector_length(self):
        results = {n: _vr(n, 0.5) for n in ALL_VERIFIERS}
        vec = get_verifier_vector(results, ALL_VERIFIERS)
        assert len(vec) == len(ALL_VERIFIERS)


class TestGetFailedVerifiers:
    def test_below_threshold(self):
        results = {"a": _vr("a", 0.3), "b": _vr("b", 0.9)}
        failed = get_failed_verifiers(results, threshold=0.5)
        assert len(failed) == 1
        assert failed[0][0] == "a"

    def test_sorted_weakest_first(self):
        results = {
            "a": _vr("a", 0.4),
            "b": _vr("b", 0.1),
            "c": _vr("c", 0.3),
        }
        failed = get_failed_verifiers(results, threshold=0.5)
        scores = [v.score for _, v in failed]
        assert scores == sorted(scores)
        assert failed[0][0] == "b"

    def test_none_failed(self):
        results = {"a": _vr("a", 0.9), "b": _vr("b", 0.8)}
        assert get_failed_verifiers(results, threshold=0.5) == []

    def test_all_failed(self):
        results = {"a": _vr("a", 0.0), "b": _vr("b", 0.1)}
        failed = get_failed_verifiers(results, threshold=0.5)
        assert len(failed) == 2

    def test_custom_threshold(self):
        results = {"a": _vr("a", 0.6), "b": _vr("b", 0.9)}
        # With threshold=0.7: "a" (0.6) fails, "b" (0.9) passes
        failed = get_failed_verifiers(results, threshold=0.7)
        assert len(failed) == 1
        assert failed[0][0] == "a"
