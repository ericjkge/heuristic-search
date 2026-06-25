"""Tests for AIME answer extraction and validation. No LLM calls."""

import pytest
from src.math.common.answer import answers_match, extract_aime_answer, validate_aime_answer


class TestExtractAimeAnswer:
    def test_boxed_simple(self):
        assert extract_aime_answer(r"\boxed{123}") == "123"

    def test_boxed_zero(self):
        assert extract_aime_answer(r"\boxed{0}") == "0"

    def test_boxed_max(self):
        assert extract_aime_answer(r"\boxed{999}") == "999"

    def test_boxed_leading_zero(self):
        assert extract_aime_answer(r"\boxed{007}") == "007"

    def test_boxed_no_slash(self):
        assert extract_aime_answer("boxed{42}") == "42"

    def test_answer_is_lowercase(self):
        assert extract_aime_answer("the answer is 500") == "500"

    def test_answer_colon(self):
        assert extract_aime_answer("Answer: 999") == "999"

    def test_answer_colon_space(self):
        assert extract_aime_answer("Answer:  17") == "17"

    def test_out_of_range_high(self):
        assert extract_aime_answer(r"\boxed{1000}") is None

    def test_out_of_range_negative(self):
        assert extract_aime_answer(r"\boxed{-1}") is None

    def test_multiple_boxed_takes_last(self):
        # Final \boxed{} is the answer by convention; earlier boxes are intermediates.
        assert extract_aime_answer(r"\boxed{5} and \boxed{6}") == "6"

    def test_same_value_twice_ok(self):
        assert extract_aime_answer(r"\boxed{42} so \boxed{42}") == "42"

    def test_intermediate_then_final_boxed(self):
        # Common real case: intermediate result boxed, then the final answer boxed last.
        assert extract_aime_answer(r"so far \boxed{100}, hence \boxed{248}") == "248"

    def test_no_answer(self):
        assert extract_aime_answer("no answer here at all") is None

    def test_boxed_takes_priority_over_nl(self):
        # boxed{7} should win over "the answer is 8"
        result = extract_aime_answer(r"the answer is 8. Therefore \boxed{7}.")
        assert result == "7"

    def test_float_in_boxed_invalid(self):
        assert extract_aime_answer(r"\boxed{3.5}") is None

    def test_empty_string(self):
        assert extract_aime_answer("") is None


class TestValidateAimeAnswer:
    def test_valid_zero(self):
        assert validate_aime_answer("0") is True

    def test_valid_999(self):
        assert validate_aime_answer("999") is True

    def test_valid_leading_zero(self):
        assert validate_aime_answer("007") is True

    def test_none_invalid(self):
        assert validate_aime_answer(None) is False

    def test_1000_invalid(self):
        assert validate_aime_answer("1000") is False

    def test_negative_invalid(self):
        assert validate_aime_answer("-1") is False

    def test_float_invalid(self):
        assert validate_aime_answer("3.5") is False

    def test_empty_invalid(self):
        assert validate_aime_answer("") is False


class TestAnswersMatch:
    def test_exact_match(self):
        assert answers_match("42", "42") is True

    def test_leading_zero_match(self):
        assert answers_match("007", "7") is True

    def test_reverse_leading_zero(self):
        assert answers_match("7", "007") is True

    def test_mismatch(self):
        assert answers_match("5", "6") is False

    def test_none_predicted(self):
        assert answers_match(None, "5") is False

    def test_zero_match(self):
        assert answers_match("0", "0") is True
