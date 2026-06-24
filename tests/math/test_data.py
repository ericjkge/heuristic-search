"""Tests for dataset loading. No LLM calls."""

import json
import pytest
from pathlib import Path
from src.math.data import MathProblem, load_problems


class TestMathProblem:
    def test_defaults(self):
        p = MathProblem(id="x", source="aime", problem="Find x.", answer="1")
        assert p.answer_type == "integer_000_999"
        assert p.year is None
        assert p.contest is None
        assert p.reference_solution is None

    def test_fields(self):
        p = MathProblem(
            id="p1", source="AIME 2025", problem="...", answer="42",
            year=2025, contest="AIME I", reference_solution="sol",
        )
        assert p.year == 2025
        assert p.reference_solution == "sol"


class TestLoadProblems:
    def test_basic_load(self, tmp_path: Path):
        f = tmp_path / "test.jsonl"
        f.write_text(
            '{"id":"p1","source":"aime","problem":"Find x.","answer":"42"}\n'
            '{"id":"p2","source":"aime","problem":"Find y.","answer":"7"}\n'
        )
        problems = load_problems(f)
        assert len(problems) == 2
        assert problems[0].id == "p1"
        assert problems[0].answer == "42"
        assert problems[1].id == "p2"

    def test_optional_fields(self, tmp_path: Path):
        f = tmp_path / "test.jsonl"
        f.write_text(
            '{"id":"p1","source":"aime","problem":"...","answer":"1",'
            '"year":2025,"contest":"AIME I","reference_solution":"sol"}\n'
        )
        problems = load_problems(f)
        p = problems[0]
        assert p.year == 2025
        assert p.contest == "AIME I"
        assert p.reference_solution == "sol"

    def test_no_topic_tags_required(self, tmp_path: Path):
        f = tmp_path / "test.jsonl"
        f.write_text('{"id":"p1","source":"aime","problem":"...","answer":"5"}\n')
        problems = load_problems(f)
        assert len(problems) == 1

    def test_answer_stored_as_string(self, tmp_path: Path):
        f = tmp_path / "test.jsonl"
        f.write_text('{"id":"p1","source":"aime","problem":"...","answer":42}\n')
        problems = load_problems(f)
        assert problems[0].answer == "42"
        assert isinstance(problems[0].answer, str)

    def test_skips_blank_lines(self, tmp_path: Path):
        f = tmp_path / "test.jsonl"
        f.write_text(
            '{"id":"p1","source":"aime","problem":"...","answer":"1"}\n'
            '\n'
            '{"id":"p2","source":"aime","problem":"...","answer":"2"}\n'
        )
        problems = load_problems(f)
        assert len(problems) == 2

    def test_sample_file_loads(self):
        sample = Path("data/aime_sample.jsonl")
        if not sample.exists():
            pytest.skip("data/aime_sample.jsonl not present")
        problems = load_problems(sample)
        assert len(problems) >= 1
        for p in problems:
            assert p.id
            assert p.problem
            assert p.answer
