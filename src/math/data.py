"""MathProblem dataclass and dataset loading for AIME/HMMT problems."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MathProblem:
    id: str
    source: str
    problem: str
    answer: str  # ground truth — NEVER passed to the LLM during solving
    answer_type: str = "integer_000_999"
    year: int | None = None
    contest: str | None = None
    reference_solution: str | None = None


def load_problems(path: str | Path) -> list[MathProblem]:
    """Load problems from a local JSONL file. No topic tags required."""
    path = Path(path)
    problems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            problems.append(MathProblem(
                id=d["id"],
                source=d.get("source", "unknown"),
                problem=d["problem"],
                answer=str(d["answer"]),
                answer_type=d.get("answer_type", "integer_000_999"),
                year=d.get("year"),
                contest=d.get("contest"),
                reference_solution=d.get("reference_solution"),
            ))
    return problems


def load_problems_hf(dataset_name: str = "math-ai/aime25", split: str = "test") -> list[MathProblem]:
    """Load problems from a HuggingFace dataset."""
    from datasets import load_dataset  # type: ignore
    ds = load_dataset(dataset_name, split=split)
    problems = []
    for row in ds:
        problems.append(MathProblem(
            id=str(row["id"]),
            source="AIME 2025",
            problem=row["problem"],
            answer=str(row["answer"]),
            answer_type="integer_000_999",
        ))
    return problems
