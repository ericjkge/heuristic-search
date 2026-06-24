"""Single-attempt baseline: one generate call, extract answer, evaluate."""

from __future__ import annotations

from src.math.common.answer import answers_match
from src.math.common.candidates import generate
from src.math.common.result import MathResult
from src.math.data import MathProblem
from utils.llm import LLM


def single_attempt(llm: LLM, problem: MathProblem) -> MathResult:
    tags = {"problem_id": problem.id, "condition": "single", "phase": "generate"}
    candidate = generate(llm, problem, tags=tags)

    predicted = candidate.extracted_answer if candidate else None
    solved = answers_match(predicted, problem.answer)

    return MathResult(
        problem_id=problem.id,
        source=problem.source,
        mode="single",
        solved=solved,
        ground_truth_answer=problem.answer,
        predicted_answer=predicted,
        selected_candidate=candidate,
        all_candidates=[candidate] if candidate else [],
        model_calls=1 if candidate else 0,
        total_tokens=0,
        best_verifier_score=0.0,
        trajectory=[],
        early_stop=False,
    )
