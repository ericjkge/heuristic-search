"""Verifier rerank baseline: generate N candidates, score all with verifiers, return best."""

from __future__ import annotations

from src.math.common.answer import answers_match
from src.math.common.candidates import generate
from src.math.common.result import MathCandidate, MathResult
from src.math.common.verifiers import aggregate_score, build_verifiers, score_candidate
from src.math.data import MathProblem
from utils.concurrency import run_parallel
from utils.llm import LLM


def verifier_rerank(
    llm: LLM,
    problem: MathProblem,
    n: int = 8,
) -> MathResult:
    tags_base = {"problem_id": problem.id, "condition": "verifier_rerank"}
    verifier_names = build_verifiers(problem, llm)

    gen_thunks = [
        lambda i=i: generate(llm, problem, tags={**tags_base, "phase": "generate", "k": i}, k=i)
        for i in range(n)
    ]
    candidates_raw = run_parallel(gen_thunks)
    candidates: list[MathCandidate] = [c for c in candidates_raw if c is not None]

    if not candidates:
        return MathResult(
            problem_id=problem.id, source=problem.source, mode="verifier_rerank",
            solved=False, ground_truth_answer=problem.answer, predicted_answer=None,
            selected_candidate=None, all_candidates=[],
            model_calls=0, total_tokens=0, best_verifier_score=0.0,
            trajectory=[], early_stop=False,
        )

    score_thunks = [
        lambda c=c, i=i: _score_one(c, problem, llm, verifier_names, {**tags_base, "k": i})
        for i, c in enumerate(candidates)
    ]
    candidates = run_parallel(score_thunks)

    selected = max(candidates, key=lambda c: aggregate_score(c.verifier_scores))
    best_vs = aggregate_score(selected.verifier_scores)
    predicted = selected.extracted_answer
    solved = answers_match(predicted, problem.answer)

    return MathResult(
        problem_id=problem.id,
        source=problem.source,
        mode="verifier_rerank",
        solved=solved,
        ground_truth_answer=problem.answer,
        predicted_answer=predicted,
        selected_candidate=selected,
        all_candidates=candidates,
        model_calls=len(candidates),
        total_tokens=0,
        best_verifier_score=best_vs,
        trajectory=[],
        early_stop=False,
        extra={"n": n},
    )


def _score_one(
    candidate: MathCandidate,
    problem: MathProblem,
    llm: LLM,
    verifier_names: list[str],
    tags: dict,
) -> MathCandidate:
    candidate.verifier_scores = score_candidate(
        candidate.solution_text, candidate.extracted_answer,
        problem, llm, verifier_names,
        tags={**tags, "phase": "score"},
    )
    return candidate
