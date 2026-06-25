"""Self-Refine baseline for AIME/HMMT math problems.

Generate an initial solution, then iterate: (1) the model critiques its own
solution with GENERIC feedback ("check your work"), and (2) refines using that
self-critique. Unlike the verifier-guided method, the feedback is the model's own
generic self-assessment, not verifier-derived failed dimensions — isolating
whether TARGETED (verifier) feedback beats generic self-feedback. No verifiers and
no gold are used during the loop; solved = the final answer matches gold.

Runs a fixed number of refine steps (no early stop), so compute is comparable;
each step is two calls (critique + refine).
"""

from __future__ import annotations

from src.math.common.answer import answers_match, extract_aime_answer
from src.math.common.candidates import generate
from src.math.common.result import MathCandidate, MathResult
from src.math.data import MathProblem
from utils.llm import LLM

_GEN_SYS = (
    "You are an expert at AMC/AIME mathematics competitions. "
    "Show your full solution step by step. "
    "End with your final answer inside \\boxed{}."
)

_CRITIQUE_SYS = (
    "You carefully review proposed solutions to competition math problems and point "
    "out mistakes. Respond with concise prose feedback only."
)

_CRITIQUE_PROMPT = """\
Here is a proposed solution to the problem.

Problem:
{problem}

Proposed solution:
{solution}

Check the work carefully: go through the reasoning and arithmetic step by step, and
identify any mistakes or inconsistencies. If something is wrong, say what is wrong.
If it looks fully correct, say so. Keep it brief."""

_REFINE_PROMPT = """\
You previously proposed this solution:
{solution}

Your review of it:
{feedback}

Using your review, write a corrected full solution to the problem. Show all steps
clearly. At the end, state your final answer as an integer between 0 and 999 in
\\boxed{{N}}."""


def _critique(llm: LLM, problem: MathProblem, solution: str, tags: dict) -> str:
    prompt = _CRITIQUE_PROMPT.format(problem=problem.problem, solution=solution)
    return llm.call(
        [{"role": "system", "content": _CRITIQUE_SYS},
         {"role": "user", "content": prompt}],
        tags=tags,
    )


def _refine(
    llm: LLM, problem: MathProblem, solution: str, feedback: str, tags: dict, k: int,
) -> MathCandidate | None:
    prompt = _REFINE_PROMPT.format(solution=solution, feedback=feedback)
    msgs = [{"role": "system", "content": _GEN_SYS},
            {"role": "user", "content": prompt}]
    try:
        text = llm.call(msgs, tags=tags)
    except Exception:
        return None
    return MathCandidate(
        candidate_id=f"{problem.id}-refine-{k}",
        problem_id=problem.id,
        solution_text=text,
        extracted_answer=extract_aime_answer(text),
        verifier_scores={},
        parent_ids=[],
        search_depth=k,
        operation="refine",
        generation_metadata={},
    )


def self_refine(llm: LLM, problem: MathProblem, num_steps: int = 3) -> MathResult:
    """Initial solution + `num_steps` rounds of generic self-critique then refine."""
    base = {"problem_id": problem.id, "condition": "self_refine"}
    cand = generate(llm, problem, tags={**base, "phase": "init", "k": 0})

    all_candidates: list[MathCandidate] = [cand] if cand else []
    calls = 1 if cand else 0
    # Correctness per step (analysis only — never fed to the model).
    trajectory = [1.0 if (cand and answers_match(cand.extracted_answer, problem.answer)) else 0.0]

    for it in range(1, num_steps + 1):
        if cand is None:
            break
        feedback = _critique(llm, problem, cand.solution_text,
                             tags={**base, "phase": "critique", "iter": it})
        calls += 1
        refined = _refine(llm, problem, cand.solution_text, feedback,
                         tags={**base, "phase": "refine", "iter": it}, k=it)
        calls += 1
        if refined is not None:
            cand = refined
            all_candidates.append(refined)
        trajectory.append(1.0 if answers_match(cand.extracted_answer, problem.answer) else 0.0)

    predicted = cand.extracted_answer if cand else None
    solved = answers_match(predicted, problem.answer)

    return MathResult(
        problem_id=problem.id,
        source=problem.source,
        mode="self_refine",
        solved=solved,
        ground_truth_answer=problem.answer,
        predicted_answer=predicted,
        selected_candidate=cand,
        all_candidates=all_candidates,
        model_calls=calls,
        total_tokens=0,
        best_verifier_score=0.0,
        trajectory=trajectory,
        early_stop=False,
        extra={"num_steps": num_steps},
    )
