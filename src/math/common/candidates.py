"""
Candidate generation, revision, and combination for AIME/HMMT math problems.
All prompts instruct the model to end with \\boxed{N} for clean answer extraction.
"""

from __future__ import annotations

from src.math.common.answer import extract_aime_answer
from src.math.common.result import MathCandidate, VerifierResult
from src.math.data import MathProblem
from utils.llm import LLM

_GEN_SYS = (
    "You are an expert at AMC/AIME mathematics competitions. "
    "Show your full solution step by step. "
    "End with your final answer inside \\boxed{}."
)

_GEN_PROMPT = """\
Solve the following competition math problem completely.

Problem:
{problem}

Show all your work clearly. At the end, state your final answer as an integer \
between 0 and 999 in \\boxed{{N}}.\
"""

_REVISE_PROMPTS: dict[str, str] = {
    "answer_format_range": """\
Your previous solution did not produce a valid integer answer in [0, 999].

Problem:
{problem}

Previous solution:
{solution}

Issue: {feedback}

Re-solve the problem completely. Ensure your final answer is an integer between \
0 and 999 stated as \\boxed{{N}}.\
""",
    "problem_condition_coverage": """\
Your previous solution missed or ignored some conditions stated in the problem.

Problem:
{problem}

Previous solution:
{solution}

Issue: {feedback}

Re-read the problem carefully. Identify ALL conditions and constraints. Provide a \
new complete solution that explicitly uses every condition. End with \\boxed{{N}}.\
""",
    "local_step_validity": """\
Your previous solution contained unjustified logical leaps or invalid steps.

Problem:
{problem}

Previous solution:
{solution}

Issue: {feedback}

Rewrite the solution making every step explicit and fully justified. Each step must \
follow rigorously from the previous. End with \\boxed{{N}}.\
""",
    "algebra_arithmetic_consistency": """\
Your previous solution contained algebra or arithmetic errors.

Problem:
{problem}

Previous solution:
{solution}

Issue: {feedback}

Redo the computation carefully. Check every algebraic manipulation and numerical \
calculation. End with \\boxed{{N}}.\
""",
    "case_coverage": """\
Your previous solution failed to cover all necessary cases.

Problem:
{problem}

Previous solution:
{solution}

Issue: {feedback}

Identify all cases that must be considered. Handle each case completely, verify no \
cases are missed or double-counted. End with \\boxed{{N}}.\
""",
    "theorem_applicability": """\
Your previous solution misapplied or incorrectly stated a theorem or identity.

Problem:
{problem}

Previous solution:
{solution}

Issue: {feedback}

Check that any theorems, formulas, or identities you use are correctly stated and \
applicable under the right assumptions. Provide a corrected solution. End with \\boxed{{N}}.\
""",
    "final_answer_consistency": """\
Your previous solution's final answer does not follow from the work shown.

Problem:
{problem}

Previous solution:
{solution}

Issue: {feedback}

Re-examine your work carefully. Ensure the value in \\boxed{{}} is exactly what your \
solution computes. Provide a corrected solution ending with \\boxed{{N}}.\
""",
}

_COMBINE_PROMPT = """\
Two candidate solutions to this problem each have different strengths and weaknesses.

Problem:
{problem}

--- Solution A ---
{solution_a}

Solution A — strong on: {strengths_a}
Solution A — weak on:   {weaknesses_a}

--- Solution B ---
{solution_b}

Solution B — strong on: {strengths_b}
Solution B — weak on:   {weaknesses_b}

Write a new combined solution that incorporates the correct reasoning from both. \
Preserve what each got right and fix what each got wrong. Show all steps clearly. \
End with \\boxed{{N}}.\
"""


def _make_id(problem_id: str, operation: str, depth: int, k: int) -> str:
    return f"{problem_id}-{operation}-d{depth}-{k}"


def _bullets(items: list[str]) -> str:
    return "\n".join(f"  • {x}" for x in items) if items else "  (none)"


def _verifier_summary(verifier_scores: dict[str, VerifierResult], threshold: float = 0.7) -> tuple[str, str]:
    """Return (strengths_str, weaknesses_str) based on verifier scores."""
    strong = [f"{n} ({v.score:.2f})" for n, v in verifier_scores.items() if v.score >= threshold]
    weak = [f"{n} ({v.score:.2f}): {v.feedback}" for n, v in verifier_scores.items() if v.score < threshold]
    return _bullets(strong), _bullets(weak)


def _build_candidate(
    problem: MathProblem,
    solution_text: str,
    operation: str,
    depth: int,
    k: int,
    parent_ids: list[str],
    metadata: dict,
) -> MathCandidate:
    extracted = extract_aime_answer(solution_text)
    return MathCandidate(
        candidate_id=_make_id(problem.id, operation, depth, k),
        problem_id=problem.id,
        solution_text=solution_text,
        extracted_answer=extracted,
        verifier_scores={},  # filled in by the caller after scoring
        parent_ids=parent_ids,
        search_depth=depth,
        operation=operation,
        generation_metadata=metadata,
    )


def generate(
    llm: LLM,
    problem: MathProblem,
    tags: dict,
    k: int = 0,
    depth: int = 0,
    temperature: float = 0.6,
) -> MathCandidate | None:
    """Fresh full-solution attempt."""
    messages = [
        {"role": "system", "content": _GEN_SYS},
        {"role": "user", "content": _GEN_PROMPT.format(problem=problem.problem)},
    ]
    try:
        response = llm.call(messages, tags=tags, temperature=temperature)
    except Exception:
        return None
    return _build_candidate(
        problem, response, "seed", depth, k,
        parent_ids=[],
        metadata={"temperature": temperature},
    )


def revise(
    llm: LLM,
    problem: MathProblem,
    candidate: MathCandidate,
    failed_verifiers: list[tuple[str, VerifierResult]],
    tags: dict,
    depth: int = 1,
    k: int = 0,
    temperature: float = 0.6,
) -> MathCandidate | None:
    """
    Targeted repair using the prompt for the single weakest verifier.
    failed_verifiers should be sorted weakest-first (from get_failed_verifiers).
    """
    if not failed_verifiers:
        return None

    weakest_name, weakest_result = failed_verifiers[0]
    prompt_template = _REVISE_PROMPTS.get(weakest_name, _REVISE_PROMPTS["local_step_validity"])
    prompt = prompt_template.format(
        problem=problem.problem,
        solution=candidate.solution_text,
        feedback=weakest_result.feedback,
    )
    messages = [
        {"role": "system", "content": _GEN_SYS},
        {"role": "user", "content": prompt},
    ]
    try:
        response = llm.call(messages, tags={**tags, "repair_target": weakest_name}, temperature=temperature)
    except Exception:
        return None
    return _build_candidate(
        problem, response, "revise", depth, k,
        parent_ids=[candidate.candidate_id],
        metadata={"temperature": temperature, "repair_target": weakest_name},
    )


def combine(
    llm: LLM,
    problem: MathProblem,
    cand_a: MathCandidate,
    cand_b: MathCandidate,
    tags: dict,
    depth: int = 1,
    k: int = 0,
    temperature: float = 0.6,
) -> MathCandidate | None:
    """Combine two candidates, surfacing per-dimension strengths/weaknesses."""
    strengths_a, weaknesses_a = _verifier_summary(cand_a.verifier_scores)
    strengths_b, weaknesses_b = _verifier_summary(cand_b.verifier_scores)
    prompt = _COMBINE_PROMPT.format(
        problem=problem.problem,
        solution_a=cand_a.solution_text,
        strengths_a=strengths_a,
        weaknesses_a=weaknesses_a,
        solution_b=cand_b.solution_text,
        strengths_b=strengths_b,
        weaknesses_b=weaknesses_b,
    )
    messages = [
        {"role": "system", "content": _GEN_SYS},
        {"role": "user", "content": prompt},
    ]
    try:
        response = llm.call(
            messages,
            tags={**tags, "parent_a": cand_a.candidate_id, "parent_b": cand_b.candidate_id},
            temperature=temperature,
        )
    except Exception:
        return None
    return _build_candidate(
        problem, response, "combine", depth, k,
        parent_ids=[cand_a.candidate_id, cand_b.candidate_id],
        metadata={"temperature": temperature},
    )
