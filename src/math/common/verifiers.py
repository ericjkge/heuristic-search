"""
LLM-judge verifier system for AIME/HMMT math problems.

Scoring architecture per candidate (2 LLM calls, fired in parallel):
  1. answer_format_range      — code check, free
  2. combined adversarial judge — 1 LLM call, 6 dimensions (think=False, fast)
  3. correctness_confidence   — 1 LLM call, holistic step-by-step review (think=True)

The combined judge (2) checks specific failure dimensions.
correctness_confidence (3) synthesises holistically: "how confident am I this is right?"
It is used as the primary final-selection key.
"""

from __future__ import annotations

import json
import re

from src.math.common.answer import validate_aime_answer
from src.math.common.result import VerifierResult
from src.math.data import MathProblem
from utils.concurrency import run_parallel
from utils.llm import LLM

# ── Verifier name lists ────────────────────────────────────────────────────────

SOFT_VERIFIERS = [
    "problem_condition_coverage",
    "local_step_validity",
    "algebra_arithmetic_consistency",
    "case_coverage",
    "theorem_applicability",
    "final_answer_consistency",
]
ALL_VERIFIERS = ["answer_format_range"] + SOFT_VERIFIERS + ["correctness_confidence"]

_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

# ── Combined adversarial judge (6 soft dimensions, think=False) ───────────────

_JUDGE_SYS = (
    "You are a harsh, adversarial mathematics competition grader. "
    "Your job is to find errors. Assume the solution below contains at least one mistake "
    "and actively look for it. Only award a high score if you cannot find any error "
    "despite genuinely trying. "
    "Output ONLY a JSON code block, no prose."
)

_JUDGE_PROMPT = """\
Problem:
{problem}

Proposed Solution:
{solution}

Actively search for errors across the following six dimensions. For each, give:
- score: a float in [0.0, 1.0] (1.0 = no errors found after active search; 0.0 = clear error)
- feedback: one concise sentence naming the specific error found, or "No error found."

Output ONLY this JSON structure:
```json
{{
  "problem_condition_coverage": {{"score": <float>, "feedback": "<str>"}},
  "local_step_validity": {{"score": <float>, "feedback": "<str>"}},
  "algebra_arithmetic_consistency": {{"score": <float>, "feedback": "<str>"}},
  "case_coverage": {{"score": <float>, "feedback": "<str>"}},
  "theorem_applicability": {{"score": <float>, "feedback": "<str>"}},
  "final_answer_consistency": {{"score": <float>, "feedback": "<str>"}}
}}
```

Dimension definitions (be adversarial in each):
- problem_condition_coverage: Did the solution explicitly use EVERY condition in the problem? \
Check each condition one by one. Score 0 if any is ignored or only implicitly assumed.
- local_step_validity: Is every step rigorously justified? Look for unjustified leaps, \
circular reasoning, or steps that don't follow from what precedes them.
- algebra_arithmetic_consistency: Recompute the key calculations numerically yourself. \
Do not trust the solution's arithmetic — redo it and check if the numbers match.
- case_coverage: If the problem requires case analysis, verify all cases are covered and \
disjoint. Try to construct a counterexample or missing case.
- theorem_applicability: For each theorem, identity, or lemma invoked, verify that all \
required preconditions hold. Assume at least one may be misapplied.
- final_answer_consistency: Plug the stated answer back into the original problem's \
constraints and verify it satisfies ALL of them directly. Check against the problem \
statement itself, not just whether the answer follows from the written work.\
"""

# ── Correctness confidence review (holistic, think=True) ─────────────────────

_CONF_SYS = (
    "You are a careful mathematics competition reviewer. "
    "Read the solution below thoroughly and give an honest assessment of its correctness. "
    "Output ONLY a JSON code block, no prose."
)

_CONF_PROMPT = """\
Problem:
{problem}

Candidate Solution:
{solution}

Read through this solution carefully, step by step. Check the logic and arithmetic as you go.
Flag anything that seems uncertain or suspicious — even if you cannot pinpoint exactly why.

Be honest about your uncertainty. Do not assume the solution is wrong, but do not rubber-stamp
it either. If a step seems off but you cannot say why, that should lower your confidence.

Output ONLY this JSON:
```json
{{
  "concerns": ["<specific concern 1>", "<specific concern 2>"],
  "confidence": <float 0.0-1.0, your honest estimate that the final answer is correct>,
  "feedback": "<one sentence: your overall assessment>"
}}
```

Guidelines for confidence:
- 1.0: Every step checked, arithmetic verified, answer confirmed correct
- 0.8-0.9: Mostly convincing, one minor concern or unverified step
- 0.5-0.7: Plausible approach but something feels uncertain or suspicious
- 0.2-0.4: Clear logical or arithmetic issue found
- 0.0-0.1: Solution is wrong or the answer is clearly incorrect\
"""


def _call_correctness_confidence(
    solution_text: str,
    problem: MathProblem,
    llm: LLM,
    tags: dict,
) -> VerifierResult:
    """
    Holistic confidence review. Sees problem + solution, returns calibrated confidence.
    Uses think=True for careful step-by-step reasoning. Falls back to 0.5 on failure.
    """
    prompt = _CONF_PROMPT.format(problem=problem.problem, solution=solution_text)
    messages = [
        {"role": "system", "content": _CONF_SYS},
        {"role": "user", "content": prompt},
    ]
    try:
        response = llm.call(messages, tags={**tags, "phase": "confidence_review"}, think=True)
        match = _JSON_RE.search(response)
        raw = match.group(1) if match else response.strip()
        data = json.loads(raw)
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        concerns = data.get("concerns", [])
        feedback = str(data.get("feedback", ""))
        if concerns:
            feedback = f"{feedback} Concerns: {'; '.join(str(c) for c in concerns)}"
        return VerifierResult(
            name="correctness_confidence",
            score=confidence,
            passed=confidence >= 0.8,
            feedback=feedback,
            is_hard=False,
        )
    except Exception as e:
        # Parse or call failure → uncertain (0.5), flagged clearly
        return VerifierResult(
            name="correctness_confidence",
            score=0.5, passed=None,
            feedback=f"[CONFIDENCE REVIEW FAILED: {e}]",
            is_hard=False,
        )


# ── Combined adversarial judge ─────────────────────────────────────────────────

def _call_llm_judge(
    solution_text: str,
    problem: MathProblem,
    llm: LLM,
    tags: dict,
) -> dict[str, VerifierResult]:
    """
    Single adversarial LLM call for all 6 soft dimensions.
    All-zero response is treated as a failed call (mapped to 0.5 each).
    """
    prompt = _JUDGE_PROMPT.format(problem=problem.problem, solution=solution_text)
    messages = [
        {"role": "system", "content": _JUDGE_SYS},
        {"role": "user", "content": prompt},
    ]
    try:
        response = llm.call(messages, tags={**tags, "phase": "judge"}, think=False)
        results = _parse_judge_response(response)
        if all(v.score == 0.0 for v in results.values()):
            return _degraded_results("[JUDGE CALL FAILED: all-zero response]")
        return results
    except Exception as e:
        return _degraded_results(f"[JUDGE CALL FAILED: {e}]")


def _parse_judge_response(response: str) -> dict[str, VerifierResult]:
    match = _JSON_RE.search(response)
    raw = match.group(1) if match else response.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _degraded_results("[JUDGE CALL FAILED: JSON parse error]")

    results: dict[str, VerifierResult] = {}
    for name in SOFT_VERIFIERS:
        if name not in data:
            results[name] = VerifierResult(
                name=name, score=0.5, passed=None,
                feedback="Missing from judge response.", is_hard=False,
            )
            continue
        dim = data[name]
        try:
            score = float(dim.get("score", 0.5))
            score = max(0.0, min(1.0, score))
            feedback = str(dim.get("feedback", ""))
        except (TypeError, ValueError):
            score, feedback = 0.5, "Malformed judge entry."
        results[name] = VerifierResult(
            name=name, score=score, passed=None, feedback=feedback, is_hard=False,
        )
    return results


def _degraded_results(reason: str) -> dict[str, VerifierResult]:
    """Score=0.5 (uncertain) on failure — not penalised, not rewarded."""
    return {
        name: VerifierResult(name=name, score=0.5, passed=None, feedback=reason, is_hard=False)
        for name in SOFT_VERIFIERS
    }


def _check_format_range(extracted_answer: str | None) -> VerifierResult:
    ok = validate_aime_answer(extracted_answer)
    return VerifierResult(
        name="answer_format_range",
        score=1.0 if ok else 0.0,
        passed=ok,
        feedback="Valid AIME integer." if ok else
                 f"Invalid answer: {extracted_answer!r} is not an integer in [0, 999].",
        is_hard=True,
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def build_verifiers(
    problem: MathProblem,
    llm: LLM | None = None,
    reference_solution: str | None = None,
) -> list[str]:
    """Return list of active verifier names. Always returns ALL_VERIFIERS in v1."""
    if reference_solution is not None and llm is not None:
        _call_llm_judge(
            reference_solution, problem, llm,
            tags={"problem_id": problem.id, "phase": "verifier_calibration"},
        )
    return list(ALL_VERIFIERS)


def score_candidate(
    solution_text: str,
    extracted_answer: str | None,
    problem: MathProblem,
    llm: LLM,
    verifier_names: list[str],
    tags: dict,
) -> dict[str, VerifierResult]:
    """
    Score one candidate:
      - 1 code check (free)
      - adversarial judge + confidence review fired in parallel
    """
    results: dict[str, VerifierResult] = {}

    if "answer_format_range" in verifier_names:
        results["answer_format_range"] = _check_format_range(extracted_answer)

    run_judge = any(n in verifier_names for n in SOFT_VERIFIERS)
    run_conf  = "correctness_confidence" in verifier_names

    thunks = []
    if run_judge:
        thunks.append(lambda: ("judge", _call_llm_judge(solution_text, problem, llm, tags=tags)))
    if run_conf:
        thunks.append(lambda: ("conf",  _call_correctness_confidence(solution_text, problem, llm, tags=tags)))

    for label, outcome in run_parallel(thunks):
        if label == "judge":
            for name in SOFT_VERIFIERS:
                if name in verifier_names:
                    results[name] = outcome.get(
                        name,
                        VerifierResult(name=name, score=0.5, passed=None,
                                       feedback="Not evaluated.", is_hard=False),
                    )
        else:
            results["correctness_confidence"] = outcome

    return results


def aggregate_score(verifier_results: dict[str, VerifierResult]) -> float:
    if not verifier_results:
        return 0.0
    return sum(v.score for v in verifier_results.values()) / len(verifier_results)


def get_verifier_vector(
    verifier_results: dict[str, VerifierResult],
    verifier_names: list[str],
) -> list[float]:
    return [verifier_results[n].score if n in verifier_results else 0.0 for n in verifier_names]


def get_failed_verifiers(
    verifier_results: dict[str, VerifierResult],
    threshold: float = 0.5,
) -> list[tuple[str, VerifierResult]]:
    failed = [(n, v) for n, v in verifier_results.items() if v.score < threshold]
    return sorted(failed, key=lambda nv: nv[1].score)
