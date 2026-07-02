"""
LLM-judge verifier system for AIME/HMMT math problems.

Scoring uses discrete rubric grades (A/B/C/D) instead of continuous [0,1] scores.
LLM-assigned continuous scores are too noisy to be meaningful; discrete rubrics force
the model to make categorical, observable judgments.

Grade mapping:  A → 0.00  |  B → 0.33  |  C → 0.66  |  D → 1.00

Scoring architecture per candidate (2 LLM calls, fired in parallel):
  1. answer_format_range      — code check, free
  2. combined adversarial judge — 1 LLM call, 6 dimensions (think=False, fast)
  3. correctness_confidence   — 1 LLM call, holistic step-by-step review (think=True)
"""

from __future__ import annotations

import json
import re

from src.math.common.answer import validate_aime_answer
from src.math.common.result import VerifierResult
from src.math.data import MathProblem
from utils.concurrency import run_parallel
from utils.llm import LLM

# ── Grade → score mapping ──────────────────────────────────────────────────────

GRADE_TO_SCORE: dict[str, float] = {"A": 0.0, "B": 0.33, "C": 0.66, "D": 1.0}

def _grade_to_score(grade: str) -> float:
    return GRADE_TO_SCORE.get(grade.strip().upper(), 0.0)

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

# ── Verifier weights ───────────────────────────────────────────────────────────
# Three tiers, each 2× the previous (1 → 2 → 4).
# Tier 4: verifiers most directly tied to answer correctness.
# Tier 2: important structural checks, one level removed from direct correctness.
# Tier 1: apply to a subset of problems; auto-grade D when not applicable.
VERIFIER_WEIGHTS: dict[str, float] = {
    "answer_format_range":            1.0,
    "problem_condition_coverage":     2.0,
    "local_step_validity":            2.0,
    "algebra_arithmetic_consistency": 4.0,
    "case_coverage":                  1.0,
    "theorem_applicability":          2.0,
    "final_answer_consistency":       4.0,
    "correctness_confidence":         4.0,
}  # total weight = 20

_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

# ── Combined adversarial judge (6 dimensions, rubric-graded) ──────────────────

_JUDGE_SYS = (
    "You are a rigorous mathematics competition grader. "
    "Evaluate the solution according to the rubric below. "
    "For each dimension pick exactly one grade: A, B, C, or D. "
    "Output ONLY a JSON code block, no prose."
)

_JUDGE_PROMPT = """\
Problem:
{problem}

Proposed Solution:
{solution}

Grade the solution on each dimension below. Use exactly the rubric given — do not
invent intermediate grades. Output ONLY this JSON:

```json
{{
  "problem_condition_coverage":     {{"grade": "<A|B|C|D>", "feedback": "<one sentence>"}},
  "local_step_validity":            {{"grade": "<A|B|C|D>", "feedback": "<one sentence>"}},
  "algebra_arithmetic_consistency": {{"grade": "<A|B|C|D>", "feedback": "<one sentence>"}},
  "case_coverage":                  {{"grade": "<A|B|C|D>", "feedback": "<one sentence>"}},
  "theorem_applicability":          {{"grade": "<A|B|C|D>", "feedback": "<one sentence>"}},
  "final_answer_consistency":       {{"grade": "<A|B|C|D>", "feedback": "<one sentence>"}}
}}
```

━━━ RUBRICS ━━━

PROBLEM_CONDITION_COVERAGE — go through each condition in the problem one by one:
  A: One or more conditions are never used or referenced anywhere in the solution.
  B: All conditions appear, but at least one is only mentioned without driving any reasoning step.
  C: All conditions actively drive the reasoning, but one condition's use could be more explicit.
  D: Every condition is explicitly identified and directly drives at least one reasoning step.

LOCAL_STEP_VALIDITY — read each logical transition; does it follow from what came before?
  A: At least one step asserts something that does not follow from the previous steps.
  B: No outright logical errors, but at least one step makes a leap requiring unstated reasoning.
  C: Every step is justified, but one transition lacks enough detail to verify independently.
  D: Every step follows rigorously from what precedes it; no gaps or unstated assumptions.

ALGEBRA_ARITHMETIC_CONSISTENCY — independently re-evaluate key computations:
  A: At least one key computation, when re-evaluated independently, yields a different result.
  B: No outright arithmetic error, but at least one key computation lacks enough working to verify.
  C: All visible computations check out, but one intermediate result is asserted without derivation.
  D: All key computations independently re-evaluated and confirmed correct.

CASE_COVERAGE — does the problem require case analysis? If not, assign D automatically:
  A: At least one necessary case is missing, or two cases overlap (double-counting).
  B: All necessary cases named, but at least one is unresolved or exhaustiveness is unargued.
  C: All cases resolved, but exhaustiveness/disjointness is implicit rather than explicit.
  D: All cases identified, fully resolved, exhaustiveness and disjointness explicitly argued. Or: no case analysis needed.

THEOREM_APPLICABILITY — for each theorem/identity/lemma invoked, check preconditions. If none invoked, assign D:
  A: A theorem is applied when at least one required precondition does not hold here.
  B: Preconditions are met, but at least one application does not explicitly verify them.
  C: All preconditions verified, but one theorem application could be stated more precisely.
  D: Every theorem correctly stated, all preconditions explicitly verified, precisely applied. Or: none invoked.

FINAL_ANSWER_CONSISTENCY — substitute the stated answer into the original problem's constraints:
  A: The stated answer, when substituted into the problem's constraints, fails at least one.
  B: Substitution reveals an unverified or violated condition not accounted for in the solution.
  C: Answer satisfies all constraints on substitution, but at least one is not explicitly checked.
  D: Answer derived from the work and explicitly verified against every constraint in the problem.\
"""

# ── Correctness confidence (holistic rubric, think=True) ──────────────────────

_CONF_SYS = (
    "You are a careful mathematics competition reviewer. "
    "Read the solution below step by step and assign one grade using the rubric. "
    "Output ONLY a JSON code block, no prose."
)

_CONF_PROMPT = """\
Problem:
{problem}

Candidate Solution:
{solution}

Read through this solution carefully, step by step. Then assign exactly one grade.
Be honest — do not upgrade a grade unless the evidence clearly warrants it.

Output ONLY this JSON:
```json
{{
  "grade": "<A|B|C|D>",
  "concerns": ["<specific concern 1>", "<specific concern 2>"],
  "feedback": "<one sentence overall assessment>"
}}
```

━━━ RUBRIC FOR CORRECTNESS_CONFIDENCE ━━━
  A: A specific error was identified during review that would change the final answer.
  B: No definite error found, but at least one step raised enough concern that the answer may be wrong.
  C: No errors or serious concerns found, but at least one step could not be independently verified.
  D: Every step reviewed and independently verified; no errors, gaps, or concerns of any kind.\
"""


def _call_correctness_confidence(
    solution_text: str,
    problem: MathProblem,
    llm: LLM,
    tags: dict,
) -> VerifierResult:
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
        grade = str(data.get("grade", "B")).strip().upper()
        score = _grade_to_score(grade)
        concerns = data.get("concerns", [])
        feedback = str(data.get("feedback", ""))
        if concerns:
            feedback = f"{feedback} Concerns: {'; '.join(str(c) for c in concerns)}"
        return VerifierResult(
            name="correctness_confidence",
            score=score,
            passed=score >= 0.66,
            feedback=f"[{grade}] {feedback}",
            is_hard=False,
        )
    except Exception as e:
        return VerifierResult(
            name="correctness_confidence",
            score=0.33, passed=None,
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
    prompt = _JUDGE_PROMPT.format(problem=problem.problem, solution=solution_text)
    messages = [
        {"role": "system", "content": _JUDGE_SYS},
        {"role": "user", "content": prompt},
    ]
    try:
        response = llm.call(messages, tags={**tags, "phase": "judge"}, think=False)
        return _parse_judge_response(response)
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
                name=name, score=0.33, passed=None,
                feedback="[Missing from judge response]", is_hard=False,
            )
            continue
        dim = data[name]
        try:
            grade = str(dim.get("grade", "B")).strip().upper()
            score = _grade_to_score(grade)
            feedback = str(dim.get("feedback", ""))
        except (TypeError, ValueError):
            grade, score, feedback = "B", 0.33, "[Malformed judge entry]"
        results[name] = VerifierResult(
            name=name, score=score, passed=None,
            feedback=f"[{grade}] {feedback}", is_hard=False,
        )
    return results


def _degraded_results(reason: str) -> dict[str, VerifierResult]:
    """Grade B (0.33) on failure — uncertain, not penalised to zero."""
    return {
        name: VerifierResult(name=name, score=0.33, passed=None, feedback=reason, is_hard=False)
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
    Score one candidate. Fires adversarial judge + confidence review in parallel.
    All scores are discrete: {0.0, 0.33, 0.66, 1.0}.
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
                        VerifierResult(name=name, score=0.33, passed=None,
                                       feedback="[Not evaluated]", is_hard=False),
                    )
        else:
            results["correctness_confidence"] = outcome

    return results


def aggregate_score(verifier_results: dict[str, VerifierResult]) -> float:
    """Weighted mean: Σ(score_i × weight_i) / Σ(weight_i). Unknown verifiers get weight 1."""
    if not verifier_results:
        return 0.0
    total_w = sum(VERIFIER_WEIGHTS.get(name, 1.0) for name in verifier_results)
    weighted = sum(v.score * VERIFIER_WEIGHTS.get(name, 1.0) for name, v in verifier_results.items())
    return weighted / total_w if total_w > 0 else 0.0


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
