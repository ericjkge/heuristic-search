"""Build one deterministic Python checker per clue, sanity-gated against gold.

Each checker is a self-contained `def check(candidate) -> bool`, where
`candidate = {"header": [...], "rows": [[...], ...]}` matches the dataset's
solution schema. The LLM writes the code once; scoring then just runs it.
"""

import re
from dataclasses import dataclass
from typing import Callable

from utils.concurrency import run_parallel
from utils.data import Puzzle
from utils.llm import LLM

# Allowed builtins in checker code (no import/open/exec/eval).
_SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in [
        "len", "range", "enumerate", "zip", "all", "any", "abs", "sorted",
        "min", "max", "next", "filter", "map", "sum", "str", "int", "bool",
        "list", "dict", "set", "tuple", "isinstance", "reversed",
    ]
}

_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)

CHECKER_SYS = (
    "You write deterministic Python validators for logic-puzzle clues. "
    "Output ONLY one Python code block, no prose."
)

CHECKER_PROMPT = """\
A candidate solution is a dict: {{"header": [...], "rows": [[...], ...]}}.
- header columns: {header}
- Column "House" holds the position as a string "1".."{n_houses}" (left to right).
- Other cells are strings using EXACTLY the value spellings from the puzzle below.

Puzzle (for attribute value spellings and the house count):
{puzzle}

Write a function `check(candidate) -> bool` that returns True iff the candidate
satisfies ONLY this clue:

    "{clue}"

Rules:
- Read column indices from candidate["header"]; do not hardcode positions.
- Compare strings case-insensitively (e.g. .lower()) since clues may differ in case.
- Be robust: if a referenced value is absent, return False rather than raising.
- Self-contained, standard Python only, no imports.
Output only the code block.
"""


@dataclass
class Verifier:
    clue: str
    code: str
    fn: Callable[[dict], bool]
    passed_gold: bool


def passes(verifiers: list["Verifier"], candidate: dict) -> list[bool]:
    """Per-verifier pass/fail for a candidate."""
    return [v.fn(candidate) for v in verifiers]


def verifier_scores(verifiers: list["Verifier"], candidate: dict) -> list[float]:
    """The per-verifier score vector for a candidate (0/1 per pass/fail checker)."""
    return [float(v.fn(candidate)) for v in verifiers]


def score(verifiers: list["Verifier"], candidate: dict) -> float:
    """Heuristic s(y) = #{checkers passed} / K. Empty verifier set -> 0.0."""
    if not verifiers:
        return 0.0
    return sum(passes(verifiers, candidate)) / len(verifiers)


def failed_clues(verifiers: list["Verifier"], candidate: dict) -> list[str]:
    """Verbatim clue text for every checker the candidate fails (for feedback)."""
    return [v.clue for v, ok in zip(verifiers, passes(verifiers, candidate)) if not ok]


def satisfied_clues(verifiers: list["Verifier"], candidate: dict) -> list[str]:
    """Verbatim clue text for every checker the candidate passes (for combine context)."""
    return [v.clue for v, ok in zip(verifiers, passes(verifiers, candidate)) if ok]


def extract_code(text: str) -> str:
    m = _CODE_RE.search(text)
    return (m.group(1) if m else text).strip()


def _safe_call(fn: Callable[[dict], bool], candidate: dict) -> bool:
    """Run a checker, returning False on any error."""
    try:
        return bool(fn(candidate))
    except Exception:
        return False


def compile_checker(code: str) -> Callable[[dict], bool]:
    """Exec checker code in a restricted namespace; return a safe callable."""
    ns = {"__builtins__": _SAFE_BUILTINS} # define namespace for restricting built-ins
    exec(compile(code, "<checker>", "exec"), ns) # exec checker code in restricted namespace (also prevents cross-thread contamination)
    fn = ns.get("check") # pull out check function
    if not callable(fn):
        raise ValueError("no check() defined")
    return lambda candidate: _safe_call(fn, candidate)


def _build_one(llm: LLM, puzzle: Puzzle, idx: int, clue: str, max_retries: int) -> Verifier:
    """Build one checker for `clue`, regenerating up to max_retries on gold failure."""
    prompt = CHECKER_PROMPT.format(
        header=puzzle.header,
        n_houses=puzzle.n_houses,
        puzzle=puzzle.puzzle,
        clue=clue,
    )
    verifier = None
    for attempt in range(max_retries + 1):
        text = llm.call(
            [
                {"role": "system", "content": CHECKER_SYS},
                {"role": "user", "content": prompt},
            ],
            tags={
                "puzzle_id": puzzle.id,
                "phase": "build_verifier",
                "clue_idx": idx,
                "attempt": attempt,
            },
        )
        code = extract_code(text)
        try:
            fn = compile_checker(code)
            passed = fn(puzzle.gold)
        except Exception:
            fn, passed = (lambda c: False), False
        if passed:
            return Verifier(clue, code, fn, True)
        verifier = Verifier(clue, code, fn, False)
    return verifier  # type: ignore[return-value]


def build_verifiers(
    llm: LLM, puzzle: Puzzle, max_retries: int = 2
) -> list[Verifier]:
    """One checker per clue, built concurrently; order matches puzzle.clues."""
    return run_parallel([
        lambda idx=idx, clue=clue: _build_one(llm, puzzle, idx, clue, max_retries)
        for idx, clue in enumerate(puzzle.clues)
    ])

# Smoke test verifier generation on first 3 puzzles
if __name__ == "__main__":
    from utils.data import load_smoke_set

    llm = LLM()
    puzzles = load_smoke_set()[:3]  # first 3 for the checkpoint
    for p in puzzles:
        vs = build_verifiers(llm, p)
        n_pass = sum(v.passed_gold for v in vs)
        print(f"=== {p.id} (K={p.k}): {n_pass}/{p.k} checkers pass gold")
        for i, v in enumerate(vs, 1):
            flag = "ok " if v.passed_gold else "FAIL"
            print(f"  [{flag}] clue {i}: {v.clue}")
    print("\ntrace:", llm.trace_path)
