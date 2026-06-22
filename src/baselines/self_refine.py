"""Self-Refine baseline.

Generate an initial solution, then iterate: (1) the model critiques its own
answer with GENERIC feedback ("check your work"), and (2) refines using that
self-critique. Unlike the main method, the feedback is the model's own generic
self-assessment, not verifier-derived failed clues — isolating whether TARGETED
(verifier) feedback beats generic self-feedback. No verifiers and no gold are
used during the loop; solved = the final answer matches gold.

Runs a fixed number of refine steps (no early stop), so compute is comparable;
each step is two calls (critique + refine).
"""

import json

from src.common.candidates import GEN_SYS, generate, matches_gold, parse_candidate
from src.common.result import Result
from utils.data import Puzzle
from utils.llm import LLM

CRITIQUE_SYS = (
    "You carefully review proposed solutions to logic grid puzzles and point out "
    "mistakes. Respond with concise prose feedback only, no code."
)

CRITIQUE_PROMPT = """\
Here is a proposed solution to the puzzle.

{puzzle}

Proposed solution:
{candidate}

Check the work carefully: go through each clue and the solution, and identify any
mistakes or inconsistencies. If something is wrong, say what is wrong. If it looks
fully correct, say so. Keep it brief."""

REFINE_PROMPT = """\
You previously proposed this solution:
{candidate}

Your review of it:
{feedback}

Using your review, output a corrected full solution as one JSON code block in the
schema {{"header": {header}, "rows": [<{n} rows>]}} ({n} rows aligned to header,
the "House" column is "1".."{n}"). Output only the JSON code block."""


def _critique(llm: LLM, puzzle: Puzzle, candidate: dict, tags: dict) -> str:
    prompt = CRITIQUE_PROMPT.format(puzzle=puzzle.puzzle, candidate=json.dumps(candidate))
    return llm.call(
        [{"role": "system", "content": CRITIQUE_SYS},
         {"role": "user", "content": prompt}],
        tags=tags,
    )


def _refine(llm: LLM, puzzle: Puzzle, candidate: dict, feedback: str,
            tags: dict, retries: int = 1) -> dict | None:
    prompt = REFINE_PROMPT.format(
        candidate=json.dumps(candidate), feedback=feedback,
        header=puzzle.header, n=puzzle.n_houses,
    )
    msgs = [{"role": "system", "content": GEN_SYS},
            {"role": "user", "content": prompt}]
    for attempt in range(retries + 1):
        text = llm.call(msgs, tags={**tags, "attempt": attempt})
        cand = parse_candidate(text, puzzle)
        if cand is not None:
            return cand
    return None


def self_refine(llm: LLM, puzzle: Puzzle, num_steps: int = 4) -> Result:
    """Initial solution + `num_steps` rounds of generic self-critique then refine."""
    base = {"puzzle_id": puzzle.id, "condition": "self_refine"}
    cand = generate(llm, puzzle, tags={**base, "phase": "init", "k": 0})
    trajectory = [1.0 if matches_gold(cand, puzzle) else 0.0]  # correctness per step (analysis only)

    for it in range(1, num_steps + 1):
        if cand is None:
            break
        feedback = _critique(llm, puzzle, cand, tags={**base, "phase": "critique", "iter": it})
        refined = _refine(llm, puzzle, cand, feedback, tags={**base, "phase": "refine", "iter": it})
        if refined is not None:
            cand = refined
        trajectory.append(1.0 if matches_gold(cand, puzzle) else 0.0)

    solved = matches_gold(cand, puzzle)
    return Result(
        puzzle_id=puzzle.id,
        solved=solved,
        best_score=float(solved),
        best_candidate=cand,
        trajectory=trajectory,
        condition="self_refine",
        extra={"num_steps": num_steps},
    )


if __name__ == "__main__":
    from utils.data import load_puzzle

    llm = LLM()
    p = load_puzzle("lgp-test-3x3-24")
    res = self_refine(llm, p, num_steps=4)
    print(f"{p.id} solved={res.solved} trajectory={res.trajectory}")
    print("trace:", llm.trace_path)
