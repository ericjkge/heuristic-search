"""Best-of-N baseline: sample the puzzle N times, pass if ANY sample is correct.

Runs a fixed number of samples; cost is measured post-hoc in tokens.
"""

from src.common.candidates import generate, matches_gold
from src.common.result import Result
from utils.concurrency import run_parallel
from utils.data import Puzzle
from utils.llm import LLM


def best_of_n(llm: LLM, puzzle: Puzzle, samples: int = 16) -> Result:
    """Draw `samples` independent attempts concurrently; solved iff any matches gold."""
    cands = run_parallel([
        lambda k=k: generate(
            llm, puzzle, tags={"puzzle_id": puzzle.id, "condition": "bon",
                               "phase": "sample", "k": k}
        )
        for k in range(samples)
    ])

    # Reduce in sample order so solved_at / trajectory match a sequential run.
    solved = False
    solved_at = None
    winner = None
    trajectory = []  # cumulative solved flag after each sample
    for n, cand in enumerate(cands, 1):
        if not solved and matches_gold(cand, puzzle):
            solved, solved_at, winner = True, n, cand
        trajectory.append(1.0 if solved else 0.0)

    return Result(
        puzzle_id=puzzle.id,
        solved=solved,
        best_score=float(solved),
        best_candidate=winner,
        trajectory=trajectory,
        condition="bon",
        extra={"n_samples": samples, "solved_at_sample": solved_at},
    )


if __name__ == "__main__":
    from utils.data import load_smoke_set

    llm = LLM()
    p = load_smoke_set()[1]
    res = best_of_n(llm, p, samples=16)
    print(f"{p.id} solved={res.solved} samples={res.extra['n_samples']} "
          f"solved_at={res.extra['solved_at_sample']}")
