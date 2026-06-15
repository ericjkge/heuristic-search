"""Verifier-guided beam search — the core MVP algorithm.

Treats solving as search over candidate grids, using the verifier pass-rate
s(y) = #{checkers passed}/K as the heuristic. Selects the best candidates,
gives them their verbatim failed clues as feedback, and revises. Runs for a
fixed number of search `steps` (iterations); cost is measured post-hoc in tokens.
"""

import random

from src.common.candidates import generate, matches_gold, revise
from src.common.result import Result
from src.common.verifiers import Verifier, failed_clues, score
from utils.data import Puzzle
from utils.llm import LLM


def beam_search(
    llm: LLM,
    puzzle: Puzzle,
    verifiers: list[Verifier],
    steps: int = 4,
    beam_w: int = 2,
    revisions: int = 2,
    seed: int = 0,
) -> Result:
    """Run verifier-guided beam search for a fixed number of `steps` iterations."""
    rng = random.Random(seed)
    # Drop checkers that fail gold (per the doc): a buggy verifier would make
    # s=1.0 unreachable even for the correct answer.
    verifiers = [v for v in verifiers if v.passed_gold]

    # POOL: M independent full attempts.
    pool = []
    for i in range(revisions):
        cand = generate(
            llm, puzzle, tags={"puzzle_id": puzzle.id, "condition": "main",
                               "phase": "init", "iter": 0, "k": i}
        )
        if cand is not None:
            pool.append((cand, score(verifiers, cand)))

    trajectory = []
    best = max((s for _, s in pool), default=0.0)
    trajectory.append(best)

    it = 0
    while it < steps and best < 1.0 and pool:
        it += 1
        ranked = sorted(pool, key=lambda cs: cs[1], reverse=True)
        # SELECT: top-W parents, plus one random low-scorer for exploration.
        parents = ranked[:beam_w]
        tail = ranked[beam_w:]
        if tail:
            parents = parents + [rng.choice(tail)]

        for cand, _ in parents:
            failed = failed_clues(verifiers, cand)
            if not failed:
                continue
            for m in range(revisions):
                child = revise(
                    llm, puzzle, cand, failed,
                    tags={"puzzle_id": puzzle.id, "condition": "main",
                          "phase": "revise", "iter": it, "k": m,
                          "parent_score": round(score(verifiers, cand), 3)},
                )
                if child is not None:
                    pool.append((child, score(verifiers, child)))

        best = max(s for _, s in pool)
        trajectory.append(best)

    best_cand, best_score = max(pool, key=lambda cs: cs[1], default=(None, 0.0))
    return Result(
        puzzle_id=puzzle.id,
        solved=matches_gold(best_cand, puzzle),  # ground truth, not s=1.0
        best_score=best_score,
        best_candidate=best_cand,
        trajectory=trajectory,
        condition="main",
        extra={"pool_size": len(pool), "iterations": it,
               "n_verifiers": len(verifiers)},
    )

# End-to-end search on 3x3
if __name__ == "__main__":
    from src.common.verifiers import build_verifiers
    from utils.data import load_smoke_set

    llm = LLM()
    p = load_smoke_set()[1]  # 3x3
    print(f"Puzzle {p.id} (K={p.k})")
    vs = build_verifiers(llm, p)
    print(f"verifiers: {sum(v.passed_gold for v in vs)}/{p.k} pass gold")
    res = beam_search(llm, p, vs, steps=4, beam_w=2, revisions=2)
    print(f"solved={res.solved} best_score={res.best_score:.2f} "
          f"traj={[round(t,2) for t in res.trajectory]}")
    print("trace:", llm.trace_path)
