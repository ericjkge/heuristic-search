"""Verifier-guided beam search — the Phase 1 method.

Treats solving as search over candidate grids, using the verifier pass-rate
score(cand) = #{checkers passed}/K as the heuristic. Selects the best candidates,
gives them their verbatim failed clues as feedback, and revises. Runs up to
`num_steps` iterations, early-stopping when a candidate is verifier-perfect.
"""

import random

from src.common.candidates import generate, matches_gold, revise
from src.common.result import Result
from src.common.verifiers import Verifier, failed_clues, score
from utils.concurrency import run_parallel
from utils.data import Puzzle
from utils.llm import LLM


def beam_search(
    llm: LLM,
    puzzle: Puzzle,
    verifiers: list[Verifier],
    num_seeds: int = 2,
    num_steps: int = 4,
    beam_width: int = 2,
    branching: int = 2,
    seed: int = 0,
) -> Result:
    """Verifier-guided beam search (up to num_steps; early-stop on score == 1)."""
    rng = random.Random(seed)
    # Drop checkers that fail gold: a buggy verifier would make score = 1.0
    # unreachable even for the correct answer.
    verifiers = [v for v in verifiers if v.passed_gold]

    # INIT: num_seeds independent full attempts (fired concurrently).
    seeds = run_parallel([
        lambda i=i: generate(
            llm, puzzle, tags={"puzzle_id": puzzle.id, "condition": "beam",
                               "phase": "init", "iter": 0, "k": i}
        )
        for i in range(num_seeds)
    ])
    pool = [(cand, score(verifiers, cand)) for cand in seeds if cand is not None]

    trajectory = []
    best = max((s for _, s in pool), default=0.0)
    trajectory.append(best)

    it = 0
    while it < num_steps and best < 1.0 and pool:
        it += 1
        ranked = sorted(pool, key=lambda cs: cs[1], reverse=True)
        # SELECT: top beam_width parents, plus one random low-scorer (explore).
        parents = ranked[:beam_width]
        tail = ranked[beam_width:]
        if tail:
            parents = parents + [rng.choice(tail)]

        for cand, _ in parents:
            failed = failed_clues(verifiers, cand)
            if not failed:
                continue
            for k in range(branching):
                child = revise(
                    llm, puzzle, cand, failed,
                    tags={"puzzle_id": puzzle.id, "condition": "beam",
                          "phase": "revise", "iter": it, "k": k,
                          "parent_score": round(score(verifiers, cand), 3)},
                )
                if child is not None:
                    pool.append((child, score(verifiers, child)))

        best = max(s for _, s in pool)
        trajectory.append(best)

    best_cand, best_score = max(pool, key=lambda cs: cs[1], default=(None, 0.0))
    return Result(
        puzzle_id=puzzle.id,
        solved=matches_gold(best_cand, puzzle),  # ground truth, not score == 1.0
        best_score=best_score,
        best_candidate=best_cand,
        trajectory=trajectory,
        condition="beam",
        extra={"pool_size": len(pool), "iterations": it,
               "n_verifiers": len(verifiers)},
    )


if __name__ == "__main__":
    from src.common.verifiers import build_verifiers
    from utils.data import load_smoke_set

    llm = LLM()
    p = load_smoke_set()[1]  # 3x3
    print(f"Puzzle {p.id} (K={p.k})")
    vs = build_verifiers(llm, p)
    print(f"verifiers: {sum(v.passed_gold for v in vs)}/{p.k} pass gold")
    res = beam_search(llm, p, vs, num_seeds=2, num_steps=4, beam_width=2, branching=2)
    print(f"solved={res.solved} best_score={res.best_score:.2f} "
          f"traj={[round(t, 2) for t in res.trajectory]}")
    print("trace:", llm.trace_path)
