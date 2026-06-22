"""Quality-diversity (QD) search — the Phase 2 method.

The population is selected by quality + verifier-coverage
diversity, and each step both EXPANDS chosen candidates (revise with failed-clue
feedback) and COMBINES complementary pairs into new candidates. Runs up to
`num_steps`, early-stopping when a candidate is verifier-perfect. The population
only grows (no pruning).
"""

from src.common.candidates import combine, generate, matches_gold, revise
from src.common.result import Result
from src.common.verifiers import Verifier, failed_clues, verifier_scores
from utils.data import Puzzle
from utils.llm import LLM


# --- selection primitives: pure functions over verifier-score vectors (no LLM) ---

def _quality(scores: list[float]) -> float:
    """Mean verifier score = #passed / m."""
    return sum(scores) / len(scores) if scores else 0.0


def _satisfied(scores: list[float]) -> set[int]:
    """Indices of verifiers this candidate fully satisfies."""
    return {i for i, v in enumerate(scores) if v >= 1.0}


def select_expansion(score_vectors: list[list[float]], num_expansions: int,
                     quality_weight: float = 1.0, diversity_weight: float = 0.5) -> list[int]:
    """Greedily pick `num_expansions` indices maximizing quality + marginal coverage.

    Each pick maximizes `quality_weight * total_score + diversity_weight * new_coverage`,
    both in verifier-count units: total_score = sum(scores); new_coverage = # verifiers it
    satisfies that no already-selected candidate does. So default weights value a newly
    covered verifier as half a satisfied one; raise diversity_weight for more spread.
    """
    if not score_vectors:
        return []
    remaining = set(range(len(score_vectors)))
    covered: set[int] = set()
    selected: list[int] = []
    while remaining and len(selected) < num_expansions:
        best = max(remaining, key=lambda i: (
            quality_weight * sum(score_vectors[i])
            + diversity_weight * len(_satisfied(score_vectors[i]) - covered)
        ))
        selected.append(best)
        covered |= _satisfied(score_vectors[best])
        remaining.discard(best)
    return selected


def select_combination(score_vectors: list[list[float]],
                       num_combinations: int) -> list[tuple[int, int]]:
    """Pick the `num_combinations` most complementary index pairs.

    Pair value = |sat(a) ∪ sat(b)| - max(|sat(a)|, |sat(b)|): how many more
    verifiers the pair *could* jointly cover than the better parent alone.
    """
    scored = []
    for a in range(len(score_vectors)):
        sat_a = _satisfied(score_vectors[a])
        for b in range(a + 1, len(score_vectors)):
            sat_b = _satisfied(score_vectors[b])
            gain = len(sat_a | sat_b) - max(len(sat_a), len(sat_b))
            scored.append((gain, a, b))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))  # gain desc, then deterministic
    return [(a, b) for _, a, b in scored[:num_combinations]]


def qd_search(
    llm: LLM,
    puzzle: Puzzle,
    verifiers: list[Verifier],
    num_seeds: int = 2,
    num_steps: int = 4,
    num_expansions: int = 2,
    num_combinations: int = 2,
    quality_weight: float = 1.0,
    diversity_weight: float = 0.5,
) -> Result:
    """QD search over candidate grids (up to num_steps; early-stop on score == 1)."""
    verifiers = [v for v in verifiers if v.passed_gold]

    # INIT: num_seeds independent attempts. population: list of (candidate, scores).
    population: list[tuple[dict, list[float]]] = []
    for i in range(num_seeds):
        cand = generate(
            llm, puzzle, tags={"puzzle_id": puzzle.id, "condition": "qd",
                               "phase": "init", "iter": 0, "k": i}
        )
        if cand is not None:
            population.append((cand, verifier_scores(verifiers, cand)))

    # Best score at each step (for analysis/debugging)
    trajectory = []
    best = max((_quality(sc) for _, sc in population), default=0.0)
    trajectory.append(best)

    step = 0
    while step < num_steps and best < 1.0 and population:
        step += 1
        score_vectors = [sc for _, sc in population]
        expand_idx = select_expansion(score_vectors, num_expansions,
                                      quality_weight, diversity_weight)
        combine_idx = select_combination(score_vectors, num_combinations)

        children = []
        # EXPAND: revise each selected candidate with its failed clues.
        for i in expand_idx:
            cand, scores = population[i]
            failed = failed_clues(verifiers, cand)
            if not failed:
                continue
            child = revise(
                llm, puzzle, cand, failed,
                tags={"puzzle_id": puzzle.id, "condition": "qd", "phase": "expand",
                      "iter": step, "k": i, "parent_score": round(_quality(scores), 3)},
            )
            if child is not None:
                children.append(child)
        # COMBINE: recombine each selected complementary pair.
        for k, (a, b) in enumerate(combine_idx):
            child = combine(
                llm, puzzle, verifiers, population[a][0], population[b][0],
                tags={"puzzle_id": puzzle.id, "condition": "qd", "phase": "combine",
                      "iter": step, "k": k, "parents": [a, b]},
            )
            if child is not None:
                children.append(child)

        for child in children:
            population.append((child, verifier_scores(verifiers, child)))

        best = max(_quality(sc) for _, sc in population)
        trajectory.append(best)

    best_cand, best_scores = max(population, key=lambda cs: _quality(cs[1]),
                                 default=(None, []))
    return Result(
        puzzle_id=puzzle.id,
        solved=matches_gold(best_cand, puzzle),  # ground truth, not score == 1.0
        best_score=_quality(best_scores),
        best_candidate=best_cand,
        trajectory=trajectory,
        condition="qd",
        extra={"pool_size": len(population), "iterations": step,
               "n_verifiers": len(verifiers)},
    )


if __name__ == "__main__":
    from src.common.verifiers import build_verifiers
    from utils.data import load_smoke_set

    llm = LLM()
    p = load_smoke_set()[1]  # 3x3
    print(f"Puzzle {p.id} (K={p.k})")
    vs = build_verifiers(llm, p)
    res = qd_search(llm, p, vs, num_seeds=2, num_steps=3,
                    num_expansions=2, num_combinations=1)
    print(f"solved={res.solved} best_score={res.best_score:.2f} "
          f"traj={[round(t, 2) for t in res.trajectory]} extra={res.extra}")
    print("trace:", llm.trace_path)
