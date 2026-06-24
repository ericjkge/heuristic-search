"""
Verifier-guided quality-diversity search for AIME/HMMT math problems.

Algorithm:
1. Build verifier set (8-dimension stack in v1)
2. Seed num_seeds candidates in parallel
3. Score all with verifiers (parallel)
4. For each step:
   a. Early stop: aggregate_score >= 0.99 AND valid extracted answer
   b. select_expansion + select_combination
   c. Generate children (parallel wave)
   d. Score children (parallel wave)
   e. truncate_population to population_size
5. Final selection: confidence_first | best_verifier | majority_vote
6. Compute solved = answers_match(predicted, ground_truth) — only here
"""

from __future__ import annotations

import random
from collections import Counter

from src.math.common.answer import answers_match, validate_aime_answer
from src.math.common.candidates import combine, generate, revise
from src.math.common.result import MathCandidate, MathResult
from src.math.common.verifiers import (
    aggregate_score,
    build_verifiers,
    get_failed_verifiers,
    score_candidate,
)
from src.math.data import MathProblem
from src.math.method.selection import select_combination, select_expansion, truncate_population
from utils.concurrency import run_parallel
from utils.llm import LLM


def _score_candidates(
    candidates: list[MathCandidate],
    problem: MathProblem,
    llm: LLM,
    verifier_names: list[str],
    tags_base: dict,
) -> list[MathCandidate]:
    """Score a batch of candidates in parallel; mutates verifier_scores in place."""
    def _score_one(c: MathCandidate) -> MathCandidate:
        c.verifier_scores = score_candidate(
            c.solution_text, c.extracted_answer, problem, llm, verifier_names,
            tags={**tags_base, "candidate_id": c.candidate_id, "phase": "score"},
        )
        return c

    thunks = [lambda c=c: _score_one(c) for c in candidates]
    return run_parallel(thunks)


def _confidence_score(c: MathCandidate) -> float:
    """Primary selection key: correctness_confidence score, fallback to 0.0."""
    vr = c.verifier_scores.get("correctness_confidence")
    return vr.score if vr is not None else 0.0


def _select_final(population: list[MathCandidate], mode: str) -> MathCandidate | None:
    if not population:
        return None
    if mode == "confidence_first":
        # Sort by correctness_confidence descending, break ties by aggregate_score.
        valid = [c for c in population if c.extracted_answer is not None]
        pool = valid if valid else population
        return max(pool, key=lambda c: (_confidence_score(c), aggregate_score(c.verifier_scores)))
    if mode == "majority_vote":
        valid = [c for c in population if c.extracted_answer is not None]
        if not valid:
            return max(population, key=lambda c: aggregate_score(c.verifier_scores))
        counts = Counter(c.extracted_answer for c in valid)
        majority_answer = counts.most_common(1)[0][0]
        candidates_with_majority = [c for c in valid if c.extracted_answer == majority_answer]
        return max(candidates_with_majority, key=lambda c: aggregate_score(c.verifier_scores))
    # default: best_verifier
    return max(population, key=lambda c: aggregate_score(c.verifier_scores))


def math_qd_search(
    llm: LLM,
    problem: MathProblem,
    num_seeds: int = 8,
    num_steps: int = 3,
    num_expansions: int = 4,
    num_combinations: int = 2,
    population_size: int = 8,
    quality_weight: float = 1.0,
    diversity_weight: float = 0.3,
    final_selection: str = "confidence_first",
    seed: int = 0,
) -> MathResult:
    rng = random.Random(seed)
    _ = rng  # reserved for future stochastic tie-breaking

    verifier_names = build_verifiers(problem, llm)
    tags_base = {"problem_id": problem.id, "condition": "verifier_guided_search"}

    # --- INIT: seed population ---
    seed_thunks = [
        lambda i=i: generate(
            llm, problem,
            tags={**tags_base, "phase": "seed", "k": i},
            k=i, depth=0,
        )
        for i in range(num_seeds)
    ]
    seeds_raw = run_parallel(seed_thunks)
    seeds = [c for c in seeds_raw if c is not None]
    if not seeds:
        return MathResult(
            problem_id=problem.id, source=problem.source, mode="verifier_guided_search",
            solved=False, ground_truth_answer=problem.answer, predicted_answer=None,
            selected_candidate=None, all_candidates=[],
            model_calls=0, total_tokens=0, best_verifier_score=0.0,
            trajectory=[], early_stop=False,
        )

    population = _score_candidates(seeds, problem, llm, verifier_names, tags_base)
    all_candidates: list[MathCandidate] = list(population)
    trajectory: list[float] = [max(aggregate_score(c.verifier_scores) for c in population)]
    early_stop = False

    # --- SEARCH LOOP ---
    for step in range(1, num_steps + 1):
        # Early stop: verifier-perfect AND valid format — never checks ground truth
        best_score = max(aggregate_score(c.verifier_scores) for c in population)
        best_with_answer = next(
            (c for c in sorted(population, key=lambda c: aggregate_score(c.verifier_scores), reverse=True)
             if validate_aime_answer(c.extracted_answer)),
            None,
        )
        if best_score >= 0.99 and best_with_answer is not None:
            early_stop = True
            break

        # --- SELECT ---
        expand_idx = select_expansion(
            population, num_expansions, verifier_names, quality_weight, diversity_weight,
        )
        combine_pairs = select_combination(population, num_combinations, verifier_names)

        # --- EXPAND: revise selected candidates ---
        expand_thunks = []
        for k_idx, idx in enumerate(expand_idx):
            cand = population[idx]
            failed = get_failed_verifiers(cand.verifier_scores)
            if not failed:
                continue
            expand_thunks.append(
                lambda cand=cand, failed=failed, k_idx=k_idx: revise(
                    llm, problem, cand, failed,
                    tags={**tags_base, "phase": "expand", "step": step, "k": k_idx},
                    depth=step, k=k_idx,
                )
            )

        # --- COMBINE: merge complementary pairs ---
        combine_thunks = []
        for k_idx, (i, j) in enumerate(combine_pairs):
            ca, cb = population[i], population[j]
            combine_thunks.append(
                lambda ca=ca, cb=cb, k_idx=k_idx: combine(
                    llm, problem, ca, cb,
                    tags={**tags_base, "phase": "combine", "step": step, "k": k_idx},
                    depth=step, k=k_idx,
                )
            )

        children_raw = run_parallel(expand_thunks + combine_thunks)
        children = [c for c in children_raw if c is not None]

        if children:
            children = _score_candidates(children, problem, llm, verifier_names, tags_base)
            all_candidates.extend(children)
            population = truncate_population(
                population + children, population_size, verifier_names, diversity_weight,
            )

        best_score = max(aggregate_score(c.verifier_scores) for c in population)
        trajectory.append(best_score)

    # --- FINAL SELECTION ---
    selected = _select_final(population, final_selection)
    predicted_answer = selected.extracted_answer if selected else None
    solved = answers_match(predicted_answer, problem.answer)

    total_tokens = sum(
        c.generation_metadata.get("prompt_tokens", 0) + c.generation_metadata.get("completion_tokens", 0)
        for c in all_candidates
    )
    best_score = max(aggregate_score(c.verifier_scores) for c in all_candidates) if all_candidates else 0.0

    return MathResult(
        problem_id=problem.id,
        source=problem.source,
        mode="verifier_guided_search",
        solved=solved,
        ground_truth_answer=problem.answer,
        predicted_answer=predicted_answer,
        selected_candidate=selected,
        all_candidates=all_candidates,
        model_calls=len(all_candidates),
        total_tokens=total_tokens,
        best_verifier_score=best_score,
        trajectory=trajectory,
        early_stop=early_stop,
        extra={
            "num_steps_taken": len(trajectory) - 1,
            "population_size_final": len(population),
            "verifier_names": verifier_names,
        },
    )
