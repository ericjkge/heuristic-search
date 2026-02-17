"""Entry point: Generator-Verifier Sudoku solver.

Usage: python run_sudoku.py [--max_rounds 5] [--log] [--dataset data/sudoku/puzzles.json]
"""

import argparse
import json
import time

from utils.llm import GeminiLLM
from utils.logging import setup_logging
from tasks.sudoku import (
    grid_to_str, validate_givens, check_solution,
)
from agents.generator import GeneratorAgent
from agents.verifier import VerifierAgent
from prompts.sudoku.generator import initial_prompt, revision_prompt
from prompts.sudoku.verifier import row_prompt, column_prompt, box_prompt


def solve_baseline(
    puzzle: list[list[int]],
    gen_llm: GeminiLLM,
) -> dict:
    """Baseline: single generator call, no verification loop."""
    generator = GeneratorAgent(gen_llm)
    puzzle_str = grid_to_str(puzzle)
    prompt = initial_prompt(puzzle_str)
    grid, raw = generator.generate(prompt)

    solved = False
    if grid is not None:
        givens_ok = len(validate_givens(puzzle, grid)) == 0
        correct = check_solution(grid)
        solved = givens_ok and correct

    return {
        "solved": solved,
        "rounds": 1,
        "total_tokens": generator.total_tokens,
    }


def solve_puzzle(
    puzzle: list[list[int]],
    max_rounds: int,
    gen_llm: GeminiLLM,
    ver_llms: list[GeminiLLM],
) -> dict:
    """Run generator-verifier loop on a single puzzle. Returns result dict."""
    generator = GeneratorAgent(gen_llm)
    verifiers = [
        VerifierAgent("row", ver_llms[0], row_prompt),
        VerifierAgent("column", ver_llms[1], column_prompt),
        VerifierAgent("box", ver_llms[2], box_prompt),
    ]

    puzzle_str = grid_to_str(puzzle)
    solved = False
    rounds_used = 0

    # Initial generation
    prompt = initial_prompt(puzzle_str)
    grid, raw = generator.generate(prompt)
    rounds_used = 1

    for round_num in range(1, max_rounds + 1):
        # Parse check
        if grid is None:
            feedback = "ERROR: Could not parse your grid. Output EXACTLY 9 lines of 9 space-separated digits."
            grid, raw = generator.generate(revision_prompt(puzzle_str, raw, feedback))
            rounds_used = round_num + 1 if round_num < max_rounds else rounds_used
            continue

        # Programmatic correctness check
        if check_solution(grid):
            solved = True
            break

        # Tier 2: LLM verifiers (generate natural language feedback)
        grid_str = grid_to_str(grid)
        all_correct = True
        feedback_parts = []

        for verifier in verifiers:
            is_correct, response = verifier.verify(grid_str)
            if not is_correct:
                all_correct = False
                feedback_parts.append(f"[{verifier.name} verifier]:\n{response}")

        if all_correct:
            solved = True
            break

        # Aggregate feedback and revise
        feedback = "\n\n".join(feedback_parts)
        if round_num < max_rounds:
            grid, raw = generator.generate(revision_prompt(puzzle_str, grid_str, feedback))
        rounds_used = round_num

    total_tokens = generator.total_tokens + sum(v.total_tokens for v in verifiers)

    return {
        "solved": solved,
        "rounds": rounds_used,
        "total_tokens": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Generator-Verifier Sudoku Solver")
    parser.add_argument("--max_rounds", type=int, default=5)
    parser.add_argument("--baseline", action="store_true", help="Single-shot baseline (no verifiers)")
    parser.add_argument("--log", action="store_true", help="Enable file logging")
    parser.add_argument("--dataset", type=str, default="data/sudoku/puzzles.json")
    parser.add_argument("--puzzle_ids", type=int, nargs="+", help="Specific puzzle IDs to run")
    args = parser.parse_args()

    mode = "baseline" if args.baseline else "gen-ver"
    suffix = f"sudoku_{mode}"

    if args.log:
        setup_logging(log_dir="logs", suffix=suffix)

    with open(args.dataset) as f:
        dataset = json.load(f)

    if args.puzzle_ids is not None:
        dataset = [e for e in dataset if e["id"] in args.puzzle_ids]

    print(f"Mode: {mode}")
    print(f"Loaded {len(dataset)} puzzles from {args.dataset}")
    if not args.baseline:
        print(f"Max rounds: {args.max_rounds}")
    print()

    gen_llm = GeminiLLM()
    ver_llms = [GeminiLLM() for _ in range(3)] if not args.baseline else []

    results = []
    start_time = time.time()

    for entry in dataset:
        puzzle_id = entry["id"]
        puzzle = entry["puzzle"]
        print(f"Puzzle {puzzle_id}...", end=" ", flush=True)

        if args.baseline:
            result = solve_baseline(puzzle, gen_llm)
        else:
            result = solve_puzzle(puzzle, args.max_rounds, gen_llm, ver_llms)
        result["id"] = puzzle_id
        results.append(result)

        status = "SOLVED" if result["solved"] else "FAILED"
        print(f"{status} in {result['rounds']} rounds ({result['total_tokens']} tokens)")

    elapsed = time.time() - start_time

    # Summary
    solved_count = sum(1 for r in results if r["solved"])
    avg_rounds = sum(r["rounds"] for r in results) / len(results) if results else 0
    total_tokens = sum(r["total_tokens"] for r in results)

    print(f"\n{'='*40}")
    print(f"Solve rate: {solved_count}/{len(results)} ({100*solved_count/len(results):.0f}%)")
    print(f"Avg rounds: {avg_rounds:.1f}")
    print(f"Total tokens: {total_tokens}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
