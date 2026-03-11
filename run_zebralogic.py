"""Entry point: Generator-Verifier ZebraLogic solver.

Usage: python3 run_zebralogic.py [--baseline] [--max_rounds 5] [--dataset data/zebralogic/puzzles_2x2.json]
"""

import argparse
import json
import time

from utils.llm import GeminiLLM
from utils.logging import setup_logging, get_logger
from tasks.zebralogic import parse_solution, check_solution
from agents.generator import GeneratorAgent
from agents.verifier import VerifierAgent
from prompts.zebralogic.generator import initial_prompt, revision_prompt
from prompts.zebralogic.verifier import (
    bijection_prompt,
    equality_prompt,
    positional_prompt,
)

logger = get_logger(__name__)


def solve_baseline(
    puzzle_text: str,
    solution: dict,
    gen_llm: GeminiLLM,
) -> dict:
    """Baseline: single generator call, no verification loop."""
    generator = GeneratorAgent(gen_llm)
    prompt = initial_prompt(puzzle_text)
    _, raw = generator.generate(prompt)

    candidate = parse_solution(raw)
    solved = candidate is not None and check_solution(candidate, solution)

    return {
        "solved": solved,
        "rounds": 1,
        "total_tokens": generator.total_tokens,
    }


def solve_puzzle(
    puzzle_text: str,
    solution: dict,
    max_rounds: int,
    gen_llm: GeminiLLM,
    ver_llms: list[GeminiLLM],
) -> dict:
    """Run generator-verifier loop on a single puzzle."""
    generator = GeneratorAgent(gen_llm)

    # Build 3 verifiers with closure-based prompt_fn
    verifiers = [
        VerifierAgent(
            "bijection",
            ver_llms[0],
            lambda tbl: bijection_prompt(tbl, puzzle_text),
        ),
        VerifierAgent(
            "equality",
            ver_llms[1],
            lambda tbl: equality_prompt(tbl, puzzle_text),
        ),
        VerifierAgent(
            "positional",
            ver_llms[2],
            lambda tbl: positional_prompt(tbl, puzzle_text),
        ),
    ]

    solved = False
    rounds_used = 0

    # Initial generation
    prompt = initial_prompt(puzzle_text)
    _, raw = generator.generate(prompt)
    rounds_used = 1

    for round_num in range(1, max_rounds + 1):
        candidate = parse_solution(raw)

        if candidate is None:
            feedback = (
                "ERROR: Could not parse your solution. Output a JSON object with "
                '"header" and "rows" keys inside <OUTPUT> tags.'
            )
            _, raw = generator.generate(
                revision_prompt(puzzle_text, raw, feedback)
            )
            rounds_used = round_num + 1 if round_num < max_rounds else rounds_used
            continue

        # Programmatic check against ground truth
        if check_solution(candidate, solution):
            solved = True
            break

        # LLM verifiers — pass only the JSON solution, not the full reasoning
        candidate_str = json.dumps(candidate)
        feedback_parts: list[str] = []

        for verifier in verifiers:
            errors = verifier.verify(candidate_str)
            if errors is not None:
                feedback_parts.append(f"[{verifier.name} verifier]:\n{errors}")

        feedback = "\n\n".join(feedback_parts) if feedback_parts else (
            "The solution is incorrect. Re-examine all clues carefully."
        )

        if round_num < max_rounds:
            _, raw = generator.generate(
                revision_prompt(puzzle_text, candidate_str, feedback)
            )
        rounds_used = round_num

    total_tokens = generator.total_tokens + sum(v.total_tokens for v in verifiers)

    return {
        "solved": solved,
        "rounds": rounds_used,
        "total_tokens": total_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generator-Verifier ZebraLogic Solver")
    parser.add_argument("--max_rounds", type=int, default=5)
    parser.add_argument("--baseline", action="store_true", help="Single-shot baseline")
    parser.add_argument("--log", action="store_true", help="Enable file logging")
    parser.add_argument("--dataset", type=str, default="data/zebralogic/puzzles_2x2.json")
    parser.add_argument("--puzzle_ids", type=str, nargs="+", help="Specific puzzle IDs to run")
    args = parser.parse_args()

    mode = "baseline" if args.baseline else "gen-ver"
    suffix = f"zebralogic_{mode}"

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
        puzzle_text = entry["puzzle"]
        solution = entry["solution"]
        print(f"Puzzle {puzzle_id}...", end=" ", flush=True)

        if args.baseline:
            result = solve_baseline(puzzle_text, solution, gen_llm)
        else:
            result = solve_puzzle(
                puzzle_text, solution, args.max_rounds, gen_llm, ver_llms
            )

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
