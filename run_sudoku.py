"""Entry point: Generator-Verifier Sudoku solver.

Usage: python run_sudoku.py [--max_rounds 5] [--log] [--dataset data/sudoku/puzzles.json]
"""

import argparse
import json
import time

from utils.llm import GeminiLLM
from utils.logging import setup_logging
from tasks.sudoku import (
    Grid, grid_to_str, validate_givens, check_solution,
)
from agents.generator import GeneratorAgent
from agents.verifier import VerifierAgent
from agents.code_verifier import CodeVerifierAgent
from prompts.sudoku.generator import initial_prompt, revision_prompt
from prompts.sudoku.verifier import row_prompt, column_prompt, box_prompt
from prompts.sudoku.code_verifier import (
    row_code_gen_prompt, column_code_gen_prompt, box_code_gen_prompt,
    row_interpret_prompt, column_interpret_prompt, box_interpret_prompt,
)
try:
    from prompts.sudoku.debate import propose_prompt, critique_prompt, vote_prompt
    from agents.debate import DebateAgent
except ImportError:
    pass

try:
    from agents.tot import solve_tot
except ImportError:
    pass


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
        feedback_parts = []

        for verifier in verifiers:
            errors = verifier.verify(grid_str)
            if errors is not None:
                feedback_parts.append(f"[{verifier.name} verifier]:\n{errors}")

        # Aggregate feedback and revise
        feedback = "\n\n".join(feedback_parts) if feedback_parts else (
            "The solution is incorrect. Double-check all rows, columns, and 3x3 boxes for duplicates."
        )
        if round_num < max_rounds:
            grid, raw = generator.generate(revision_prompt(puzzle_str, grid_str, feedback))
        rounds_used = round_num

    total_tokens = generator.total_tokens + sum(v.total_tokens for v in verifiers)

    return {
        "solved": solved,
        "rounds": rounds_used,
        "total_tokens": total_tokens,
    }


def solve_puzzle_code_verifier(
    puzzle: list[list[int]],
    max_rounds: int,
    gen_llm: GeminiLLM,
    ver_llms: list[GeminiLLM],
) -> dict:
    """Run generator–code-verifier loop on a single puzzle."""
    generator = GeneratorAgent(gen_llm)
    verifiers = [
        CodeVerifierAgent("row", ver_llms[0], row_code_gen_prompt, row_interpret_prompt),
        CodeVerifierAgent("column", ver_llms[1], column_code_gen_prompt, column_interpret_prompt),
        CodeVerifierAgent("box", ver_llms[2], box_code_gen_prompt, box_interpret_prompt),
    ]

    # Generate checking code upfront
    for v in verifiers:
        v.generate_code()

    puzzle_str = grid_to_str(puzzle)
    solved = False
    rounds_used = 0

    # Initial generation
    prompt = initial_prompt(puzzle_str)
    grid, raw = generator.generate(prompt)
    rounds_used = 1

    for round_num in range(1, max_rounds + 1):
        if grid is None:
            feedback = "ERROR: Could not parse your grid. Output EXACTLY 9 lines of 9 space-separated digits."
            grid, raw = generator.generate(revision_prompt(puzzle_str, raw, feedback))
            rounds_used = round_num + 1 if round_num < max_rounds else rounds_used
            continue

        # Programmatic correctness check
        if check_solution(grid):
            solved = True
            break

        # Code-writing verifiers
        all_correct = True
        feedback_parts: list[str] = []

        for verifier in verifiers:
            is_correct, response = verifier.verify(grid)
            if not is_correct:
                all_correct = False
                feedback_parts.append(f"[{verifier.name} verifier]:\n{response}")

        if all_correct:
            solved = True
            break

        feedback = "\n\n".join(feedback_parts)
        if round_num < max_rounds:
            grid_str = grid_to_str(grid)
            grid, raw = generator.generate(revision_prompt(puzzle_str, grid_str, feedback))
        rounds_used = round_num

    total_tokens = generator.total_tokens + sum(v.total_tokens for v in verifiers)
    return {
        "solved": solved,
        "rounds": rounds_used,
        "total_tokens": total_tokens,
    }


def solve_debate(
    puzzle: list[list[int]],
    debate_llms: list[GeminiLLM],
    critique_rounds: int = 2,
) -> dict:
    """Debate baseline: 3 agents propose, critique, and vote."""
    puzzle_str = grid_to_str(puzzle)

    agents = [
        DebateAgent(f"agent_{i}", debate_llms[i]) for i in range(3)
    ]

    # 1. Propose
    grids: list[Grid | None] = []
    raws: list[str] = []
    for agent in agents:
        grid, raw = agent.propose(propose_prompt(puzzle_str))
        grids.append(grid)
        raws.append(raw)

    # 2. Critique rounds
    for _ in range(critique_rounds):
        new_grids: list[Grid | None] = []
        new_raws: list[str] = []
        for i, agent in enumerate(agents):
            other_raws = [raws[j] for j in range(3) if j != i]
            prompt = critique_prompt(puzzle_str, raws[i], other_raws)
            grid, raw = agent.critique(prompt)
            new_grids.append(grid)
            new_raws.append(raw)
        grids = new_grids
        raws = new_raws

    # 3. Vote
    votes: list[int] = []
    for agent in agents:
        choice = agent.vote(vote_prompt(puzzle_str, raws))
        votes.append(choice)

    # Majority vote; ties broken by first agent to get a vote
    vote_counts: dict[int, int] = {}
    for v in votes:
        vote_counts[v] = vote_counts.get(v, 0) + 1
    winner = max(vote_counts, key=lambda k: vote_counts[k])

    final_grid = grids[winner]
    solved = False
    if final_grid is not None:
        givens_ok = len(validate_givens(puzzle, final_grid)) == 0
        correct = check_solution(final_grid)
        solved = givens_ok and correct

    total_tokens = sum(a.total_tokens for a in agents)
    # rounds = 1 propose + critique_rounds + 1 vote
    rounds_used = 1 + critique_rounds + 1

    return {
        "solved": solved,
        "rounds": rounds_used,
        "total_tokens": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Generator-Verifier Sudoku Solver")
    parser.add_argument("--max_rounds", type=int, default=5)
    parser.add_argument("--baseline", action="store_true", help="Single-shot baseline (no verifiers)")
    parser.add_argument("--debate", action="store_true", help="Debate baseline (3 agents debate)")
    parser.add_argument("--tot", action="store_true", help="Tree-of-Thought DFS with backtracking")
    parser.add_argument("--code_verifier", action="store_true", help="Use code-writing verifiers")
    parser.add_argument("--log", action="store_true", help="Enable file logging")
    parser.add_argument("--dataset", type=str, default="data/sudoku/puzzles.json")
    parser.add_argument("--puzzle_ids", type=int, nargs="+", help="Specific puzzle IDs to run")
    args = parser.parse_args()

    if args.baseline:
        mode = "baseline"
    elif args.debate:
        mode = "debate"
    elif args.tot:
        mode = "tot"
    elif args.code_verifier:
        mode = "code-ver"
    else:
        mode = "gen-ver"
    suffix = f"sudoku_{mode}"

    if args.log:
        setup_logging(log_dir="logs", suffix=suffix)

    with open(args.dataset) as f:
        dataset = json.load(f)

    if args.puzzle_ids is not None:
        dataset = [e for e in dataset if e["id"] in args.puzzle_ids]

    print(f"Mode: {mode}")
    print(f"Loaded {len(dataset)} puzzles from {args.dataset}")
    if not args.baseline and not args.debate:
        max_rounds = args.max_rounds if not args.tot else min(args.max_rounds, 30)
        print(f"Max rounds: {max_rounds}")
    print()

    gen_llm = GeminiLLM()
    if args.debate:
        debate_llms = [GeminiLLM() for _ in range(3)]
        ver_llms: list[GeminiLLM] = []
    elif args.baseline or args.tot:
        debate_llms: list[GeminiLLM] = []
        ver_llms = []
    else:
        debate_llms = []
        ver_llms = [GeminiLLM() for _ in range(3)]

    results = []
    start_time = time.time()

    for entry in dataset:
        puzzle_id = entry["id"]
        puzzle = entry["puzzle"]
        print(f"Puzzle {puzzle_id}...", end=" ", flush=True)

        if args.baseline:
            result = solve_baseline(puzzle, gen_llm)
        elif args.debate:
            result = solve_debate(puzzle, debate_llms)
        elif args.tot:
            result = solve_tot(puzzle, gen_llm, args.max_rounds)
        elif args.code_verifier:
            result = solve_puzzle_code_verifier(puzzle, args.max_rounds, gen_llm, ver_llms)
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
