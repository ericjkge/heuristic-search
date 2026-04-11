"""Entry point: Generator-Verifier Sudoku solver.

Usage: python run_sudoku.py [--max_rounds 5] [--log] [--dataset data/sudoku/puzzles.json]
"""

import argparse
import json
import time
from pathlib import Path

from utils.llm import HarvardGPTLLM, QwenLLM
from utils.logging import setup_logging, write_trace
from tasks.sudoku import (
    Grid, grid_to_str, str_to_grid, validate_givens, check_solution,
)
from agents.generator import GeneratorAgent
from agents.verifier import VerifierAgent
from agents.debate import DebateAgent
from prompts.sudoku.generator import initial_prompt, revision_prompt
from prompts.sudoku.verifier import row_prompt, column_prompt, box_prompt
from prompts.sudoku.cot import cot_prompt
from prompts.sudoku.self_refine import self_critique_prompt
from prompts.sudoku.debate import propose_prompt, critique_prompt


def make_llm(model: str) -> HarvardGPTLLM | QwenLLM:
    if model == "gpt-none":
        return HarvardGPTLLM(reasoning_effort="none")
    if model == "gpt-high":
        return HarvardGPTLLM(reasoning_effort="high")
    if model == "qwen-none":
        return QwenLLM(thinking=False)
    if model == "qwen-thinking":
        return QwenLLM(thinking=True)
    raise ValueError(f"Unknown model: {model}")


def solve_baseline(
    puzzle: list[list[int]],
    puzzle_id: int,
    gen_llm: HarvardGPTLLM | QwenLLM,
    trace_file: Path | None = None,
) -> dict:
    """Baseline: single generator call, no verification loop."""
    generator = GeneratorAgent(gen_llm)
    puzzle_str = grid_to_str(puzzle)
    prompt = initial_prompt(puzzle_str)

    t0 = time.time()
    grid, raw = generator.generate(prompt)
    if trace_file:
        write_trace(trace_file, {
            "problem_id": str(puzzle_id),
            "round": 0,
            "agent": "gen",
            "output": raw,
            "tokens_in": gen_llm.last_input_tokens,
            "tokens_out": gen_llm.last_output_tokens,
            "time_s": round(time.time() - t0, 3),
        })

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


def solve_cot(
    puzzle: list[list[int]],
    gen_llm: HarvardGPTLLM,
) -> dict:
    """CoT baseline: single call with chain-of-thought prompting."""
    generator = GeneratorAgent(gen_llm)
    puzzle_str = grid_to_str(puzzle)
    prompt = cot_prompt(puzzle_str)
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


def solve_self_consistency(
    puzzle: list[list[int]],
    sc_llms: list[HarvardGPTLLM],
) -> dict:
    """Self-Consistency: K=3 independent solutions, majority vote."""
    puzzle_str = grid_to_str(puzzle)
    prompt = initial_prompt(puzzle_str)

    grids: list[Grid | None] = []
    total_tokens = 0
    for llm in sc_llms:
        gen = GeneratorAgent(llm)
        grid, _ = gen.generate(prompt)
        total_tokens += gen.total_tokens
        grids.append(grid)

    # Majority vote on parsed grids
    best_grid: Grid | None = None
    best_count = 0
    for i, g in enumerate(grids):
        if g is None:
            continue
        count = sum(1 for other in grids if other == g)
        if count > best_count:
            best_count = count
            best_grid = g

    # Fallback: first parseable
    if best_grid is None:
        for g in grids:
            if g is not None:
                best_grid = g
                break

    solved = False
    if best_grid is not None:
        givens_ok = len(validate_givens(puzzle, best_grid)) == 0
        correct = check_solution(best_grid)
        solved = givens_ok and correct

    return {
        "solved": solved,
        "rounds": 1,
        "total_tokens": total_tokens,
    }


def solve_self_refine(
    puzzle: list[list[int]],
    max_rounds: int,
    gen_llm: HarvardGPTLLM,
) -> dict:
    """Self-Refine: generate, self-critique, revise (single LLM)."""
    import re

    generator = GeneratorAgent(gen_llm)
    puzzle_str = grid_to_str(puzzle)

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

        # Self-critique
        grid_str = grid_to_str(grid)
        critique_resp = gen_llm.generate(self_critique_prompt(puzzle_str, grid_str))
        generator.total_tokens += gen_llm.last_tokens

        # Extract feedback from <OUTPUT> tags
        matches = re.findall(r"<OUTPUT>(.*?)</OUTPUT>", critique_resp, re.DOTALL)
        feedback = matches[-1].strip() if matches else ""
        if not feedback:
            feedback = "The solution is incorrect. Double-check all rows, columns, and 3x3 boxes for duplicates."

        # Revise
        if round_num < max_rounds:
            grid, raw = generator.generate(revision_prompt(puzzle_str, grid_str, feedback))
        rounds_used = round_num

    solved = grid is not None and check_solution(grid)
    return {
        "solved": solved,
        "rounds": rounds_used,
        "total_tokens": generator.total_tokens,
    }


def solve_puzzle(
    puzzle: list[list[int]],
    puzzle_id: int,
    max_rounds: int,
    gen_llm: HarvardGPTLLM | QwenLLM,
    ver_llms: list[HarvardGPTLLM | QwenLLM],
    trace_file: Path | None = None,
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

    # Initial generation (round 0)
    prompt = initial_prompt(puzzle_str)
    t0 = time.time()
    grid, raw = generator.generate(prompt)
    if trace_file:
        write_trace(trace_file, {
            "problem_id": str(puzzle_id),
            "round": 0,
            "agent": "gen",
            "output": raw,
            "tokens_in": gen_llm.last_input_tokens,
            "tokens_out": gen_llm.last_output_tokens,
            "time_s": round(time.time() - t0, 3),
        })
    rounds_used = 1

    for round_num in range(1, max_rounds + 1):
        # Parse check
        if grid is None:
            feedback = "ERROR: Could not parse your grid. Output EXACTLY 9 lines of 9 space-separated digits."
            t0 = time.time()
            grid, raw = generator.generate(revision_prompt(puzzle_str, raw, feedback))
            if trace_file:
                write_trace(trace_file, {
                    "problem_id": str(puzzle_id),
                    "round": round_num,
                    "agent": "gen",
                    "output": raw,
                    "tokens_in": gen_llm.last_input_tokens,
                    "tokens_out": gen_llm.last_output_tokens,
                    "time_s": round(time.time() - t0, 3),
                })
            rounds_used = round_num + 1 if round_num < max_rounds else rounds_used
            continue

        # Tier 2: LLM verifiers (generate natural language feedback)
        grid_str = grid_to_str(grid)
        feedback_parts = []

        for verifier in verifiers:
            t0 = time.time()
            errors = verifier.verify(grid_str)
            if trace_file:
                write_trace(trace_file, {
                    "problem_id": str(puzzle_id),
                    "round": round_num,
                    "agent": f"ver_{verifier.name}",
                    "output": errors or "",
                    "tokens_in": verifier.llm.last_input_tokens,
                    "tokens_out": verifier.llm.last_output_tokens,
                    "time_s": round(time.time() - t0, 3),
                })
            if errors is not None:
                feedback_parts.append(f"[{verifier.name} verifier]:\n{errors}")

        feedback = "\n\n".join(feedback_parts)
        t0 = time.time()
        grid, raw = generator.generate(revision_prompt(puzzle_str, grid_str, feedback))
        if trace_file:
            write_trace(trace_file, {
                "problem_id": str(puzzle_id),
                "round": round_num,
                "agent": "gen",
                "output": raw,
                "tokens_in": gen_llm.last_input_tokens,
                "tokens_out": gen_llm.last_output_tokens,
                "time_s": round(time.time() - t0, 3),
            })
        rounds_used = round_num

    # Final correctness check after all rounds
    if grid is not None and check_solution(grid):
        solved = True

    total_tokens = generator.total_tokens + sum(v.total_tokens for v in verifiers)

    return {
        "solved": solved,
        "rounds": rounds_used,
        "total_tokens": total_tokens,
    }



def solve_debate(
    puzzle: list[list[int]],
    debate_llms: list[HarvardGPTLLM],
    critique_rounds: int = 2,
) -> dict:
    """Debate baseline: 3 agents propose, critique, and vote."""
    puzzle_str = grid_to_str(puzzle)

    agents = [
        DebateAgent(f"agent_{i}", debate_llms[i], parse_fn=str_to_grid)
        for i in range(3)
    ]

    # 1. Propose
    grids: list[Grid | None] = []
    raws: list[str] = []
    for agent in agents:
        grid, raw = agent.propose(propose_prompt(puzzle_str))
        grids.append(grid)
        raws.append(raw)

    # 2. Critique rounds
    for cr in range(critique_rounds):
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

    # 3. Majority vote on parsed grids
    best_grid: Grid | None = None
    best_count = 0
    for g in grids:
        if g is None:
            continue
        count = sum(1 for other in grids if other == g)
        if count > best_count:
            best_count = count
            best_grid = g

    if best_grid is None:
        for g in grids:
            if g is not None:
                best_grid = g
                break

    solved = False
    if best_grid is not None:
        givens_ok = len(validate_givens(puzzle, best_grid)) == 0
        correct = check_solution(best_grid)
        solved = givens_ok and correct

    total_tokens = sum(a.total_tokens for a in agents)
    rounds_used = 1 + critique_rounds

    return {
        "solved": solved,
        "rounds": rounds_used,
        "total_tokens": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Generator-Verifier Sudoku Solver")
    parser.add_argument("--max_rounds", type=int, default=5)
    parser.add_argument("--baseline", action="store_true", help="Single-shot baseline (no verifiers)")
    parser.add_argument("--cot", action="store_true", help="Chain-of-thought baseline")
    parser.add_argument("--self_consistency", action="store_true", help="Self-consistency (K=3) baseline")
    parser.add_argument("--self_refine", action="store_true", help="Self-refine baseline")
    parser.add_argument("--debate", action="store_true", help="Debate baseline (3 agents debate)")
    parser.add_argument("--log", action="store_true", help="Enable file logging")
    parser.add_argument("--dataset", type=str, default="data/sudoku/puzzles.json")
    parser.add_argument("--puzzle_ids", type=int, nargs="+", help="Specific puzzle IDs to run")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory for JSONL results")
    parser.add_argument("--model", type=str, default="gpt-none",
                        choices=["gpt-none", "gpt-high", "qwen-none", "qwen-thinking"],
                        help="LLM config to use")
    args = parser.parse_args()

    if args.baseline:
        mode = "baseline"
    elif args.cot:
        mode = "cot"
    elif args.self_consistency:
        mode = "self-consistency"
    elif args.self_refine:
        mode = "self-refine"
    elif args.debate:
        mode = "debate"
    else:
        mode = "gen-ver"
    suffix = f"sudoku_{mode}"

    with open(args.dataset) as f:
        dataset = json.load(f)

    if args.puzzle_ids is not None:
        dataset = [e for e in dataset if e["id"] in args.puzzle_ids]

    if args.log:
        pid = args.puzzle_ids[0] if args.puzzle_ids and len(args.puzzle_ids) == 1 else None
        setup_logging(log_dir="logs", suffix=suffix, puzzle_id=pid)

    # JSONL results file
    dataset_stem = Path(args.dataset).stem  # e.g. "puzzles_0.3"
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"sudoku_{mode}_{args.model}_{dataset_stem}.jsonl"

    # Trace directory
    traces_dir = Path("traces") / f"sudoku_{mode}_{args.model}_{dataset_stem}"

    print(f"Mode: {mode} | Model: {args.model}")
    print(f"Loaded {len(dataset)} puzzles from {args.dataset}")
    if mode in ("gen-ver", "self-refine"):
        max_rounds = args.max_rounds
        print(f"Max rounds: {max_rounds}")
    print()

    gen_llm = make_llm(args.model)
    if args.debate:
        debate_llms = [make_llm(args.model) for _ in range(3)]
        sc_llms: list[HarvardGPTLLM | QwenLLM] = []
        ver_llms: list[HarvardGPTLLM | QwenLLM] = []
    elif args.self_consistency:
        debate_llms = []
        sc_llms = [make_llm(args.model) for _ in range(3)]
        ver_llms = []
    elif mode == "gen-ver":
        debate_llms = []
        sc_llms = []
        ver_llms = [make_llm(args.model) for _ in range(3)]
    else:
        debate_llms = []
        sc_llms = []
        ver_llms = []

    results = []
    start_time = time.time()

    for entry in dataset:
        puzzle_id = entry["id"]
        puzzle = entry["puzzle"]
        print(f"Puzzle {puzzle_id}...", end=" ", flush=True)

        trace_file = traces_dir / f"p{puzzle_id}.jsonl"

        t0 = time.time()
        if args.baseline:
            result = solve_baseline(puzzle, puzzle_id, gen_llm, trace_file=trace_file)
        elif args.cot:
            result = solve_cot(puzzle, gen_llm)
        elif args.self_consistency:
            result = solve_self_consistency(puzzle, sc_llms)
        elif args.self_refine:
            result = solve_self_refine(puzzle, args.max_rounds, gen_llm)
        elif args.debate:
            result = solve_debate(puzzle, debate_llms)
        else:
            result = solve_puzzle(puzzle, puzzle_id, args.max_rounds, gen_llm, ver_llms, trace_file=trace_file)
        result["id"] = puzzle_id
        result["elapsed"] = round(time.time() - t0, 2)
        results.append(result)

        # Append to JSONL
        with open(results_file, "a") as rf:
            rf.write(json.dumps(result) + "\n")

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
