"""Prompt templates for the Sudoku debate agents."""


def propose_prompt(puzzle_str: str) -> str:
    return f"""You are a Sudoku solver. Solve the following puzzle.

Rules:
- Each row must contain digits 1-9 with no repeats.
- Each column must contain digits 1-9 with no repeats.
- Each 3x3 box must contain digits 1-9 with no repeats.
- You must keep all given (non-dot) digits in their positions.

Puzzle:
{puzzle_str}

Output: Represent the board with one row per line, 9 space-separated digits per row

d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d"""


def critique_prompt(
    puzzle_str: str, own_solution: str, other_solutions: list[str]
) -> str:
    others_block = ""
    for i, sol in enumerate(other_solutions, 1):
        others_block += f"\n--- Other agent {i} ---\n{sol}\n"

    return f"""You are a Sudoku solver participating in a debate. Below is the original puzzle, your current solution, and solutions from other agents.

Rules:
- Each row must contain digits 1-9 with no repeats.
- Each column must contain digits 1-9 with no repeats.
- Each 3x3 box must contain digits 1-9 with no repeats.
- You must keep all given (non-dot) digits in their positions.

Puzzle:
{puzzle_str}

Your current solution:
{own_solution}

Other agents' solutions:
{others_block}
Critique the other solutions. Identify any errors you see. Then decide: keep your current solution or revise it based on what you learned. Output your final solution.

Output: Represent the board with one row per line, 9 space-separated digits per row

d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d"""


def vote_prompt(puzzle_str: str, all_solutions: list[str]) -> str:
    solutions_block = ""
    for i, sol in enumerate(all_solutions, 1):
        solutions_block += f"\n--- Solution {i} ---\n{sol}\n"

    return f"""You are a Sudoku solver. Below is the original puzzle and candidate solutions from 3 agents (including yourself). Pick the best solution.

Rules:
- Each row must contain digits 1-9 with no repeats.
- Each column must contain digits 1-9 with no repeats.
- Each 3x3 box must contain digits 1-9 with no repeats.
- All given (non-dot) digits must remain in their positions.

Puzzle:
{puzzle_str}

Candidate solutions:
{solutions_block}
Check each solution for correctness. Output ONLY the number of the best solution (1, 2, or 3)."""
