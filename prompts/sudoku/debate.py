"""Debate (MAD) prompts for the Sudoku task."""


def propose_prompt(puzzle_str: str) -> str:
    return f"""You are a Sudoku solver. Solve the following puzzle.

Rules:
- Each row must contain digits 1-9 with no repeats.
- Each column must contain digits 1-9 with no repeats.
- Each 3x3 box must contain digits 1-9 with no repeats.
- You must keep all given (non-dot) digits in their positions.

Coordinates:
- Top-left is R1C1.
- Bottom-right is R9C9.

Puzzle:
{puzzle_str}

Output: Wrap your final answer in <OUTPUT> tags with one row per line, 9 space-separated digits per row:

<OUTPUT>
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
</OUTPUT>"""


def critique_prompt(
    puzzle_str: str,
    own_solution: str,
    other_solutions: list[str],
) -> str:
    others_text = "\n\n".join(
        f"--- Solution {i+1} ---\n{s}" for i, s in enumerate(other_solutions)
    )
    return f"""You are a Sudoku solver engaged in a debate. You have proposed a solution and have seen other agents' solutions. Revise your answer if you find errors.

Rules:
- Each row must contain digits 1-9 with no repeats.
- Each column must contain digits 1-9 with no repeats.
- Each 3x3 box must contain digits 1-9 with no repeats.
- You must keep all given (non-dot) digits in their positions.

Puzzle:
{puzzle_str}

Your previous solution:
{own_solution}

Other agents' solutions:
{others_text}

Compare all solutions carefully. Check for row, column, and box constraint violations. Produce your best revised solution.

Output: Wrap your final answer in <OUTPUT> tags with one row per line, 9 space-separated digits per row:

<OUTPUT>
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
</OUTPUT>"""


