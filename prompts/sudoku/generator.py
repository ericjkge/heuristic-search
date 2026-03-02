"""Prompt templates for the Sudoku generator agent."""


def initial_prompt(puzzle_str: str) -> str:
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


def revision_prompt(puzzle_str: str, previous_grid_str: str, feedback: str) -> str:
    return f"""You are a Sudoku solver. Fix your previous solution based on verifier feedback.

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

Your previous solution:
{previous_grid_str}

Verifier feedback:
{feedback}

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
