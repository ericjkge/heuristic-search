"""Self-critique prompt for the Sudoku self-refine baseline."""


def self_critique_prompt(puzzle_str: str, grid_str: str) -> str:
    return f"""You are a Sudoku verifier. Carefully check whether this solution is correct.

Rules:
- Each row must contain digits 1-9 with no repeats.
- Each column must contain digits 1-9 with no repeats.
- Each 3x3 box must contain digits 1-9 with no repeats.
- All given (non-dot) digits must be in their original positions.

Coordinates:
- Top-left is R1C1.
- Bottom-right is R9C9.

Puzzle (given digits, dots are blanks):
{puzzle_str}

Candidate solution:
{grid_str}

Check every row, column, and 3x3 box for duplicate or missing digits. Also verify that all given digits are preserved.

Output: If you find errors, list them inside <OUTPUT> tags. If the solution is correct, output empty <OUTPUT></OUTPUT> tags.

<OUTPUT>
R3: duplicate 5 (C2, C7), missing 4
C8: duplicate 2 (R1, R6), missing 9
</OUTPUT>"""
