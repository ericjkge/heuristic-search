"""Prompt templates for the Sudoku verifier agents (row, column, box)."""


def row_prompt(grid_str: str) -> str:
    return f"""You are a Sudoku row verifier. Check whether each row contains the digits 1-9 with no repeats.

Coordinates:
- Top-left is R1C1.
- Bottom-right is R9C9.

Candidate solution:
{grid_str}

For each row, check if it contains exactly the digits 1-9 with no duplicates.

Write down your process as you work then wrap any errors in <OUTPUT> tags with one error per line. If all rows are correct, output empty <OUTPUT></OUTPUT> tags.

<OUTPUT>
R3: duplicate 5 (C2, C7), missing 4
R5: duplicate 3 (C1, C9), missing 2
</OUTPUT>"""


def column_prompt(grid_str: str) -> str:
    return f"""You are a Sudoku column verifier. Check whether each column contains the digits 1-9 with no repeats.

Coordinates:
- Top-left is R1C1.
- Bottom-right is R9C9.

Candidate solution:
{grid_str}


Write down your process as you work then wrap any errors in <OUTPUT> tags with one error per line. If all columns are correct, output empty <OUTPUT></OUTPUT> tags.

<OUTPUT>
C2: duplicate 3 (R1, R6), missing 8
C8: duplicate 5 (R8, R9), missing 1
</OUTPUT>"""


def box_prompt(grid_str: str) -> str:
    return f"""You are a Sudoku box verifier. Check whether each 3x3 box contains the digits 1-9 with no repeats.

Coordinates:
- Top-left is R1C1.
- Bottom-right is R9C9.

The 9 boxes are:
- Box1: R1-R3, C1-C3
- Box2: R1-R3, C4-C6
- Box3: R1-R3, C7-C9
- Box4: R4-R6, C1-C3
- Box5: R4-R6, C4-C6
- Box6: R4-R6, C7-C9
- Box7: R7-R9, C1-C3
- Box8: R7-R9, C4-C6
- Box9: R7-R9, C7-C9

Candidate solution:
{grid_str}

Write down your process as you work then wrap any errors in <OUTPUT> tags with one error per line. If all boxes are correct, output empty <OUTPUT></OUTPUT> tags.

<OUTPUT>
Box1: duplicate 6 (R1C1, R3C3), missing 1
Box7: duplicate 2 (R8C2, R9C2), missing 3
</OUTPUT>"""