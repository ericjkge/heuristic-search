"""Prompt templates for Tree-of-Thought Sudoku solver."""


def initial_propose_prompt(puzzle_str: str) -> str:
    return f"""You are a Sudoku solver using a step-by-step approach.

Rules:
- Each row must contain digits 1-9 with no repeats.
- Each column must contain digits 1-9 with no repeats.
- Each 3x3 box must contain digits 1-9 with no repeats.
- You must keep all given (non-dot) digits in their positions.

Puzzle:
{puzzle_str}

Fill in a few cells you are MOST confident about. Use . for cells you are not yet sure about. Do NOT guess — only fill cells where you can logically deduce the answer.

Output EXACTLY 9 lines, each with 9 space-separated tokens (digits 1-9 or .):

d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d"""


def continue_propose_prompt(puzzle_str: str, current_str: str) -> str:
    return f"""You are a Sudoku solver continuing to fill in a partial board.

Rules:
- Each row must contain digits 1-9 with no repeats.
- Each column must contain digits 1-9 with no repeats.
- Each 3x3 box must contain digits 1-9 with no repeats.
- You must keep all given (non-dot) digits in their positions.
- You must keep all previously filled digits in their positions.

Original puzzle:
{puzzle_str}

Current partial solution:
{current_str}

Fill in a few MORE cells you are confident about. Keep all existing digits. Use . for cells you are still unsure about.

Output EXACTLY 9 lines, each with 9 space-separated tokens (digits 1-9 or .):

d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d"""


def backtrack_propose_prompt(
    puzzle_str: str, rolled_back_str: str, error_msg: str
) -> str:
    return f"""You are a Sudoku solver. Your previous attempt had errors and was rolled back.

Rules:
- Each row must contain digits 1-9 with no repeats.
- Each column must contain digits 1-9 with no repeats.
- Each 3x3 box must contain digits 1-9 with no repeats.
- You must keep all given (non-dot) digits in their positions.
- You must keep all previously filled digits in their positions.

Original puzzle:
{puzzle_str}

Error in previous attempt:
{error_msg}

Rolled-back state (your starting point):
{rolled_back_str}

Try filling DIFFERENT cells this time, or use different values. Use . for cells you are unsure about.

Output EXACTLY 9 lines, each with 9 space-separated tokens (digits 1-9 or .):

d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d
d d d d d d d d d"""
