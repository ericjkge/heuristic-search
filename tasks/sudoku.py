"""Sudoku puzzle utilities: formatting, parsing, validation."""

import re
from typing import Optional

import numpy as np


Grid = list[list[int]]


def grid_to_str(puzzle: Grid) -> str:
    """Format a 9x9 grid for display in prompts. 0 = blank cell shown as '.'."""
    lines = []
    for row in puzzle:
        cells = [str(c) if c != 0 else "." for c in row]
        lines.append(" ".join(cells))
    return "\n".join(lines)


def _extract_output_tag(text: str) -> Optional[str]:
    """Extract content between the last <OUTPUT> and </OUTPUT> tags."""
    matches = re.findall(r"<OUTPUT>(.*?)</OUTPUT>", text, re.DOTALL)
    return matches[-1] if matches else None


def str_to_grid(text: str) -> Optional[Grid]:
    """Parse lines that contain exactly 9 digits. Tries <OUTPUT> tags first."""
    tagged = _extract_output_tag(text)
    source = tagged if tagged is not None else text
    grid: Grid = []
    for line in source.strip().splitlines():
        digits = re.findall(r"\d", line)
        if len(digits) == 9:
            grid.append([int(d) for d in digits])
    if len(grid) == 9:
        return grid
    return None


def str_to_partial_grid(text: str) -> Optional[Grid]:
    """Parse lines with exactly 9 tokens that are each a digit or '.'. Dots become 0."""
    grid: Grid = []
    for line in text.strip().splitlines():
        tokens = re.findall(r"[\d.]", line)
        if len(tokens) == 9 and all(t in "0123456789." for t in tokens):
            grid.append([0 if t == "." else int(t) for t in tokens])
    if len(grid) == 9:
        return grid
    return None


def check_partial_solution(
    puzzle: Grid, candidate: Grid
) -> tuple[bool, bool, str]:
    """Validate a partial Sudoku solution.

    Returns (is_valid, is_complete, error_message).
    - Checks givens preserved, filled cells in 1-9, no duplicate digits in row/col/box (ignoring 0s).
    - is_complete = all 81 cells filled AND valid.
    """
    errors: list[str] = []

    # Check givens preserved
    for r in range(9):
        for c in range(9):
            if puzzle[r][c] != 0 and candidate[r][c] != puzzle[r][c]:
                errors.append(
                    f"R{r+1}C{c+1}: given={puzzle[r][c]}, got={candidate[r][c]}"
                )

    # Check filled cells are 1-9
    for r in range(9):
        for c in range(9):
            v = candidate[r][c]
            if v != 0 and not (1 <= v <= 9):
                errors.append(f"R{r+1}C{c+1}: invalid value {v}")

    # Check row uniqueness (ignoring 0s)
    for r in range(9):
        filled = [candidate[r][c] for c in range(9) if candidate[r][c] != 0]
        seen: set[int] = set()
        for v in filled:
            if v in seen:
                errors.append(f"Row {r+1}: duplicate {v}")
                break
            seen.add(v)

    # Check column uniqueness (ignoring 0s)
    for c in range(9):
        filled = [candidate[r][c] for r in range(9) if candidate[r][c] != 0]
        seen = set()
        for v in filled:
            if v in seen:
                errors.append(f"Col {c+1}: duplicate {v}")
                break
            seen.add(v)

    # Check box uniqueness (ignoring 0s)
    for br in range(3):
        for bc in range(3):
            filled = []
            for r in range(br * 3, br * 3 + 3):
                for c in range(bc * 3, bc * 3 + 3):
                    if candidate[r][c] != 0:
                        filled.append(candidate[r][c])
            seen = set()
            for v in filled:
                if v in seen:
                    errors.append(f"Box ({br+1},{bc+1}): duplicate {v}")
                    break
                seen.add(v)

    is_valid = len(errors) == 0
    all_filled = all(candidate[r][c] != 0 for r in range(9) for c in range(9))
    is_complete = is_valid and all_filled
    error_msg = "; ".join(errors) if errors else ""

    return is_valid, is_complete, error_msg


def validate_givens(puzzle: Grid, candidate: Grid) -> list[str]:
    """Check that all non-zero cells in puzzle match candidate."""
    errors = []
    for r in range(9):
        for c in range(9):
            if puzzle[r][c] != 0 and puzzle[r][c] != candidate[r][c]:
                errors.append(
                    f"R{r+1}C{c+1}: given={puzzle[r][c]}, got={candidate[r][c]}"
                )
    return errors


def check_solution(candidate: Grid) -> bool:
    """Programmatic correctness check using numpy."""
    arr = np.array(candidate)
    # Check all values are 1-9
    if arr.min() < 1 or arr.max() > 9:
        return False
    # Check rows
    for r in range(9):
        if len(set(arr[r])) != 9:
            return False
    # Check columns
    for c in range(9):
        if len(set(arr[:, c])) != 9:
            return False
    # Check 3x3 boxes
    for br in range(3):
        for bc in range(3):
            box = arr[br*3:(br+1)*3, bc*3:(bc+1)*3].flatten()
            if len(set(box)) != 9:
                return False
    return True
