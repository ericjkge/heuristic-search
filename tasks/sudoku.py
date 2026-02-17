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


def str_to_grid(text: str) -> Optional[Grid]:
    """Parse lines that contain exactly 9 digits."""
    grid: Grid = []
    for line in text.strip().splitlines():
        digits = re.findall(r"\d", line)
        if len(digits) == 9:
            grid.append([int(d) for d in digits])
    if len(grid) == 9:
        return grid
    return None


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
