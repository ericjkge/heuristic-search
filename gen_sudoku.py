"""Generate Sudoku puzzle dataset.

Usage: python gen_dataset.py --n 10 --mask_rate 0.6
"""

import argparse
import json
from pathlib import Path

import sudokum


def generate_puzzles(n: int, mask_rate: float) -> list[dict]:
    """Generate n puzzles with given mask_rate (fraction of cells to hide)."""
    puzzles = []
    for i in range(n):
        puzzle = sudokum.generate(mask_rate=mask_rate)
        success, solution = sudokum.solve(puzzle)
        if not success:
            print(f"Warning: puzzle {i} could not be solved, skipping")
            continue
        puzzles.append({
            "id": i,
            "puzzle": puzzle,
            "solution": solution,
            "mask_rate": mask_rate,
        })
    return puzzles


def main():
    parser = argparse.ArgumentParser(description="Generate Sudoku dataset")
    parser.add_argument("--n", type=int, default=10, help="Number of puzzles")
    parser.add_argument("--mask_rate", type=float, default=0.6, help="Fraction of cells to mask")
    parser.add_argument("--output", type=str, default="data/sudoku/puzzles.json")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    puzzles = generate_puzzles(args.n, args.mask_rate)

    with open(args.output, "w") as f:
        json.dump(puzzles, f, indent=2)

    print(f"Generated {len(puzzles)} puzzles → {args.output}")


if __name__ == "__main__":
    main()
