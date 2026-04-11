"""Generate Sudoku puzzle dataset.

Usage: python gen_dataset.py --n 10 --mask_rate 0.6
"""

import argparse
import json
from pathlib import Path

import sudokum


def generate_puzzles(
    n: int,
    mask_rate: float,
    start_id: int = 0,
    existing_hashes: set[tuple] | None = None,
) -> list[dict]:
    """Generate n unique puzzles with given mask_rate (fraction of cells to hide)."""
    if existing_hashes is None:
        existing_hashes = set()

    puzzles: list[dict] = []
    attempts = 0
    max_attempts = n * 100

    while len(puzzles) < n and attempts < max_attempts:
        attempts += 1
        puzzle = sudokum.generate(mask_rate=mask_rate)
        key = tuple(tuple(row) for row in puzzle)
        if key in existing_hashes:
            continue
        success, solution = sudokum.solve(puzzle)
        if not success:
            continue
        existing_hashes.add(key)
        puzzles.append({
            "id": start_id + len(puzzles),
            "puzzle": puzzle,
            "solution": solution,
            "mask_rate": mask_rate,
        })

    if len(puzzles) < n:
        print(f"Warning: only generated {len(puzzles)} unique puzzles (asked {n})")

    return puzzles


def main():
    parser = argparse.ArgumentParser(description="Generate Sudoku dataset")
    parser.add_argument("--n", type=int, default=10, help="Number of puzzles")
    parser.add_argument("--mask_rate", type=float, default=0.6, help="Fraction of cells to mask")
    parser.add_argument("--output", type=str, default="data/sudoku/puzzles.json")
    parser.add_argument("--append", action="store_true", help="Append N new unique puzzles to existing file")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    existing_hashes: set[tuple] = set()
    if args.append and Path(args.output).exists():
        with open(args.output) as f:
            existing = json.load(f)
        for e in existing:
            existing_hashes.add(tuple(tuple(row) for row in e["puzzle"]))

    new_puzzles = generate_puzzles(args.n, args.mask_rate, start_id=len(existing), existing_hashes=existing_hashes)
    all_puzzles = existing + new_puzzles

    with open(args.output, "w") as f:
        json.dump(all_puzzles, f, indent=2)

    verb = "Appended" if args.append and existing else "Generated"
    print(f"{verb} {len(new_puzzles)} puzzles → {args.output} (total: {len(all_puzzles)})")


if __name__ == "__main__":
    main()
