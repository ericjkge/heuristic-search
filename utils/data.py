"""Load and preprocess ZebraLogic puzzles for the verifier-guided search MVP."""

import re
from dataclasses import dataclass, field

from datasets import load_dataset

# Small/medium sizes where Qwen3-8B (no thinking) should have headroom.
# One puzzle is sampled per size for the smoke test.
SMOKE_SIZES = ["2*3", "3*3", "3*4", "4*3", "4*4", "4*5", "5*4", "3*5"]

_CLUE_RE = re.compile(r"^\s*(\d+)\.\s+(.*\S)\s*$")


@dataclass
class Puzzle:
    id: str
    size: str
    n_houses: int
    n_attrs: int
    puzzle: str  # full original text
    clues: list[str]  # parsed clue strings, in order (c1..cK)
    header: list[str]  # solution column names, e.g. ["House", "Name", ...]
    gold_rows: list[list[str]]  # gold solution rows

    @property
    def k(self) -> int:
        return len(self.clues)

    @property
    def gold(self) -> dict:
        return {"header": self.header, "rows": self.gold_rows}


def parse_size(size: str) -> tuple[int, int]:
    """'5*6' -> (5 houses, 6 attributes)."""
    houses, attrs = size.split("*")
    return int(houses), int(attrs)


def split_clues(puzzle_text: str) -> list[str]:
    """Extract numbered clues that follow the '## Clues:' header."""
    _, _, clue_block = puzzle_text.partition("## Clues:")
    clues = []
    for line in clue_block.splitlines():
        m = _CLUE_RE.match(line)
        if m:
            clues.append(m.group(2))
    return clues


def to_puzzle(ex: dict) -> Puzzle:
    n_houses, n_attrs = parse_size(ex["size"])
    return Puzzle(
        id=ex["id"],
        size=ex["size"],
        n_houses=n_houses,
        n_attrs=n_attrs,
        puzzle=ex["puzzle"],
        clues=split_clues(ex["puzzle"]),
        header=ex["solution"]["header"],
        gold_rows=ex["solution"]["rows"],
    )


def load_smoke_set(sizes: list[str] = SMOKE_SIZES) -> list[Puzzle]:
    """First puzzle of each requested size."""
    ds = load_dataset("WildEval/ZebraLogic", "grid_mode", split="test")
    wanted = set(sizes)
    first = {}
    for ex in ds:
        if ex["size"] in wanted and ex["size"] not in first:
            first[ex["size"]] = ex
    return [to_puzzle(first[s]) for s in sizes if s in first]


def load_puzzle(puzzle_id: str) -> Puzzle:
    """Load a single puzzle by its dataset id (e.g. 'lgp-test-3x3-24')."""
    ds = load_dataset("WildEval/ZebraLogic", "grid_mode", split="test")
    for ex in ds:
        if ex["id"] == puzzle_id:
            return to_puzzle(ex)
    raise ValueError(f"puzzle id not found: {puzzle_id}")


def load_puzzles(sizes: list[str], per_size: int = 5) -> list[Puzzle]:
    """First `per_size` puzzles of each requested size, in `sizes` order."""
    ds = load_dataset("WildEval/ZebraLogic", "grid_mode", split="test")
    buckets = {s: [] for s in sizes}
    for ex in ds:
        bucket = buckets.get(ex["size"])
        if bucket is not None and len(bucket) < per_size:
            bucket.append(ex)
    return [to_puzzle(ex) for s in sizes for ex in buckets[s]]

# Smoke test puzzle loading and pre-processing
if __name__ == "__main__":
    puzzles = load_smoke_set()
    print(f"Loaded {len(puzzles)} puzzles\n")
    for p in puzzles:
        print(f"=== {p.id}  (size {p.size} -> {p.n_houses} houses x {p.n_attrs} attrs)")
        print(f"    header: {p.header}")
        print(f"    K = {p.k} clues parsed")
        for i, c in enumerate(p.clues, 1):
            print(f"      {i}. {c}")
        print()
