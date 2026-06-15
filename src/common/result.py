"""Shared result type returned by the method and every baseline."""

from dataclasses import dataclass, field


@dataclass
class Result:
    puzzle_id: str
    solved: bool
    best_score: float
    best_candidate: dict | None
    trajectory: list[float]  # best score after each iteration
    condition: str = "main"
    extra: dict = field(default_factory=dict)
