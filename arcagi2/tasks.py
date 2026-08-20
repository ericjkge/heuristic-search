"""ARC-AGI-2 task loading and exact scoring.

Data from github.com/arcprize/ARC-AGI-2 (public sets: 1000 training, 120
evaluation tasks). Scoring follows github.com/arcprize/arc-agi-benchmarking
(the official leaderboard harness): each test input gets up to 2 attempts,
a task's score is the FRACTION of its test inputs with an exactly-correct
attempt, and the run score averages task scores. `solved` (all test inputs
correct — the stricter rule in the data repo's readme) is also reported.

    uv run python arcagi2/tasks.py          # dataset sanity check
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Grid = list[list[int]]

DATA_DIR = Path(__file__).parent / "data"


@dataclass
class Task:
    id: str
    split: str  # training | evaluation
    train: list[dict[str, Grid]]  # [{"input": Grid, "output": Grid}, ...]
    test: list[dict[str, Grid]]


def load_task(path: Path, split: str) -> Task:
    d = json.loads(path.read_text())
    return Task(id=path.stem, split=split, train=d["train"], test=d["test"])


def load_tasks(split: str = "training", ids: list[str] | None = None) -> list[Task]:
    files = sorted((DATA_DIR / split).glob("*.json"))
    if ids is not None:
        wanted = set(ids)
        files = [f for f in files if f.stem in wanted]
    return [load_task(f, split) for f in files]


# ---------------------------------------------------------------- grids

def is_grid(g: Any) -> bool:
    return (isinstance(g, list) and len(g) > 0
            and all(isinstance(row, list) and len(row) == len(g[0])
                    and all(isinstance(c, int) and 0 <= c <= 9 for c in row)
                    for row in g))


def grids_equal(a: Any, b: Any) -> bool:
    return a == b and is_grid(a)


def cell_accuracy(pred: Any, gold: Grid) -> float:
    """Fraction of matching cells; 0.0 on shape mismatch. Diagnostic only —
    official scoring is exact match."""
    if not is_grid(pred) or len(pred) != len(gold) or len(pred[0]) != len(gold[0]):
        return 0.0
    total = len(gold) * len(gold[0])
    hits = sum(pc == gc for pr, gr in zip(pred, gold) for pc, gc in zip(pr, gr))
    return hits / total


def render_grid(g: Grid) -> str:
    return "\n".join(" ".join(str(c) for c in row) for row in g)


# ---------------------------------------------------------------- scoring

def score_task(task: Task, attempts: list[list[Grid]], max_attempts: int = 2) -> dict:
    """attempts[i] = candidate output grids for test input i (first
    `max_attempts` count). Returns per-test correctness and task-level solved."""
    per_test = []
    for i, pair in enumerate(task.test):
        cands = (attempts[i] if i < len(attempts) else [])[:max_attempts]
        correct = any(grids_equal(c, pair["output"]) for c in cands)
        best_cell_acc = max((cell_accuracy(c, pair["output"]) for c in cands),
                            default=0.0)
        per_test.append({"correct": correct, "n_attempts": len(cands),
                         "best_cell_acc": round(best_cell_acc, 4)})
    n_correct = sum(t["correct"] for t in per_test)
    return {"id": task.id,
            "score": n_correct / len(per_test) if per_test else 0.0,
            "solved": bool(per_test) and n_correct == len(per_test),
            "per_test": per_test}


def score_run(tasks: list[Task], attempts_by_id: dict[str, list[list[Grid]]],
              max_attempts: int = 2) -> dict:
    """attempts_by_id[task.id] -> attempts as in score_task. `score` = mean of
    per-task fractional scores (leaderboard metric); `n_solved` = tasks with
    all test inputs correct."""
    results = [score_task(t, attempts_by_id.get(t.id, []), max_attempts)
               for t in tasks]
    return {"n": len(tasks),
            "score": sum(r["score"] for r in results) / len(tasks) if tasks else 0.0,
            "n_solved": sum(r["solved"] for r in results),
            "per_task": results}


if __name__ == "__main__":
    for split in ("training", "evaluation"):
        tasks = load_tasks(split)
        grids = [g for t in tasks for p in t.train + t.test
                 for g in (p["input"], p["output"])]
        assert all(is_grid(g) for g in grids)
        multi = sum(len(t.test) > 1 for t in tasks)
        print(f"{split}: {len(tasks)} tasks, {multi} with >1 test input, "
              f"{len(grids)} grids ok")
    t = load_tasks("training", ids=["00576224"])[0]
    gold = [a["output"] for a in t.test]
    assert score_task(t, [gold])["solved"]
    assert not score_task(t, [[t.test[0]["input"]]])["solved"]
    print("scoring sanity ok:", t.id)
