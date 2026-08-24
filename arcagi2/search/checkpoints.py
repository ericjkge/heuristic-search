"""Extract round-k checkpoints from a search run's logs.

For each pair-search log, replay events in order; at each round marker,
snapshot every node's latest V and take the top-2 distinct candidates —
exactly what run() would submit if max_rounds were k. Scores each checkpoint
with the official metric.

    uv run python arcagi2/search/checkpoints.py runs/<name> [--rounds 2,4,6,8,10]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SEARCH_DIR = Path(__file__).resolve().parent
ROOT = SEARCH_DIR.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arcagi2.tasks import load_tasks, score_run


def pair_checkpoints(log_path: Path, rounds: list[int]) -> dict[int, list]:
    """round -> top-2 distinct candidate grids at that point."""
    nodes: dict[int, dict] = {}  # id -> {candidate, value}
    out: dict[int, list] = {}
    for line in log_path.read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = ev.get("event")
        if kind == "node" and ev.get("candidate") is not None:
            nodes[ev["id"]] = {"candidate": ev["candidate"], "value": ev["value"]}
        elif kind == "rescore":
            for nid, val in (ev.get("values") or {}).items():
                if int(nid) in nodes:
                    nodes[int(nid)]["value"] = val
        elif kind == "round" and ev.get("round") in rounds:
            attempts = []
            for n in sorted(nodes.values(), key=lambda n: -n["value"]):
                if n["candidate"] not in attempts:
                    attempts.append(n["candidate"])
                if len(attempts) == 2:
                    break
            out[ev["round"]] = attempts
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--rounds", default="2,4,6,8,10")
    ap.add_argument("--split", default="evaluation")
    args = ap.parse_args()
    rounds = [int(r) for r in args.rounds.split(",")]

    run_dir = args.run_dir if args.run_dir.is_absolute() else SEARCH_DIR / args.run_dir
    by_task: dict[str, dict[int, dict[int, list]]] = defaultdict(dict)
    for f in sorted((run_dir / "logs").glob("*.jsonl")):
        stem = f.stem
        tid, pair = (stem.rpartition("_p")[0], int(stem.rpartition("_p")[2])) \
            if "_p" in stem and stem.rpartition("_p")[2].isdigit() else (stem, 0)
        by_task[tid][pair] = pair_checkpoints(f, rounds)

    tasks = load_tasks(args.split, ids=list(by_task))
    print(f"{'round':>5} {'fractional':>11} {'strict':>7}")
    for r in rounds:
        attempts_by_id = {
            tid: [pairs.get(i, {}).get(r, []) for i in range(len(pairs))]
            for tid, pairs in by_task.items()}
        sc = score_run(tasks, attempts_by_id)
        print(f"{r:>5} {sc['score']:>11.4f} {sc['n_solved']:>7}")


if __name__ == "__main__":
    main()
