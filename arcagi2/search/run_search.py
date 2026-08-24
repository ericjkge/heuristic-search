"""Run verifier-guided search on ARC-AGI-2 tasks.

    uv run python arcagi2/search/run_search.py --split training --ids 00576224
    uv run python arcagi2/search/run_search.py --dev

Writes runs/<name>/ with attempts/ (top-2 grids per test pair), logs/ (one
JSONL event stream per test-pair search: <tid>.jsonl for pair 0,
<tid>_p<i>.jsonl beyond), and summary.json scored via the official
fractional metric. Multi-test-input tasks run one independent search per
test input.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import time
from pathlib import Path

SEARCH_DIR = Path(__file__).resolve().parent
ARC_DIR = SEARCH_DIR.parent
ROOT = ARC_DIR.parent
for p in (str(ROOT), str(SEARCH_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tqdm.asyncio import tqdm_asyncio

import llm
from arcagi2.tasks import load_tasks, score_run
from search import ArcSearch, SearchConfig

RUNS_DIR = SEARCH_DIR / "runs"


async def run_one(task, args, cfg: SearchConfig, out_dir: Path) -> tuple[dict, list]:
    attempts: list[list] = []
    row: dict = {"id": task.id, "n_nodes": 0, "n_verifiers": 0,
                 "llm_calls": 0, "parse_failures": 0}
    for i in range(len(task.test)):
        suffix = "" if i == 0 else f"_p{i}"
        log_file = (out_dir / "logs" / f"{task.id}{suffix}.jsonl").open("w")

        def log_event(ev: dict) -> None:
            log_file.write(json.dumps(ev, ensure_ascii=False) + "\n")
            log_file.flush()

        log_event({"event": "start", "id": task.id, "pair": i,
                   "n_train": len(task.train)})
        try:
            search = ArcSearch(task, pair_index=i, model=args.model,
                               provider=args.provider,
                               reasoning_effort=args.reasoning_effort,
                               cfg=cfg, log=log_event)
            result = await search.run()
            attempts.append(result.attempts)
            row["n_nodes"] += result.n_nodes
            row["n_verifiers"] += len(search.vset.verifiers)
            row["llm_calls"] += search.llm_calls
            row["parse_failures"] += search.parse_failures
        except Exception as e:
            log_event({"event": "error", "error": repr(e)})
            row.setdefault("errors", []).append(repr(e))
            attempts.append([])
        finally:
            log_file.close()
    (out_dir / "attempts" / f"{task.id}.json").write_text(json.dumps(attempts))
    return row, attempts


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="evaluation", choices=["training", "evaluation"])
    ap.add_argument("--ids", default=None, help="comma-separated task ids")
    ap.add_argument("--dev", action="store_true",
                    help="use the 20-task dev slice (arcagi2/dev_slice.txt)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--provider", default="abaka", choices=sorted(llm.PROVIDERS))
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--name", default=None)
    # search config — defaults come from SearchConfig so file edits take effect
    dflt = SearchConfig()
    ap.add_argument("--n-seed", type=int, default=dflt.n_seed)
    ap.add_argument("--n-parents", type=int, default=dflt.n_parents)
    ap.add_argument("--k-children", type=int, default=dflt.k_children)
    ap.add_argument("--max-rounds", type=int, default=dflt.max_rounds)
    ap.add_argument("--evolve-every", type=int, default=dflt.evolve_every,
                    help="add verifiers every N rounds; 0 disables")
    ap.add_argument("--seed-verifiers", default=dflt.seed_verifiers,
                    help="verifier count range asked for in the seed prompt")
    ap.add_argument("--evolve-verifiers", default=dflt.evolve_verifiers,
                    help="verifier count range asked for per evolve round")
    ap.add_argument("--tau", type=float, default=dflt.tau)
    ap.add_argument("--deg-coef", type=float, default=dflt.deg_coef)
    ap.add_argument("--temperature", type=float, default=dflt.expand_temperature)
    ap.add_argument("--no-feedback", action="store_true",
                    help="do not show verifier statements+scores to the generator")
    ap.add_argument("--n-inspirations", type=int, default=2,
                    help="extra tried candidates shown on expansion; 0 disables")
    ap.add_argument("--random-parents", action="store_true",
                    help="ablation: sample parents uniformly, ignoring V")
    args = ap.parse_args()

    cfg = SearchConfig(
        n_seed=args.n_seed, n_parents=args.n_parents, k_children=args.k_children,
        max_rounds=args.max_rounds, evolve_every=args.evolve_every,
        tau=args.tau, deg_coef=args.deg_coef,
        feedback=not args.no_feedback, expand_temperature=args.temperature,
        random_parents=args.random_parents, n_inspirations=args.n_inspirations,
        seed_verifiers=args.seed_verifiers, evolve_verifiers=args.evolve_verifiers)

    ids = args.ids.split(",") if args.ids else None
    if args.dev:
        ids = (ARC_DIR / "dev_slice.txt").read_text().split()
    tasks = load_tasks(args.split, ids=ids)[: args.limit]

    name = args.name or f"search-{time.strftime('%m%d-%H%M%S')}"
    out_dir = RUNS_DIR / name
    for sub in ("attempts", "logs"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    results = await tqdm_asyncio.gather(
        *(run_one(t, args, cfg, out_dir) for t in tasks), desc="tasks")
    rows = [r for r, _ in results]
    scores = score_run(tasks, {t.id: a for t, (_, a) in zip(tasks, results)})

    summary = {"name": name, "split": args.split, "model": args.model,
               "provider": args.provider, "reasoning_effort": args.reasoning_effort,
               "config": dataclasses.asdict(cfg),
               "n": scores["n"], "score": scores["score"],
               "n_solved": scores["n_solved"], "rows": rows,
               "per_task": scores["per_task"], "usage": llm.USAGE.summary()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("rows", "per_task")}, indent=2))
    print("saved to", out_dir)


if __name__ == "__main__":
    asyncio.run(main())
