"""Run verifier-guided best-first search on DeepPlanning travel tasks.

    uv run python deepplanning/travelplanning/search/run_search.py --ids 0
    uv run python deepplanning/travelplanning/search/run_search.py --limit 10

Writes runs/<name>/ with reports/ (plan per task), logs/ (JSONL events for the
viewer), evaluation via their official conversion + rule pipeline, and
summary.json. The greedy baseline is their own pipeline (travelplanning/run.py),
not duplicated here.
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
TP_DIR = SEARCH_DIR.parent
ROOT = TP_DIR.parent.parent
for p in (str(ROOT), str(SEARCH_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tqdm.asyncio import tqdm_asyncio

import llm
from scorer import score_run
from search import SearchConfig, TravelSearch

RUNS_DIR = SEARCH_DIR / "runs"
TEST_DATA = TP_DIR / "data" / "travelplanning_query_en.json"


async def run_one(ex: dict, args, cfg: SearchConfig, out_dir: Path) -> dict:
    tid = f"id_{ex['id']}"
    log_file = (out_dir / "logs" / f"{tid}.jsonl").open("w")

    def log_event(ev: dict) -> None:
        log_file.write(json.dumps(ev, ensure_ascii=False) + "\n")
        log_file.flush()

    log_event({"event": "start", "id": tid, "question": ex["query"]})
    try:
        search = TravelSearch(ex, model=args.model, provider=args.provider,
                              cfg=cfg, log=log_event,
                              reasoning_effort=args.reasoning_effort)
        result = await search.run()
        row = {"id": tid, "reason": result.reason, "n_nodes": result.n_nodes,
               "rounds": result.rounds, "llm_calls": search.llm_calls,
               "parse_failures": search.parse_failures,
               "plan_chars": len(result.plan)}
        (out_dir / "reports" / f"{tid}.txt").write_text(result.plan)
    except Exception as e:
        log_event({"event": "error", "error": repr(e)})
        row = {"id": tid, "reason": "error", "error": repr(e)}
    finally:
        log_file.close()
    return row


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=None, help="comma-separated task ids, e.g. 0,3,7")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--provider", default="abaka", choices=sorted(llm.PROVIDERS))
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-eval", action="store_true", help="skip conversion+evaluation")
    # search config
    # search config — defaults come from SearchConfig so file edits take effect
    dflt = SearchConfig()
    ap.add_argument("--n-seed", type=int, default=dflt.n_seed)
    ap.add_argument("--n-parents", type=int, default=dflt.n_parents)
    ap.add_argument("--k-children", type=int, default=dflt.k_children)
    ap.add_argument("--max-rounds", type=int, default=dflt.max_rounds)
    ap.add_argument("--evolve-every", type=int, default=dflt.evolve_every,
                    help="add verifiers every N rounds; 0 disables")
    ap.add_argument("--tau", type=float, default=dflt.tau)
    ap.add_argument("--deg-coef", type=float, default=dflt.deg_coef)
    ap.add_argument("--seed-verifiers", default=dflt.seed_verifiers,
                    help="how many rubric items to seed, e.g. '8-12'")
    ap.add_argument("--evolve-verifiers", default=dflt.evolve_verifiers,
                    help="how many rubric items to add per evolution, e.g. '1-3'")
    ap.add_argument("--max-turns", type=int, default=dflt.max_turns,
                    help="tool-call turns per episode before a forced plan")
    ap.add_argument("--no-feedback", action="store_true",
                    help="do not show rubric scores to the agent on revision")
    args = ap.parse_args()

    cfg = SearchConfig(
        n_seed=args.n_seed, n_parents=args.n_parents, k_children=args.k_children,
        max_rounds=args.max_rounds, tau=args.tau, deg_coef=args.deg_coef,
        evolve_every=args.evolve_every,
        seed_verifiers=args.seed_verifiers, evolve_verifiers=args.evolve_verifiers,
        max_turns=args.max_turns,
        feedback=not args.no_feedback,
    )
    examples = json.loads(TEST_DATA.read_text())
    if args.ids:
        wanted = {s.strip() for s in args.ids.split(",")}
        examples = [e for e in examples if str(e["id"]) in wanted]
    examples = examples[: args.limit]

    name = args.name or time.strftime("%m%d-%H%M%S")
    out_dir = RUNS_DIR / name
    for sub in ("reports", "logs"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    rows = await tqdm_asyncio.gather(
        *(run_one(ex, args, cfg, out_dir) for ex in examples), desc="tasks")

    summary = {
        "name": name, "model": args.model, "provider": args.provider,
        "config": dataclasses.asdict(cfg), "n": len(rows), "rows": rows,
        "reasons": {r: sum(1 for x in rows if x.get("reason") == r)
                    for r in sorted({x.get("reason") for x in rows})},
        "usage": dataclasses.asdict(llm.USAGE),
    }
    if not args.no_eval:
        print("\nscoring via official conversion + evaluation ...")
        summary["scores"] = score_run(out_dir)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    printable = {k: v for k, v in summary.items() if k not in ("usage", "rows")}
    if "scores" in printable:
        printable["scores"] = {k: v for k, v in printable["scores"].items() if k != "per_task"}
    print(json.dumps(printable, indent=2))
    print("usage:", llm.USAGE.summary())
    print("saved to", out_dir)


if __name__ == "__main__":
    asyncio.run(main())
