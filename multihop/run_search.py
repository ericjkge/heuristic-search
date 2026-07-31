"""Run best-first search with self-improving verifiers on MuSiQue.

    uv run python -m multihop.run_search --limit 10           # search
    uv run python -m multihop.run_search --limit 10 --greedy  # no-search baseline

Writes per-question traces and a summary under runs/<name>/.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import time
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio

import llm
from .search import SearchConfig, best_first_search

from .env import MusiqueEnv
from .es_retriever import DEFAULT_URL, ESRetriever
from .metrics import score

PKG = Path(__file__).parent


async def run_one(
    ex: dict, args: argparse.Namespace, cfg: SearchConfig, out_dir: Path,
    retriever: ESRetriever,
) -> dict:
    env = MusiqueEnv(
        ex, retriever, model=args.model, provider=args.provider,
        retrieval_k=args.retrieval_k, max_searches=args.max_searches,
    )
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = (log_dir / f"{ex['id']}.jsonl").open("w")

    def log_event(ev: dict) -> None:
        log_file.write(json.dumps(ev) + "\n")
        log_file.flush()  # so the viewer sees events as they happen

    log_event({"event": "start", "id": ex["id"], "question": ex["question"], "gold": ex["answer"]})
    try:
        if args.greedy:
            answer = await env.run_greedy(log=log_event)
            trace: dict = {"mode": "greedy"}
            log_event({"event": "done", "answer": answer, "reason": "greedy"})
        else:
            result = await best_first_search(env, cfg, log=log_event)
            answer = result.answer
            trace = dataclasses.asdict(result)
    except Exception as e:  # keep one bad question from killing the batch
        answer, trace = "", {"error": repr(e)}
        log_event({"event": "error", "error": repr(e)})
    finally:
        log_file.close()
    m = score(answer or "", ex["answer"], ex.get("answer_aliases", []))
    row = {
        "id": ex["id"],
        "hop": ex["hop"],
        "question": ex["question"],
        "gold": ex["answer"],
        "prediction": answer,
        **m,
        "reason": trace.get("reason", "greedy" if args.greedy else "error"),
        "n_nodes": trace.get("n_nodes", 0),
        "parse_failures": env.parse_failures,
    }
    (out_dir / "traces").mkdir(exist_ok=True)
    with (out_dir / "traces" / f"{ex['id']}.json").open("w") as f:
        json.dump({"example_id": ex["id"], "row": row, "trace": trace}, f, indent=1)
    return row


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PKG / "data/dev_small.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=llm.DEFAULT_MODEL)
    ap.add_argument("--provider", default=llm.DEFAULT_PROVIDER, choices=sorted(llm.PROVIDERS))
    ap.add_argument("--name", default=None, help="run name (default: timestamp)")
    ap.add_argument("--greedy", action="store_true", help="greedy no-search baseline")
    ap.add_argument("--es-url", default=DEFAULT_URL)
    ap.add_argument("--retrieval-k", type=int, default=3)
    ap.add_argument("--max-searches", type=int, default=5)
    # search config
    ap.add_argument("--n-seed", type=int, default=2)
    ap.add_argument("--n-parents", type=int, default=1)
    ap.add_argument("--k-children", type=int, default=1)
    ap.add_argument("--max-rounds", type=int, default=12)
    ap.add_argument("--evolve-every", type=int, default=0,
                    help="verifier self-improvement every N rounds; 0 disables")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--satisfy-threshold", type=float, default=0.55)
    args = ap.parse_args()

    cfg = SearchConfig(
        n_seed=args.n_seed, n_parents=args.n_parents, k_children=args.k_children,
        max_rounds=args.max_rounds, evolve_every=args.evolve_every, tau=args.tau,
        satisfy_threshold=args.satisfy_threshold,
    )
    examples = [json.loads(l) for l in Path(args.data).open()][: args.limit]
    name = args.name or time.strftime("%m%d-%H%M%S") + ("-greedy" if args.greedy else "")
    out_dir = PKG / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    retriever = ESRetriever(args.es_url)
    n_docs = retriever.count()
    print(f"ES index ready: {n_docs:,} docs")
    rows = await tqdm_asyncio.gather(
        *(run_one(ex, args, cfg, out_dir, retriever) for ex in examples), desc="questions"
    )

    with (out_dir / "results.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n = len(rows)
    summary = {
        "name": name,
        "mode": "greedy" if args.greedy else "search",
        "model": args.model,
        "provider": args.provider,
        "config": dataclasses.asdict(cfg),
        "n": n,
        "em": sum(r["em"] for r in rows) / n,
        "f1": sum(r["f1"] for r in rows) / n,
        "by_hop": {
            hop: {
                "n": len(g),
                "em": sum(r["em"] for r in g) / len(g),
                "f1": sum(r["f1"] for r in g) / len(g),
            }
            for hop in sorted({r["hop"] for r in rows})
            if (g := [r for r in rows if r["hop"] == hop])
        },
        "reasons": {
            reason: sum(1 for r in rows if r["reason"] == reason)
            for reason in sorted({r["reason"] for r in rows})
        },
        "usage": dataclasses.asdict(llm.USAGE),
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k != "usage"}, indent=2))
    print("usage:", llm.USAGE.summary())
    print("saved to", out_dir)


if __name__ == "__main__":
    asyncio.run(main())
