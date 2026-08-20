"""Majority-vote baseline: sample K grids per test pair, submit the 2 most
frequent (ties broken by first occurrence).

    uv run python arcagi2/baselines/majority_vote.py --dev --samples 5
"""

from __future__ import annotations

import asyncio
import json

from common import build_prompt, run_baseline_cli, sample_grid


async def solve_pair(task, pair_index, args):
    prompt = build_prompt(task, pair_index)
    results = await asyncio.gather(
        *(sample_grid(prompt, args) for _ in range(args.samples)))
    counts: dict[str, dict] = {}
    for r in results:
        if r["answer"] is None:
            continue
        key = json.dumps(r["answer"])
        counts.setdefault(key, {"grid": r["answer"], "votes": 0})["votes"] += 1
    ranked = sorted(counts.values(), key=lambda c: -c["votes"])
    grids = [c["grid"] for c in ranked[:2]]
    record = {"n_samples": len(results),
              "n_parsed": sum(r["answer"] is not None for r in results),
              "votes": [c["votes"] for c in ranked],
              # distinct grids with counts, so the sample pool can be reused
              # (e.g. verifier reranking) without regenerating
              "pool": [{"grid": c["grid"], "votes": c["votes"]} for c in ranked]}
    return grids, record


if __name__ == "__main__":
    run_baseline_cli(
        "majority_vote", solve_pair,
        lambda ap: ap.add_argument("--samples", type=int, default=5,
                                   help="independent samples per test pair"))
