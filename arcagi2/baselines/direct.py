"""Direct baseline: N independent attempts per test pair (official protocol).

    uv run python arcagi2/baselines/direct.py --dev
"""

from __future__ import annotations

import asyncio

from common import build_prompt, run_baseline_cli, sample_grid


async def solve_pair(task, pair_index, args):
    prompt = build_prompt(task, pair_index)
    results = await asyncio.gather(
        *(sample_grid(prompt, args) for _ in range(args.attempts)))
    grids = [r["answer"] for r in results if r["answer"] is not None]
    return grids, [{k: v for k, v in r.items() if k != "text"} for r in results]


if __name__ == "__main__":
    run_baseline_cli(
        "direct", solve_pair,
        lambda ap: ap.add_argument("--attempts", type=int, default=2,
                                   help="independent attempts per test pair"))
