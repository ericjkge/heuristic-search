"""Shared prompt, parsing, and run harness for ARC-AGI-2 baselines.

The prompt and grid parsing replicate the official harness
(github.com/arcprize/arc-agi-benchmarking) verbatim. Each baseline defines
solve_pair(task, pair_index, args) -> (attempt_grids, record) and calls
run_baseline_cli(); the harness handles task loading, per-pair fan-out,
official fractional scoring, and summary output.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

BASELINES_DIR = Path(__file__).resolve().parent
ARC_DIR = BASELINES_DIR.parent
ROOT = ARC_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm.asyncio import tqdm_asyncio

import llm
from arcagi2.tasks import Grid, Task, load_tasks, score_run

RUNS_DIR = BASELINES_DIR / "runs"

# Verbatim from arc-agi-benchmarking prompts/system_prompt.txt + prompt_manager.py
PROMPT_TEMPLATE = """You are participating in a puzzle solving competition. You are an expert at solving puzzles.

Below is a list of input and output pairs with a pattern. Your goal is to identify the pattern or transformation in the training examples that maps the input to the output, then apply that pattern to the test input to give a final output.

Respond in the format of the training output examples

--Training Examples--
{training_examples}
--End of Training Examples--

--Test Input--
{test_input}
--End of Test Input--

Your response:"""


def build_prompt(task: Task, pair_index: int) -> str:
    training_examples = ""
    for i, pair in enumerate(task.train):
        training_examples += f"--Example {i}-- \n\n INPUT: \n\n"
        training_examples += json.dumps(pair["input"]) + "\n\n"
        training_examples += "OUTPUT: \n\n"
        training_examples += json.dumps(pair["output"]) + "\n\n"
    return PROMPT_TEMPLATE.format(
        training_examples=training_examples,
        test_input=json.dumps(task.test[pair_index]["input"]))


# ---- parsing, ported from arc_agi_benchmarking/utils/parsing.py ----

def _backscan_json(text: str) -> list | None:
    last, closing = -1, None
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ("]", "}"):
            last, closing = i, text[i]
            break
    if last == -1:
        return None
    opening = "[" if closing == "]" else "{"
    depth, start = 1, -1
    for i in range(last - 1, -1, -1):
        if text[i] == closing:
            depth += 1
        elif text[i] == opening:
            depth -= 1
            if depth == 0:
                start = i
                break
    if start == -1:
        return None
    try:
        parsed = json.loads(text[start:last + 1])
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, list) and parsed and all(isinstance(r, list) for r in parsed):
        return parsed
    return None


def _from_boxed(text: str) -> list | None:
    m = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(1).strip())
            if isinstance(parsed, list) and all(isinstance(r, list) for r in parsed):
                return parsed
        except json.JSONDecodeError:
            pass
    return None


def parse_grid(text: str) -> Grid | None:
    for parser in (_from_boxed, _backscan_json):
        out = parser(text or "")
        if out is not None:
            return out
    return None


# ---- sampling ----

async def sample_grid(prompt: str | list[dict[str, str]], args,
                      retry_attempts: int | None = None) -> dict:
    """One sample with internal retries on unparseable output (mirrors the
    official harness's retry_attempts, which don't consume attempts)."""
    retries = args.retry_attempts if retry_attempts is None else retry_attempts
    text = ""
    for _ in range(retries + 1):
        try:
            text = await llm.generate(prompt, model=args.model, provider=args.provider,
                                      max_tokens=args.max_tokens,
                                      reasoning_effort=args.reasoning_effort)
        except Exception as e:
            return {"answer": None, "error": repr(e)}
        grid = parse_grid(text)
        if grid is not None:
            return {"answer": grid, "chars": len(text), "text": text}
    return {"answer": None, "error": "parse_failure", "raw": (text or "")[-2000:]}


# ---- CLI harness ----

SolvePair = Callable[[Task, int, argparse.Namespace], Awaitable[tuple[list[Grid], Any]]]


def run_baseline_cli(method: str, solve_pair: SolvePair,
                     add_args: Callable[[argparse.ArgumentParser], None] | None = None,
                     ) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="evaluation", choices=["training", "evaluation"])
    ap.add_argument("--ids", default=None, help="comma-separated task ids")
    ap.add_argument("--dev", action="store_true",
                    help="use the 20-task dev slice (arcagi2/dev_slice.txt)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--provider", default="abaka", choices=sorted(llm.PROVIDERS))
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--retry-attempts", type=int, default=2,
                    help="internal retries per sample on unparseable output")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--name", default=None)
    if add_args:
        add_args(ap)
    args = ap.parse_args()

    async def run_task(task: Task, out_dir: Path) -> tuple[dict, list[list[Grid]]]:
        attempts, records = [], []
        for i in range(len(task.test)):
            try:
                grids, record = await solve_pair(task, i, args)
            except Exception as e:
                grids, record = [], {"error": repr(e)}
            attempts.append(grids)
            records.append(record)
        (out_dir / "attempts" / f"{task.id}.json").write_text(
            json.dumps(records, ensure_ascii=False))
        return {"id": task.id}, attempts

    async def main() -> None:
        ids = args.ids.split(",") if args.ids else None
        if args.dev:
            ids = (ARC_DIR / "dev_slice.txt").read_text().split()
        tasks = load_tasks(args.split, ids=ids)[: args.limit]
        name = args.name or f"{method}-{time.strftime('%m%d-%H%M%S')}"
        out_dir = RUNS_DIR / name
        (out_dir / "attempts").mkdir(parents=True, exist_ok=True)

        results = await tqdm_asyncio.gather(
            *(run_task(t, out_dir) for t in tasks), desc="tasks")
        scores = score_run(tasks, {t.id: a for t, (_, a) in zip(tasks, results)})

        summary = {"name": name, "method": method, "split": args.split,
                   "model": args.model, "provider": args.provider,
                   "reasoning_effort": args.reasoning_effort,
                   "config": {k: v for k, v in vars(args).items()
                              if k not in ("ids", "name")},
                   "n": scores["n"], "score": scores["score"],
                   "n_solved": scores["n_solved"],
                   "per_task": scores["per_task"], "usage": llm.USAGE.summary()}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({k: v for k, v in summary.items() if k != "per_task"},
                         indent=2))
        print("saved to", out_dir)

    asyncio.run(main())
