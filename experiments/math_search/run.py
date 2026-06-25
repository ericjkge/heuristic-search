"""
CLI entry point for AIME/HMMT math reasoning search experiments.

Usage:
  python -m experiments.math_search.run \\
    --dataset data/aime_sample.jsonl \\
    --mode verifier_guided_search \\
    --output runs/aime_test.jsonl

  python -m experiments.math_search.run \\
    --dataset data/aime_sample.jsonl \\
    --mode verifier_guided_search \\
    --num_seeds 8 --num_steps 3 --num_expansions 4 --num_combinations 2 \\
    --population_size 8 --quality_weight 1.0 --diversity_weight 0.3 \\
    --config configs/math_search.yaml \\
    --output runs/aime_search_test.jsonl

  # HuggingFace dataset:
  python -m experiments.math_search.run \\
    --hf_dataset math-ai/aime25 \\
    --mode bon \\
    --output runs/aime25_bon.jsonl

  # Single problem (debug):
  python -m experiments.math_search.run \\
    --dataset data/aime_sample.jsonl \\
    --problem_id synthetic_001 \\
    --mode single \\
    --output runs/debug.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from src.math.baselines.bon import best_of_n
from src.math.baselines.self_refine import self_refine
from src.math.baselines.single import single_attempt
from src.math.baselines.verifier_rerank import verifier_rerank
from src.math.data import MathProblem, load_problems, load_problems_hf
from src.math.method.qd_search import math_qd_search
from utils.llm import LLM


def _load_yaml(path: str) -> dict:
    if not _HAS_YAML:
        print("Warning: PyYAML not installed; ignoring --config.", file=sys.stderr)
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _build_cfg(args: argparse.Namespace) -> dict:
    """Merge YAML config (base) with CLI flag overrides."""
    cfg: dict = {}
    if args.config and Path(args.config).exists():
        raw = _load_yaml(args.config)
        cfg.update(raw.get("search", {}))
        cfg.update(raw.get("bon", {}))
        cfg.update(raw.get("verifier_rerank", {}))
        cfg["model"] = raw.get("model", {})
    # CLI overrides
    for key in (
        "num_seeds", "num_steps", "num_expansions", "num_combinations",
        "population_size", "quality_weight", "diversity_weight",
        "n", "selection",
    ):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    return cfg


def _run_one(llm: LLM, problem: MathProblem, mode: str, cfg: dict):
    if mode == "single":
        return single_attempt(llm, problem)
    if mode == "bon":
        return best_of_n(
            llm, problem,
            n=cfg.get("n", 8),
            selection=cfg.get("selection", "majority_vote"),
        )
    if mode == "verifier_rerank":
        return verifier_rerank(llm, problem, n=cfg.get("n", 8))
    if mode == "self_refine":
        return self_refine(llm, problem, num_steps=cfg.get("num_steps", 3))
    if mode == "verifier_guided_search":
        return math_qd_search(
            llm, problem,
            num_seeds=cfg.get("num_seeds", 8),
            num_steps=cfg.get("num_steps", 3),
            num_expansions=cfg.get("num_expansions", 4),
            num_combinations=cfg.get("num_combinations", 2),
            population_size=cfg.get("population_size", 8),
            quality_weight=cfg.get("quality_weight", 1.0),
            diversity_weight=cfg.get("diversity_weight", 0.3),
            final_selection=cfg.get("final_selection", "confidence_first"),
        )
    raise ValueError(f"Unknown mode: {mode}")


def _count_tokens_from_trace(trace_path) -> int:
    """Sum prompt_tokens + completion_tokens across all calls in the trace file."""
    total = 0
    try:
        with open(trace_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                total += row.get("prompt_tokens", 0) + row.get("completion_tokens", 0)
    except FileNotFoundError:
        pass
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src_group = ap.add_mutually_exclusive_group()
    src_group.add_argument("--dataset", help="Path to local JSONL dataset file")
    src_group.add_argument("--hf_dataset", help="HuggingFace dataset name, e.g. math-ai/aime25")
    ap.add_argument("--hf_split", default="test", help="HuggingFace split (default: test)")
    ap.add_argument("--problem_id", default=None, help="Run a single problem by ID")
    ap.add_argument("--mode",
                    choices=["single", "bon", "verifier_rerank", "self_refine", "verifier_guided_search"],
                    default="verifier_guided_search")
    ap.add_argument("--config", default="configs/math_search.yaml")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    # Search params (override YAML)
    ap.add_argument("--num_seeds", type=int, default=None)
    ap.add_argument("--num_steps", type=int, default=None)
    ap.add_argument("--num_expansions", type=int, default=None)
    ap.add_argument("--num_combinations", type=int, default=None)
    ap.add_argument("--population_size", type=int, default=None)
    ap.add_argument("--quality_weight", type=float, default=None)
    ap.add_argument("--diversity_weight", type=float, default=None)
    # BoN params
    ap.add_argument("--n", type=int, default=None, help="N for bon/verifier_rerank")
    ap.add_argument("--selection", default=None, help="Selection mode for bon")
    args = ap.parse_args()

    if not args.dataset and not args.hf_dataset:
        ap.error("One of --dataset or --hf_dataset is required.")

    cfg = _build_cfg(args)

    # Load problems
    if args.hf_dataset:
        problems = load_problems_hf(args.hf_dataset, split=args.hf_split)
    else:
        problems = load_problems(args.dataset)

    if args.problem_id:
        problems = [p for p in problems if p.id == args.problem_id]
        if not problems:
            sys.exit(f"Problem ID {args.problem_id!r} not found in dataset.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    workers = min(len(problems), 15)
    llm = LLM()
    print(f"Run dir: {llm.run_dir}")
    print(f"Mode: {args.mode} | Problems: {len(problems)} | Workers: {workers}")
    print()

    solved_count = 0
    print_lock = threading.Lock()
    counter = [0]  # mutable for closure

    results_map: dict[str, object] = {}

    def _work(problem: MathProblem):
        result = _run_one(llm, problem, args.mode, cfg)
        with print_lock:
            counter[0] += 1
            i = counter[0]
            status = "✓" if result.solved else "✗"
            print(f"[{i}/{len(problems)}] {problem.id} ... "
                  f"{status}  predicted={result.predicted_answer}  gt={problem.answer}  "
                  f"score={result.best_verifier_score:.2f}  calls={result.model_calls}")
        return result

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_work, p): p for p in problems}
        for fut in as_completed(futures):
            result = fut.result()
            results_map[result.problem_id] = result

    # Write output in original problem order
    with open(out_path, "w") as out_f:
        for problem in problems:
            result = results_map[problem.id]
            if result.solved:
                solved_count += 1
            out_f.write(json.dumps(result.to_dict()) + "\n")

    total_tokens = _count_tokens_from_trace(llm.trace_path)

    n = len(problems)
    acc = solved_count / n * 100 if n else 0.0
    print()
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Solved: {solved_count}/{n} ({acc:.1f}%)")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Output: {out_path}")
    print(f"Trace: {llm.trace_path}")


if __name__ == "__main__":
    main()
