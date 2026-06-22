"""Run a single condition — on one puzzle (debug) or across all 75 puzzles (--all).

Usage:
    python -m experiments.run_single <puzzle_id> [--method beam|bon|self_refine|qd] [opts]
    python -m experiments.run_single --all --method qd [opts]

Examples:
    python -m experiments.run_single lgp-test-3x3-24 --method qd     # one puzzle, verbose
    python -m experiments.run_single --all --method qd               # all 75, writes summary
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from experiments.run import (BEAM_WIDTH, BRANCHING, DIVERSITY_WEIGHT, NUM_COMBINATIONS,
                             NUM_EXPANSIONS, NUM_SEEDS, NUM_STEPS, PER_SIZE, QUALITY_WEIGHT,
                             SAMPLES, SIZES, WORKERS, _aggregate_trace)
from src.baselines.bon import best_of_n
from src.baselines.self_refine import self_refine
from src.method.beam_search import beam_search
from src.method.qd_search import qd_search
from src.common.verifiers import build_verifiers
from utils.data import load_puzzle, load_puzzles
from utils.llm import LLM


def _trace_tokens(trace_path) -> int:
    total = 0
    for line in trace_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            total += r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)
    return total


def _run_one(llm: LLM, p, args):
    """Run the chosen method on one puzzle; returns (verifiers_kept | None, Result)."""
    kept, verifiers = None, None
    if args.method in ("beam", "qd"):
        verifiers = build_verifiers(llm, p)
        kept = sum(v.passed_gold for v in verifiers)
    if args.method == "beam":
        res = beam_search(llm, p, verifiers, num_seeds=args.num_seeds,
                          num_steps=args.num_steps, beam_width=args.beam_width,
                          branching=args.branching)
    elif args.method == "qd":
        res = qd_search(llm, p, verifiers, num_seeds=args.num_seeds,
                        num_steps=args.num_steps, num_expansions=args.num_expansions,
                        num_combinations=args.num_combinations,
                        quality_weight=args.quality_weight,
                        diversity_weight=args.diversity_weight)
    elif args.method == "bon":
        res = best_of_n(llm, p, samples=args.samples)
    else:  # self_refine
        res = self_refine(llm, p, num_steps=args.num_steps)
    return kept, res


def _run_all(llm: LLM, args):
    """Run the chosen method across all puzzles; write rollups + summary.json."""
    m = args.method
    puzzles = load_puzzles(SIZES, PER_SIZE)
    results_dir = llm.run_dir / "results"
    results_dir.mkdir(exist_ok=True)
    print(f"Run dir: {llm.run_dir}\n{m} on {len(puzzles)} puzzles | workers: {WORKERS}\n")

    done = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(_run_one, llm, p, args): p for p in puzzles}
        for i, fut in enumerate(as_completed(futures), 1):
            p = futures[fut]
            try:
                kept, res = fut.result()
                done[p.id] = (p, kept, res)
                kept_str = f"kept={kept}/{p.k}" if kept is not None else ""
                print(f"[{i:2}/{len(puzzles)}] {p.id:18} {kept_str:10} "
                      f"{m}={'Y' if res.solved else 'n'}", flush=True)
            except Exception as e:
                print(f"[{i:2}/{len(puzzles)}] {p.id:18} FAILED: {e}", flush=True)

    agg = _aggregate_trace(llm.trace_path)
    by_size = {s: {"n": 0, f"{m}_solved": 0, f"{m}_tokens": 0, "build_tokens": 0} for s in SIZES}
    overall = {m: {"solved": 0, "tokens": 0, "calls": 0}}
    for pid, (p, kept, res) in done.items():
        calls, tokens = agg[pid][m]
        build_tokens = agg[pid]["build"][1]
        (results_dir / f"{p.id}.json").write_text(json.dumps({
            "puzzle_id": p.id, "size": p.size, "K": p.k, "verifiers_kept": kept,
            "build": {"calls": agg[pid]["build"][0], "tokens": build_tokens},
            "conditions": {m: {
                "solved": res.solved, "best_score": round(res.best_score, 3),
                "trajectory": [round(t, 3) for t in res.trajectory],
                "calls": calls, "tokens": tokens,
                "best_candidate": res.best_candidate, "extra": res.extra,
            }},
        }, indent=2))
        overall[m]["solved"] += int(res.solved)
        overall[m]["tokens"] += tokens
        overall[m]["calls"] += calls
        a = by_size[p.size]
        a["n"] += 1
        a[f"{m}_solved"] += int(res.solved)
        a[f"{m}_tokens"] += tokens
        a["build_tokens"] += build_tokens

    n = len(done)
    print(f"\n{m} solved {overall[m]['solved']}/{n} | "
          f"mean tokens {overall[m]['tokens'] / n:.0f}")
    out = {"sizes": SIZES, "per_size": PER_SIZE, "conditions": [m],
           "n_puzzles": n, "overall": overall, "by_size": by_size}
    (results_dir / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {n} rollups + summary.json to {results_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("puzzle_id", nargs="?", help="dataset id, e.g. lgp-test-3x3-24")
    ap.add_argument("--all", action="store_true", help="run across all 75 puzzles")
    ap.add_argument("--method", choices=["beam", "bon", "self_refine", "qd"], default="beam")
    ap.add_argument("--num-seeds", type=int, default=NUM_SEEDS, help="beam/qd: initial attempts")
    ap.add_argument("--num-steps", type=int, default=NUM_STEPS, help="beam/qd/self_refine iterations")
    ap.add_argument("--beam-width", type=int, default=BEAM_WIDTH, help="beam: top parents kept")
    ap.add_argument("--branching", type=int, default=BRANCHING, help="beam: revisions per parent")
    ap.add_argument("--num-expansions", type=int, default=NUM_EXPANSIONS, help="qd: expansions per step")
    ap.add_argument("--num-combinations", type=int, default=NUM_COMBINATIONS, help="qd: combinations per step")
    ap.add_argument("--quality-weight", type=float, default=QUALITY_WEIGHT, help="qd: selection quality weight")
    ap.add_argument("--diversity-weight", type=float, default=DIVERSITY_WEIGHT, help="qd: selection diversity weight")
    ap.add_argument("--samples", type=int, default=SAMPLES, help="bon: independent samples")
    args = ap.parse_args()

    if not args.all and not args.puzzle_id:
        ap.error("give a puzzle_id, or --all to run across all puzzles")

    llm = LLM()
    if args.all:
        _run_all(llm, args)
        return

    p = load_puzzle(args.puzzle_id)
    print(f"Puzzle {p.id} (size {p.size}, K={p.k}) | method={args.method}")
    print(f"Run dir: {llm.run_dir}\n")
    kept, res = _run_one(llm, p, args)
    if kept is not None:
        print(f"verifiers: {kept}/{p.k} pass gold\n")

    print(f"solved={res.solved}  best_score={res.best_score:.2f}  "
          f"tokens={_trace_tokens(llm.trace_path)}")
    print(f"trajectory: {[round(t, 2) for t in res.trajectory]}")
    print(f"extra: {res.extra}")
    if res.best_candidate is not None:
        print("\nbest_candidate:")
        print(json.dumps(res.best_candidate, indent=2))
    if not res.solved:
        print("\ngold:")
        print(json.dumps(p.gold, indent=2))
    print(f"\ntrace: {llm.trace_path}")


if __name__ == "__main__":
    main()
