"""Run one ZebraLogic puzzle by id through a single method (search or bon).

Usage:
    python -m experiments.run_single <puzzle_id> [--method search|bon] [opts]

Examples:
    python -m experiments.run_single lgp-test-3x3-24
    python -m experiments.run_single lgp-test-4x4-6 --method search --steps 6
    python -m experiments.run_single lgp-test-4x4-6 --method bon --samples 32
"""

import argparse
import json

from experiments.run import BEAM_W, REVISIONS, SAMPLES, STEPS
from src.baselines.bon import best_of_n
from src.method.search import beam_search
from src.common.verifiers import build_verifiers
from utils.data import load_puzzle
from utils.llm import LLM


def _trace_tokens(trace_path) -> int:
    total = 0
    for line in trace_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            total += r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("puzzle_id", help="dataset id, e.g. lgp-test-3x3-24")
    ap.add_argument("--method", choices=["search", "bon"], default="search")
    ap.add_argument("--steps", type=int, default=STEPS, help="search: beam iterations")
    ap.add_argument("--beam-w", type=int, default=BEAM_W, help="search: beam width W")
    ap.add_argument("--revisions", type=int, default=REVISIONS, help="search: M (init pool + revisions/parent)")
    ap.add_argument("--samples", type=int, default=SAMPLES, help="bon: independent samples")
    args = ap.parse_args()

    llm = LLM()
    p = load_puzzle(args.puzzle_id)
    print(f"Puzzle {p.id} (size {p.size}, K={p.k}) | method={args.method}")
    print(f"Run dir: {llm.run_dir}\n")

    if args.method == "search":
        verifiers = build_verifiers(llm, p)
        kept = sum(v.passed_gold for v in verifiers)
        print(f"verifiers: {kept}/{p.k} pass gold\n")
        res = beam_search(llm, p, verifiers, steps=args.steps,
                          beam_w=args.beam_w, revisions=args.revisions)
    else:
        res = best_of_n(llm, p, samples=args.samples)

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
