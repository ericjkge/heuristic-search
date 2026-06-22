"""Full run: 3x2..5x6 (15 sizes x 5 puzzles = 75) across {beam, bon, self_refine, qd}.

beam = verifier-guided beam search; bon = pure sampling (pass@N oracle check);
self_refine = generic self-critique + refine; qd = quality-diversity search
(expand + combine). Puzzles run concurrently across a thread pool. Per-puzzle
build/<condition> tokens + calls are derived from the trace afterward (each row is
tagged by puzzle_id/condition/phase), robust to concurrent execution. Per-puzzle
rollups + overall and per-size summaries are written under runs/<ts>/results/.
"""

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.baselines.bon import best_of_n
from src.baselines.self_refine import self_refine
from src.method.qd_search import qd_search
from src.method.beam_search import beam_search
from src.common.verifiers import build_verifiers
from utils.data import load_puzzles
from utils.llm import LLM

SIZES = [f"{r}*{c}" for r in (3, 4, 5) for c in (2, 3, 4, 5, 6)]  # 15 sizes
PER_SIZE = 5            # puzzles per size -> 75 total
NUM_SEEDS = 2           # beam + qd: independent initial attempts
NUM_STEPS = 3           # beam iters + self_refine rounds + qd steps
BEAM_WIDTH = 2          # beam: top candidates kept per step
BRANCHING = 2           # beam: revisions generated per parent
NUM_EXPANSIONS = 2      # qd: candidates expanded per step
NUM_COMBINATIONS = 1    # qd: pairs combined per step
QUALITY_WEIGHT = 1.0    # qd: selection weight on quality
DIVERSITY_WEIGHT = 0.5  # qd: selection weight on verifier diversity
SAMPLES = 4             # bon: independent samples
WORKERS = 75            # concurrent puzzles (= one per puzzle; I/O-bound, pool caps at 1000)

CONDITIONS = ("beam", "bon", "self_refine", "qd")


def _work(llm: LLM, p, cfg: dict):
    """Process one puzzle through every condition; returns (p, kept, {name: Result})."""
    verifiers = build_verifiers(llm, p)
    kept = sum(v.passed_gold for v in verifiers)
    results = {
        "beam": beam_search(llm, p, verifiers, num_seeds=cfg["num_seeds"],
                            num_steps=cfg["num_steps"], beam_width=cfg["beam_width"],
                            branching=cfg["branching"]),
        "bon": best_of_n(llm, p, samples=cfg["samples"]),
        "self_refine": self_refine(llm, p, num_steps=cfg["num_steps"]),
        "qd": qd_search(llm, p, verifiers, num_seeds=cfg["num_seeds"],
                        num_steps=cfg["num_steps"], num_expansions=cfg["num_expansions"],
                        num_combinations=cfg["num_combinations"],
                        quality_weight=cfg["quality_weight"],
                        diversity_weight=cfg["diversity_weight"]),
    }
    return p, kept, results


def _aggregate_trace(trace_path) -> dict:
    """Per-puzzle compute from the trace: {puzzle_id: {build/<condition>: [calls, tokens]}}."""
    keys = ("build", *CONDITIONS)
    agg = defaultdict(lambda: {k: [0, 0] for k in keys})
    for line in trace_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        t = r.get("tags", {})
        pid = t.get("puzzle_id")
        if pid is None:
            continue
        if t.get("phase") == "build_verifier":
            key = "build"
        elif t.get("condition") in CONDITIONS:
            key = t["condition"]
        else:
            continue
        agg[pid][key][0] += 1
        agg[pid][key][1] += r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)
    return agg


def run(sizes: list[str] = SIZES, per_size: int = PER_SIZE, num_seeds: int = NUM_SEEDS,
        num_steps: int = NUM_STEPS, beam_width: int = BEAM_WIDTH, branching: int = BRANCHING,
        num_expansions: int = NUM_EXPANSIONS, num_combinations: int = NUM_COMBINATIONS,
        quality_weight: float = QUALITY_WEIGHT, diversity_weight: float = DIVERSITY_WEIGHT,
        samples: int = SAMPLES, workers: int = WORKERS) -> dict:
    llm = LLM()
    puzzles = load_puzzles(sizes, per_size)
    results_dir = llm.run_dir / "results"
    results_dir.mkdir(exist_ok=True)
    cfg = dict(num_seeds=num_seeds, num_steps=num_steps, beam_width=beam_width,
               branching=branching, num_expansions=num_expansions,
               num_combinations=num_combinations, quality_weight=quality_weight,
               diversity_weight=diversity_weight, samples=samples)

    print(f"Run dir: {llm.run_dir}\nPuzzles: {len(puzzles)} "
          f"({len(sizes)} sizes x {per_size}) | workers: {workers}\n")

    # --- run all puzzles concurrently ---
    done = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_work, llm, p, cfg): p for p in puzzles}
        for i, fut in enumerate(as_completed(futures), 1):
            p = futures[fut]
            try:
                _, kept, res = fut.result()
                done[p.id] = (p, kept, res)
                marks = " ".join(f"{name}={'Y' if res[name].solved else 'n'}"
                                 for name in CONDITIONS)
                print(f"[{i:2}/{len(puzzles)}] {p.id:18} kept={kept:2}/{p.k:<2} {marks}",
                      flush=True)
            except Exception as e:  # don't lose the whole run to one puzzle
                print(f"[{i:2}/{len(puzzles)}] {p.id:18} FAILED: {e}", flush=True)

    # --- derive per-puzzle compute from the trace, then write rollups + summaries ---
    agg = _aggregate_trace(llm.trace_path)
    overall = {name: {"solved": 0, "tokens": 0, "calls": 0}
               for name in ("build", *CONDITIONS)}
    by_size = {s: {"n": 0, "build_tokens": 0,
                   **{f"{c}_solved": 0 for c in CONDITIONS},
                   **{f"{c}_tokens": 0 for c in CONDITIONS}} for s in sizes}

    for pid, (p, kept, res) in done.items():
        compute = agg[pid]

        def cond(name):
            calls, tokens = compute[name]
            r = res[name]
            return {
                "solved": r.solved,
                "best_score": round(r.best_score, 3),
                "trajectory": [round(t, 3) for t in r.trajectory],
                "calls": calls,
                "tokens": tokens,
                "best_candidate": r.best_candidate,
                "extra": r.extra,
            }

        rollup = {
            "puzzle_id": p.id,
            "size": p.size,
            "K": p.k,
            "verifiers_kept": kept,
            "build": {"calls": compute["build"][0], "tokens": compute["build"][1]},
            "conditions": {name: cond(name) for name in CONDITIONS},
        }
        (results_dir / f"{p.id}.json").write_text(json.dumps(rollup, indent=2))

        overall["build"]["calls"] += compute["build"][0]
        overall["build"]["tokens"] += compute["build"][1]
        for name in CONDITIONS:
            overall[name]["solved"] += int(res[name].solved)
            overall[name]["calls"] += compute[name][0]
            overall[name]["tokens"] += compute[name][1]

        a = by_size[p.size]
        a["n"] += 1
        a["build_tokens"] += compute["build"][1]
        for name in CONDITIONS:
            a[f"{name}_solved"] += int(res[name].solved)
            a[f"{name}_tokens"] += compute[name][1]

    n = len(done)
    print("\n=== OVERALL ===")
    print(f"{'phase':12} {'solve_rate':>12} {'mean_tokens':>12} {'mean_calls':>11}")
    for name, o in overall.items():
        rate = "-" if name == "build" else f"{o['solved']}/{n}"
        print(f"{name:12} {rate:>12} {o['tokens'] / n:>12.0f} {o['calls'] / n:>11.1f}")

    print("\n=== SOLVE RATE BY SIZE ===")
    print(f"{'size':6} {'n':>3} " + " ".join(f"{c:>11}" for c in CONDITIONS))
    for s in sizes:
        a = by_size[s]
        if not a["n"]:
            continue
        cells = " ".join(f"{a[f'{c}_solved']:>3}/{a['n']:<7}" for c in CONDITIONS)
        print(f"{s:6} {a['n']:>3} {cells}")

    out = {"sizes": sizes, "per_size": per_size, **cfg, "workers": workers,
           "conditions": list(CONDITIONS), "n_puzzles": n,
           "overall": overall, "by_size": by_size}
    (results_dir / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {n} rollups + summary.json to {results_dir}")
    return out


if __name__ == "__main__":
    run()
