"""Experiment harness: sweep instances x conditions, aggregate, plot.

Conditions (all LLM conditions share ONE total candidate budget; comparisons are
best-s-vs-candidates curves plus final bars -- unbounded best-of-N is banned):

  multistart      -- no LLM: random restarts + greedy local moves (numerics floor)
  best_of_n       -- flat sampling, N = search budget; population, no feedback
  hillclimb       -- iters = search budget; feedback, no population
  search_raw      -- QD loop, w_soft=0, div_weight=0 (selection on raw s only)
  search_quality  -- + hand-coded quality verifiers (w_soft=0.3)
  search_qd       -- + disjoint diversity descriptors (div_weight=0.3): full method

    python pentagon_packing/run.py --instances 10
    python pentagon_packing/run.py --instances 10,25 --conditions search_qd,best_of_n
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baselines import best_of_n, hillclimb, multistart
from llm import LLM
from search import search

# Best-known outer sides from Erich Friedman's "Pentagons in Squares" page.
# NOTE: records there are moving monthly in 2026 -- refresh before any claims.
SOTA = {10: 4.906, 25: 7.454}

HP = dict(num_seeds=4, num_steps=3, top_k=2, branching=2, pop_size=8,
          w_soft=0.3, div_weight=0.3)
BUDGET = HP["num_seeds"] + HP["num_steps"] * HP["top_k"] * HP["branching"]  # 16
CONDITIONS = ["multistart", "best_of_n", "hillclimb",
              "search_raw", "search_quality", "search_qd"]


def run_one(cond, n, llm, s_target, work_dir):
    if cond == "multistart":
        s, _ = multistart(n)
        return {"best_s": s, "curve": [s]}
    if cond == "best_of_n":
        best, hist = best_of_n(n, llm, s_target=s_target, N=BUDGET, work_dir=work_dir)
    elif cond == "hillclimb":
        best, hist = hillclimb(n, llm, s_target=s_target, iters=BUDGET, work_dir=work_dir)
    else:
        w_soft = 0.0 if cond == "search_raw" else HP["w_soft"]
        div_w = HP["div_weight"] if cond == "search_qd" else 0.0
        best, hist = search(
            n, llm, s_target=s_target, num_seeds=HP["num_seeds"],
            num_steps=HP["num_steps"], top_k=HP["top_k"], branching=HP["branching"],
            pop_size=HP["pop_size"], w_soft=w_soft, div_weight=div_w,
            work_dir=work_dir, condition=cond,
        )
    # anytime curve: best feasible s after each candidate, in evaluation order
    curve, cur = [], float("inf")
    for c in hist:
        if c.feasible:
            cur = min(cur, c.raw_s)
        curve.append(cur if cur < float("inf") else None)
    return {"best_s": best.raw_s if best else None, "curve": curve}


def plot_summary(results, instances, conditions, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    width = 0.8 / max(1, len(conditions))
    base = np.arange(len(instances))
    for ci, cond in enumerate(conditions):
        ys = [results.get((n, cond), {}).get("best_s") or np.nan for n in instances]
        ax.bar(base + ci * width, ys, width=width, label=cond)
    for i, n in enumerate(instances):
        if n in SOTA:
            ax.hlines(SOTA[n], base[i] - width / 2,
                      base[i] + len(conditions) * width - width / 2,
                      colors="k", linestyles="--", lw=1)
    ax.set_xticks(base + (len(conditions) - 1) * width / 2)
    ax.set_xticklabels([f"n={n}" for n in instances])
    ax.set_ylabel("best s (lower better; dashes = best known)")
    ax.legend(fontsize=7)

    ax = axes[1]  # anytime curves for the first instance
    n0 = instances[0]
    for cond in conditions:
        curve = results.get((n0, cond), {}).get("curve")
        if curve:
            xs = range(1, len(curve) + 1)
            ys = [v if v is not None else np.nan for v in curve]
            ax.plot(xs, ys, marker=".", label=cond)
    if n0 in SOTA:
        ax.axhline(SOTA[n0], color="k", ls="--", lw=1)
    ax.set_xlabel("candidates evaluated")
    ax.set_ylabel(f"best s so far (n={n0})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="10")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    args = ap.parse_args()
    instances = [int(x) for x in args.instances.split(",")]
    conditions = args.conditions.split(",")

    llm = LLM()
    print(f"run dir: {llm.run_dir} | budget: {BUDGET} candidates per LLM condition")
    results = {}
    for n in instances:
        s_target = SOTA.get(n)
        if s_target is None:
            raise SystemExit(f"no SOTA entry for n={n}; add it to run.py")
        for cond in conditions:
            work_dir = llm.run_dir / f"n{n}_{cond}"
            out = run_one(cond, n, llm, s_target, work_dir)
            results[(n, cond)] = out
            print(f"[n={n}] {cond:16} best_s={out['best_s']}")

    (llm.run_dir / "summary.json").write_text(json.dumps(
        {f"{n}/{c}": v for (n, c), v in results.items()}, indent=2))
    plot_summary(results, instances, conditions, llm.run_dir / "summary.png")
    print(f"summary: {llm.run_dir}/summary.json + summary.png")


if __name__ == "__main__":
    main()
