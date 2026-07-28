"""Experiment harness: sweep instances x conditions x repeats, aggregate, plot.

Conditions (all share ONE total candidate budget; comparisons are
best-s-vs-candidates curves plus final bars -- unbounded best-of-N is banned):

  best_of_n   -- flat sampling, N = search budget; population, no feedback
  self_refine -- iters = search budget; feedback, no population
  search_raw  -- bucket-rank + power-law parent sampling, verifiers OFF
  search      -- + soft-verifier tiebreak within raw buckets: full method

Every (instance, condition, repeat) cell is independent and runs concurrently.

    python pentagon_packing/run.py --instances 10
    python pentagon_packing/run.py --instances 10,25 --conditions search,best_of_n --repeats 3
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baselines import best_of_n, self_refine
from concurrency import run_parallel
from llm import LLM
from search import search

# Best-known sides, Erich Friedman's "Pentagons in Squares" (records as of 2026-07;
# only n=1,2 proved). Page shows "s = X+", i.e. best known to 5 decimals.
SOTA = {
    1: 1.59811, 2: 2.46345, 3: 2.90812, 4: 3.10252, 5: 3.54903,
    6: 3.88299, 7: 4.18037, 8: 4.38191, 9: 4.60033, 10: 4.90581,
    11: 5.11556, 12: 5.23221, 13: 5.52102, 14: 5.69659, 15: 5.90119,
    16: 6.06452, 17: 6.23761, 18: 6.32566, 19: 6.55781, 20: 6.66135,
    21: 6.85368, 22: 7.04445, 23: 7.20624, 24: 7.33690, 25: 7.45360,
}

HP = dict(num_seeds=2, num_steps=5, expansions=4, pop_size=8, alpha=1.0)
BUDGET = HP["num_seeds"] + HP["num_steps"] * HP["expansions"]  # 22
CONDITIONS = ["best_of_n", "self_refine", "search_raw", "search"]


def run_one(cond, n, llm, s_target, work_dir, rep=0, label=None):
    label = label or cond  # repeat-tagged condition (e.g. search_r1) for the trace/viewer
    if cond == "best_of_n":
        best, hist = best_of_n(n, llm, s_target=s_target, N=BUDGET, work_dir=work_dir,
                               condition=label)
    elif cond == "self_refine":
        best, hist = self_refine(n, llm, s_target=s_target, iters=BUDGET, work_dir=work_dir,
                                 condition=label)
    else:
        best, hist = search(
            n, llm, s_target=s_target, num_seeds=HP["num_seeds"],
            num_steps=HP["num_steps"], expansions=HP["expansions"],
            pop_size=HP["pop_size"], alpha=HP["alpha"],
            use_soft=(cond != "search_raw"), seed=rep,
            work_dir=work_dir, condition=label,
        )
    # anytime curve: best feasible s after each candidate, in evaluation order
    curve, cur = [], float("inf")
    for c in hist:
        if c.feasible:
            cur = min(cur, c.raw_s)
        curve.append(cur if cur < float("inf") else None)
    return {"best_s": best.raw_s if best else None, "curve": curve}


def plot_summary(results, instances, conditions, out_path):
    """results: {(n, cond, rep): {best_s, curve}}. Bars = best over repeats."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    width = 0.8 / max(1, len(conditions))
    base = np.arange(len(instances))
    for ci, cond in enumerate(conditions):
        ys = []
        for n in instances:
            vals = [v["best_s"] for (nn, cc, _), v in results.items()
                    if nn == n and cc == cond and v["best_s"] is not None]
            ys.append(min(vals) if vals else np.nan)
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

    ax = axes[1]  # anytime curves for the first instance, one line per repeat
    n0 = instances[0]
    colors = {c: f"C{i}" for i, c in enumerate(conditions)}
    seen = set()
    for (n, cond, rep), v in sorted(results.items()):
        if n != n0 or not v.get("curve"):
            continue
        ys = [y if y is not None else np.nan for y in v["curve"]]
        ax.plot(range(1, len(ys) + 1), ys, marker=".", color=colors.get(cond),
                alpha=0.8, label=cond if cond not in seen else None)
        seen.add(cond)
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
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()
    instances = [int(x) for x in args.instances.split(",")]
    conditions = args.conditions.split(",")

    llm = LLM()
    print(f"run dir: {llm.run_dir} | budget: {BUDGET} candidates per LLM condition")

    cells = []
    for n in instances:
        if n not in SOTA:
            raise SystemExit(f"no SOTA entry for n={n}; add it to run.py")
        for cond in conditions:
            for rep in range(args.repeats):
                cells.append((n, cond, rep))

    def thunk(n, cond, rep):
        def go():
            label = cond if args.repeats == 1 else f"{cond}_r{rep}"
            out = run_one(cond, n, llm, SOTA[n], llm.run_dir / f"n{n}_{label}",
                          rep=rep, label=label)
            print(f"[n={n}] {cond:16} r{rep}: best_s={out['best_s']}")
            return ((n, cond, rep), out)
        return go

    results = dict(run_parallel([thunk(*c) for c in cells]))

    (llm.run_dir / "summary.json").write_text(json.dumps(
        {f"{n}/{c}/r{r}": v for (n, c, r), v in results.items()}, indent=2))
    plot_summary(results, instances, conditions, llm.run_dir / "summary.png")
    print(f"summary: {llm.run_dir}/summary.json + summary.png")


if __name__ == "__main__":
    main()
