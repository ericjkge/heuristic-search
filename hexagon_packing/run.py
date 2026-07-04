"""Experiment harness: sweep instances x conditions, aggregate, plot.

Conditions:
  search_soft -- verifier-guided search (w_soft > 0)
  search_raw  -- same search with soft verifiers off (the ablation)
  best_of_n   -- flat sampling, N matched to the search's total candidate count

All conditions for a run share one LLM run_dir (one trace.jsonl, tagged by
instance/condition) so the viewer can show everything together. Writes
summary.json + summary.png into the run_dir.

    python hexagon_packing/run.py --instances 11
    python hexagon_packing/run.py --instances 11,12 --repeats 3
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from best_of_n import best_of_n
from concurrency import run_parallel
from llm import LLM
from search import search

SOTA = {11: 3.931, 12: 3.942}  # AlphaEvolve best-known outer side per instance
CONDITIONS = ["search_soft", "search_raw", "best_of_n"]

HP = dict(num_seeds=2, num_steps=6, top_k=2, branching=2, k_verifiers=3, w_soft=0.3)
BON_N = 4


def run_one(cond, n, llm, s_target, hp, work_dir):
    if cond == "best_of_n":
        return best_of_n(n, llm, s_target=s_target, N=BON_N, work_dir=work_dir)
    w_soft = hp["w_soft"] if cond == "search_soft" else 0.0
    return search(
        n, llm, s_target=s_target, num_seeds=hp["num_seeds"], num_steps=hp["num_steps"],
        top_k=hp["top_k"], branching=hp["branching"], k_verifiers=hp["k_verifiers"],
        w_soft=w_soft, work_dir=work_dir,
    )


def plot_summary(results, instances, conditions, out_path):
    fig, ax = plt.subplots(figsize=(2 + 1.6 * len(instances), 4))
    width = 0.8 / max(1, len(conditions))
    base = np.arange(len(instances))
    for ci, cond in enumerate(conditions):
        ys = [min([r["best_s"] for r in results if r["n"] == n and r["condition"] == cond],
                  default=float("nan")) for n in instances]
        ax.bar(base + ci * width, ys, width=width, label=cond)
    for i, n in enumerate(instances):  # SOTA reference line per instance
        if n in SOTA:
            ax.hlines(SOTA[n], base[i] - width / 2, base[i] + len(conditions) * width - width / 2,
                      colors="k", linestyles="--", lw=1)
    ax.set_xticks(base + (len(conditions) - 1) * width / 2)
    ax.set_xticklabels([f"n={n}" for n in instances])
    ax.set_ylabel("best outer side s (lower is better)")
    ax.set_title("best s by condition (dashed = SOTA)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="11,12")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    hp = HP
    instances = [int(x) for x in args.instances.split(",")]
    conditions = args.conditions.split(",")

    llm = LLM()
    print(f"run dir: {llm.run_dir}  |  hp={hp}  |  BoN N={BON_N}")

    cells = []  # every (instance, condition, repeat) is independent -> run concurrently
    for n in instances:
        if n not in SOTA:
            print(f"skip n={n}: no SOTA target in SOTA dict")
            continue
        for cond in conditions:
            for rep in range(args.repeats):
                cells.append((n, cond, rep))

    def thunk(n, cond, rep):
        def go():
            suffix = f"n{n}_{cond}" if args.repeats == 1 else f"n{n}_{cond}_r{rep}"
            best, history = run_one(cond, n, llm, SOTA[n], hp, llm.run_dir / suffix)
            rec = {
                "n": n, "condition": cond, "repeat": rep,
                "best_s": best.raw_s, "s_target": SOTA[n],
                "n_candidates": len(history),
                "n_feasible": sum(c.feasible for c in history),
            }
            print(f"  n={n} {cond} r{rep}: best_s={best.raw_s:.4f} "
                  f"({rec['n_feasible']}/{rec['n_candidates']} feasible)")
            return rec
        return go

    results = run_parallel([thunk(*c) for c in cells])

    summary = {"hp": hp, "bon_n": BON_N, "results": results}
    (llm.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_summary(results, instances, conditions, llm.run_dir / "summary.png")
    print(f"\nsummary: {llm.run_dir / 'summary.json'}\nplot:    {llm.run_dir / 'summary.png'}")


if __name__ == "__main__":
    main()
