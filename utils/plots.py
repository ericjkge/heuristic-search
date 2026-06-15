"""Graph experiment results. Usage: python -m utils.plots [run_dir]

Reads runs/<ts>/results/summary.json and writes results/solve_by_size.png:
solve count (out of per_size) vs puzzle size, main vs bon.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = Path("runs")
MAIN_C, BON_C = "#2563eb", "#dc2626"


def latest_run() -> Path:
    runs = sorted((d for d in RUNS.glob("*") if (d / "results" / "summary.json").exists()),
                  reverse=True)
    if not runs:
        raise SystemExit("no runs with results/summary.json found")
    return runs[0]


def _size_key(s: str) -> tuple[int, int]:
    h, a = (int(x) for x in s.split("*"))
    return h, a  # order by houses, then attrs (3*2, 3*3, ..., 4*2, ...)


def solve_by_size(run_dir: Path):
    summary = json.loads((run_dir / "results" / "summary.json").read_text())
    bs = summary["by_size"]
    sizes = sorted((s for s, a in bs.items() if a["n"]), key=_size_key)
    main = [bs[s]["main_solved"] for s in sizes]
    bon = [bs[s]["bon_solved"] for s in sizes]
    top = max((bs[s]["n"] for s in sizes), default=5)
    x = range(len(sizes))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, main, marker="o", color=MAIN_C, label="main")
    ax.plot(x, bon, marker="o", color=BON_C, label="bon")
    ax.set_xticks(list(x))
    ax.set_xticklabels(sizes, rotation=45)
    ax.set_xlabel("Puzzle size")
    ax.set_ylabel(f"Solved (out of {top})")
    ax.set_ylim(-0.2, top + 0.2)
    ax.set_yticks(range(top + 1))
    ax.set_title("Solve count by puzzle size")
    ax.legend()
    fig.tight_layout()

    out = run_dir / "results" / "solve_by_size.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run()
    solve_by_size(run_dir)
