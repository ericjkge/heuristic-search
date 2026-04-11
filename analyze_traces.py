"""Analyze per-puzzle trace JSONL files from an experiment directory.

Usage:
    python3 analyze_traces.py traces/sudoku_gen-ver_gpt-none_puzzles_0.6/
    python3 analyze_traces.py traces/sudoku_gen-ver_gpt-none_puzzles_0.6/ --merge merged.jsonl
"""

import argparse
import json
import statistics
from pathlib import Path


def load_puzzle_traces(trace_dir: Path) -> list[tuple[str, list[dict]]]:
    """Load all per-puzzle trace files, sorted by puzzle ID."""
    files = sorted(trace_dir.glob("p*.jsonl"), key=lambda p: p.stem[1:])
    result = []
    for f in files:
        pid = f.stem[1:]  # strip leading 'p'
        rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        result.append((pid, rows))
    return result


def compute_puzzle_stats(rows: list[dict]) -> dict:
    total_in = sum(r["tokens_in"] for r in rows)
    total_out = sum(r["tokens_out"] for r in rows)
    total_tokens = total_in + total_out
    total_time = sum(r["time_s"] for r in rows)

    gen_in = sum(r["tokens_in"] for r in rows if r["agent"] == "gen")
    gen_out = sum(r["tokens_out"] for r in rows if r["agent"] == "gen")
    ver_in = sum(r["tokens_in"] for r in rows if r["agent"].startswith("ver_"))
    ver_out = sum(r["tokens_out"] for r in rows if r["agent"].startswith("ver_"))

    return {
        "total_tokens": total_tokens,
        "tokens_in": total_in,
        "tokens_out": total_out,
        "gen_tokens": gen_in + gen_out,
        "ver_tokens": ver_in + ver_out,
        "total_time_s": round(total_time, 3),
        "num_calls": len(rows),
    }


def fmt_mean_std(values: list[float], fmt: str = ".1f") -> str:
    if not values:
        return "N/A"
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:{fmt}} ± {std:{fmt}}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze experiment trace files")
    parser.add_argument("trace_dir", type=Path, help="Experiment trace directory")
    parser.add_argument("--merge", type=Path, default=None,
                        help="Write merged flat JSONL (all rows in puzzle-ID order)")
    parser.add_argument("--csv", type=Path, default=None, help="Write per-puzzle stats CSV")
    args = parser.parse_args()

    if not args.trace_dir.exists():
        print(f"Directory not found: {args.trace_dir}")
        return

    puzzles = load_puzzle_traces(args.trace_dir)
    if not puzzles:
        print("No trace files found.")
        return

    per_puzzle_stats = [(pid, compute_puzzle_stats(rows)) for pid, rows in puzzles]

    # Print per-puzzle table
    print(f"{'PID':<20} {'Tokens':>8} {'In':>7} {'Out':>7} {'Gen':>7} {'Ver':>7} {'Time(s)':>8} {'Calls':>6}")
    print("-" * 80)
    for pid, s in per_puzzle_stats:
        print(f"{pid:<20} {s['total_tokens']:>8} {s['tokens_in']:>7} {s['tokens_out']:>7} "
              f"{s['gen_tokens']:>7} {s['ver_tokens']:>7} {s['total_time_s']:>8.1f} {s['num_calls']:>6}")

    # Aggregate stats
    all_tokens = [s["total_tokens"] for _, s in per_puzzle_stats]
    all_time = [s["total_time_s"] for _, s in per_puzzle_stats]
    all_gen = [s["gen_tokens"] for _, s in per_puzzle_stats]
    all_ver = [s["ver_tokens"] for _, s in per_puzzle_stats]

    print(f"\n{'='*80}")
    print(f"Puzzles:      {len(per_puzzle_stats)}")
    print(f"Total tokens: {fmt_mean_std(all_tokens, '.0f')} per puzzle")
    print(f"  Gen tokens: {fmt_mean_std(all_gen, '.0f')} per puzzle")
    print(f"  Ver tokens: {fmt_mean_std(all_ver, '.0f')} per puzzle")
    print(f"Total time:   {fmt_mean_std(all_time, '.1f')}s per puzzle")

    # Optional CSV output
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["puzzle_id", "total_tokens", "tokens_in", "tokens_out",
                                                    "gen_tokens", "ver_tokens", "total_time_s", "num_calls"])
            writer.writeheader()
            for pid, s in per_puzzle_stats:
                writer.writerow({"puzzle_id": pid, **s})
        print(f"\nCSV written to {args.csv}")

    # Optional merge output
    if args.merge:
        with open(args.merge, "w") as f:
            for _, rows in puzzles:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
        print(f"Merged JSONL written to {args.merge}")


if __name__ == "__main__":
    main()
