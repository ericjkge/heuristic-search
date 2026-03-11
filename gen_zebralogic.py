"""Download ZebraLogic grid_mode dataset from HuggingFace.

Usage: python3 download_zebralogic.py [--limit 10]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ZebraLogic dataset")
    parser.add_argument("--limit", type=int, default=0, help="Max puzzles per size (0=all)")
    parser.add_argument("--output_dir", type=str, default="data/zebralogic")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("WildEval/ZebraLogic", "grid_mode", split="test")

    # Group by size
    by_size: dict[str, list[dict]] = defaultdict(list)
    for row in ds:
        size = row["size"]
        entry = {
            "id": row["id"],
            "size": size,
            "puzzle": row["puzzle"],
            "solution": row["solution"],
        }
        by_size[size].append(entry)

    total = 0
    for size, entries in sorted(by_size.items()):
        if args.limit > 0:
            entries = entries[: args.limit]

        # "5*6" -> "5x6"
        size_label = size.replace("*", "x")
        out_path = output_dir / f"puzzles_{size_label}.json"
        with open(out_path, "w") as f:
            json.dump(entries, f, indent=2)

        total += len(entries)
        print(f"  {size_label}: {len(entries)} puzzles -> {out_path}")

    print(f"\nTotal: {total} puzzles saved to {output_dir}/")


if __name__ == "__main__":
    main()
