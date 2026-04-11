"""Download miniF2F_v2c dataset from GitHub, split into valid/test JSON.

Usage: python3 gen_minif2f.py [--output_dir data/minif2f]
"""

import argparse
import json
import urllib.request
from pathlib import Path


DATASET_URL = (
    "https://raw.githubusercontent.com/roozbeh-yz/miniF2F_v2/"
    "main/datasets/miniF2F_v2c.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download miniF2F_v2c dataset")
    parser.add_argument("--output_dir", type=str, default="data/minif2f")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading miniF2F_v2c from {DATASET_URL}...")
    with urllib.request.urlopen(DATASET_URL) as resp:
        raw_data = json.loads(resp.read().decode())

    # Normalize field names (some have spaces -> underscores)
    problems: list[dict] = []
    for entry in raw_data:
        problem = {}
        for key, value in entry.items():
            norm_key = key.strip().replace(" ", "_")
            problem[norm_key] = value
        problems.append(problem)

    # Split by 'split' field
    valid = [p for p in problems if p.get("split") == "valid"]
    test = [p for p in problems if p.get("split") == "test"]

    valid_path = output_dir / "problems_valid.json"
    test_path = output_dir / "problems_test.json"

    with open(valid_path, "w") as f:
        json.dump(valid, f, indent=2, ensure_ascii=False)

    with open(test_path, "w") as f:
        json.dump(test, f, indent=2, ensure_ascii=False)

    print(f"  valid: {len(valid)} problems -> {valid_path}")
    print(f"  test:  {len(test)} problems -> {test_path}")
    print(f"\nTotal: {len(problems)} problems saved to {output_dir}/")


if __name__ == "__main__":
    main()
