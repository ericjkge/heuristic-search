"""Download and prepare the MuSiQue (answerable) dataset.

Pulls MuSiQue-Ans from HuggingFace (dgslibisey/MuSiQue, mirror of the official
release from github.com/stonybrooknlp/musique) and writes JSONL files under
multihop/data/:

    train.jsonl        full training split      (~19.9k)
    dev.jsonl          full validation split    (~2.4k; test labels are hidden)
    dev_small.jsonl    hop-stratified dev subset for cheap iteration

Each row keeps the official fields:
    id, question, answer, answer_aliases, hop, paragraphs (20 per question,
    with is_supporting flags), question_decomposition

Run:  uv run python multihop/prepare_data.py [--small-size 120] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).parent / "data"
HF_DATASET = "dgslibisey/MuSiQue"


def hop_of(row: dict) -> int:
    # ids look like "2hop__482757_12019"; fall back to decomposition length
    rid = row.get("id", "")
    if "hop" in rid:
        return int(rid.split("hop")[0])
    return len(row.get("question_decomposition", []))


def to_record(row: dict) -> dict:
    return {
        "id": row["id"],
        "hop": hop_of(row),
        "question": row["question"],
        "answer": row["answer"],
        "answer_aliases": row.get("answer_aliases", []),
        "paragraphs": [
            {
                "idx": p["idx"],
                "title": p["title"],
                "text": p["paragraph_text"],
                "is_supporting": p["is_supporting"],
            }
            for p in row["paragraphs"]
        ],
        "question_decomposition": row.get("question_decomposition", []),
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hops = Counter(r["hop"] for r in records)
    hop_str = ", ".join(f"{h}-hop: {n}" for h, n in sorted(hops.items()))
    print(f"wrote {len(records):>6} rows -> {path}  ({hop_str})")


def stratified_sample(records: list[dict], size: int, seed: int) -> list[dict]:
    """Sample evenly across hop counts (2/3/4), preserving official order within strata."""
    by_hop: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_hop[r["hop"]].append(r)
    rng = random.Random(seed)
    per_hop = size // len(by_hop)
    out: list[dict] = []
    for hop in sorted(by_hop):
        pool = by_hop[hop]
        out.extend(rng.sample(pool, min(per_hop, len(pool))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small-size", type=int, default=120, help="size of dev_small.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(HF_DATASET)

    train = [to_record(r) for r in ds["train"]]
    dev = [to_record(r) for r in ds["validation"]]

    write_jsonl(DATA_DIR / "train.jsonl", train)
    write_jsonl(DATA_DIR / "dev.jsonl", dev)
    write_jsonl(DATA_DIR / "dev_small.jsonl", stratified_sample(dev, args.small_size, args.seed))


if __name__ == "__main__":
    main()
