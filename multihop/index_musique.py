"""Index the pooled MuSiQue paragraphs into Elasticsearch.

The corpus is the union of all 20-paragraph candidate sets across the MuSiQue
train + dev splits, deduplicated. This is the IRCoT-style retrieval setting:
harder than reading the 20 paragraphs given per question, but every question's
supporting paragraphs are guaranteed to be in the corpus.

    uv run python -m multihop.index_musique            # index (skips if done)
    uv run python -m multihop.index_musique --recreate # drop and re-index
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from elasticsearch import Elasticsearch, helpers

from .es_retriever import DEFAULT_URL, INDEX

DATA_DIR = Path(__file__).parent / "data"
SPLITS = ("train.jsonl", "dev.jsonl")


def stream_docs():
    seen: set[str] = set()
    for split in SPLITS:
        for line in (DATA_DIR / split).open():
            row = json.loads(line)
            for p in row["paragraphs"]:
                key = hashlib.md5((p["title"] + "\x00" + p["text"]).encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                yield {
                    "_index": INDEX,
                    "_id": key,
                    "contents": f'"{p["title"]}"\n{p["text"]}',
                }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--recreate", action="store_true")
    args = ap.parse_args()

    es = Elasticsearch(args.url, request_timeout=120)
    if es.indices.exists(index=INDEX):
        if not args.recreate and es.count(index=INDEX)["count"] > 0:
            print(f"index '{INDEX}' already has {es.count(index=INDEX)['count']:,} docs; "
                  "use --recreate to redo")
            return
        es.indices.delete(index=INDEX)

    es.indices.create(
        index=INDEX,
        settings={"number_of_shards": 1, "number_of_replicas": 0},
        mappings={"properties": {"contents": {"type": "text"}}},
    )

    t0 = time.time()
    done = 0
    for ok, item in helpers.streaming_bulk(
        es, stream_docs(), chunk_size=2000, request_timeout=120,
        raise_on_error=False, max_retries=3,
    ):
        done += 1
        if not ok:
            print("failed:", item)

    es.indices.refresh(index=INDEX)
    print(f"done: {es.count(index=INDEX)['count']:,} unique paragraphs "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
