"""LiveCodeBench-Pro problem loading.

Problems come from the gated HF dataset QAQAQAQAQ/LiveCodeBench-Pro (needs
HF_TOKEN in .env); official test data per problem comes from the open dataset
QAQAQAQAQ/LiveCodeBench-Pro-Testcase and is fetched lazily by the judge.

The HF splits are time windows (biannual_2024_7_12, quater_2024_10_12, ...)
that overlap; problems are deduplicated by problem_id like their benchmark.py.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

LCB_DIR = Path(__file__).resolve().parent
ROOT = LCB_DIR.parent
for p in (str(ROOT), str(LCB_DIR)):  # LCB_DIR holds their harness files (api_interface, judge, util)
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

DATASET = "QAQAQAQAQ/LiveCodeBench-Pro"
DIFFICULTIES = ("easy", "medium", "hard")


@dataclass
class Problem:
    problem_id: str
    title: str
    difficulty: str
    platform: str
    statement: str
    link: str
    time_limit_ms: int
    memory_limit_mb: int
    split: str

    @property
    def examples(self) -> list[tuple[str, str]]:
        """(input, output) pairs from the statement's Example section."""
        # fences may close on the last content line ("1 3```"); the markdown puts
        # blank lines between input rows, which are dropped
        ins = re.findall(r"#### Input #\d+\s*\n```(.*?)```", self.statement, re.S)
        outs = re.findall(r"#### Output #\d+\s*\n```(.*?)```", self.statement, re.S)
        ins = ["\n".join(l for l in b.split("\n") if l.strip()) for b in ins]
        outs = [o.strip("\n") for o in outs]
        return list(zip(ins, outs))


TESTCASE_DATASET = "QAQAQAQAQ/LiveCodeBench-Pro-Testcase"
DATA_DIR = LCB_DIR / "data"
MEDIUM100 = DATA_DIR / "medium100.txt"  # eval set: 100 newest non-interactive testable mediums


def read_id_list(path: Path = MEDIUM100) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def load_medium100() -> list[Problem]:
    """The eval set, in the saved (newest-first) order."""
    ids = read_id_list()
    by_id = {p.problem_id: p for p in load_problems(ids=ids)}
    return [by_id[i] for i in ids]


def available_testcase_ids() -> set[str]:
    """problem_ids that have official test data (706 of the 864 statements)."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(TESTCASE_DATASET)
    return {s.rfilename[:-4] for s in info.siblings if s.rfilename.endswith(".zip")}


def load_problems(splits: list[str] | None = None,
                  difficulties: list[str] | None = None,
                  ids: list[str] | None = None,
                  with_tests: bool = True) -> list[Problem]:
    from datasets import load_dataset

    ds = load_dataset(DATASET, token=os.environ.get("HF_TOKEN"))
    testable = available_testcase_ids() if with_tests else None
    seen: dict[str, Problem] = {}
    for split in (splits or list(ds.keys())):
        for row in ds[split]:
            pid = row["problem_id"]
            if pid in seen:
                continue
            if difficulties and row["difficulty"] not in difficulties:
                continue
            if ids and pid not in ids:
                continue
            if testable is not None and pid not in testable:
                continue
            seen[pid] = Problem(
                problem_id=pid, title=row["problem_title"], difficulty=row["difficulty"],
                platform=row["platform"], statement=row["problem_statement"],
                link=row.get("link", ""), time_limit_ms=int(row["time_limit"]),
                memory_limit_mb=int(row["memory_limit"]), split=split)
    return list(seen.values())


if __name__ == "__main__":
    import collections

    probs = load_problems()
    print(len(probs), "unique problems")
    print(collections.Counter(p.difficulty for p in probs))
    p = probs[0]
    print(p.problem_id, p.title, p.difficulty, p.time_limit_ms, "ms", p.memory_limit_mb, "MB")
    print("examples:", len(p.examples))
