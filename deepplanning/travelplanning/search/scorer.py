"""Black-box scoring of search outputs via their official pipeline.

Our run directory mimics their results layout (reports/ -> converted_plans/ ->
evaluation/), so we can call their conversion (qwen-plus) and rule-based
evaluation unchanged and read back the per-task score JSONs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SEARCH_DIR = Path(__file__).resolve().parent
TP_DIR = SEARCH_DIR.parent
for p in (str(TP_DIR), str(TP_DIR / "agent"), str(TP_DIR / "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluation.convert_report import convert_reports
from evaluation.eval_converted import evaluate_plans

TEST_DATA = TP_DIR / "data" / "travelplanning_query_en.json"
DATABASE_DIR = TP_DIR / "database" / "database_en"


def score_run(run_dir: Path, workers: int = 10) -> dict:
    """Convert + evaluate every report in run_dir/reports/. Returns summary."""
    convert_reports(run_dir, language="en", workers=workers, skip_existing=True)
    evaluate_plans(run_dir, TEST_DATA, DATABASE_DIR, workers=workers)

    per_task = {}
    for f in (run_dir / "evaluation").glob("id_*_score.json"):
        s = json.loads(f.read_text())
        per_task[f.stem.replace("_score", "")] = s.get("scores", {})
    n = len(per_task)
    if n == 0:
        return {"n": 0, "per_task": {}}
    agg = {
        key: sum(t.get(key, 0.0) for t in per_task.values()) / n
        for key in ("commonsense_weighted_score", "personalized_score",
                    "composite_score", "case_acc")
    }
    return {"n": n, **agg, "per_task": per_task}
