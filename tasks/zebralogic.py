"""ZebraLogic puzzle utilities: parsing, validation."""

import json
import re
from typing import Optional


def _extract_output_tag(text: str) -> Optional[str]:
    """Extract content between the last <OUTPUT> and </OUTPUT> tags."""
    matches = re.findall(r"<OUTPUT>(.*?)</OUTPUT>", text, re.DOTALL)
    return matches[-1] if matches else None


def parse_solution(text: str) -> Optional[dict]:
    """Parse JSON solution dict from <OUTPUT> tags. Returns None on failure."""
    tagged = _extract_output_tag(text)
    if tagged is None:
        return None

    try:
        return json.loads(tagged.strip())
    except json.JSONDecodeError:
        return None


def _normalize(s: str | int) -> str:
    """Normalize a value for comparison: lowercase, remove spaces."""
    return str(s).lower().replace(" ", "")


def check_solution(candidate: dict, solution: dict) -> bool:
    """Check candidate against ground truth solution.

    Header-agnostic: compares each row as a set of normalized values.
    Rows are ordered by house number so row i must match row i.
    """
    cand_rows: list[list] = candidate["rows"]
    sol_rows: list[list] = solution["rows"]

    if len(cand_rows) != len(sol_rows):
        return False

    for sol_row, cand_row in zip(sol_rows, cand_rows):
        if set(_normalize(v) for v in sol_row) != set(_normalize(v) for v in cand_row):
            return False

    return True
