"""Candidate generation, revision, combination, and parsing.

A candidate is a dict {"header": [...], "rows": [[...], ...]} matching the
dataset solution schema. Generation asks the LLM for fenced JSON; parsing
validates shape against the puzzle and returns None on malformed output.
"""

import json
import re
from typing import Any

from src.common.verifiers import Verifier, failed_clues, satisfied_clues
from utils.data import Puzzle
from utils.llm import LLM

_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

GEN_SYS = (
    "You solve logic grid puzzles. Output ONLY the final solution as one JSON "
    "code block, no prose or reasoning."
)

GEN_PROMPT = """\
Solve this logic puzzle.

{puzzle}

Output the solution as JSON in this exact schema:
{{"header": {header}, "rows": [<{n} rows>]}}
- Exactly {n} rows, one per house.
- Each row is a list aligned to header; the "House" column is "1".."{n}".
- Use the attribute value spellings from the puzzle.
Output only the JSON code block."""

REVISE_PROMPT = """\
You previously proposed this solution to the puzzle:
{candidate}

It VIOLATES these clues (your assignment is inconsistent with each):
{feedback}

Revise the solution so every clue holds. Re-read the conflicting clues
carefully and change only what is needed to satisfy them without breaking
the others. Output the corrected full solution as one JSON code block in the
same schema ({n} rows aligned to {header}). Output only the JSON code block."""

COMBINE_PROMPT = """\
Two candidate solutions to this puzzle each satisfy some clues and violate others.

{puzzle}

--- Solution A ---
{cand_a}
A satisfies these clues:
{sat_a}
A violates these clues:
{failed_a}

--- Solution B ---
{cand_b}
B satisfies these clues:
{sat_b}
B violates these clues:
{failed_b}

Combine A and B into a single solution that satisfies ALL clues: keep the
assignments each got right and resolve the conflicts between them. Output the
combined full solution as one JSON code block in the schema ({n} rows aligned to
{header}, the "House" column is "1".."{n}"). Output only the JSON code block."""


def matches_gold(candidate: dict | None, puzzle: Puzzle) -> bool:
    """True iff the candidate's assignment equals the gold solution.

    Compared by house (case-insensitive cells), ignoring row order.
    This is the ground-truth solve metric, independent of the verifiers.
    """
    if candidate is None:
        return False
    house_col = puzzle.header.index("House")

    def keyed(rows: list[list[str]]) -> dict[str, list[str]]:
        return {
            str(r[house_col]).strip(): [str(c).strip().lower() for c in r]
            for r in rows
        }

    return keyed(candidate["rows"]) == keyed(puzzle.gold_rows)


def _norm(s) -> str:
    return str(s).strip().lower()


def _reorder_to_canonical(
    rows: list[list[str]], model_header, canonical: list[str]
) -> list[list[str]]:
    """Realign each row's cells into canonical column order (e.g. if gold header
    is ["House", "Name"] and response is ["Name", "House"], this function realigns to gold).

    Uses the model's header to map columns by (normalized) name. If that
    header is missing or isn't a permutation of the canonical columns, we can't
    trust it, so rows are assumed already canonical (the prompt asks for that).
    """
    if not isinstance(model_header, list) or len(model_header) != len(canonical):
        return rows
    norm_model = [_norm(h) for h in model_header]
    if set(norm_model) != {_norm(h) for h in canonical}:
        return rows  # different column set (or duplicates) -> don't remap
    src_index = {h: i for i, h in enumerate(norm_model)}
    perm = [src_index[_norm(h)] for h in canonical]
    return [[r[i] for i in perm] for r in rows]


def parse_candidate(text: str, puzzle: Puzzle) -> dict | None:
    """Extract + shape-validate a candidate. Returns None if malformed.

    Cells are realigned to the canonical column order (via the model's echoed
    header), so downstream column lookups are order-invariant.
    """
    m = _JSON_RE.search(text)
    raw = (m.group(1) if m else text).strip()
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    if not isinstance(obj, dict) or "rows" not in obj:
        return None
    rows = obj.get("rows")
    if not isinstance(rows, list) or len(rows) != puzzle.n_houses:
        return None
    width = len(puzzle.header)
    if not all(isinstance(r, list) and len(r) == width for r in rows):
        return None

    rows = [[str(c) for c in r] for r in rows]
    rows = _reorder_to_canonical(rows, obj.get("header"), puzzle.header)
    return {"header": puzzle.header, "rows": rows}


def _gen(llm: LLM, puzzle: Puzzle, messages: list[dict[str, str]],
         tags: dict[str, Any], temperature: float, retries: int = 1) -> dict | None:
    """Call the LLM and parse; retry once on malformed output."""
    for attempt in range(retries + 1):
        text = llm.call(
            messages, tags={**tags, "attempt": attempt}, temperature=temperature
        )
        cand = parse_candidate(text, puzzle)
        if cand is not None:
            return cand
    return None


def generate(llm: LLM, puzzle: Puzzle, tags: dict[str, Any],
             temperature: float = 0.6) -> dict | None:
    """A fresh full-solution attempt."""
    prompt = GEN_PROMPT.format(
        puzzle=puzzle.puzzle, header=puzzle.header, n=puzzle.n_houses
    )
    msgs = [
        {"role": "system", "content": GEN_SYS},
        {"role": "user", "content": prompt},
    ]
    return _gen(llm, puzzle, msgs, tags, temperature)


def revise(llm: LLM, puzzle: Puzzle, candidate: dict, failed: list[str],
           tags: dict[str, Any], temperature: float = 0.6) -> dict | None:
    """A revision of `candidate` guided by the verbatim failed clues."""
    feedback = "\n".join(f"- {c}" for c in failed)
    prompt = REVISE_PROMPT.format(
        candidate=json.dumps(candidate),
        feedback=feedback,
        n=puzzle.n_houses,
        header=puzzle.header,
    )
    msgs = [
        {"role": "system", "content": GEN_SYS},
        {"role": "user", "content": prompt},
    ]
    return _gen(llm, puzzle, msgs, tags, temperature)


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {c}" for c in items) if items else "(none)"


def combine(llm: LLM, puzzle: Puzzle, verifiers: list[Verifier], cand_a: dict,
            cand_b: dict, tags: dict[str, Any], temperature: float = 0.6) -> dict | None:
    """Combine two candidates, given each one's satisfied/failed clues (constraint-only)."""
    prompt = COMBINE_PROMPT.format(
        puzzle=puzzle.puzzle,
        cand_a=json.dumps(cand_a),
        sat_a=_bullets(satisfied_clues(verifiers, cand_a)),
        failed_a=_bullets(failed_clues(verifiers, cand_a)),
        cand_b=json.dumps(cand_b),
        sat_b=_bullets(satisfied_clues(verifiers, cand_b)),
        failed_b=_bullets(failed_clues(verifiers, cand_b)),
        n=puzzle.n_houses,
        header=puzzle.header,
    )
    msgs = [
        {"role": "system", "content": GEN_SYS},
        {"role": "user", "content": prompt},
    ]
    return _gen(llm, puzzle, msgs, tags, temperature)
