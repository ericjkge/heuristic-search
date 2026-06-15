"""Local viewer for LLM call traces. Run: streamlit run utils/viewer.py

Reads runs/<ts>/trace.jsonl and shows each call's prompt + response, grouped
by puzzle -> phase. Defaults to the latest run; refresh the browser to update.
"""

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root on path

from src.common.candidates import matches_gold, parse_candidate  # noqa: E402
from utils.data import to_puzzle  # noqa: E402

RUNS = Path("runs")
PHASE_ORDER = ["build_verifier", "init", "revise", "critique", "refine", "sample", "smoke"]
CANDIDATE_PHASES = {"init", "revise", "refine", "sample"}  # phases whose response is a solution


@st.cache_resource
def puzzle_index() -> dict:
    """id -> Puzzle for every dataset puzzle (cached). {} if the dataset is unavailable."""
    try:
        from datasets import load_dataset
        ds = load_dataset("WildEval/ZebraLogic", "grid_mode", split="test")
        return {ex["id"]: to_puzzle(ex) for ex in ds}
    except Exception:
        return {}


def matched_gold(row: dict, puzzle) -> bool | None:
    """True/False if the response is a gold/non-gold solution, None if not applicable."""
    if puzzle is None or row.get("tags", {}).get("phase") not in CANDIDATE_PHASES:
        return None
    return matches_gold(parse_candidate(row.get("response", ""), puzzle), puzzle)


def list_runs() -> list[Path]:
    return sorted(
        (d for d in RUNS.glob("*") if (d / "trace.jsonl").exists()),
        reverse=True,
    )


def load_trace(run_dir: Path) -> list[dict]:
    rows = []
    for line in (run_dir / "trace.jsonl").read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def group(rows: list[dict]) -> dict:
    """puzzle_id -> phase -> [call, ...], preserving first-seen order."""
    out: dict = {}
    for r in rows:
        tags = r.get("tags", {})
        pid = tags.get("puzzle_id", "misc")
        phase = tags.get("phase", "?")
        out.setdefault(pid, {}).setdefault(phase, []).append(r)
    return out


def phase_key(phase: str) -> int:
    return PHASE_ORDER.index(phase) if phase in PHASE_ORDER else len(PHASE_ORDER)


def _gold_badge(r: dict, puzzle) -> str:
    if puzzle is None or r.get("tags", {}).get("phase") not in CANDIDATE_PHASES:
        return ""
    cand = parse_candidate(r.get("response", ""), puzzle)
    if cand is None:
        return "  ·  ⚠ unparseable"
    return "  ·  ✅ gold" if matches_gold(cand, puzzle) else "  ·  ❌ not gold"


def render_call(r: dict, idx: int, puzzle):
    tags = r.get("tags", {})
    bits = [f"{k}={v}" for k, v in tags.items() if k not in ("puzzle_id", "phase")]
    head = (
        f"#{idx}  " + "  ".join(bits)
        + f"  ·  {r.get('prompt_tokens', '?')}+{r.get('completion_tokens', '?')} tok"
        + f"  ·  {r.get('latency_s', '?')}s"
        + _gold_badge(r, puzzle)
    )
    with st.expander(head):
        for m in r.get("messages", []):
            st.caption(m.get("role", "?"))
            st.code(m.get("content", ""), language="markdown")
        reasoning = r.get("reasoning")
        if reasoning:
            with st.expander(f"reasoning ({len(reasoning)} chars)"):
                st.markdown(reasoning)
        st.caption("response")
        st.code(r.get("response", ""), language="markdown")


def render(run_dir: Path, phase_filter: list[str]):
    rows = load_trace(run_dir)
    total_tok = sum(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in rows)
    st.write(f"**{len(rows)}** calls · **{total_tok:,}** tokens · `{run_dir.name}`")

    index = puzzle_index()
    grouped = group(rows)
    for pid, phases in grouped.items():
        n = sum(len(c) for ph, c in phases.items() if ph in phase_filter)
        if not n:
            continue
        puzzle = index.get(pid)
        st.header(f"{pid}  ({n} calls)")
        for phase in sorted(phases, key=phase_key):
            if phase not in phase_filter:
                continue
            calls = phases[phase]
            sub = f"{phase} · {len(calls)} calls"
            if phase in CANDIDATE_PHASES and puzzle is not None:
                n_gold = sum(matched_gold(r, puzzle) is True for r in calls)
                sub += f" · {n_gold}/{len(calls)} matched gold"
            st.subheader(sub)
            for i, r in enumerate(calls):
                render_call(r, i, puzzle)


def main():
    st.set_page_config(page_title="LLM trace viewer", layout="wide")
    st.title("LLM trace viewer")

    runs = list_runs()
    if not runs:
        st.warning("No runs found in runs/")
        return

    names = [d.name for d in runs]
    chosen = st.sidebar.selectbox("Run", names, index=0)
    run_dir = RUNS / chosen

    all_phases = PHASE_ORDER
    phase_filter = st.sidebar.multiselect("Phases", all_phases, default=all_phases)

    render(run_dir, phase_filter)


if __name__ == "__main__":
    main()
