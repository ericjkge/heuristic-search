"""DeepPlanning trajectory viewer.

    uv run streamlit run viewer_deepplanning.py

Renders the trajectory JSONs their pipeline writes under
deepplanning/<domain>/results/<model_lang>/trajectories/id_*.json as a
readable conversation (tool calls paired with their results), alongside the
report, converted plan, and per-dimension evaluation scores.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

DP = Path(__file__).parent / "deepplanning"

st.set_page_config(
    page_title="DeepPlanning viewer",
    page_icon=":material/travel:",
    layout="wide",
)


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def score_for(run_dir: Path, tid: str):
    return load_json(run_dir / "evaluation" / f"{tid}_score.json")


def traj_label(run_dir: Path, p: Path) -> str:
    s = score_for(run_dir, p.stem)
    if not s:
        return f"⚪ {p.stem}"
    sc = s.get("scores", {})
    if sc.get("case_acc", 0) >= 1:
        mark = "✅"
    elif sc.get("composite_score", 0) >= 0.5:
        mark = "🟡"
    else:
        mark = "❌"
    return f"{mark} {p.stem} · {sc.get('composite_score', 0):.2f}"


def md_breaks(text: str) -> str:
    """Prepare LLM text for st.markdown: escape '<' so tags like <plan> render
    literally instead of opening a raw-HTML block (inside which markdown
    ignores line breaks entirely), then turn single newlines into hard breaks
    so line-based plan rows (Day 1: / 07:03-07:51 | ...) stay on their own
    lines."""
    text = text.replace("<", "\\<")
    return re.sub(r"(?<!\n)\n(?!\n)", "  \n", text)


def compact_args(arguments: str, limit: int = 80) -> str:
    """One-line preview of a tool call's JSON arguments."""
    try:
        text = json.dumps(json.loads(arguments), ensure_ascii=False)
    except Exception:
        text = str(arguments)
    text = " ".join(text.split())
    return text[:limit] + "…" if len(text) > limit else text


def render_tool_call(tc: dict, result_msg: dict | None) -> None:
    fn = tc.get("function", {})
    name = fn.get("name", "?")
    args = fn.get("arguments", "")
    with st.expander(f":material/build: {name} — {compact_args(args)}"):
        try:
            st.code(json.dumps(json.loads(args), indent=2, ensure_ascii=False),
                    language="json", wrap_lines=True)
        except Exception:
            st.code(str(args), language=None, wrap_lines=True)
        if result_msg is None:
            st.caption("No result recorded for this call.")
            return
        st.caption("Result")
        content = result_msg.get("content") or ""
        try:
            st.json(json.loads(content))
        except Exception:
            st.code(content, language=None, wrap_lines=True)


def render_trajectory(msgs: list[dict]) -> None:
    results_by_call = {
        m.get("tool_call_id"): m for m in msgs
        if m.get("role") == "tool" and m.get("tool_call_id")
    }
    claimed: set[str] = set()

    for m in msgs:
        role = m.get("role", "?")
        content = m.get("content") or ""
        if role == "system":
            with st.expander(":material/settings: system prompt"):
                st.text(content)
        elif role == "user":
            with st.chat_message("user"):
                st.text(content)
        elif role == "assistant":
            with st.chat_message("assistant"):
                if content.strip():
                    st.markdown(md_breaks(content))
                for tc in m.get("tool_calls") or []:
                    result = results_by_call.get(tc.get("id"))
                    if result is not None:
                        claimed.add(tc.get("id"))
                    render_tool_call(tc, result)
        elif role == "tool" and m.get("tool_call_id") not in claimed:
            # orphan tool message (no matching call, e.g. from broken runs)
            with st.chat_message("assistant"):
                with st.expander(":material/build: tool result (unmatched)"):
                    st.code(content, language=None, wrap_lines=True)


# ------------------------------------------------------------------- sidebar
run_dirs = sorted(
    (p for p in DP.glob("*/results/*") if (p / "trajectories").is_dir()),
    key=lambda p: p.stat().st_mtime, reverse=True,
)
if not run_dirs:
    st.info("No DeepPlanning results yet. Run their pipeline first "
            "(deepplanning/travelplanning/run.py).")
    st.stop()

st.sidebar.subheader(":material/travel: DeepPlanning viewer")
run_dir = st.sidebar.selectbox(
    "Run", run_dirs,
    format_func=lambda p: f"{p.parts[-3].replace('planning', '')}/{p.name}",
)
traj_files = sorted(
    (run_dir / "trajectories").glob("id_*.json"),
    key=lambda p: int(p.stem.split("_")[1]),
)
if not traj_files:
    st.sidebar.info("No trajectories in this run.")
    st.stop()
traj_file = st.sidebar.selectbox(
    "Task", traj_files, format_func=lambda p: traj_label(run_dir, p)
)

traj = load_json(traj_file) or {}
tid = traj_file.stem
score = score_for(run_dir, tid)
report_file = run_dir / "reports" / f"{tid}.txt"
converted = load_json(run_dir / "converted_plans" / f"{tid}_converted.json")
msgs = traj.get("messages", [])

# -------------------------------------------------------------------- header
with st.container(border=True):
    st.text(traj.get("query", "(no query)"))

n_tools = sum(len(m.get("tool_calls") or []) for m in msgs if m.get("role") == "assistant")
sc = (score or {}).get("scores", {})
cols = st.columns(6)
cols[0].metric("Composite", f"{sc.get('composite_score', 0):.3f}" if score else "—")
cols[1].metric("Commonsense", f"{sc.get('commonsense_weighted_score', 0):.3f}" if score else "—")
cols[2].metric("Personalized", f"{sc.get('personalized_score', 0):.3f}" if score else "—")
cols[3].metric("Case acc", f"{sc.get('case_acc', 0):.0f}" if score else "—")
cols[4].metric("Tool calls", n_tools)
cols[5].metric("Elapsed", f"{traj.get('elapsed_time', 0):.0f}s")

traj_tab, report_tab, plan_tab, score_tab = st.tabs(
    ["Trajectory", "Report", "Converted plan", "Score detail"]
)

with traj_tab:
    if msgs:
        render_trajectory(msgs)
    else:
        st.caption("No messages in this trajectory.")

with report_tab:
    if report_file.exists():
        st.markdown(md_breaks(report_file.read_text()))
    elif traj.get("final_plan"):
        st.markdown(md_breaks(traj["final_plan"]))
    else:
        st.caption("No report available.")

with plan_tab:
    if converted is not None:
        st.json(converted)
    else:
        st.caption("No converted plan available.")

with score_tab:
    if not score:
        st.caption("No evaluation for this task.")
    else:
        check_cfg = {
            "passed": st.column_config.CheckboxColumn("passed", width="small"),
            "message": st.column_config.TextColumn("message", width="large"),
        }

        checks = [
            {
                "dimension": dim,
                "check": c.get("name", ""),
                "passed": bool(c.get("passed")),
                "message": c.get("message", ""),
            }
            for dim, dd in score.get("commonsense_dimension_details", {}).items()
            for c in dd.get("checks", [])
        ]
        if checks:
            n_pass = sum(c["passed"] for c in checks)
            st.markdown(f"**Commonsense checks** — {n_pass}/{len(checks)} passed")
            st.dataframe(
                pd.DataFrame(checks), hide_index=True, width="stretch",
                column_config=check_cfg,
            )

        pers = score.get("personalized_dimension_score", {})
        constraints = [
            {
                "constraint": name,
                "passed": bool(v.get("passed")),
                "message": v.get("message", ""),
            }
            for name, v in pers.get("constraints", {}).items()
        ]
        if constraints:
            n_pass = sum(c["passed"] for c in constraints)
            st.markdown(
                f"**Personalized constraints** — {n_pass}/{len(constraints)} passed "
                f"(score {pers.get('score', 0):.2f})"
            )
            st.dataframe(
                pd.DataFrame(constraints), hide_index=True, width="stretch",
                column_config=check_cfg,
            )

        with st.expander("full score JSON"):
            st.json(score)
