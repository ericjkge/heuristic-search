"""Live search-tree viewer.

    uv run streamlit run multihop/viewer.py

Replays the append-only JSONL event logs that runs write to
multihop/runs/<name>/logs/<question_id>.jsonl, re-reading on an interval so a running
search can be watched live. Each node shows the exact input prompt sent to
the LLM and the raw completion that produced it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import sys

sys.path.insert(0, str(Path(__file__).parent))
from metrics import f1 as answer_f1
from metrics import exact_match

RUNS = Path(__file__).parent / "runs"

st.set_page_config(page_title="Search Tree Viewer", page_icon="🌳", layout="wide")

st.markdown("""
<style>
  .block-container { padding-top: 2.2rem; padding-bottom: 1rem; }
  div[data-testid="stSidebarHeader"] { padding-bottom: 0; }
  .qtitle { font-size: 1.15rem; font-weight: 650; line-height: 1.35; margin: 0 0 .3rem 0; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- log replay
def read_events(path: Path) -> list[dict]:
    events = []
    try:
        for line in path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # tolerate a half-written last line during live runs
    except OSError:
        pass
    return events


def replay(events: list[dict]) -> dict:
    """Fold the event log into the current search state."""
    s: dict = {
        "question": "", "gold": "", "status": "running", "round": 0,
        "answer": "", "reason": "", "best_node_id": None, "error": "",
        "nodes": {}, "verifiers": [],
    }
    for ev in events:
        kind = ev.get("event")
        if kind == "start":
            s["question"], s["gold"] = ev.get("question", ""), ev.get("gold", "")
        elif kind == "node":
            s["nodes"][ev["id"]] = ev
        elif kind == "verifiers":
            s["verifiers"] = ev.get("verifiers", [])
            for nid, upd in ev.get("values", {}).items():
                node = s["nodes"].get(int(nid))
                if node is not None:
                    node["value"] = upd["value"]
                    node["scores"] = upd["scores"]
        elif kind == "round":
            s["round"] = ev.get("round", s["round"])
        elif kind == "done":
            s["status"] = "done"
            s["answer"] = ev.get("answer", "")
            s["reason"] = ev.get("reason", "")
            s["best_node_id"] = ev.get("best_node_id")
            s["round"] = ev.get("rounds", s["round"])
        elif kind == "error":
            s["status"], s["error"] = "error", ev.get("error", "")
    return s


# ------------------------------------------------------------------ tree view
def lerp(a: str, b: str, t: float) -> str:
    ah, bh = int(a[1:], 16), int(b[1:], 16)
    out = 0
    for shift in (16, 8, 0):
        ca, cb = (ah >> shift) & 255, (bh >> shift) & 255
        out |= int(ca + (cb - ca) * t) << shift
    return f"#{out:06x}"


def tree_dot(nodes: list[dict], best_id: int | None) -> str:
    lines = [
        "digraph G {",
        'bgcolor="transparent"; rankdir=TB; ranksep=0.35; nodesep=0.25;',
        'node [fontname="Helvetica,Arial,sans-serif", fontsize=10, shape=box,'
        ' style="filled,rounded", color="#c8cdd2", penwidth=1, margin="0.12,0.06"];',
        'edge [color="#b6bcc2", arrowsize=0.6];',
    ]
    for n in nodes:
        v = max(0.0, min(1.0, n.get("value", 0.0)))
        fill = lerp("#f4f6f4", "#7bc47f", v)
        style = ""
        if n.get("terminal"):
            style = ' color="#4a7fd4" penwidth=1.8'
        if n["id"] == best_id:
            style = ' color="#d4574a" penwidth=2.4'
        label = f"n{n['id']}\\nV={n.get('value', 0):.2f}"
        lines.append(f'{n["id"]} [label="{label}", fillcolor="{fill}"{style}];')
        if n.get("parent") not in (None, 0):
            lines.append(f'{n["parent"]} -> {n["id"]};')
    lines.append("}")
    return "\n".join(lines)


# -------------------------------------------------------------------- render
def header(s: dict) -> None:
    st.markdown(f'<div class="qtitle">{s["question"] or "(unknown question)"}</div>',
                unsafe_allow_html=True)
    if s["error"]:
        st.error(s["error"])
    st.write("")


def score_table(scores: dict[str, float]) -> None:
    df = pd.DataFrame(
        [{"verifier": k, "score": float(v)} for k, v in scores.items()]
    )
    st.dataframe(
        df, hide_index=True, width="stretch",
        column_config={
            "score": st.column_config.ProgressColumn(
                "score", min_value=0.0, max_value=1.0, format="%.2f"
            ),
        },
    )


def verifier_table(verifiers: list[dict]) -> None:
    if not verifiers:
        st.caption("No verifiers yet.")
        return
    df = pd.DataFrame([
        {
            "name": v["name"],
            "kind": v.get("kind", ""),
            "statement": v.get("statement", ""),
        }
        for v in verifiers
    ])
    st.dataframe(df, hide_index=True, width="stretch")


def node_inspector(nodes: list[dict], by_id: dict, best_id: int | None) -> None:
    ids = [n["id"] for n in nodes]
    default = ids.index(best_id) if best_id in by_id else len(ids) - 1
    sel = st.selectbox(
        "Node", ids, index=default,
        format_func=lambda i: (
            f"n{i} · V={by_id[i].get('value', 0):.3f} · depth {by_id[i].get('depth', '?')}"
            + (" · answered" if by_id[i].get("terminal") else "")
        ),
        key="node_select",
    )
    n = by_id[sel]
    if n.get("scores"):
        score_table(n["scores"])

    prompt_tab, out_tab, info_tab, state_tab = st.tabs(
        ["Input prompt", "LLM output", "Information", "State"]
    )
    with prompt_tab:
        st.code(n.get("prompt") or "(not recorded)", language=None, wrap_lines=True)
    with out_tab:
        st.code(n.get("output") or "(not recorded)", language=None, wrap_lines=True)
    with info_tab:
        docs = n.get("information", [])
        if not docs:
            st.caption("No documents retrieved at this step.")
        for d in docs:
            st.markdown(f"**{d.get('title', '(untitled)')}**")
            st.write(d.get("text", ""))
    with state_tab:
        st.json(n.get("state", []))


def render(s: dict) -> None:
    header(s)
    nodes = sorted(s["nodes"].values(), key=lambda n: n["id"])
    by_id = {n["id"]: n for n in nodes}

    tree_col, detail_col = st.columns([6, 5], gap="large")
    with tree_col:
        st.markdown("##### Search tree")
        if not nodes:
            st.caption("No nodes yet — waiting for first expansions.")
        else:
            st.graphviz_chart(tree_dot(nodes, s["best_node_id"]), width="stretch")
        st.markdown("##### Verifiers")
        verifier_table(s["verifiers"])

    with detail_col:
        st.markdown("##### Node inspector")
        if nodes:
            node_inspector(nodes, by_id, s["best_node_id"])
        else:
            st.caption("Nothing to inspect yet.")


# ------------------------------------------------------------------- sidebar
run_dirs = sorted(
    (p for p in RUNS.iterdir() if (p / "logs").is_dir()),
    key=lambda p: p.stat().st_mtime, reverse=True,
) if RUNS.exists() else []
if not run_dirs:
    st.info("No runs with logs yet. Start one with "
            "`uv run python -m multihop.run_search ...` and reload.")
    st.stop()

st.sidebar.markdown("#### 🌳 Search viewer")
run_dir = st.sidebar.selectbox("Run", run_dirs, format_func=lambda p: p.name)
q_files = sorted(
    (run_dir / "logs").glob("*.jsonl"),
    key=lambda p: p.stat().st_mtime, reverse=True,
)
if not q_files:
    st.sidebar.info("No question logs in this run yet.")
    st.stop()


def q_label(p: Path) -> str:
    s = replay(read_events(p))
    if s["status"] == "error":
        mark = "💥"
    elif s["status"] != "done":
        mark = "⏳"
    elif exact_match(s["answer"] or "", s["gold"] or ""):
        mark = "✅"
    elif answer_f1(s["answer"] or "", s["gold"] or "") > 0:
        mark = "🟡"
    else:
        mark = "❌"
    return f"{mark} {p.stem}"


q_file = st.sidebar.selectbox("Question", q_files, format_func=q_label)
auto = st.sidebar.toggle("Auto-refresh (2s)", value=True)


@st.fragment(run_every="2s" if auto else None)
def live_view() -> None:
    render(replay(read_events(q_file)))


live_view()
