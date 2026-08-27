"""Live tree viewer for travel-planning search runs.

    uv run streamlit run deepplanning/travelplanning/search/viewer.py

Replays runs/<name>/logs/id_*.jsonl (same event scheme as multihop) and shows
scores from the evaluation dir once a run has been scored.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

RUNS = Path(__file__).parent / "runs"

st.set_page_config(page_title="Travel search viewer", page_icon=":material/travel:",
                   layout="wide")


def read_events(path: Path) -> list[dict]:
    events = []
    try:
        for line in path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    except OSError:
        pass
    return events


def replay(events: list[dict]) -> dict:
    s: dict = {"question": "", "status": "running", "round": 0, "reason": "",
               "best_node_id": None, "error": "", "nodes": {}, "verifiers": [],
               "plan": ""}
    for ev in events:
        kind = ev.get("event")
        if kind == "start":
            s["question"] = ev.get("question", "")
        elif kind == "node":
            s["nodes"][ev["id"]] = ev
        elif kind == "verifiers":
            s["verifiers"] = ev.get("verifiers", [])
            for nid, upd in (ev.get("values") or {}).items():
                node = s["nodes"].get(int(nid))
                if node is not None:
                    node["value"] = upd["value"]
                    node["scores"] = upd["scores"]
                    if "reasons" in upd:
                        node["reasons"] = upd["reasons"]
        elif kind == "round":
            s["round"] = ev.get("round", s["round"])
        elif kind == "done":
            s.update(status="done", reason=ev.get("reason", ""),
                     best_node_id=ev.get("best_node_id"),
                     round=ev.get("rounds", s["round"]), plan=ev.get("plan", ""))
        elif kind == "error":
            s.update(status="error", error=ev.get("error", ""))
    return s


def scores_for(run_dir: Path, tid: str) -> dict | None:
    f = run_dir / "evaluation" / f"{tid}_score.json"
    if f.exists():
        try:
            return json.loads(f.read_text()).get("scores", {})
        except Exception:
            return None
    return None


def q_label(run_dir: Path, p: Path) -> str:
    sc = scores_for(run_dir, p.stem)
    if sc is None:
        status = replay(read_events(p))["status"]
        return {"done": "⚪", "error": "💥"}.get(status, "⏳") + " " + p.stem
    mark = "✅" if sc.get("case_acc", 0) >= 1 else ("🟡" if sc.get("composite_score", 0) >= 0.5 else "❌")
    return f"{mark} {p.stem} · {sc.get('composite_score', 0):.2f}"


def lerp(a: str, b: str, t: float) -> str:
    ah, bh = int(a[1:], 16), int(b[1:], 16)
    out = 0
    for shift in (16, 8, 0):
        ca, cb = (ah >> shift) & 255, (bh >> shift) & 255
        out |= int(ca + (cb - ca) * t) << shift
    return f"#{out:06x}"


def tree_dot(nodes: list[dict], best_id: int | None) -> str:
    lines = ["digraph G {",
             'bgcolor="transparent"; rankdir=TB; ranksep=0.35; nodesep=0.25;',
             'node [fontname="Helvetica,Arial,sans-serif", fontsize=10, shape=box,'
             ' style="filled,rounded", color="#c8cdd2", penwidth=1, margin="0.12,0.06"];',
             'edge [color="#b6bcc2", arrowsize=0.6];']
    for n in nodes:
        v = max(0.0, min(1.0, n.get("value", 0.0)))
        fill = lerp("#f4f6f4", "#7bc47f", v)
        style = ' color="#4a7fd4" penwidth=1.8' if n.get("terminal") else ""
        if n["id"] == best_id:
            style = ' color="#d4574a" penwidth=2.4'
        lines.append(f'{n["id"]} [label="n{n["id"]}\\nV={n.get("value", 0):.2f}", '
                     f'fillcolor="{fill}"{style}];')
        if n.get("parent") not in (None, 0):
            lines.append(f'{n["parent"]} -> {n["id"]};')
    lines.append("}")
    return "\n".join(lines)


run_dirs = sorted((p for p in RUNS.glob("*") if (p / "logs").is_dir()),
                  key=lambda p: p.stat().st_mtime, reverse=True) if RUNS.exists() else []
if not run_dirs:
    st.info("No search runs yet.")
    st.stop()

st.sidebar.subheader(":material/travel: Travel search viewer")
run_dir = st.sidebar.selectbox("Run", run_dirs, format_func=lambda p: p.name)
files = sorted((run_dir / "logs").glob("id_*.jsonl"),
               key=lambda p: int(p.stem.split("_")[1]))
if not files:
    st.stop()
q_file = st.sidebar.selectbox("Task", files, format_func=lambda p: q_label(run_dir, p))
auto = st.sidebar.toggle("Auto-refresh (2s)", value=True)


def render() -> None:
    s = replay(read_events(q_file))
    sc = scores_for(run_dir, q_file.stem)
    with st.container(border=True):
        st.text(s["question"] or q_file.stem)
    cols = st.columns(6)
    cols[0].metric("Status", s["status"] if not s["reason"] else s["reason"])
    cols[1].metric("Round", s["round"])
    cols[2].metric("Nodes", len(s["nodes"]))
    cols[3].metric("Composite", f"{sc.get('composite_score', 0):.3f}" if sc else "—")
    cols[4].metric("Personalized", f"{sc.get('personalized_score', 0):.3f}" if sc else "—")
    cols[5].metric("Commonsense", f"{sc.get('commonsense_weighted_score', 0):.3f}" if sc else "—")
    if s["error"]:
        st.error(s["error"])

    nodes = sorted(s["nodes"].values(), key=lambda n: n["id"])
    by_id = {n["id"]: n for n in nodes}
    tree_col, detail_col = st.columns([5, 6], gap="large")

    with tree_col:
        st.markdown("##### Search tree")
        if nodes:
            st.graphviz_chart(tree_dot(nodes, s["best_node_id"]), width="stretch")
        else:
            st.caption("No nodes yet.")
        st.markdown("##### Verifiers")
        if s["verifiers"]:
            st.dataframe(pd.DataFrame(
                [{"name": v["name"], "kind": v.get("kind", ""),
                  "statement": v.get("statement", "")} for v in s["verifiers"]]),
                hide_index=True, width="stretch")

    with detail_col:
        st.markdown("##### Node inspector")
        if not nodes:
            return
        ids = [n["id"] for n in nodes]
        default = ids.index(s["best_node_id"]) if s["best_node_id"] in by_id else len(ids) - 1
        sel = st.selectbox("Node", ids, index=default,
                           format_func=lambda i: (f"n{i} · V={by_id[i].get('value', 0):.3f}"
                                                  + f" · {by_id[i].get('n_turns', 0)} turns"
                                                  + (" · forced" if by_id[i].get("forced") else "")),
                           key="node_select")
        n = by_id[sel]
        if n.get("scores"):
            st.dataframe(
                pd.DataFrame([{"verifier": k, "score": v,
                               "reason": (n.get("reasons") or {}).get(k, "")}
                              for k, v in n["scores"].items()]),
                hide_index=True, width="stretch",
                column_config={"score": st.column_config.ProgressColumn(
                    "score", min_value=0.0, max_value=1.0, format="%.2f")})
        prompt_tab, out_tab, info_tab, plan_tab = st.tabs(
            ["Input prompt", "Turn output", "Tool results", "Plan"])
        with prompt_tab:
            st.code(n.get("prompt") or "(not recorded)", language=None, wrap_lines=True)
        with out_tab:
            st.code(n.get("output") or "(empty)", language=None, wrap_lines=True)
        with info_tab:
            infos = n.get("information") or []
            if not infos:
                st.caption("No tool results at this turn.")
            for item in infos:
                with st.expander(item.get("name", "tool")):
                    try:
                        st.json(json.loads(item.get("content", "")))
                    except Exception:
                        st.code(item.get("content", ""), language=None, wrap_lines=True)
        with plan_tab:
            plan = n.get("plan") or ""
            if plan:
                st.code(plan, language=None, wrap_lines=True)
            else:
                st.caption("Non-terminal node — no plan.")


view = st.fragment(run_every="2s" if auto else None)(render)
view()
