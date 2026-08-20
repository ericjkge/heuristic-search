"""Live tree viewer for ARC-AGI-2 search runs.

    uv run streamlit run arcagi2/search/viewer.py

Replays runs/<name>/logs/<task_id>.jsonl (same event scheme as travel) and
shows official scores from summary.json once the run has finished.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

SEARCH_DIR = Path(__file__).resolve().parent
RUNS = SEARCH_DIR / "runs"
DATA = SEARCH_DIR.parent / "data"

PALETTE = ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
           "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"]

st.set_page_config(page_title="ARC search viewer", page_icon=":material/grid_on:",
                   layout="wide")


def grid_html(g, cell: int = 13) -> str:
    if not g or not isinstance(g, list) or not isinstance(g[0], list):
        return "<i>(no grid)</i>"
    rows = []
    for row in g:
        tds = "".join(
            f'<td style="width:{cell}px;height:{cell}px;padding:0;'
            f'background:{PALETTE[c] if isinstance(c, int) and 0 <= c <= 9 else "#fff"};'
            f'border:1px solid #555;"></td>' for c in row)
        rows.append(f"<tr>{tds}</tr>")
    return (f'<table style="border-collapse:collapse;display:inline-block;'
            f'margin:2px 8px 2px 0;">{"".join(rows)}</table>')


def stem_to_task_pair(stem: str) -> tuple[str, int]:
    """Log stems are '<tid>' (pair 0) or '<tid>_p<i>' for later pairs."""
    if "_p" in stem:
        tid, _, pair = stem.rpartition("_p")
        try:
            return tid, int(pair)
        except ValueError:
            pass
    return stem, 0


def node_candidates(n: dict) -> list | None:
    """Node's grids, one per test input (old logs stored a single grid)."""
    if n.get("candidates"):
        return n["candidates"]
    return [n["candidate"]] if n.get("candidate") else None


def node_scores(n: dict) -> list[dict]:
    """Per-input score dicts (old logs stored one flat dict)."""
    s = n.get("scores")
    return s if isinstance(s, list) else ([s] if s else [])


def norm_attempts(a: list) -> list[list]:
    """attempts[pair_i] = list of grids (old logs: flat list of grids)."""
    if a and a[0] and isinstance(a[0][0][0], int):
        return [a]  # old single-pair format
    return a


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
    s: dict = {"status": "running", "round": 0, "nodes": {}, "verifiers": [],
               "attempts": [], "error": ""}
    for ev in events:
        kind = ev.get("event")
        if kind == "node":
            s["nodes"][ev["id"]] = ev
        elif kind == "verifiers":
            s["verifiers"] = ev.get("verifiers", [])
        elif kind == "rescore":
            for nid, val in (ev.get("values") or {}).items():
                node = s["nodes"].get(int(nid))
                if node is not None:
                    node["value"] = val
        elif kind == "round":
            s["round"] = ev.get("round", s["round"])
        elif kind == "done":
            s.update(status="done", attempts=ev.get("attempts", []),
                     round=ev.get("rounds", s["round"]))
        elif kind == "error":
            s.update(status="error", error=ev.get("error", ""))
    return s


@st.cache_data
def load_task(split: str, tid: str) -> dict | None:
    for sp in (split, "training", "evaluation"):
        f = DATA / sp / f"{tid}.json"
        if f.exists():
            return json.loads(f.read_text())
    return None


def run_summary(run_dir: Path) -> dict:
    f = run_dir / "summary.json"
    if f.exists():
        try:
            d = json.loads(f.read_text())
            return {"split": d.get("split", "evaluation"),
                    "scores": {r["id"]: r for r in d.get("per_task", [])}}
        except Exception:
            pass
    return {"split": "evaluation", "scores": {}}


def q_label(summary: dict, p: Path) -> str:
    tid, pair = stem_to_task_pair(p.stem)
    label = tid + (f" · pair {pair}" if pair else "")
    sc = summary["scores"].get(tid)
    if sc is None:
        status = replay(read_events(p))["status"]
        return {"done": "⚪", "error": "💥"}.get(status, "⏳") + " " + label
    mark = "✅" if sc.get("solved") else ("🟡" if sc.get("score", 0) > 0 else "❌")
    return f"{mark} {label} · {sc.get('score', 0):.2f}"


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
        style = ' color="#d4574a" penwidth=2.4' if n["id"] == best_id else ""
        no_grid = "" if node_candidates(n) else "\\n(no grid)"
        lines.append(f'{n["id"]} [label="n{n["id"]}\\nV={n.get("value", 0):.2f}'
                     f'{no_grid}", fillcolor="{fill}"{style}];')
        if n.get("parent") is not None:
            lines.append(f'{n["parent"]} -> {n["id"]};')
    lines.append("}")
    return "\n".join(lines)


run_dirs = sorted((p for p in RUNS.glob("*") if (p / "logs").is_dir()),
                  key=lambda p: p.stat().st_mtime, reverse=True) if RUNS.exists() else []
if not run_dirs:
    st.info("No search runs yet.")
    st.stop()

st.sidebar.subheader(":material/grid_on: ARC search viewer")
run_dir = st.sidebar.selectbox("Run", run_dirs, format_func=lambda p: p.name)
summary = run_summary(run_dir)
files = sorted((run_dir / "logs").glob("*.jsonl"))
if not files:
    st.stop()
q_file = st.sidebar.selectbox("Task", files, format_func=lambda p: q_label(summary, p))
auto = st.sidebar.toggle("Auto-refresh (2s)", value=True)


def render() -> None:
    tid, pair = stem_to_task_pair(q_file.stem)
    s = replay(read_events(q_file))
    task = load_task(summary["split"], tid)
    sc = summary["scores"].get(tid)

    cols = st.columns(6)
    cols[0].metric("Status", s["status"])
    cols[1].metric("Round", s["round"])
    cols[2].metric("Nodes", len(s["nodes"]))
    cols[3].metric("Verifiers", len(s["verifiers"]))
    cols[4].metric("Score", f"{sc['score']:.2f}" if sc else "—")
    cols[5].metric("Solved", ("yes" if sc.get("solved") else "no") if sc else "—")
    if s["error"]:
        st.error(s["error"])

    if task:
        with st.expander("Task (train pairs + test)", expanded=False):
            for i, p in enumerate(task["train"]):
                st.markdown(f"**train {i}**  " + grid_html(p["input"]) +
                            " → " + grid_html(p["output"]), unsafe_allow_html=True)
            for i, p in enumerate(task["test"]):
                st.markdown(f"**test {i}**  " + grid_html(p["input"]) +
                            " → " + grid_html(p.get("output")) + " (gold)",
                            unsafe_allow_html=True)

    nodes = sorted(s["nodes"].values(), key=lambda n: n["id"])
    by_id = {n["id"]: n for n in nodes}
    scored = [n for n in nodes if node_candidates(n)]
    best_id = max(scored, key=lambda n: n.get("value", 0))["id"] if scored else None
    tree_col, detail_col = st.columns([4, 7], gap="large")

    with tree_col:
        st.markdown("##### Search tree")
        if nodes:
            st.graphviz_chart(tree_dot(nodes, best_id), width="stretch")
        else:
            st.caption("No nodes yet.")
        if s["attempts"]:
            st.markdown("##### Submitted attempts")
            for i, pair_attempts in enumerate(norm_attempts(s["attempts"])):
                label = f"**pair {i}**  " if len(s["attempts"]) > 1 else ""
                st.markdown(label + "".join(grid_html(a) for a in pair_attempts),
                            unsafe_allow_html=True)
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
        default = ids.index(best_id) if best_id in by_id else len(ids) - 1
        sel = st.selectbox("Node", ids, index=default,
                           format_func=lambda i: (f"n{i} · V={by_id[i].get('value', 0):.3f}"
                                                  + ("" if node_candidates(by_id[i]) else " · no grid")),
                           key="node_select")
        n = by_id[sel]
        cands = node_candidates(n)
        scores = node_scores(n)
        stmts = {v["name"]: v.get("statement", "") for v in s["verifiers"]}
        for i, cand in enumerate(cands or []):
            if task and len(cands) > 1:
                st.markdown(f"**Test input {i}**")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Candidate")
                st.markdown(grid_html(cand), unsafe_allow_html=True)
            gold_idx = pair if len(cands) == 1 else i
            gold = (task["test"][gold_idx].get("output")
                    if task and gold_idx < len(task["test"]) else None)
            if gold:
                with c2:
                    st.caption("Gold")
                    st.markdown(grid_html(gold), unsafe_allow_html=True)
            if i < len(scores) and scores[i]:
                st.dataframe(
                    pd.DataFrame([{"verifier": k, "score": v, "statement": stmts.get(k, "")}
                                  for k, v in sorted(scores[i].items(), key=lambda kv: kv[1])]),
                    hide_index=True, width="stretch",
                    column_config={"score": st.column_config.ProgressColumn(
                        "score", min_value=0.0, max_value=1.0, format="%.2f")})
        prompt_tab, out_tab = st.tabs(["Input prompt", "Turn output"])
        with prompt_tab:
            st.code(n.get("prompt") or "(not recorded)", language=None, wrap_lines=True)
        with out_tab:
            st.code(n.get("output") or "(empty)", language=None, wrap_lines=True)


view = st.fragment(run_every="2s" if auto else None)(render)
view()
