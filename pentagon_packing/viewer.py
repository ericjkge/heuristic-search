"""Streamlit viewer for pentagon-packing runs: LLM traces + packing visualizations.

    streamlit run pentagon_packing/viewer.py

Reads runs/<ts>/trace.jsonl plus the per-candidate geometry.npz/metrics.json that
evaluate.py writes, and renders each candidate's packing next to its
prompt/response. A per-condition diagnostics panel scores the hand-coded quality
verifiers across all feasible candidates (corr with s = is each verifier pulling
the right way?).

Repeats: run.py tags each repeat's calls with a labeled condition (search_r0,
search_r1, ...), so repeats appear as separate sections with exact artifact
resolution. Runs recorded before 2026-07-05 lack the labels and collapse repeats
(last trace row wins; artifacts resolve to the first repeat dir).
"""

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Polygon, Rectangle

sys.path.insert(0, str(Path(__file__).parent))  # sibling imports when run via streamlit

import verifiers as V

RUNS = Path("runs")
R = 1.0 / (2.0 * math.sin(math.pi / 5.0))  # circumradius of a unit pentagon


def list_runs():
    return sorted((d for d in RUNS.glob("*") if (d / "trace.jsonl").exists()), reverse=True)


def load_trace(run_dir):
    return [json.loads(ln) for ln in (run_dir / "trace.jsonl").read_text().splitlines() if ln.strip()]


def _pent(cx, cy, a0):
    return [(cx + R * math.cos(a0 + k * 2 * math.pi / 5), cy + R * math.sin(a0 + k * 2 * math.pi / 5))
            for k in range(5)]


def cand_dir(run_dir, tags):
    """Locate a candidate's artifact dir from its trace tags."""
    inst, cond, cid = tags.get("instance"), tags.get("condition"), tags.get("cand_id")
    if not (inst and cond and cid):
        return None
    p = run_dir / f"{inst}_{cond}" / cid
    if p.exists():
        return p
    hits = list(run_dir.glob(f"{inst}_{cond}*/{cid}"))  # repeats fallback
    return hits[0] if hits else None


def read_metrics(npz):
    m = npz.parent / "metrics.json"
    return json.loads(m.read_text()) if m.exists() else {}


def draw_packing(npz):
    d = np.load(npz)
    centers, angles, s = d["centers"], d["angles"], float(d["s"])
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.add_patch(Rectangle((-s / 2, -s / 2), s, s, fill=False, ec="red", lw=1.5))
    for (cx, cy), a in zip(centers, angles):
        ax.add_patch(Polygon(_pent(cx, cy, float(a)), closed=True,
                             fc="steelblue", ec="k", alpha=0.75, lw=0.5))
    ax.set_aspect("equal")
    ax.autoscale()
    ax.axis("off")
    ax.set_title(f"s={s:.4f}  n={len(centers)}", fontsize=9)
    return fig


def quality_stats(run_dir, inst, cond):
    """Score every hand-coded quality verifier over the condition's feasible
    candidates. corr(score, s): negative = rewards smaller s (pulling right)."""
    evs, sides = [], []
    for g in sorted(run_dir.glob(f"{inst}_{cond}*/*/geometry.npz")):
        if read_metrics(g).get("feasible"):
            d = np.load(g)
            evs.append(V.geometry_vars(d["centers"], d["angles"], float(d["s"])))
            sides.append(float(d["s"]))
    rows = []
    for name in V.QUALITY_WEIGHTS:
        scores = [V.quality_scores(ev)[name] for ev in evs]
        row = {"verifier": name, "weight": V.QUALITY_WEIGHTS[name]}
        if scores:
            row |= {"min": round(min(scores), 3), "mean": round(float(np.mean(scores)), 3),
                    "max": round(max(scores), 3)}
            if len(scores) >= 3 and np.std(scores) > 1e-9 and np.std(sides) > 1e-9:
                row["corr_with_s"] = round(float(np.corrcoef(scores, sides)[0, 1]), 3)
            else:
                row["corr_with_s"] = None  # constant scores or too few points
        rows.append(row)
    return rows


def render_quality(run_dir, inst, cond):
    rows = quality_stats(run_dir, inst, cond)
    if not rows or "mean" not in rows[0]:
        st.caption("not enough feasible candidates for verifier diagnostics")
        return
    for row in rows:
        corr = row.get("corr_with_s")
        note = ""
        if row.get("min") == row.get("max"):
            note = " · ⚠ constant (no gradient)"
        elif corr is not None and corr > 0:
            note = " · ⚠ scores HIGHER for larger s (pulling the wrong way)"
        st.markdown(f"**{row['verifier']}** (w={row['weight']})")
        st.caption(f"min {row.get('min')} · mean {row.get('mean')} · max {row.get('max')}"
                   f" · corr(score, s) = {corr}{note}")
    st.divider()


def main():
    st.set_page_config(page_title="pentagon packing viewer", layout="wide")
    runs = list_runs()
    if not runs:
        st.warning("No runs with trace.jsonl found under runs/")
        return

    run_dir = RUNS / st.sidebar.selectbox("run", [d.name for d in runs])
    rows = load_trace(run_dir)
    instances = sorted({r["tags"].get("instance") for r in rows if r["tags"].get("instance")})
    conditions = sorted({r["tags"].get("condition") for r in rows if r["tags"].get("condition")})
    sel_inst = st.sidebar.multiselect("instances", instances, default=instances)
    sel_cond = st.sidebar.multiselect("conditions", conditions, default=conditions)

    total_tok = sum(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in rows)
    st.title(f"{run_dir.name}")
    st.caption(f"{len(rows)} calls · {total_tok:,} tokens")

    # summary table: best feasible s per (instance, condition)
    summary = []
    for inst in sel_inst:
        for cond in sel_cond:
            geoms = list(run_dir.glob(f"{inst}_{cond}*/*/geometry.npz"))
            feasible_s = [float(np.load(g)["s"]) for g in geoms if read_metrics(g).get("feasible")]
            summary.append({"instance": inst, "condition": cond,
                            "best_s": round(min(feasible_s), 5) if feasible_s else None,
                            "feasible": len(feasible_s), "total": len(geoms)})
    if summary:
        st.dataframe(summary, use_container_width=True)

    for inst in sel_inst:
        for cond in sel_cond:
            sub = [r for r in rows
                   if r["tags"].get("instance") == inst and r["tags"].get("condition") == cond]
            if not sub:
                continue
            st.header(f"{inst} · {cond}")

            if cond.startswith("search") and not cond.startswith("search_raw"):  # where quality steers
                with st.expander("quality verifier diagnostics", expanded=True):
                    render_quality(run_dir, inst, cond)

            byid = {}  # retries share a cand_id; the last (chronological) attempt wins
            for r in sub:
                if r["tags"].get("phase") == "revise":
                    byid[r["tags"]["cand_id"]] = r
            def _order(r):  # step asc, then numeric child index (bon.10 after bon.2)
                cid = str(r["tags"].get("cand_id"))
                tail = cid.rsplit(".", 1)[-1]
                return (r["tags"].get("step") or 0,
                        int(tail) if tail.isdigit() else -1, cid)
            cands = sorted(byid.values(), key=_order)
            for r in cands:
                t = r["tags"]
                cd = cand_dir(run_dir, t)
                gp = cd / "geometry.npz" if cd else None
                gp = gp if (gp and gp.exists()) else None
                mfile = cd / "metrics.json" if cd else None
                m = json.loads(mfile.read_text()) if mfile and mfile.exists() else {}
                if m.get("feasible"):
                    badge = f"✅ s={m['raw_s']:.4f}"
                elif m:  # ran but crashed/timed out/infeasible -- show why
                    badge = f"❌ {m.get('reason', '')[:60]}"
                else:  # no artifacts at all: revision attempts exhausted
                    badge = "❌ no usable revision (attempts exhausted)"
                att = t.get("attempt")
                head = (f"{t['cand_id']} · {badge} · step {t.get('step')} · "
                        f"parent {t.get('parent_id')} · {r.get('completion_tokens', '?')} tok"
                        + (f" · {t['mode']}" if t.get("mode") else "")
                        + (f" · attempt {att}" if att and att > 1 else ""))
                with st.expander(head):
                    left, right = st.columns([1, 3])
                    with left:
                        if gp:
                            st.pyplot(draw_packing(gp))
                            plt.close("all")
                    with right:
                        for msg in r["messages"]:
                            st.caption(msg["role"])
                            st.code(msg["content"][:6000], language="markdown")
                        st.caption("response")
                        st.code(r["response"][:6000], language="markdown")


if __name__ == "__main__":
    main()
