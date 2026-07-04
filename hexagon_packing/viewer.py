"""Streamlit viewer for hexagon-packing runs: LLM traces + packing visualizations.

    streamlit run hexagon_packing/viewer.py

Reads runs/<ts>/trace.jsonl (same schema as utils/llm.py) plus the per-candidate
geometry.npz that evaluate.py writes, and renders each candidate's packing next to
its prompt/response. Fork of utils/viewer.py, decoupled from ZebraLogic.
"""

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Polygon

sys.path.insert(0, str(Path(__file__).parent))  # sibling imports when run via streamlit

import verifiers as V
from decompose import _parse_verifiers

RUNS = Path("runs")


def list_runs():
    return sorted((d for d in RUNS.glob("*") if (d / "trace.jsonl").exists()), reverse=True)


def load_trace(run_dir):
    return [json.loads(ln) for ln in (run_dir / "trace.jsonl").read_text().splitlines() if ln.strip()]


def _hex(cx, cy, R, a0):
    return [(cx + R * math.cos(a0 + k * math.pi / 3), cy + R * math.sin(a0 + k * math.pi / 3))
            for k in range(6)]


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


def geom_path(run_dir, tags):
    """Locate a candidate's geometry.npz from its trace tags (None if it never ran)."""
    d = cand_dir(run_dir, tags)
    p = d / "geometry.npz" if d else None
    return p if (p and p.exists()) else None


def read_metrics(npz):
    m = npz.parent / "metrics.json"
    return json.loads(m.read_text()) if m.exists() else {}


def draw_packing(npz):
    d = np.load(npz)
    centers, angles, s = d["centers"], d["angles"], float(d["s"])
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    # container: origin-centered, flat-top, side s (vertices at k*60 deg)
    ax.add_patch(Polygon(_hex(0, 0, s, 0.0), closed=True, fill=False, ec="red", lw=1.5))
    for (cx, cy), a in zip(centers, angles):
        ax.add_patch(Polygon(_hex(cx, cy, 1.0, float(a)), closed=True,
                             fc="steelblue", ec="k", alpha=0.75, lw=0.5))
    ax.set_aspect("equal")
    ax.autoscale()
    ax.axis("off")
    ax.set_title(f"s={s:.4f}  n={len(centers)}", fontsize=9)
    return fig


def load_verifiers(run_dir, inst, cond, all_rows):
    """The condition's final verifier set: verifiers.json, else last decompose row.

    Decompose trace rows are tagged condition="decompose", so the fallback matches
    on instance across ALL rows, not the condition-filtered subset.
    """
    hits = sorted(run_dir.glob(f"{inst}_{cond}*/verifiers.json"))
    if hits:
        return json.loads(hits[0].read_text())
    dec = [r for r in all_rows
           if r["tags"].get("phase") == "decompose" and r["tags"].get("instance") == inst]
    if not dec:
        return []
    try:
        return _parse_verifiers(dec[-1]["response"])
    except ValueError:
        return []


def verifier_stats(run_dir, inst, cond, parsed):
    """Score each verifier on every feasible candidate.

    Returns rows: description, code, min/mean/max score, and the correlation of
    the score with s (negative = rewards smaller s, i.e. pulling the right way;
    ~0 or positive = useless/harmful).
    """
    evs, sides = [], []
    for g in sorted(run_dir.glob(f"{inst}_{cond}*/*/geometry.npz")):
        if read_metrics(g).get("feasible"):
            d = np.load(g)
            s = float(d["s"])
            evs.append(V.geometry_vars(d["centers"], d["angles"], s))
            sides.append(s)
    rows = []
    for v in parsed:
        scores = [V.verify_one(v["verify_code"], ev) for ev in evs]
        row = {"description": v["description"], "verify_code": v["verify_code"]}
        if scores:
            row |= {"min": round(min(scores), 3), "mean": round(float(np.mean(scores)), 3),
                    "max": round(max(scores), 3)}
            if len(scores) >= 3 and np.std(scores) > 1e-9 and np.std(sides) > 1e-9:
                row["corr_with_s"] = round(float(np.corrcoef(scores, sides)[0, 1]), 3)
            else:
                row["corr_with_s"] = None  # constant scores or too few points
        rows.append(row)
    return rows


def render_verifiers(run_dir, inst, cond, parsed):
    rows = verifier_stats(run_dir, inst, cond, parsed)
    if not rows:
        st.caption("no verifiers found for this condition")
        return
    for i, row in enumerate(rows):
        st.markdown(f"**{i + 1}. {row['description']}**")
        st.code(row["verify_code"], language="python")
        if "mean" in row:
            corr = row["corr_with_s"]
            note = ""
            if row["min"] == row["max"]:
                note = " · ⚠ constant (no gradient)"
            elif corr is not None and corr > 0:
                note = " · ⚠ scores HIGHER for larger s (pulling the wrong way)"
            st.caption(
                f"scores over feasible candidates: min {row['min']} · mean {row['mean']}"
                f" · max {row['max']} · corr(score, s) = {corr}{note}"
            )
        st.divider()


def main():
    st.set_page_config(page_title="hexagon packing viewer", layout="wide")
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
                            "best_s": round(min(feasible_s), 4) if feasible_s else None,
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

            if cond == "decompose":  # raw decompose conversation, full width
                for r in sub:
                    att = r["tags"].get("attempt")
                    head = (f"decompose call · {r.get('completion_tokens', '?')} tok"
                            + (f" · attempt {att}" if att else ""))
                    with st.expander(head):
                        for msg in r["messages"]:
                            st.caption(msg["role"])
                            st.code(msg["content"][:6000], language="markdown")
                        st.caption("response")
                        st.code(r["response"][:6000], language="markdown")
                continue

            if cond == "search_soft":  # verifier stats only steer the soft condition
                vlist = load_verifiers(run_dir, inst, cond, rows)
                if vlist:
                    with st.expander("decompose → verifiers", expanded=True):
                        render_verifiers(run_dir, inst, cond, vlist)

            byid = {}  # retries share a cand_id; the last (chronological) attempt wins
            for r in sub:
                if r["tags"].get("phase") in ("revise", "best_of_n"):
                    byid[r["tags"]["cand_id"]] = r
            cands = sorted(byid.values(),
                           key=lambda r: (r["tags"].get("step") or 0, str(r["tags"].get("cand_id"))))
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
