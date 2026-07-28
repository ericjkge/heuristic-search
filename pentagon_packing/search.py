"""Verifier-guided greedy search for one pentagon-packing instance.

seed: revise initial.py x num_seeds -> hard gate (evaluate.py subprocess) ->
rank by effective score (scoring.py: raw-s buckets, soft-verifier tiebreak
within a bucket) -> sample `expansions` parents per step by POWER LAW over
ranks (p_i ~ (i+1)^-alpha, i.i.d. with replacement; alpha=0 uniform,
alpha->inf hill-climb) -> population = best pop_size by effective score.
Final answer = best raw_s over the full history (true objective -- never the
effective score, which is set-relative and verifier-tinted).

Revisions are DIFF-ONLY (SEARCH/REPLACE edits): full rewrites regressed against
the strong ice-ray seed (test run 2026-07-23: 2/11 rewrites improved their
parent vs 5/11 diffs, and rewrites produced the degenerate outliers). Feedback
is constraint-only: measurements (raw s, pack() wall-clock, verifier scores),
never suggested fixes. Anti-redundancy context: each revise prompt shows up to
2 randomly drawn SIBLING edits (prior gated children of the same parent) as
unified diffs against the shown source plus their measured outcomes -- without
this, repeated sampling of one parent kept rediscovering the same edit
(2026-07-23 run 2: 10/16 diffs were semantic no-ops, population collapsed to
clones). Ablation: use_soft=False drops verifier scores from both ranking and
prompt feedback (raw-only).
"""

import difflib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import diff
import scoring
import verifiers
from concurrency import run_parallel

_HERE = Path(__file__).parent
INITIAL_PY = _HERE / "initial.py"
EVALUATE_PY = str(_HERE / "evaluate.py")


@dataclass
class Candidate:
    cand_id: str
    parent_id: str | None
    step: int
    source: str
    feasible: bool
    raw_s: float | None
    quality: dict = field(default_factory=dict)
    eval_seconds: float | None = None
    reason: str = ""


_GEOMETRY_SPEC = """\
You are improving a Python program that packs {n} unit regular pentagons (side 1) into
the smallest axis-aligned origin-centered SQUARE: a point p is inside iff
max(|px|, |py|) <= s/2. MINIMIZE the outer side s. The program defines
pack(n) -> (centers, angles, s); keep that signature, keep the container convention,
and keep every pentagon non-overlapping and inside the square. Each pentagon's position
and rotation are free. Pentagons do not tile the plane: the densest known plane packings
pair pentagons in OPPOSITE orientations (double-lattice motifs), and the best finite
packings often mix orientations and tilt boundary rows against the walls."""

REVISE_PROMPT = _GEOMETRY_SPEC + """

Current program:
```python
{source}
```

Measurements of the current program's packing:
{stats}
{siblings}
Propose one or more edits that reduce s. You may add optimization inside pack() (e.g.
scipy.optimize); the program is run with a {timeout}-second limit. Prefer editing inside
the EVOLVE-BLOCK. Present each edit as a SEARCH/REPLACE block in EXACTLY this format,
where the SEARCH text matches the current program verbatim:
<<<<<<< SEARCH
<lines to find>
=======
<replacement lines>
>>>>>>> REPLACE

Output only diff blocks, no prose.
"""


def _format_stats(cand, use_soft, s_target, timeout):
    lines = [f"- outer side s = {cand.raw_s:.4f} (lower is better; best known ~ {s_target})"]
    if cand.eval_seconds is not None:
        lines.append(f"- pack(n) wall-clock: {cand.eval_seconds:.1f}s "
                     f"(hard limit {timeout}s; exceeding it fails the candidate)")
    if use_soft and cand.quality:
        lines.append("- structural measurements (each in [0,1], higher tends to mean smaller s):")
        for name, sc in cand.quality.items():
            lines.append(f"    - {name}: {sc:.3f}")
    return "\n".join(lines)


def _format_siblings(parent, siblings):
    """Render prior gated edits of this parent as unified diffs + outcomes.
    Empty string when there are none (prompt collapses to the old shape)."""
    if not siblings:
        return ""
    out = ["\nEdits already attempted on this exact program (as unified diffs "
           "against it, with their measured outcomes):\n"]
    for k, sib in enumerate(siblings, 1):
        d = list(difflib.unified_diff(parent.source.splitlines(),
                                      sib.source.splitlines(), lineterm="", n=2))[2:]
        out.append(f"[edit {k}]\n```diff\n" + "\n".join(d) + "\n```")
        if sib.feasible:
            delta = sib.raw_s - parent.raw_s
            change = "unchanged" if abs(delta) < 1e-9 else f"{delta:+.4f}"
            secs = (f", pack(n) wall-clock {sib.eval_seconds:.1f}s"
                    if sib.eval_seconds is not None else "")
            out.append(f"Outcome: s = {sib.raw_s:.4f} ({change} vs this program){secs}\n")
        else:
            out.append(f"Outcome: INFEASIBLE -- {sib.reason}\n")
    return "\n".join(out)


def _gate(source, n, cand_id, work_dir, timeout):
    cand_dir = work_dir / cand_id
    cand_dir.mkdir(parents=True, exist_ok=True)
    prog = cand_dir / "source.py"
    prog.write_text(source)
    t0 = time.monotonic()
    try:
        subprocess.run(
            [sys.executable, EVALUATE_PY, str(prog), str(n), str(cand_dir)],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        metrics = {"feasible": False, "raw_s": None, "reason": f"timeout>{timeout}s",
                   "n": n, "eval_seconds": float(timeout)}
        (cand_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        return metrics, cand_dir
    elapsed = round(time.monotonic() - t0, 2)
    mfile = cand_dir / "metrics.json"
    if not mfile.exists():
        return {"feasible": False, "raw_s": None, "reason": "evaluate.py wrote no metrics",
                "n": n, "eval_seconds": elapsed}, cand_dir
    metrics = json.loads(mfile.read_text())
    metrics["eval_seconds"] = elapsed  # subprocess wall-clock incl. interpreter startup
    mfile.write_text(json.dumps(metrics, indent=2))
    return metrics, cand_dir


def _make_candidate(cand_id, parent_id, step, source, metrics, cand_dir):
    secs = metrics.get("eval_seconds")
    if not metrics["feasible"]:
        return Candidate(cand_id, parent_id, step, source, False, None,
                         eval_seconds=secs, reason=metrics.get("reason", ""))
    geo = np.load(cand_dir / "geometry.npz")
    ev = verifiers.geometry_vars(geo["centers"], geo["angles"], float(geo["s"]))
    quality = verifiers.quality_scores(ev)
    return Candidate(cand_id, parent_id, step, source, True, metrics["raw_s"],
                     quality, secs, "ok")


def _sample_parents(population, num, alpha, use_soft, rng):
    """Draw `num` parents i.i.d. (with replacement): rank the population by
    effective score (raw-only when use_soft is off), then p_i ~ (i+1)^-alpha."""
    if use_soft:
        ranked = scoring.rank_by_effective(population)
    else:
        ranked = sorted(population, key=lambda c: c.raw_s)
    probs = np.array([(i + 1) ** -float(alpha) for i in range(len(ranked))])
    probs /= probs.sum()
    picks = rng.choice(len(ranked), size=num, p=probs)
    return [ranked[i] for i in picks]


def _revise(llm, parent, n, cand_id, step, condition, use_soft, s_target,
            max_patch_attempts, timeout, siblings=()):
    prompt = REVISE_PROMPT.format(
        n=n, source=parent.source, timeout=timeout,
        stats=_format_stats(parent, use_soft, s_target, timeout),
        siblings=_format_siblings(parent, siblings),
    )
    messages = [{"role": "user", "content": prompt}]
    error = None
    for attempt in range(1, max_patch_attempts + 1):
        response = llm.call(
            messages,
            tags={"instance": f"n{n}", "condition": condition, "phase": "revise",
                  "step": step, "cand_id": cand_id, "parent_id": parent.cand_id,
                  "attempt": attempt, "mode": "diff"},
        )
        try:
            return diff.apply_diff(parent.source, response)
        except ValueError as e:
            error = e
            messages = messages + [
                {"role": "assistant", "content": response},
                {"role": "user",
                 "content": f"The previous edit was not successful. Error: {e}\n\nTry again."},
            ]
    raise ValueError(f"no usable diff after {max_patch_attempts} attempts: {error}")


def search(n, llm, *, s_target, num_seeds=2, num_steps=5, expansions=4,
           pop_size=8, alpha=1.0, use_soft=True, timeout=300,
           max_patch_attempts=3, work_dir=None, condition=None, seed=None):
    """Run the search for instance n. Returns (best_candidate_or_None, history).

    Budget: num_seeds + num_steps * expansions gated candidates.
    """
    condition = condition or "search"
    work_dir = Path(work_dir) if work_dir else (llm.run_dir / f"n{n}_{condition}")
    rng = np.random.default_rng(seed)
    root_source = INITIAL_PY.read_text()

    root_metrics, root_dir = _gate(root_source, n, f"n{n}.root", work_dir, timeout)
    if not root_metrics["feasible"]:
        raise RuntimeError(f"seed program infeasible: {root_metrics.get('reason')} "
                           "-- fix initial.py before searching")
    root = _make_candidate(f"n{n}.root", None, -1, root_source, root_metrics, root_dir)
    population = [root]
    history = [root]
    print(f"[n={n} {condition}] baseline s={root.raw_s:.4f}")

    for step in range(num_steps + 1):  # step 0 = seeds; 1..num_steps = expansions
        if step == 0:
            parents = [root] * num_seeds
        else:
            parents = _sample_parents(population, expansions, alpha, use_soft, rng)

        def thunk(i, parent, siblings):
            def run():
                cid = f"n{n}.s{step}.{i}"
                try:
                    child_source = _revise(llm, parent, n, cid, step, condition,
                                           use_soft, s_target, max_patch_attempts,
                                           timeout, siblings)
                except ValueError as e:  # revision attempts exhausted
                    return Candidate(cid, parent.cand_id, step, parent.source, False, None,
                                     reason=str(e))
                metrics, cdir = _gate(child_source, n, cid, work_dir, timeout)
                return _make_candidate(cid, parent.cand_id, step, child_source, metrics, cdir)
            return run

        def draw_siblings(parent):
            # prior gated children of this parent; skip revise-failures whose
            # source is just the parent's (no edit to show)
            sibs = [c for c in history
                    if c.parent_id == parent.cand_id and c.source != parent.source]
            if len(sibs) <= 2:
                return sibs
            return [sibs[j] for j in rng.choice(len(sibs), size=2, replace=False)]

        children = run_parallel(
            [thunk(i, p, draw_siblings(p)) for i, p in enumerate(parents)])
        history += children
        feasible = [c for c in population + children if c.feasible]
        if use_soft:
            population = scoring.rank_by_effective(feasible)[:pop_size]
        else:
            population = sorted(feasible, key=lambda c: c.raw_s)[:pop_size]
        n_feas = sum(c.feasible for c in children)
        best_s = min((c.raw_s for c in population), default=float("nan"))
        print(f"[n={n} {condition}] step {step}: {n_feas}/{len(children)} feasible; best s={best_s:.4f}")

    feasible = [c for c in history if c.feasible]
    best = min(feasible, key=lambda c: c.raw_s) if feasible else None
    return best, history


if __name__ == "__main__":
    from llm import LLM

    llm = LLM()
    best, history = search(10, llm, s_target=4.906,
                           num_seeds=2, num_steps=1, expansions=2)  # tiny smoke
    if best is None:
        print("\nno feasible candidate found")
    else:
        print(f"\nbest: {best.cand_id}  s={best.raw_s:.4f}")
    print(f"artifacts: {llm.run_dir}")
