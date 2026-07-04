"""Verifier-guided QD search for one pentagon-packing instance.

seed: revise initial.py x num_seeds -> hard gate (evaluate.py subprocess) ->
quality = w_raw*raw01 + w_soft*weighted_mean(hand-coded verifiers) ->
diversity = descriptor-vector distance (diversity.py) ->
parents: greedy by quality + div_weight * min-distance-to-picked ->
population: greedy frontier truncation (quality + div_weight * min-distance) ->
repeat num_steps. Final answer = best raw_s (true objective, never the
combined score -- avoids verifier-overconfidence at the final pick).

Quality and diversity are DISJOINT (AIME v4 design): quality never sees
structure descriptors, diversity never sees scores. Revisions alternate 50/50
between diffs and full rewrites. Feedback is constraint-only: measurements,
never suggested fixes. Ablations: w_soft=0 (raw-only), div_weight=0 (no QD).
"""

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import diff
import diversity as div
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
    descriptor: list = field(default_factory=list)
    score: float | None = None
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

REWRITE_PROMPT = _GEOMETRY_SPEC + """

Current program:
```python
{source}
```

Measurements of the current program's packing:
{stats}

Rewrite the program to reduce s. You may restructure it completely -- change the
construction strategy, add numerical optimization (e.g. scipy.optimize), anything --
as long as pack(n) keeps its signature and returns a valid packing. The program is
run with a {timeout}-second limit.

Output the COMPLETE new program in a single ```python code block, nothing else.
"""


def _extract_program(text):
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if not blocks:
        raise ValueError("no ```python code block found in response")
    code = max(blocks, key=len).strip() + "\n"
    if "def pack(" not in code:
        raise ValueError("code block does not define pack(n)")
    return code


def _format_stats(cand, use_soft, s_target):
    lines = [f"- outer side s = {cand.raw_s:.4f} (lower is better; best known ~ {s_target})"]
    if use_soft and cand.quality:
        lines.append("- structural measurements (each in [0,1], higher tends to mean smaller s):")
        for name, sc in cand.quality.items():
            lines.append(f"    - {name}: {sc:.3f}")
    return "\n".join(lines)


def _gate(source, n, cand_id, work_dir, timeout):
    cand_dir = work_dir / cand_id
    cand_dir.mkdir(parents=True, exist_ok=True)
    prog = cand_dir / "source.py"
    prog.write_text(source)
    try:
        subprocess.run(
            [sys.executable, EVALUATE_PY, str(prog), str(n), str(cand_dir)],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        metrics = {"feasible": False, "raw_s": None, "reason": f"timeout>{timeout}s", "n": n}
        (cand_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        return metrics, cand_dir
    mfile = cand_dir / "metrics.json"
    if not mfile.exists():
        return {"feasible": False, "raw_s": None, "reason": "evaluate.py wrote no metrics", "n": n}, cand_dir
    return json.loads(mfile.read_text()), cand_dir


def _make_candidate(cand_id, parent_id, step, source, metrics, cand_dir,
                    s_ref, s_target, w_raw, w_soft):
    if not metrics["feasible"]:
        return Candidate(cand_id, parent_id, step, source, False, None,
                         reason=metrics.get("reason", ""))
    geo = np.load(cand_dir / "geometry.npz")
    ev = verifiers.geometry_vars(geo["centers"], geo["angles"], float(geo["s"]))
    quality = verifiers.quality_scores(ev)
    descriptor = div.descriptor_vector(ev, source=source)
    score = scoring.combined_score(metrics["raw_s"], quality, s_ref, s_target, w_raw, w_soft)
    return Candidate(cand_id, parent_id, step, source, True, metrics["raw_s"],
                     quality, descriptor, score, "ok")


def _select_parents(population, top_k, div_weight):
    """Greedy: repeatedly pick the feasible candidate maximising
    combined_score + div_weight * min_descriptor_distance_to_already_picked.
    div_weight=0 reduces to plain top-k by score."""
    feasible = [c for c in population if c.feasible]
    picked = []
    while len(picked) < top_k and len(picked) < len(feasible):
        remaining = [c for c in feasible if c not in picked]
        picked.append(max(
            remaining,
            key=lambda c: c.score + div_weight * div.min_distance_to(
                c.descriptor, [p.descriptor for p in picked]),
        ))
    return picked


def _truncate_population(population, pop_size, div_weight):
    """Greedy frontier: seed with best score, then add by
    score + div_weight * min_distance_to_selected. Infeasibles are dropped."""
    feasible = [c for c in population if c.feasible]
    if len(feasible) <= pop_size:
        return feasible
    ranked = sorted(feasible, key=lambda c: c.score, reverse=True)
    kept = [ranked[0]]
    rest = ranked[1:]
    while len(kept) < pop_size and rest:
        best = max(rest, key=lambda c: c.score + div_weight * div.min_distance_to(
            c.descriptor, [k.descriptor for k in kept]))
        kept.append(best)
        rest.remove(best)
    return kept


def _revise(llm, parent, n, cand_id, step, condition, use_soft, s_target,
            max_patch_attempts, timeout, mode):
    template = REVISE_PROMPT if mode == "diff" else REWRITE_PROMPT
    prompt = template.format(
        n=n, source=parent.source, timeout=timeout,
        stats=_format_stats(parent, use_soft, s_target),
    )
    messages = [{"role": "user", "content": prompt}]
    error = None
    for attempt in range(1, max_patch_attempts + 1):
        response = llm.call(
            messages,
            tags={"instance": f"n{n}", "condition": condition, "phase": "revise",
                  "step": step, "cand_id": cand_id, "parent_id": parent.cand_id,
                  "attempt": attempt, "mode": mode},
        )
        try:
            if mode == "diff":
                return diff.apply_diff(parent.source, response)
            return _extract_program(response)
        except ValueError as e:
            error = e
            messages = messages + [
                {"role": "assistant", "content": response},
                {"role": "user",
                 "content": f"The previous edit was not successful. Error: {e}\n\nTry again."},
            ]
    raise ValueError(f"no usable {mode} after {max_patch_attempts} attempts: {error}")


def search(n, llm, *, s_target, s_ref=None, num_seeds=4, num_steps=3, top_k=2,
           branching=2, pop_size=8, w_raw=1.0, w_soft=0.3, div_weight=0.3,
           timeout=300, max_patch_attempts=3, work_dir=None, condition=None):
    """Run the search for instance n. Returns (best_candidate_or_None, history)."""
    use_soft = w_soft > 0
    condition = condition or "search_qd"
    work_dir = Path(work_dir) if work_dir else (llm.run_dir / f"n{n}_{condition}")
    root_source = INITIAL_PY.read_text()

    root_metrics, root_dir = _gate(root_source, n, f"n{n}.root", work_dir, timeout)
    if not root_metrics["feasible"]:
        raise RuntimeError(f"seed program infeasible: {root_metrics.get('reason')} "
                           "-- fix initial.py before searching")
    if s_ref is None:
        s_ref = root_metrics["raw_s"]

    root = _make_candidate(f"n{n}.root", None, -1, root_source, root_metrics, root_dir,
                           s_ref, s_target, w_raw, w_soft)
    population = [root]
    history = [root]
    print(f"[n={n} {condition}] baseline s={s_ref:.4f}")

    for step in range(num_steps + 1):  # step 0 = seeds; 1..num_steps = expansions
        if step == 0:
            parents = [root] * num_seeds
        else:
            top = _select_parents(population, top_k, div_weight)
            if not top:
                break
            parents = [p for p in top for _ in range(branching)]

        def thunk(i, parent):
            mode = "diff" if i % 2 == 0 else "rewrite"  # 50/50; one of each per parent
            def run():
                cid = f"n{n}.s{step}.{i}"
                try:
                    child_source = _revise(llm, parent, n, cid, step, condition,
                                           use_soft, s_target, max_patch_attempts,
                                           timeout, mode)
                except ValueError as e:  # revision attempts exhausted
                    return Candidate(cid, parent.cand_id, step, parent.source, False, None,
                                     reason=str(e))
                metrics, cdir = _gate(child_source, n, cid, work_dir, timeout)
                return _make_candidate(cid, parent.cand_id, step, child_source, metrics,
                                       cdir, s_ref, s_target, w_raw, w_soft)
            return run

        children = run_parallel([thunk(i, p) for i, p in enumerate(parents)])
        history += children
        population = _truncate_population(population + children, pop_size, div_weight)
        n_feas = sum(c.feasible for c in children)
        best_s = min((c.raw_s for c in population if c.feasible), default=float("nan"))
        print(f"[n={n} {condition}] step {step}: {n_feas}/{len(children)} feasible; best s={best_s:.4f}")

    feasible = [c for c in history if c.feasible]
    best = min(feasible, key=lambda c: c.raw_s) if feasible else None
    return best, history


if __name__ == "__main__":
    from llm import LLM

    llm = LLM()
    best, history = search(10, llm, s_target=4.906,
                           num_seeds=2, num_steps=1, top_k=1, branching=2)  # tiny smoke
    if best is None:
        print("\nno feasible candidate found")
    else:
        print(f"\nbest: {best.cand_id}  s={best.raw_s:.4f}  score={best.score:.4f}")
    print(f"artifacts: {llm.run_dir}")
